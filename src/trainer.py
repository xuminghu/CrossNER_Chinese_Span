
import torch
import torch.nn as nn

from src.conll2002_metrics import *
from src.dataloader import domain2labels, pad_token_label_id,only_index_labels
from transformers import AutoTokenizer,get_cosine_schedule_with_warmup
import os
import numpy as np
from tqdm import tqdm
import logging
import torch.nn.functional as F
from src.dataloader import pad_token_label_id

logger = logging.getLogger()

class BaseTrainer(object):
    def __init__(self, params, model):
        self.params = params
        self.model = model
        
        self.optimizer = torch.optim.AdamW(
            filter(lambda x:x.requires_grad,self.model.parameters()),
            lr = params.lr
        )
        self.loss_fn = nn.CrossEntropyLoss()
        # self.loss_fn = nn.BCELoss(reduction="none")
        # self.early_stop = params.early_stop
        self.no_improvement_num = 0
        self.best_acc = 0
        self.scheduler = None
    def train_step(self, X, indexs,raw_types,span_types,entity_positions,pad_masks,index_subword_masks,type_subword_masks,real_span_masks):
        self.model.train()

        index_logits,type_logits = self.model(X,entity_positions,pad_masks,index_subword_masks)
        index_total_mask = torch.logical_and(pad_masks,index_subword_masks)
        indexs[index_total_mask==0] = pad_token_label_id
        span_types[real_span_masks==0] = pad_token_label_id
        index_logits = index_logits.view(index_logits.size(0)*index_logits.size(1), index_logits.size(2))
        type_logits = type_logits.view(type_logits.size(0)*type_logits.size(1),type_logits.size(2))
        indexs = indexs.view(indexs.size(0)*indexs.size(1))
        span_types = span_types.view(span_types.size(0)*span_types.size(1))
        index_loss = self.loss_fn(index_logits,indexs)
        type_loss = self.loss_fn(type_logits,span_types)        

        self.optimizer.zero_grad()
        loss = self.params.alpha * index_loss + (1-self.params.alpha) * type_loss
        loss.backward()
        self.optimizer.step()
        if self.scheduler is not None:
            self.scheduler.step()
        return index_loss.item(),type_loss.item(),loss.item()
    
    def train_step_for_bilstm(self, X, lengths, y):
        self.model.train()
        preds = self.model(X)
        loss = self.model.crf_loss(preds, lengths, y)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def evaluate(self, dataloader, tgt_dm,case_study=False,type_study=None):
        self.model.eval()

        index_pred_list = []
        type_pred_list = []
        index_list = []
        type_list = []
        pbar = tqdm(enumerate(dataloader), total=len(dataloader))
        sentences = []
        # auto_tokenizer = AutoTokenizer.from_pretrained(self.params.model_name)
        for i, (X,lengths,indexs,raw_types,span_types,entity_positions,pad_masks,index_subword_masks,type_subword_masks,real_span_masks) in pbar:

            X,indexs,raw_types,span_types = X.cuda(),indexs.cuda(),raw_types.cuda(),span_types.cuda()
            entity_positions,pad_masks,index_subword_masks,type_subword_masks,real_span_masks \
                = entity_positions.cuda(),pad_masks.cuda(),index_subword_masks.cuda(),type_subword_masks.cuda(),real_span_masks.cuda()
            
            index_total_mask = torch.logical_and(pad_masks,index_subword_masks)
            type_total_mask = torch.logical_and(pad_masks,type_subword_masks)

            index_logits,type_logits,entity_positions,valid_span_masks = self.model(X,entity_positions,pad_masks,index_subword_masks,evaluate = True)
            index_preds = index_logits.argmax(dim=-1)

            type_logits = type_logits.argmax(dim=-1)
            type_preds = torch.zeros_like(raw_types).to(raw_types.device)
            # entity_positions = torch.masked_select(entity_positions,valid_span_masks.unsqueeze(-1).repeat(1,1,2))
            # entity_positions = entity_positions.view(-1,2)[:,0].view(-1)
            for j in range(valid_span_masks.shape[0]):
                for k in range(torch.sum(valid_span_masks[j])):
                    start,end = entity_positions[j][k][0],entity_positions[j][k][1]
                    type_preds[j,start:end+1] = type_logits[j,k]
            
            type_preds[type_total_mask==0] = pad_token_label_id

            index_pred_list.extend(index_preds.data.cpu().numpy())
            type_pred_list.extend(type_preds.data.cpu().numpy())
            

            indexs[index_total_mask==0] = pad_token_label_id
            raw_types[type_total_mask==0] = pad_token_label_id

            index_list.extend(indexs.cpu().data.numpy()) # y is a list
            type_list.extend(raw_types.cpu().data.numpy())
            # if i == 0:
            #     logging.info("Batch Entity Positions")
            #     logging.info("{}".format(entity_positions))
            #     logging.info("Batch Type Pred")
            #     logging.info("{}".format(type_preds))
            #     logging.info("Batch Ground Truth Type")
            #     logging.info("{}".format(raw_types))

            if case_study:
                X = X.data.cpu()
                for i in range(X.shape[0]):
                    sentences.append(X[i].numpy().tolist())
        # concatenation
        index_pred_list = np.concatenate(index_pred_list, axis=0)   # (length, num_tag)
        type_pred_list = np.concatenate(type_pred_list, axis=0)   # (length, num_tag)

        # if not use_bilstm:
        #     index_pred_list = np.argmax(index_pred_list, axis=1)
        #     type_pred_list = np.argmax(type_pred_list,axis=1)
        index_list = np.concatenate(index_list, axis=0)
        type_list = np.concatenate(type_list, axis=0)
        # calcuate f1 score
        index_pred_list = list(index_pred_list)
        type_pred_list = list(type_pred_list)
        index_list = list(index_list)
        type_list = list(type_list)
        lines = []

        if case_study:
            num_sentence = 0
            num_tokens = 0
            flags = []
            flag = []
        for pred_index, pred_type,gold_index,gold_type in zip(index_pred_list, type_pred_list,index_list,type_list):
            gold_index = int(gold_index)
            if gold_index != pad_token_label_id:
                pred_index = only_index_labels[pred_index]
                if pred_type == pad_token_label_id:
                    pred_type = ""
                else:
                    pred_type = domain2labels[tgt_dm if not self.params.only_index else "only_index"][pred_type]

                if gold_type != pad_token_label_id:
                    gold_type = domain2labels[tgt_dm if not self.params.only_index else "only_index"][gold_type]
                else:
                    gold_type = ""
                pred_token = pred_index + "-" + pred_type
                gold_index = only_index_labels[gold_index]
                gold_token = gold_index + "-" + gold_type
                lines.append("w" + " " + pred_token + " " + gold_token)
                if case_study:
                    flag.append(True)
            else:
                if case_study:
                    flag.append(False)
            if case_study:
                num_tokens +=1
                if (num_tokens >=len(sentences[num_sentence])):
                    num_tokens = 0
                    num_sentence += 1
                    flags.append(flag)
                    flag = []
        if case_study:
            auto_tokenizer = AutoTokenizer.from_pretrained(self.params.model_name)
            results = conll2002_measure(lines,sentences = sentences,flags = flags,tokenizer = auto_tokenizer,type_study=type_study)
        else:
            results = conll2002_measure(lines,type_study=type_study)
        if self.params.only_index:
            return results["fb1"]
        else:
            f1 = results["fb1"]
            precision = results["precision"]
            recall = results["recall"]
            f1_index = results["fb1_index"]
            by_type = results["by_type"]
            by_index = results["by_index"]
            
            if case_study:
                cases = results["cases"]
                return f1,precision,recall,f1_index,by_type,by_index,cases
            elif type_study is not None:
                studied_type_preds = results["studied_type_preds"]
                return f1,precision,recall,f1_index,by_type,by_index,studied_type_preds
            else:
                return f1,precision,recall,f1_index,by_type,by_index    
                
    def train_source(self, dataloader_train, tgt_dm):
        logger.info("Pretraining on source NER dataset ...")
        no_improvement_num = 0
        best_f1 = 0
        # f1_dev,prec_dev,recall_dev,f1_dev_index,dev_by_type,dev_by_index = self.evaluate(dataloader_dev, tgt_dm)

        for e in range(self.params.epoch):
            logger.info("============== epoch %d ==============" % e)
            loss_list = []
        
            pbar = tqdm(enumerate(dataloader_train), total=len(dataloader_train))
            for i, (X, lengths,indexs,raw_types,span_types,entity_positions,pad_masks,index_subword_masks,type_subword_masks,real_span_masks) in pbar:
                X,indexs,raw_types,span_types = X.cuda(), indexs.cuda(),raw_types.cuda(),span_types.cuda()
                entity_positions,pad_masks,index_subword_masks,type_subword_masks,real_span_masks = \
                     entity_positions.cuda(),pad_masks.cuda(),index_subword_masks.cuda(),type_subword_masks.cuda(),real_span_masks.cuda()

                index_loss,type_loss,loss = self.train_step(X, indexs,raw_types,span_types,entity_positions,pad_masks,index_subword_masks,type_subword_masks,real_span_masks)
                loss_list.append(loss)
                pbar.set_description("(Epoch {}) LOSS:{:.4f}".format(e, np.mean(loss_list)))

            logger.info("Finish training epoch %d. loss: %.4f" % (e, np.mean(loss_list)))

            logger.info("============== Evaluate epoch %d on Train Set ==============" % e)
            if self.params.only_index:
                f1_train = self.evaluate(dataloader_train,tgt_dm)
                logger.info("Evaluate on Dev Set. F1: %.4f. " % (f1_train))
            else:
                f1_train,prec_train,recall_train,f1_train_index,train_by_type,train_by_index = self.evaluate(dataloader_train, tgt_dm)
                logger.info("Evaluate on Train Set. Prec: %.4f Recall: %.4f F1: %.4f F1_index: %.4f . " % (prec_train,recall_train,f1_train,f1_train_index))
                for t in train_by_type.keys():
                    logger.info("Entity Type %s. Prec: %.4f Recall: %.4f F1: %.4f" %(t,train_by_type[t]["precision"],train_by_type[t]["recall"],train_by_type[t]["fb1"]))
                for index in train_by_index.keys():
                    logger.info("Entity Index %s. Prec: %.4f Recall: %.4f F1: %.4f" %(index,train_by_index[index]["precision"],train_by_index[index]["recall"],train_by_index[index]["fb1"]))
            if f1_train > best_f1:
                logger.info("Found better model!!")
                best_f1 = f1_train
                no_improvement_num = 0
            else:
                no_improvement_num += 1
                logger.info("No better model found (%d/%d)" % (no_improvement_num, 1))

            # if no_improvement_num >= 1:
            #     break
            if e >= 2:
                break

    def train_target(self,dataloader_train,dataloader_dev,dataloader_test):
        train_num_steps = self.params.epoch * len(dataloader_train)
        self.scheduler = get_cosine_schedule_with_warmup(self.optimizer,self.params.warmup*train_num_steps,train_num_steps)
        best_f1 = 0
        logger.info("Training on target domain ...")        
        for e in range(self.params.epoch):
            logger.info("============== epoch %d ==============" % e)
            
            pbar = tqdm(enumerate(dataloader_train), total=len(dataloader_train))
            loss_list = []
            for i, (X,lengths,indexs,raw_types,span_types,entity_positions,pad_masks,index_subword_masks,type_subword_masks,real_span_masks) in pbar:
                X, indexs,raw_types,span_types = X.cuda(), indexs.cuda(),raw_types.cuda(),span_types.cuda()
                entity_positions,pad_masks,index_subword_masks,type_subword_masks,real_span_masks = \
                    entity_positions.cuda(),pad_masks.cuda(),index_subword_masks.cuda(),type_subword_masks.cuda(),real_span_masks.cuda()
                
                index_loss,type_loss,loss = self.train_step(X, indexs,raw_types,span_types,entity_positions,pad_masks,index_subword_masks,type_subword_masks,real_span_masks)
                loss_list.append(loss)
                pbar.set_description("(Epoch {}) LOSS:{:.4f}".format(e, np.mean(loss_list)))

            logger.info("Finish training epoch %d. loss: %.4f" % (e, np.mean(loss_list)))

            logger.info("============== Evaluate epoch %d on Train Set ==============" % e)
            if self.params.only_index:
                f1_train= self.evaluate(dataloader_train, self.params.tgt_dm)
                logger.info("Evaluate on Train Set. F1: %.4f." % (f1_train))        
            else:
                f1_train,prec_train,recall_train,f1_train_index,train_by_type,train_by_index,studied_type_preds = self.evaluate(dataloader_train, self.params.tgt_dm, type_study="person")
                logger.info("Evaluate on Train Set. Prec: %.4f Recall: %.4f F1: %.4f F1_index: %.4f . " % (prec_train,recall_train,f1_train,f1_train_index))
                logger.info("{}".format(studied_type_preds))
                for t in train_by_type.keys():
                    logger.info("Entity Type %s. Prec: %.4f Recall: %.4f F1: %.4f" %(t,train_by_type[t]["precision"],train_by_type[t]["recall"],train_by_type[t]["fb1"]))
                for index in train_by_index.keys():
                    logger.info("Entity Index %s. Prec: %.4f Recall: %.4f F1: %.4f" %(index,train_by_index[index]["precision"],train_by_index[index]["recall"],train_by_index[index]["fb1"]))
            if dataloader_dev is not None:
                logger.info("============== Evaluate epoch %d on Dev Set ==============" % e)
                if self.params.only_index:
                    f1_dev = self.evaluate(dataloader_dev, self.tgt_dm)
                    logger.info("Evaluate on Dev Set. F1: %.4f." % (f1_dev))            
                else:
                    f1_dev,prec_dev,recall_dev,f1_dev_index,dev_by_type,dev_by_index,studied_type_preds = self.evaluate(dataloader_dev, self.params.tgt_dm,type_study="person")
                    logger.info("Evaluate on Dev Set. Prec: %.4f Recall: %.4f F1: %.4f F1_index: %.4f . " % (prec_dev,recall_dev,f1_dev,f1_dev_index))
                    logger.info("{}".format(studied_type_preds))
                    for t in dev_by_type.keys():
                        logger.info("Entity Type %s. Prec: %.4f Recall: %.4f F1: %.4f" %(t,dev_by_type[t]["precision"],dev_by_type[t]["recall"],dev_by_type[t]["fb1"]))
                    for index in dev_by_index.keys():
                        logger.info("Entity Index %s. Prec: %.4f Recall: %.4f F1: %.4f" %(index,dev_by_index[index]["precision"],dev_by_index[index]["recall"],dev_by_index[index]["fb1"]))

            logger.info("============== Evaluate epoch %d on Test Set ==============" % e)
            if self.params.only_index:
                f1_test = self.evaluate(dataloader_test, self.params.tgt_dm)
                logger.info("Evaluate on Test Set. F1: %.4f." % (f1_test))
            else:
                f1_test,prec_test,recall_test,f1_test_index,test_by_type,test_by_index,studied_type_preds = self.evaluate(dataloader_test, self.params.tgt_dm,type_study="person")
                logger.info("Evaluate on Test Set. Prec: %.4f Recall: %.4f F1: %.4f F1_index: %.4f . " % (prec_test,recall_test,f1_test,f1_test_index))
                logger.info("{}".format(studied_type_preds))
                for t in test_by_type.keys():
                    logger.info("Entity Type %s. Prec: %.4f Recall: %.4f F1: %.4f" %(t,test_by_type[t]["precision"],test_by_type[t]["recall"],test_by_type[t]["fb1"]))
                for index in test_by_index.keys():
                    logger.info("Entity Index %s. Prec: %.4f Recall: %.4f F1: %.4f" %(index,test_by_index[index]["precision"],test_by_index[index]["recall"],test_by_index[index]["fb1"]))
                            
            if f1_test > best_f1:
                logger.info("Found better model!!")
                best_f1 = f1_test
                # no_improvement_num = 0
                # trainer.save_model()
            else:
                # no_improvement_num += 1
                logger.info("No better model found")

            # if no_improvement_num >= params.early_stop:
            #     if params.case_study:
            #         f1_test,f1_test_index,cases = trainer.evaluate(dataloader_test, params.tgt_dm, use_bilstm=params.bilstm,case_study=True)
            #         with open(params.tgt_dm+"_casestudy.log",'w') as f:
            #             for (raw,pred,gold) in cases:
            #                 f.write("raw:{}\npred:{}\ngold:{}\n\n".format(raw,pred,gold))
            #     break

    def save_model(self):
        """
        save the best model
        """
        saved_path = os.path.join(self.params.dump_path, "best_finetune_model.pth")
        torch.save({
            "model": self.model,
        }, saved_path)
        logger.info("Best model has been saved to %s" % saved_path)
    

