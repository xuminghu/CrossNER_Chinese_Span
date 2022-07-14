
from src.config import get_params
from src.utils import init_experiment
from src.dataloader import get_dataloader, get_source_dataloader, get_dataloader_for_bilstmtagger
from src.trainer import BaseTrainer
from src.model import BertTagger, BiLSTMTagger
from src.coach.dataloader import get_dataloader_for_coach
from src.coach.model import EntityPredictor
from src.coach.trainer import CoachTrainer

import torch
import numpy as np
from tqdm import tqdm
import random
import torch.nn as nn

def random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def main(params):
    # initialize experiment
    logger = init_experiment(params, logger_filename=params.logger_filename)
    if params.only_index:
        params.num_tags = 3
    if params.bilstm:
        # dataloader
        dataloader_train, dataloader_dev, dataloader_test, vocab = get_dataloader_for_bilstmtagger(params)
        # bilstm-crf model
        model = BiLSTMTagger(params, vocab)
        model.cuda()
        # trainer
        trainer = BaseTrainer(params, model)
    elif params.coach:
        # dataloader
        dataloader_train, dataloader_dev, dataloader_test, vocab = get_dataloader_for_coach(params)
        # coach model
        binary_tagger = BiLSTMTagger(params, vocab)
        entity_predictor = EntityPredictor(params)
        binary_tagger.cuda()
        entity_predictor.cuda()
        # trainer
        trainer = CoachTrainer(params, binary_tagger, entity_predictor)
    else:
        # dataloader
        dataloader_train, dataloader_dev, dataloader_test = get_dataloader(params)
        # BERT-based NER Tagger
        model = BertTagger(params)
        model.cuda()
        # trainer
        trainer = BaseTrainer(params, model)

    if params.source and not params.joint:
        source_trainloader = get_source_dataloader(params.batch_size, "source",params.only_index)
        trainer.train_source(source_trainloader,"source")
    
    model.type_classifier = nn.Linear(model.hidden_dim+params.span_width_embedding_dim, params.num_target_tag)
    model = model.cuda()
    trainer.train_target(dataloader_train, dataloader_dev,dataloader_test)
    # logger.info("============== Evaluate before Target Training on Train Set ==============")
    # if params.only_index:
    #     f1_train= trainer.evaluate(dataloader_train, params.tgt_dm, use_bilstm=params.bilstm)
    #     logger.info("Evaluate on Train Set. F1: %.4f." % (f1_train))        
    # else:
    #     f1_train,prec_train,recall_train,f1_train_index,train_by_type,train_by_index,studied_type_preds = trainer.evaluate(dataloader_train, params.tgt_dm, use_bilstm=params.bilstm,type_study="person")
    #     logger.info("Evaluate on Train Set. Prec: %.4f Recall: %.4f F1: %.4f F1_index: %.4f . " % (prec_train,recall_train,f1_train,f1_train_index))
    #     logger.info("{}".format(studied_type_preds))
    #     for t in train_by_type.keys():
    #         logger.info("Entity Type %s. Prec: %.4f Recall: %.4f F1: %.4f" %(t,train_by_type[t]["precision"],train_by_type[t]["recall"],train_by_type[t]["fb1"]))
    #     for index in train_by_index.keys():
    #         logger.info("Entity Index %s. Prec: %.4f Recall: %.4f F1: %.4f" %(index,train_by_index[index]["precision"],train_by_index[index]["recall"],train_by_index[index]["fb1"]))

    # logger.info("============== Evaluate before Target Training on Dev Set ==============" )
    # if params.only_index:
    #     f1_dev = trainer.evaluate(dataloader_dev, params.tgt_dm, use_bilstm=params.bilstm)
    #     logger.info("Evaluate on Dev Set. F1: %.4f." % (f1_dev))            
    # else:
    #     f1_dev,prec_dev,recall_dev,f1_dev_index,dev_by_type,dev_by_index,studied_type_preds = trainer.evaluate(dataloader_dev, params.tgt_dm, use_bilstm=params.bilstm,type_study="person")
    #     logger.info("Evaluate on Dev Set. Prec: %.4f Recall: %.4f F1: %.4f F1_index: %.4f . " % (prec_dev,recall_dev,f1_dev,f1_dev_index))
    #     logger.info("{}".format(studied_type_preds))
    #     for t in dev_by_type.keys():
    #         logger.info("Entity Type %s. Prec: %.4f Recall: %.4f F1: %.4f" %(t,dev_by_type[t]["precision"],dev_by_type[t]["recall"],dev_by_type[t]["fb1"]))
    #     for index in dev_by_index.keys():
    #         logger.info("Entity Index %s. Prec: %.4f Recall: %.4f F1: %.4f" %(index,dev_by_index[index]["precision"],dev_by_index[index]["recall"],dev_by_index[index]["fb1"]))

    # logger.info("============== Evaluate before Target Training on Test Set ==============")
    # if params.only_index:
    #     f1_test = trainer.evaluate(dataloader_test, params.tgt_dm, use_bilstm=params.bilstm)
    #     logger.info("Evaluate on Test Set. F1: %.4f." % (f1_test))
    # else:
    #     f1_test,prec_test,recall_test,f1_test_index,test_by_type,test_by_index,studied_type_preds = trainer.evaluate(dataloader_test, params.tgt_dm, use_bilstm=params.bilstm,type_study="person")
    #     logger.info("Evaluate on Test Set. Prec: %.4f Recall: %.4f F1: %.4f F1_index: %.4f . " % (prec_test,recall_test,f1_test,f1_test_index))
    #     logger.info("{}".format(studied_type_preds))
    #     for t in test_by_type.keys():
    #         logger.info("Entity Type %s. Prec: %.4f Recall: %.4f F1: %.4f" %(t,test_by_type[t]["precision"],test_by_type[t]["recall"],test_by_type[t]["fb1"]))
    #     for index in test_by_index.keys():
    #         logger.info("Entity Index %s. Prec: %.4f Recall: %.4f F1: %.4f" %(index,test_by_index[index]["precision"],test_by_index[index]["recall"],test_by_index[index]["fb1"]))




if __name__ == "__main__":
    params = get_params()

    random_seed(params.seed)
    main(params)
