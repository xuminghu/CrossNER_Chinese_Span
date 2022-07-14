
import torch
import torch.nn as nn
import torch.utils.data as data
from torch.utils.data import DataLoader
from src.conll2002_metrics import start_of_chunk,end_of_chunk
from tqdm import tqdm
import random
import logging
import os
logger = logging.getLogger()

# from transformers import BertTokenizer
# bert_tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
from src.config import get_params
params = get_params()
from transformers import AutoTokenizer
auto_tokenizer = AutoTokenizer.from_pretrained(params.model_name)
pad_token_label_id = nn.CrossEntropyLoss().ignore_index

# politics_labels = ['O', 'B-country', 'B-politician', 'I-politician', 'B-election', 'I-election', 'B-person', 'I-person', 'B-organisation', 'I-organisation', 'B-location', 'B-misc', 'I-location', 'I-country', 'I-misc', 'B-politicalparty', 'I-politicalparty', 'B-event', 'I-event']
# science_labels = ['O', 'B-scientist', 'I-scientist', 'B-person', 'I-person', 'B-university', 'I-university', 'B-organisation', 'I-organisation', 'B-country', 'I-country', 'B-location', 'I-location', 'B-discipline', 'I-discipline', 'B-enzyme', 'I-enzyme', 'B-protein', 'I-protein', 'B-chemicalelement', 'I-chemicalelement', 'B-chemicalcompound', 'I-chemicalcompound', 'B-astronomicalobject', 'I-astronomicalobject', 'B-academicjournal', 'I-academicjournal', 'B-event', 'I-event', 'B-theory', 'I-theory', 'B-award', 'I-award', 'B-misc', 'I-misc']
# music_labels = ['O', 'B-musicgenre', 'I-musicgenre', 'B-song', 'I-song', 'B-band', 'I-band', 'B-album', 'I-album', 'B-musicalartist', 'I-musicalartist', 'B-musicalinstrument', 'I-musicalinstrument', 'B-award', 'I-award', 'B-event', 'I-event', 'B-country', 'I-country', 'B-location', 'I-location', 'B-organisation', 'I-organisation', 'B-person', 'I-person', 'B-misc', 'I-misc']
# literature_labels = ["O", "B-book", "I-book", "B-writer", "I-writer", "B-award", "I-award", "B-poem", "I-poem", "B-event", "I-event", "B-magazine", "I-magazine", "B-literarygenre", "I-literarygenre", 'B-country', 'I-country', "B-person", "I-person", "B-location", "I-location", 'B-organisation', 'I-organisation', 'B-misc', 'I-misc']
# ai_labels = ["O", "B-field", "I-field", "B-task", "I-task", "B-product", "I-product", "B-algorithm", "I-algorithm", "B-researcher", "I-researcher", "B-metrics", "I-metrics", "B-programlang", "I-programlang", "B-conference", "I-conference", "B-university", "I-university", "B-country", "I-country", "B-person", "I-person", "B-organisation", "I-organisation", "B-location", "I-location", "B-misc", "I-misc"]
only_index_labels = ["B","I","O"]
source_labels = ["教育","地点","软件","事件","文化","法律法规","时间与日历","品牌","人物","自然地理","工作","网站","组织","车辆","诊断与治疗","疾病和症状","奖项","生物","食物","游戏","星座","虚拟事物","药物"]
address_labels = ["方位","行政村或社区","开发区","乡镇街道","子兴趣点","房屋编号","地市","省份","路名","兴趣点","单元号","村子组别","区县","楼层号","距离","路口","路号"]
domain2labels = {"source": source_labels, "address": address_labels,"only_index":only_index_labels}

def read_ner(datapath, tgt_dm,only_index = False):
    inputs, index_labels, raw_type_labels,span_type_labels,entity_positions,index_subword_masks,type_subword_masks = [],[],[],[],[],[],[]
    with open(datapath, "r") as fr:
        token_list, index_label_list,raw_type_label_list,span_type_label_list,entity_pos_list,index_subword_mask,type_subword_mask = [], [], [],[],[],[],[]
        last_index_label = "O"
        last_type_label = ""
        entity_num = 0
        token_idx = 0
        
        for i, line in enumerate(fr):
            line = line.strip()
            if line == "":
                if len(token_list) > 0:
                    assert len(token_list) == len(index_label_list)
                    assert len(token_list) == len(raw_type_label_list)
                    if entity_num != len(entity_pos_list):
                        entity_pos_list[-1][1] = len(token_list)
                        span_type_label_list.append(domain2labels[tgt_dm].index(type_label))
                    inputs.append([auto_tokenizer.cls_token_id] + token_list + [auto_tokenizer.sep_token_id])
                    index_labels.append([pad_token_label_id] + index_label_list + [pad_token_label_id])
                    raw_type_labels.append([pad_token_label_id] + raw_type_label_list + [pad_token_label_id])
                    span_type_labels.append(span_type_label_list)
                    entity_positions.append(entity_pos_list)
                    index_subword_masks.append([1] + index_subword_mask + [1])
                    type_subword_masks.append([1] + type_subword_mask + [1])
                token_list, index_label_list,raw_type_label_list,span_type_label_list,entity_pos_list,index_subword_mask,type_subword_mask = [], [], [],[],[],[],[]
                last_index_label = "O"
                last_type_label = ""
                entity_num = 0
                token_idx = 0            
                continue
            
            splits = line.split(" ")
            token = splits[0]
            label = splits[1]
            if only_index:
                label = label.split("-")[0]
            else:
                index_label = label.split("-")[0]
                if index_label == "O":
                    type_label = ""
                else:
                    type_label = label.split("-")[1]
            subs_ = auto_tokenizer.tokenize(token)
            if len(subs_) > 0:
                if end_of_chunk(last_index_label,index_label,last_type_label,type_label):
                    entity_pos_list[entity_num][1] = token_idx
                    entity_num += 1
                    span_type_label_list.append(domain2labels[tgt_dm].index(last_type_label))
                if start_of_chunk(last_index_label,index_label,last_type_label,type_label):
                    entity_pos_list.append([token_idx+1,token_idx+1])
                token_idx += len(subs_)
                last_index_label = index_label
                last_type_label = type_label

                # index_label_list.extend([only_index_labels.index(index_label)] + [pad_token_label_id] * (len(subs_) - 1))
                index_label_list.extend([only_index_labels.index(index_label)] * len(subs_))
                index_subword_mask.extend([1] + [0] * (len(subs_)-1))
                if type_label == "":
                    raw_type_label_list.extend([pad_token_label_id]* len(subs_))
                    type_subword_mask.extend([0]*len(subs_))
                else:
                    t = domain2labels[tgt_dm if not only_index else "only_index"].index(type_label)
                    raw_type_label_list.extend([t] * (len(subs_)))
                    type_subword_mask.extend([1]+[0]*(len(subs_)-1))
                token_list.extend(auto_tokenizer.convert_tokens_to_ids(subs_))
            else:
                print("length of subwords for %s is zero; its label is %s" % (token, label))

    return inputs, index_labels,raw_type_labels,span_type_labels,entity_positions,index_subword_masks,type_subword_masks,


def read_ner_for_bilstm(datapath, tgt_dm, vocab):
    inputs, labels = [], []
    with open(datapath, "r") as fr:
        token_list, label_list = [], []
        for i, line in enumerate(fr):
            line = line.strip()
            if line == "":
                if len(token_list) > 0:
                    assert len(token_list) == len(label_list)
                    inputs.append(token_list)
                    labels.append(label_list)
                
                token_list, label_list = [], []
                continue
            
            splits = line.split("\t")
            token = splits[0]
            label = splits[1]
            
            token_list.append(vocab.word2index[token])
            label_list.append(domain2labels[tgt_dm].index(label))

    return inputs, labels



class Dataset(data.Dataset):
    def __init__(self, inputs, index_labels,raw_type_labels,span_type_labels,entity_positions,index_subword_masks,type_subword_masks):
        self.X = inputs
        self.index = index_labels
        self.raw_type = raw_type_labels
        self.span_type = span_type_labels  
        self.entity_positions = entity_positions
        self.index_subword_masks = index_subword_masks
        self.type_subword_masks = type_subword_masks
        
    def __getitem__(self, index):
        return self.X[index], self.index[index],self.raw_type[index],self.span_type[index],self.entity_positions[index],self.index_subword_masks[index],self.type_subword_masks[index]

    def __len__(self):
        return len(self.X)


PAD_INDEX = 0
class Vocab():
    def __init__(self):
        self.word2index = {"PAD": PAD_INDEX}
        self.index2word = {PAD_INDEX: "PAD"}
        self.n_words = 1

    def index_words(self, word_list):
        for word in word_list:
            if word not in self.word2index:
                self.word2index[word] = self.n_words
                self.index2word[self.n_words] = word
                self.n_words += 1

def get_vocab(path):
    vocabulary = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            vocabulary.append(line)
    return vocabulary


def collate_fn(data):
    X, index_labels,raw_type_labels,span_type_labels,entity_positions,index_subword_masks,type_subword_masks = zip(*data)
    lengths = [len(bs_x) for bs_x in X]
    max_lengths = max(lengths)
    max_entity_num = params.max_entity_number

    padded_seqs = torch.LongTensor(len(X), max_lengths).fill_(auto_tokenizer.pad_token_id)
    padded_indexs = torch.LongTensor(len(X), max_lengths).fill_(pad_token_label_id)
    padded_raw_types = torch.LongTensor(len(X), max_lengths).fill_(pad_token_label_id)

    padded_span_types = torch.LongTensor(len(X),max_entity_num).fill_(pad_token_label_id)
    padded_entity_positions = torch.LongTensor(len(X),max_entity_num,2).fill_(0)
    real_span_masks = torch.LongTensor(len(X),max_entity_num).fill_(0)

    padded_index_subword_masks = torch.LongTensor(len(X),max_lengths).fill_(1)
    padded_type_subword_masks = torch.LongTensor(len(X),max_lengths).fill_(1)
    
    pad_masks = torch.LongTensor(len(X),max_lengths).fill_(1)

    for i, (seq, indexs,raw_types,span_types,entity_position,index_subword_mask,type_subword_mask) in enumerate(zip(X, index_labels,raw_type_labels,span_type_labels,entity_positions,index_subword_masks,type_subword_masks)):
        length = lengths[i]
        entity_num = len(entity_position)
        padded_seqs[i, :length] = torch.LongTensor(seq)
        padded_indexs[i, :length] = torch.LongTensor(indexs)
        padded_raw_types[i,:length] = torch.LongTensor(raw_types)
        if entity_num != 0:
            padded_span_types[i,:entity_num] = torch.LongTensor(span_types)
            padded_entity_positions[i,:entity_num] = torch.LongTensor(entity_position)

        real_span_masks[i,:entity_num] = 1
        pad_masks[i,0] = 0
        pad_masks[i,length:] = 0

        padded_index_subword_masks[i,:length] = torch.LongTensor(index_subword_mask)
        padded_type_subword_masks[i,:length] = torch.LongTensor(type_subword_mask)

    return padded_seqs,lengths, padded_indexs,padded_raw_types,padded_span_types,padded_entity_positions,pad_masks,padded_index_subword_masks,padded_type_subword_masks,real_span_masks


def collate_fn_for_bilstm(data):
    X, y = zip(*data)
    lengths = [len(bs_x) for bs_x in X]
    max_lengths = max(lengths)
    padded_seqs = torch.LongTensor(len(X), max_lengths).fill_(PAD_INDEX)
    for i, seq in enumerate(X):
        length = lengths[i]
        padded_seqs[i, :length] = torch.LongTensor(seq)

    lengths = torch.LongTensor(lengths)
    return padded_seqs, lengths, y


def get_dataloader_for_bilstmtagger(params):
    vocab_src = get_vocab("ner_data/conll2003/vocab.txt")
    vocab_tgt = get_vocab("ner_data/%s/vocab.txt" % params.tgt_dm)
    vocab = Vocab()
    vocab.index_words(vocab_src)
    vocab.index_words(vocab_tgt)

    logger.info("Load training set data ...")
    conll_inputs_train, conll_labels_train = read_ner_for_bilstm("ner_data/conll2003/train.txt", params.tgt_dm, vocab)
    inputs_train, labels_train = read_ner_for_bilstm("ner_data/%s/train.txt" % params.tgt_dm, params.tgt_dm, vocab)
    inputs_train = inputs_train * 10 + conll_inputs_train
    labels_train = labels_train * 10 + conll_labels_train

    logger.info("Load dev set data ...")
    inputs_dev, labels_dev = read_ner_for_bilstm("ner_data/%s/dev.txt" % params.tgt_dm, params.tgt_dm, vocab)

    logger.info("Load test set data ...")
    inputs_test, labels_test = read_ner_for_bilstm("ner_data/%s/test.txt" % params.tgt_dm, params.tgt_dm, vocab)

    logger.info("train size: %d; dev size %d; test size: %d;" % (len(inputs_train), len(inputs_dev), len(inputs_test)))

    dataset_train = Dataset(inputs_train, labels_train)
    dataset_dev = Dataset(inputs_dev, labels_dev)
    dataset_test = Dataset(inputs_test, labels_test)
    
    dataloader_train = DataLoader(dataset=dataset_train, batch_size=params.batch_size, shuffle=True, collate_fn=collate_fn_for_bilstm)
    dataloader_dev = DataLoader(dataset=dataset_dev, batch_size=params.batch_size, shuffle=False, collate_fn=collate_fn_for_bilstm)
    dataloader_test = DataLoader(dataset=dataset_test, batch_size=params.batch_size, shuffle=False, collate_fn=collate_fn_for_bilstm)

    return dataloader_train, dataloader_dev, dataloader_test, vocab


def load_corpus(tgt_dm):
    print("Loading corpus ...")
    data_path = "enwiki_corpus/%s_removebracket.tok" % tgt_dm
    sent_list = []
    with open(data_path, "r") as fr:
        for i, line in tqdm(enumerate(fr)):
            line = line.strip()
            sent_list.append(line)
    return sent_list


def get_dataloader(params):
    logger.info("Load training set data")
    inputs_train, index_labels_train,raw_type_labels_train,span_type_labels_train,\
    entity_positions_train,index_subword_masks_train,type_subword_masks_train\
            = read_ner("ner_data/%s/seed%d" % (params.tgt_dm,params.train_seed), params.tgt_dm,params.only_index)
    if params.n_samples != -1:
        logger.info("Few-shot on %d samples" % params.n_samples)
        inputs_train = inputs_train[:params.n_samples]
        index_labels_train = index_labels_train[:params.n_samples]
        type_labels_train = type_labels_train[:params.n_samples]
    logger.info("Load development set data")
    inputs_dev = None
    if os.path.exists("ner_data/%s/dev" % params.tgt_dm):
        inputs_dev, index_labels_dev,raw_type_labels_dev,span_type_labels_dev,\
        entity_positions_dev,index_subword_masks_dev,type_subword_masks_dev \
                = read_ner("ner_data/%s/dev" % params.tgt_dm, params.tgt_dm,params.only_index)
    
    logger.info("Load test set data")
    inputs_test, index_labels_test,raw_type_labels_test,span_type_labels_test,\
    entity_positions_test,index_subword_masks_test,type_subword_masks_test \
            = read_ner("ner_data/%s/test" % params.tgt_dm, params.tgt_dm,params.only_index)

    logger.info("train size: %d; test size: %d;" % (len(inputs_train),len(inputs_test)))

    dataset_train = Dataset(inputs_train, index_labels_train,raw_type_labels_train,span_type_labels_train,\
        entity_positions_train,index_subword_masks_train,type_subword_masks_train)
    if inputs_dev is not None:
        dataset_dev = Dataset(inputs_dev, index_labels_dev,raw_type_labels_dev,span_type_labels_dev,\
            entity_positions_dev,index_subword_masks_dev,type_subword_masks_dev)
    else:
        dataset_dev = None
    dataset_test = Dataset(inputs_test, index_labels_test,raw_type_labels_test,span_type_labels_test,\
        entity_positions_test,index_subword_masks_test,type_subword_masks_test)
    
    dataloader_train = DataLoader(dataset=dataset_train, batch_size=params.batch_size, shuffle=True, collate_fn=collate_fn)
    if dataset_dev is not None:
        dataloader_dev = DataLoader(dataset=dataset_dev, batch_size=params.batch_size, shuffle=False, collate_fn=collate_fn)
    else:
        dataloader_dev = None
    dataloader_test = DataLoader(dataset=dataset_test, batch_size=params.batch_size, shuffle=False, collate_fn=collate_fn)

    return dataloader_train, dataloader_dev, dataloader_test


def get_source_dataloader(batch_size, tgt_dm="source",only_index=False):
    inputs_train, index_labels_train,raw_type_labels_train,span_type_labels_train,\
    entity_positions_train,index_subword_masks_train,type_subword_masks_train\
        = read_ner("ner_data/source/train", tgt_dm,only_index)

    logger.info("source dataset: train size: %d" % (len(inputs_train)))

    dataset_train = Dataset(inputs_train, index_labels_train,raw_type_labels_train,span_type_labels_train,\
        entity_positions_train,index_subword_masks_train,type_subword_masks_train)    
    dataloader_train = DataLoader(dataset=dataset_train, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

    return dataloader_train


