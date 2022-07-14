
import torch
import torch.nn as nn
from torch.nn import functional as F
# from transformers import BertModel, BertTokenizer
from transformers import AutoConfig
from transformers import AutoModelWithLMHead
from src.dataloader import only_index_labels
from src.conll2002_metrics import end_of_chunk, start_of_chunk
from src.utils import load_embedding
from allennlp.modules.span_extractors import MaxPoolingSpanExtractor,EndpointSpanExtractor,SelfAttentiveSpanExtractor

import logging
logger = logging.getLogger()

class MLPHead(nn.Module):
    def __init__(self,input_size,hidden_size,output_size,dropout,num_layers):
        super(MLPHead,self).__init__()
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout)
        if num_layers > 1:
            self.layers = nn.ModuleList(
                [nn.Linear(input_size,hidden_size)] +
                [nn.Linear(hidden_size,hidden_size) for i in range(num_layers-2)] +
                [nn.Linear(hidden_size,output_size)]
            )
        else:
            self.layer = nn.Linear(input_size,output_size)
            self.layers = None
        self.layer_norms = nn.ModuleList([nn.LayerNorm(hidden_size) for i in range(num_layers-1)])
    def forward(self,X):
        # X shape (batch_size,seq_len,input_size)
        if self.layers is None:
            return self.layer(X)
        for i in range(len(self.layers)-1):
            X = self.layers[i](X)
            X = self.relu(X)
            X = self.dropout(X)
            X = self.layer_norms[i](X)
        output = self.layers[len(self.layers)-1](X)
        return output

class BertTagger(nn.Module):
    def __init__(self, params):
        super(BertTagger, self).__init__()
        self.num_tag = params.num_source_tag
        self.hidden_dim = params.hidden_dim
        self.max_entity_number = params.max_entity_number
        config = AutoConfig.from_pretrained(params.model_name)
        config.output_hidden_states = True
        # self.bert = BertModel.from_pretrained("bert-base-cased")
        self.model = AutoModelWithLMHead.from_pretrained(params.model_name, config=config)
        self.span_extractor = MaxPoolingSpanExtractor(
            input_dim = self.hidden_dim,
            num_width_embeddings = params.max_span_width,
            span_width_embedding_dim =  params.span_width_embedding_dim,
            bucket_widths = True
        )
        if params.ckpt != "":
            logger.info("Reloading model from %s" % params.ckpt)
            model_ckpt = torch.load(params.ckpt)
            self.model.load_state_dict(model_ckpt)
        self.index_classifier = nn.Linear(self.hidden_dim,3)
        # self.index_classifier = MLPHead(self.hidden_dim,self.hidden_dim,3,0.15,3)
        self.type_classifier = nn.Linear(self.hidden_dim+params.span_width_embedding_dim, self.num_tag)
        # self.type_classifier = MLPHead(self.hidden_dim,self.hidden_dim,self.num_tag,0.15,3)
        # self.sigmoid = nn.Sigmoid()
        # self.index_crf = CRF(3)
        # self.type_crf = CRF(self.num_tag)

    def forward(self, X,entity_positions,pad_masks,index_subword_masks,evaluate=False):
        attention_mask = pad_masks.float()
        attention_mask[:,0] = 1 
        outputs = self.model(input_ids = X,attention_mask=attention_mask) # a tuple ((bsz,seq_len,hidden_dim), (bsz, hidden_dim))
        outputs = outputs[1][-1] # (bsz, seq_len, hidden_dim)

        index_logits = self.index_classifier(outputs)
        index_preds = torch.argmax(index_logits,dim=-1)
        if evaluate:
            entity_positions,valid_span_masks = self.detect_span(index_preds,pad_masks,index_subword_masks)
        span_embeddings = self.span_extractor(outputs,entity_positions)
        type_logits = self.type_classifier(span_embeddings)

        if evaluate:
            return index_logits,type_logits,entity_positions,valid_span_masks
        else:
            return index_logits,type_logits
        # return index_pred,type_pred

    def detect_span(self,index_preds,pad_masks,index_subword_masks):
        total_mask = torch.logical_and(pad_masks,index_subword_masks)
        valid_indices = torch.nonzero(total_mask)
        batch_size,seq_len = index_preds.shape
        entity_positions = torch.empty(batch_size,self.max_entity_number,2,dtype=torch.long).fill_(0).to(index_preds.device)
        valid_span_mask = torch.zeros(batch_size,self.max_entity_number,dtype=torch.bool).to(index_preds.device)
        
        sent_num = 0
        last_tag = "O"
        last_type = ""
        last_token_idx = 0
        entity_num = 0
        still_open = False
        for i in range(len(valid_indices)):
            sent_idx , token_idx = valid_indices[i][0],valid_indices[i][1]
            if sent_idx != sent_num:
                sent_num += 1
                if still_open:
                    idx = torch.arange(0,total_mask.shape[1]).to(total_mask.device)
                    entity_positions[sent_idx-1,entity_num,1] = torch.argmax(total_mask[sent_idx-1] * idx)
                    assert entity_positions[sent_idx-1,entity_num,0] <= entity_positions[sent_idx-1,entity_num,1]
                    entity_num += 1
                    
                last_tag = "O"
                last_type = ""
                last_token_idx = 0
                valid_span_mask[sent_idx-1,:entity_num] = 1
                entity_num = 0 
                still_open=False

            current_tag = only_index_labels[index_preds[sent_idx,token_idx].item()]
            current_type = ""
            if end_of_chunk(last_tag,current_tag,last_type,current_type):
                last_token_idx += 1
                while last_token_idx < seq_len and index_subword_masks[sent_idx, last_token_idx] == 0 and pad_masks[sent_idx,last_token_idx] == 1:
                    last_token_idx += 1
                entity_positions[sent_idx,entity_num,1] = last_token_idx-1
                assert entity_positions[sent_idx,entity_num,0] <= entity_positions[sent_idx,entity_num,1]
                entity_num += 1
                still_open = False
            if start_of_chunk(last_tag,current_tag,last_type,current_type):
                assert entity_positions[sent_idx,entity_num,0] == 0
                entity_positions[sent_idx,entity_num,0] = token_idx
                still_open = True

            last_tag = current_tag
            last_type = current_type
            last_token_idx = token_idx
        
        if still_open:
            idx = torch.arange(0,total_mask.shape[1]).to(total_mask.device)
            entity_positions[sent_num,entity_num,1] = torch.argmax(total_mask[sent_num] * idx)
            assert entity_positions[sent_num,entity_num,0] <= entity_positions[sent_num,entity_num,1]
            entity_num += 1
        assert (entity_positions[:,:,0] <= entity_positions[:,:,1]).all()
        return entity_positions,valid_span_mask

class BiLSTMTagger(nn.Module):
    def __init__(self, params, vocab):
        super(BiLSTMTagger, self).__init__()
        self.embedding = nn.Embedding(vocab.n_words, params.emb_dim, padding_idx=0)
        embedding = load_embedding(vocab, params.emb_dim, params.emb_file, params.usechar)
        self.embedding.weight.data.copy_(torch.FloatTensor(embedding))
        
        self.dropout = params.dropout
        self.lstm = nn.LSTM(params.emb_dim, params.lstm_hidden_dim, num_layers=params.n_layer, dropout=params.dropout, bidirectional=True, batch_first=True)
        
        self.linear = nn.Linear(params.lstm_hidden_dim * 2, params.num_tag)
        self.crf_layer = CRF(params.num_tag)
        
    def forward(self, X, return_hiddens=False):
        """
        Input: 
            X: (bsz, seq_len)
        Output:
            prediction: (bsz, seq_len, num_tag)
            lstm_hidden: (bsz, seq_len, hidden_size)
        """
        embeddings = self.embedding(X)
        embeddings = F.dropout(embeddings, p=self.dropout, training=self.training)
        lstm_hidden, (_, _) = self.lstm(embeddings)  # (bsz, seq_len, hidden_dim)
        prediction = self.linear(lstm_hidden)

        if return_hiddens:
            return prediction, lstm_hidden
        else:
            return prediction
    
    def crf_decode(self, inputs, lengths):
        """ crf decode
        Input:
            inputs: (bsz, seq_len, num_entity)
            lengths: lengths of x (bsz, )
        Ouput:
            crf_loss: loss of crf
        """
        prediction = self.crf_layer(inputs)
        prediction = [ prediction[i, :length].data.cpu().numpy() for i, length in enumerate(lengths) ]

        return prediction
    
    def crf_loss(self, inputs, lengths, y):
        """ create crf loss
        Input:
            inputs: (bsz, seq_len, num_entity)
            lengths: lengths of x (bsz, )
            y: label of entity value (bsz, seq_len)
        Ouput:
            crf_loss: loss of crf
        """
        padded_y = self.pad_label(lengths, y)
        crf_loss = self.crf_layer.loss(inputs, padded_y)

        return crf_loss

    def pad_label(self, lengths, y):
        bsz = len(lengths)
        max_len = torch.max(lengths)
        padded_y = torch.LongTensor(bsz, max_len).fill_(0)
        for i in range(bsz):
            length = lengths[i]
            y_i = y[i]
            padded_y[i, 0:length] = torch.LongTensor(y_i)

        padded_y = padded_y.cuda()
        return padded_y


class CRF(nn.Module):

    def __init__(self, num_tags):
        super(CRF, self).__init__()
        
        self.num_tags = num_tags
        self.transitions = nn.Parameter(torch.Tensor(num_tags, num_tags))
        self.start_transitions = nn.Parameter(torch.randn(num_tags))
        self.stop_transitions = nn.Parameter(torch.randn(num_tags))

        nn.init.xavier_normal_(self.transitions)

    def forward(self, feats,masks):
        # Shape checks
        if len(feats.shape) != 3:
            raise ValueError("feats must be 3-d got {}-d".format(feats.shape))

        return self._viterbi(feats,masks)

    def loss(self, feats, tags,masks):
        """
        Computes negative log likelihood between features and tags.
        Essentially difference between individual sequence scores and 
        sum of all possible sequence scores (partition function)
        Parameters:
            feats: Input features [batch size, sequence length, number of tags]
            tags: Target tag indices [batch size, sequence length]. Should be between
                    0 and num_tags
        Returns:
            Negative log likelihood [a scalar] 
        """
        # Shape checks
        if len(feats.shape) != 3:
            raise ValueError("feats must be 3-d got {}-d".format(feats.shape))

        if len(tags.shape) != 2:
            raise ValueError('tags must be 2-d but got {}-d'.format(tags.shape))

        if feats.shape[:2] != tags.shape:
            raise ValueError('First two dimensions of feats and tags must match ', feats.shape, tags.shape)
        
        sequence_score = self._sequence_score(feats, tags,masks)
        partition_function = self._partition_function(feats,masks)
        log_probability = sequence_score - partition_function

        # -ve of l()
        # Average across batch
        return -log_probability.mean()

    def _sequence_score(self, feats, tags,feat_mask):
        """
        Parameters:
            feats: Input features [batch size, sequence length, number of tags]
            tags: Target tag indices [batch size, sequence length]. Should be between
                    0 and num_tags
        Returns: Sequence score of shape [batch size]
        """

        batch_size = feats.shape[0]
        tags[tags==-100] = 0
        # Compute feature scores
        feat_score = (feats.gather(2, tags.unsqueeze(-1)).squeeze(-1) * feat_mask).sum(-1)
        
        # find the first valid position and the last valid position
        idx = torch.arange(feats.shape[1],0,-1).view(1,feats.shape[1]).to(feat_mask.device)
        first_pos = torch.argmax(feat_mask*idx,dim=1)
        idx = torch.arange(0,feats.shape[1]).view(1,feats.shape[1]).to(feat_mask.device)
        last_pos = torch.argmax(feat_mask*idx,dim=1)

        # print(feat_score.size())

        # Compute transition scores
        # Unfold to get [from, to] tag index pairs
        tags_pairs = tags.unfold(1, 2, 1)

        # Use advanced indexing to pull out required transition scores
        indices = tags_pairs.permute(2, 0, 1).chunk(2)
        trans_score = (self.transitions[indices].squeeze(0) * feat_mask[:,1:]).sum(dim=-1)

        # Compute start and stop scores
        start_score = self.start_transitions[tags.gather(1,first_pos.view(1,batch_size))]
        stop_score = self.stop_transitions[tags.gather(1,last_pos.view(1,batch_size))]

        return feat_score + start_score + trans_score + stop_score

    def _partition_function(self, feats,masks):
        """
        Computes the partitition function for CRF using the forward algorithm.
        Basically calculate scores for all possible tag sequences for 
        the given feature vector sequence
        Parameters:
            feats: Input features [batch size, sequence length, number of tags]
        Returns:
            Total scores of shape [batch size]
        """
        batch_size, seq_size, num_tags = feats.shape

        if self.num_tags != num_tags:
            raise ValueError('num_tags should be {} but got {}'.format(self.num_tags, num_tags))

        a = feats[:, 0] + self.start_transitions.unsqueeze(0) # [batch_size, num_tags]
        
        transitions = self.transitions.unsqueeze(0).repeat(batch_size,1,1) # [1, num_tags, num_tags] from -> to

        for i in range(1, seq_size):
            feat = feats[:, i].unsqueeze(1) # [batch_size, 1, num_tags]
            mask_i = masks[:,i].view(batch_size,1,1)
            a = self._log_sum_exp(a.unsqueeze(-1) + (transitions + feat)*mask_i, 1) # [batch_size, num_tags]

        return self._log_sum_exp(a + self.stop_transitions.unsqueeze(0), 1) # [batch_size]

    def _viterbi(self, feats,masks):
        """
        Uses Viterbi algorithm to predict the best sequence
        Parameters:
            feats: Input features [batch size, sequence length, number of tags]
        Returns: Best tag sequence [batch size, sequence length]
        """
        batch_size, seq_size, num_tags = feats.shape

        if self.num_tags != num_tags:
            raise ValueError('num_tags should be {} but got {}'.format(self.num_tags, num_tags))
        
        v = feats[:, 0] + self.start_transitions.unsqueeze(0) # [batch_size, num_tags]
        transitions = self.transitions.unsqueeze(0) # [1, num_tags, num_tags] from -> to
        paths = []

        for i in range(1, seq_size):
            feat = feats[:, i] # [batch_size, num_tags]
            mask_i = masks[:,i].view(batch_size,1,1)
            v, idx = (v.unsqueeze(-1) + transitions*mask_i).max(1) # [batch_size, num_tags], [batch_size, num_tags]
            
            paths.append(idx)
            v = (v + feat*mask_i.squeeze(1)) # [batch_size, num_tags]

        
        v, tag = (v + self.stop_transitions.unsqueeze(0)).max(1, True)

        # Backtrack
        tags = [tag]
        for idx in reversed(paths):
            tag = idx.gather(1, tag)
            tags.append(tag)

        tags.reverse()
        return torch.cat(tags, 1)
    
    def _log_sum_exp(self, logits, dim):
        """
        Computes log-sum-exp in a stable way
        """
        max_val, _ = logits.max(dim)
        return max_val + (logits - max_val.unsqueeze(dim)).exp().sum(dim).log()
