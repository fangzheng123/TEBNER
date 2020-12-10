# encoding: utf-8

import torch
import torch.nn as nn
from transformers import BertModel

class BertWordModel(nn.Module):
    """
    使用BERT来实现AutoNER模型
    预测token之间的tie or break
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.bert = BertModel.from_pretrained(config.pretrain_bert_path)

        # 用于编码实体向量
        self.entity_gru_layer = nn.GRU(input_size=config.bert_hidden_size, hidden_size=config.bert_hidden_size,
                                     num_layers=1, bidirectional=False, batch_first=True)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(config.dropout)
        self.connect_dense_layer = nn.Linear(config.bert_hidden_size*2, config.dnn_hidden_size)
        self.type_dense_layer = nn.Linear(config.bert_hidden_size, config.dnn_hidden_size)
        self.connect_classifier = nn.Linear(config.dnn_hidden_size, config.connect_label_num)
        self.type_classifier = nn.Linear(config.dnn_hidden_size, config.type_label_num)

        self.connect_seq_layer = nn.Sequential(self.connect_dense_layer, self.relu, self.dropout, self.connect_classifier)
        self.type_seq_layer = nn.Sequential(self.type_dense_layer, self.relu, self.dropout, self.type_classifier)

    def forward(self, x):
        input_ids, attention_mask, token_type_ids = x
        sequence_output, pooled_output = self.bert(input_ids=input_ids,
                                                   token_type_ids=token_type_ids, attention_mask=attention_mask)

        return sequence_output
    
    def token_connecting(self, sequence_output):
        """
        序列中token的连接关系预测，tie or break(当前token与下一个token)
        拼接相邻token的向量, 然后二分类
        :param sequence_output: 序列输出，shape=(B,S,H)
        :return: shape=(B,S-1,2)
        """
        # 连接前后token向量
        current_sequence_output = sequence_output[:, :-1, :]
        next_sequence_output = sequence_output[:, 1:, :]
        
        # shape=(B, S-1, 768*2)
        connect_output = torch.cat((current_sequence_output, next_sequence_output), dim=-1)
        # shape=(B, S-1, 2)
        connect_output = self.connect_seq_layer(connect_output)

        return connect_output

    def entity_typing(self, sequence_output, entity_begins, entity_ends):
        """
        对实体进行分类
        拼接实体的token的向量, 然后多分类
        :param sequence_output: shape=(B,S,H)
        :param entity_begins: 实体在序列中开始位置, shape=(B)
        :param entity_ends: 实体在序列中结束位置, shape=(B)
        :return: shape=(B, Type Num)
        """
        # 获取实体向量
        all_entity_vec_list = []
        for batch_index in range(entity_begins.shape[0]):
            entity_begin = entity_begins[batch_index]
            entity_end = entity_ends[batch_index]
            entity_vec = sequence_output[batch_index, entity_begin:entity_end + 1, :]

            seq_gru_out, gru_hidden = self.entity_gru_layer(entity_vec.unsqueeze(dim=0))
            all_entity_vec_list.append(seq_gru_out.squeeze(dim=0)[-1, :].unsqueeze(dim=0))
        
        # shape=(B, 768)
        entity_output = torch.cat(all_entity_vec_list, dim=0)
        # shape=(B, Type Num)
        entity_output = self.type_seq_layer(entity_output)

        return entity_output
