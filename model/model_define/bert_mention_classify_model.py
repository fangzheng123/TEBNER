# encoding: utf-8

import torch
import torch.nn as nn
from transformers import BertModel

class BERTMentionClassifyModel(nn.Module):
    """
    BERT Mention分类模型
    """
    def __init__(self, config):
        super().__init__()
        self.bert = BertModel.from_pretrained(config.pretrain_bert_path)

        self.dense_layer1 = nn.Linear(config.bert_hidden_size * 3, config.dnn_hidden_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(config.dropout)
        self.dense_layer2 = nn.Linear(config.dnn_hidden_size, config.label_num)
        self.clsf_seq_layer = nn.Sequential(self.dense_layer1, self.relu, self.dropout, self.dense_layer2)

    def forward(self, x):
        # 输入的句子
        input_ids, token_type_ids, attention_mask, mention_begins, mention_ends = x
        sequence_output, pooled_output = self.bert(input_ids, token_type_ids=token_type_ids,
                                                   attention_mask=attention_mask)
        # 取mention首尾2个token与[CLS]拼接
        begin_embed = torch.cat([torch.index_select(ele, 0, i) for ele, i in zip(sequence_output, mention_begins)])
        end_embed = torch.cat([torch.index_select(ele, 0, i) for ele, i in zip(sequence_output, mention_ends)])
        out = torch.cat([pooled_output, begin_embed, end_embed], dim=-1)

        out = self.clsf_seq_layer(out)

        return out