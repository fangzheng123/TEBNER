# encoding: utf-8

import torch.nn as nn
from transformers import BertModel

class BertSentModel(nn.Module):
    """
    构建BERT序列标注模型
    """
    def __init__(self, config):
        super().__init__()

        self.bert = BertModel.from_pretrained(config.pretrain_bert_path)
        self.label_num = config.label_num
        self.dropout = nn.Dropout(config.dropout)
        self.classifier = nn.Linear(config.bert_hidden_size, config.label_num)

    def forward(self, x):
        input_ids, attention_mask, token_type_ids = x
        sequence_output, pooled_output = self.bert(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)

        return logits

