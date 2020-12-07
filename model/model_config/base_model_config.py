# encoding: utf-8

import torch
from transformers import BertTokenizer, BertConfig

class BaseConfig(object):
    """
    所有模型基础配置
    """

    def __init__(self, args):
        self.args = args

        # 模型名称
        self.model_name = self.args.task_name
        # 设备
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # 预训练存储路径
        self.pretrain_bert_path = self.args.pre_trained_model_path
        # bert分词器
        self.tokenizer = BertTokenizer.from_pretrained(self.pretrain_bert_path, do_lower_case=args.do_lower_case)
        # bert相关配置
        self.pretrain_config = BertConfig.from_pretrained(self.pretrain_bert_path)
