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

        # 超过指定batch数未提升，则结束训练
        self.require_improvement = self.args.require_improvement
        # epoch数
        self.num_epochs = self.args.num_train_epochs
        # 训练模型时batch size
        self.train_batch_size = self.args.per_gpu_train_batch_size * torch.cuda.device_count()
        # 验证模型时batch size
        self.dev_batch_size = self.args.per_gpu_dev_batch_size * torch.cuda.device_count()
        # 测试模型时batch size
        self.test_batch_size = self.args.per_gpu_test_batch_size * torch.cuda.device_count()
        # 每隔多少batch进行一次模型验证
        self.per_eval_batch_step = self.args.per_eval_batch_step

    def get_label_dict(self, label_list):
        """
        获取label字符串对应的id字典
        :return: label[str] = id, label[id] = str
        """
        label_id_dict = {}
        id_label_dict = {}
        for index, label in enumerate(label_list):
            label_id_dict[label] = index
            id_label_dict[index] = label

        return label_id_dict, id_label_dict