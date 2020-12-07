# encoding: utf-8

import torch
from model.model_config.base_model_config import BaseConfig

class BERTSentConfig(BaseConfig):
    """
    BERT+Softmax NER模型参数配置
    """

    def __init__(self, args):
        super().__init__(args)

        # 模型存储路径
        self.model_save_path = self.args.model_dir + "/" + self.args.model_type + "_sent_version_" + ".ckpt"

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
        # 最大句子长度(padding后，短填长切)
        self.max_seq_len = self.args.max_seq_length
        # 学习率
        self.learning_rate = self.args.learning_rate
        # 标签列表
        self.label_list = self.get_label_list()
        # 标签id字典
        self.label_id_dict, self.id_label_dict = self.get_label_dict(self.label_list)
        # 标签数
        self.label_num = len(self.label_list)
        # bert最后一层输出维度
        self.bert_hidden_size = self.args.bert_hidden_size
        # dropout
        self.dropout = self.args.dropout
        # 损失函数
        self.loss_type = self.args.loss_type

    def get_label_list(self):
        """
        获取所有标签
        :return:
        """
        all_names = self.args.label_names
        all_name_list = all_names.split(",")
        label_list = [item + "-" + name.strip() for name in all_name_list for item in ["B", "I"]]
        label_list = label_list + ["O"]
        return label_list

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
