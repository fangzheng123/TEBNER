# encoding: utf-8

from model.model_config.base_model_config import BaseConfig

class BERTSentConfig(BaseConfig):
    """
    BERT+Softmax NER模型参数配置
    """

    def __init__(self, args):
        super().__init__(args)

        # 模型存储路径
        self.model_save_path = self.args.model_dir + "/" + self.args.model_type + "_sent_version" + ".ckpt"

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