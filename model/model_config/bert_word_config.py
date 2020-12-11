# encoding: utf-8

from model.model_config.base_model_config import BaseConfig

class BERTWordConfig(BaseConfig):
    """
    BERT AutoNER模型参数配置
    """

    def __init__(self, args):
        super().__init__(args)

        # 模型存储路径
        self.model_save_path = self.args.model_dir + "/" + self.args.model_type + "_word_version" + ".ckpt"

        # 最大句子长度(padding后，短填长切)
        self.max_seq_len = self.args.max_seq_length
        # bert最后一层输出维度
        self.bert_hidden_size = self.args.bert_hidden_size
        # 全连接层输出维度
        self.dnn_hidden_size = self.args.dnn_hidden_size
        # dropout
        self.dropout = self.args.dropout

        # 连接关系标签列表, Tie or Break
        # self.connect_label_list = ["B", "T", "S"]
        self.connect_label_list = ["B", "T"]
        # 类别标签列表
        self.type_label_list = self.get_type_label_list()
        # 连接关系标签id字典
        self.connect_label_id_dict, self.connect_id_label_dict = self.get_label_dict(self.connect_label_list)
        # 类别标签id字典
        self.type_label_id_dict, self.type_id_label_dict = self.get_label_dict(self.type_label_list)
        # 类别标签数
        self.type_label_num = len(self.type_label_list)
        # 连接关系标签数
        self.connect_label_num = len(self.connect_label_list)

    def get_type_label_list(self):
        """
        获取所有标签
        :return:
        """
        all_names = self.args.label_names
        type_label_list = all_names.split(",")
        # type_label_list.append("None")
        return type_label_list