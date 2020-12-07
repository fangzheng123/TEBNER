# encoding: utf-8

class PhraseConfig(object):
    """
    文件路径及logging配置
    """

    def __init__(self, args):
        self.args = args

        # 短语最大长度
        self.MAX_PHRASE_LEN = args.max_phrase_len
        # 候选短语出现的最小频次
        self.MIN_PHRASE_FREQ = args.min_phrase_freq
        # 基分类器个数
        self.BASE_MODEL_NUM = args.base_model_num
        # 训练集中负例数与正例数之比
        self.NEG_POS_TIMES = args.neg_pos_times

        # 词向量文件
        self.word_vec_path = args.word_vec_path
        # 标点符号文件
        self.symbol_path = args.symbol_path
        # 词性文件
        self.pos_label_path = args.pos_label_path
        # 停用词文件
        self.stopword_path = args.stopword_path

        # 模型存储路径
        self.model_path = args.model_path

        # 是否将中间结果写入文件中
        self.IS_WRITE_INTER_RESULT = False
        if args.do_write_inter_result:
            self.IS_WRITE_INTER_RESULT = True
        # 中间文件路径
        inter_data_dir = args.inter_data_dir
        self.cut_pos_path = inter_data_dir + "/source_cut_pos"
        self.candidate_phrase_path = inter_data_dir + "/candidate_phrase"
        self.candidate_phrase_word_vec_path = inter_data_dir + "/phrase_word_vec"
        self.candidate_phrase_feature_path = inter_data_dir + "/candidate_phrase_feature"
        self.candidate_phrase_label_path = inter_data_dir + "/candidate_phrase_label"