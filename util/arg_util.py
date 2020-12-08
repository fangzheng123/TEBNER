# encoding: utf-8

import argparse


class ArgparseUtil(object):
    """
    参数解析工具类
    """

    def __init__(self):
        self.parser = argparse.ArgumentParser()

        # 任务领域
        self.parser.add_argument("--task_name", default=None, type=str, required=True)
        # 种子
        self.parser.add_argument("--seed", type=int, default=42, help="random seed for initialization")

    def phrase_argparse(self):
        """
        短语挖掘参数解析
        :return:
        """
        # 模型配置参数
        self.parser.add_argument("--word_vec_path", default=None, type=str, required=True)
        self.parser.add_argument("--symbol_path", default=None, type=str, required=True)
        self.parser.add_argument("--stopword_path", default=None, type=str, required=True)
        self.parser.add_argument("--pos_label_path", default=None, type=str, required=True)
        self.parser.add_argument("--model_path", default=None, type=str, required=True)
        self.parser.add_argument("--inter_data_dir", default=None, type=str, required=True)
        self.parser.add_argument("--do_write_inter_result", action="store_true")
        self.parser.add_argument("--do_train", action="store_true")
        self.parser.add_argument("--do_test", action="store_true")

        # 模型配置参数
        self.parser.add_argument("--max_phrase_len", default=5, type=int)
        self.parser.add_argument("--min_phrase_freq", default=30, type=int)
        self.parser.add_argument("--base_model_num", default=10, type=int)
        self.parser.add_argument("--neg_pos_times", default=8, type=int)

        # 模型训练参数
        self.parser.add_argument("--phrase_train_entity_path", default=None, type=str, required=True)
        self.parser.add_argument("--phrase_train_raw_text_path", default=None, type=str, required=True)

        # 任务文件参数
        self.parser.add_argument("--source_data_path", default=None, type=str, required=True)
        self.parser.add_argument("--seed_entity_path", default=None, type=str, required=True)
        self.parser.add_argument("--pred_result_path", default=None, type=str, required=True)
        self.parser.add_argument("--text_format_path", default=None, type=str, required=True)
        self.parser.add_argument("--candidate_phrase_entity_path", default=None, type=str, required=True)

        args = self.parser.parse_args()
        return args

    def phrase_label_argparse(self):
        """
        短语标注参数解析
        :return:
        """
        self.parser.add_argument("--word_vec_path", default=None, type=str, required=True)
        self.parser.add_argument("--seed_entity_path", default=None, type=str, required=True)
        self.parser.add_argument("--phrase_path", default=None, type=str, required=True)
        self.parser.add_argument("--gold_entity_path", default=None, type=str, required=True)

        self.parser.add_argument("--part_word_vec_path", default=None, type=str, required=True)
        self.parser.add_argument("--phrase_label_path", default=None, type=str, required=True)

        args = self.parser.parse_args()
        return args

    def distance_label_argparse(self):
        """
        远程标注参数解析
        :return:
        """
        # 用户传入文件参数
        self.parser.add_argument("--seed_entity_path", default=None, type=str, required=True)
        self.parser.add_argument("--phrase_path", default=None, type=str, required=True)
        self.parser.add_argument("--train_data_path", default=None, type=str, required=True)
        self.parser.add_argument("--dev_data_path", default=None, type=str, required=True)
        self.parser.add_argument("--test_data_path", default=None, type=str, required=True)

        # 新生成文件参数
        self.parser.add_argument("--train_distance_data_path", default=None, type=str, required=True)
        self.parser.add_argument("--dev_distance_data_path", default=None, type=str, required=True)
        self.parser.add_argument("--test_distance_data_path", default=None, type=str, required=True)

        args = self.parser.parse_args()
        return args

    def bert_model_argparse(self):
        """
        模型训练参数解析
        :return:
        """
        # GPU
        self.parser.add_argument("--gpu_devices", default=None, type=str, required=True)

        # 模型路径相关参数
        self.parser.add_argument("--pre_trained_model_path", default=None, type=str, required=True)
        self.parser.add_argument("--model_dir", default=None, type=str, required=True)
        self.parser.add_argument("--model_type", default=None, type=str, required=True)

        # 训练、测试、验证开关
        self.parser.add_argument("--do_train", action="store_true")
        self.parser.add_argument("--do_test", action="store_true")
        self.parser.add_argument("--do_predict", action="store_true")

        # 模型数据相关参数
        self.parser.add_argument("--train_data_path", default=None, type=str)
        self.parser.add_argument("--dev_data_path", default=None, type=str)
        self.parser.add_argument("--test_data_path", default=None, type=str)
        self.parser.add_argument("--pred_data_path", default=None, type=str)
        self.parser.add_argument("--pred_result_path", default=None, type=str)
        self.parser.add_argument("--output_path", default=None, type=str)


    def bert_sent_add_parse(self):
        self.bert_model_argparse()

        # 模型配置相关参数
        self.parser.add_argument("--num_train_epochs", type=int, default=2,
                                 help="Total number of training epochs to perform.")
        self.parser.add_argument("--per_gpu_train_batch_size", default=16, type=int,
                                 help="Batch size per GPU/CPU for training.")
        self.parser.add_argument("--per_gpu_dev_batch_size", default=32, type=int,
                                 help="Batch size per GPU/CPU for evaluating.")
        self.parser.add_argument("--per_gpu_test_batch_size", default=64, type=int,
                                 help="Batch size per GPU/CPU for testing.")
        self.parser.add_argument("--require_improvement", type=int, default=1000, help="Require improvement")
        self.parser.add_argument("--per_eval_batch_step", type=int, default=1000, help="Evaluate model per batch step")
        self.parser.add_argument("--max_seq_length", default=128, type=int,
                                 help="The maximum total input sequence length after tokenization. Sequences longer "
                                      "than this will be truncated, sequences shorter will be padded.", )
        self.parser.add_argument("--learning_rate", default=3e-5, type=float,
                                 help="The initial learning rate for Adam.")
        self.parser.add_argument("--do_lower_case", action="store_true",
                                 help="Set this flag if you are using an uncased model.")
        self.parser.add_argument("--loss_type", default="ce", type=str,
                                 choices=["lsr", "focal", "ce"])
        self.parser.add_argument("--label_names", type=str, required=True, help="All label Name")
        self.parser.add_argument("--bert_hidden_size", default=768, type=int,
                                 help="The hidden size of BERT.")
        self.parser.add_argument("--dropout", default=0.15, type=float,
                                 help="The dropout rate")
        self.parser.add_argument("--weight_decay", default=0.01, type=float,
                                 help="Weight decay if we apply some.")
        self.parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                                 help="Epsilon for Adam optimizer.")
        self.parser.add_argument("--warmup_proportion", default=0.1, type=float,
                                 help="Proportion of training to perform linear learning rate warmup for,E.g., 0.1 = 10% of training.")

    def bert_mention_classify_argparse(self):
        """
        bert mention分类模型参数解析
        :return:
        """
        self.bert_sent_add_parse()

        self.parser.add_argument("--dnn_hidden_size", default=256, type=int)

        args = self.parser.parse_args()

        return args

    def bert_sent_argparse(self):
        """
        bert句子级模型参数解析
        :return:
        """
        self.bert_sent_add_parse()

        args = self.parser.parse_args()

        return args

    def bert_autoner_argparse(self):
        """
        bert autoner模型参数解析
        :return:
        """
        self.model_argparse()

        # 模型配置相关参数
        self.parser.add_argument("--do_no_seg", action="store_true")
        
        self.parser.add_argument("--num_train_epochs", type=int, default=2,
                                 help="Total number of training epochs to perform.")
        self.parser.add_argument("--max_seq_length", default=128, type=int,
                                 help="The maximum total input sequence length after tokenization. Sequences longer "
                                      "than this will be truncated, sequences shorter will be padded.")
        self.parser.add_argument("--seq_max_word_num", default=90, type=int,
                                 help="The maximum word num in the sequence")
        self.parser.add_argument("--bert_hidden_size", default=768, type=int,
                                 help="The hidden size of BERT.")
        self.parser.add_argument("--dnn_hidden_size", default=128, type=int,
                                 help="The hidden size of DNN layer.")
        self.parser.add_argument("--dropout", default=0.15, type=float,
                                 help="The dropout rate")
        self.parser.add_argument("--per_gpu_train_batch_size", default=8, type=int,
                                 help="Batch size per GPU/CPU for training.")
        self.parser.add_argument("--per_gpu_dev_batch_size", default=32, type=int,
                                 help="Batch size per GPU/CPU for evaluating.")
        self.parser.add_argument("--per_gpu_test_batch_size", default=64, type=int,
                                 help="Batch size per GPU/CPU for testing.")
        self.parser.add_argument("--require_improvement", type=int, default=1000, help="Require improvement")
        self.parser.add_argument("--per_eval_batch_step", type=int, default=1000, help="Evaluate model per batch step")
        self.parser.add_argument("--learning_rate", default=5e-5, type=float,
                                 help="The initial learning rate for Adam.")
        self.parser.add_argument("--do_lower_case", action="store_true",
                                 help="Set this flag if you are using an uncased model.")
        self.parser.add_argument("--label_names", default="finance_company,loan_product,risk_word", type=str,
                                 required=True,
                                 help="All label Name")
        self.parser.add_argument("--weight_decay", default=0.01, type=float,
                                 help="Weight decay if we apply some.")
        self.parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                                 help="Epsilon for Adam optimizer.")
        self.parser.add_argument("--warmup_proportion", default=0.1, type=float,
                                 help="Proportion of training to perform linear learning rate warmup for,E.g., 0.1 = 10% of training.")

        args = self.parser.parse_args()

        return args

    def entity_extract_argparse(self):
        """
        新实体挖掘参数解析
        :return:
        """
        # PhraseMining模型配置参数
        self.phrase_argparse()

        # 用户传入文件参数
        self.parser.add_argument("--seed_entity_expand_path", default=None, type=str, required=True)
        self.parser.add_argument("--bert_softmax_pred_path", default=None, type=str, required=True)
        self.parser.add_argument("--bert_autoner_pred_path", default=None, type=str, required=True)
        self.parser.add_argument("--entity_multi_score_rank_path", default=None, type=str, required=True)

        args = self.parser.parse_args()
        return args

    def next_label_argparse(self):
        """
        下一轮模型数据标注参数解析
        :return:
        """
        # 用户传入文件参数
        self.parser.add_argument("--rank_new_entity_path", default=None, type=str, required=True)
        self.parser.add_argument("--manual_pos_label_path", default=None, type=str, required=True)
        self.parser.add_argument("--label_head_num", default=None, type=int, required=True)
        self.parser.add_argument("--neg_tail_num", default=None, type=int, required=True)
        self.parser.add_argument("--seed_entity_expand_path", default=None, type=str, required=True)
        self.parser.add_argument("--text_format_path", default=None, type=str, required=True)
        self.parser.add_argument("--train_next_data_path", default=None, type=str, required=True)
        self.parser.add_argument("--dev_next_data_path", default=None, type=str, required=True)
        self.parser.add_argument("--entity_add_extract_path", default=None, type=str, required=True)

        args = self.parser.parse_args()
        return args

    def distill_argparse(self):
        """
        蒸馏模型参数解析
        :return:
        """
        self.bert_softmax_add_parse()

        # 是否对学生模型进行单独训练
        self.parser.add_argument("--do_single", action="store_true")

        self.parser.add_argument("--distill_model_dir", type=str, required=True)
        self.parser.add_argument("--student_type", default="cnn", type=str, required=True)
        self.parser.add_argument("--T", default=4, type=int, required=True)
        self.parser.add_argument("--cnn_dropout", default=0.15, type=float)
        self.parser.add_argument("--cnn_lr", default=1e-3, type=float)
        self.parser.add_argument("--gru_lr", default=1e-3, type=float)
        self.parser.add_argument("--cnn_num_epochs", default=3, type=int)
        self.parser.add_argument("--gru_num_epochs", default=3, type=int)
        self.parser.add_argument("--cnn_num_filters", default=100, type=int)

        args = self.parser.parse_args()
        return args




