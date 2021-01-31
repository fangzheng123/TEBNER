# encoding: utf-8


import sys
sys.path.append("../TEBNER")

import os
from util.arg_util import ArgparseUtil
args = ArgparseUtil().bert_mention_classify_argparse()
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_devices

from util.log_util import LogUtil
from util.model_util import ModelUtil
from util.file_util import FileUtil
from model.model_data_process.bert_mention_data_processor import BERTMentionDataProcessor
from model.model_define.bert_mention_classify_model import BERTMentionClassifyModel
from model.model_process.bert_mention_process import BERTMentionProcess
from model.model_config.bert_mention_config import BERTMentionConfig

class MentionClassify(object):
    """
    Mention分类
    """
    def __init__(self, args):
        self.args = args
        self.model_util = ModelUtil()

        self.bert_mention_config = BERTMentionConfig(self.args)
        self.mention_data_processor = BERTMentionDataProcessor(self.bert_mention_config)
        self.bert_mention_classify_model = BERTMentionClassifyModel(self.bert_mention_config)\
            .to(self.bert_mention_config.device)
        self.bert_mention_process = BERTMentionProcess(self.bert_mention_config)

    def train(self):
        """
        训练Mention分类模型
        :return:
        """
        # 加载数据
        LogUtil.logger.info("Loading data...")
        train_dataloader = self.mention_data_processor.load_dataset(self.args.train_data_path, is_train=True)
        dev_dataloader = self.mention_data_processor.load_dataset(self.args.dev_data_path, is_dev=True)
        LogUtil.logger.info("Finished loading data ...")

        # 固定种子，保证每次运行结果一致
        self.model_util.seed_everything(self.args.seed)

        # 训练模型
        LogUtil.logger.info("Training model...")
        self.bert_mention_process.train(self.bert_mention_classify_model, train_dataloader, dev_dataloader)
        LogUtil.logger.info("Finished Training model!!!")

    def test(self):
        """
        测试模型
        :return:
        """
        # 加载数据
        LogUtil.logger.info("Loading data...")
        test_dataloader = self.mention_data_processor.load_dataset(self.args.test_data_path, is_test=True)
        LogUtil.logger.info("Finished loading data!!!")

        # 固定种子，保证每次运行结果一致
        self.model_util.seed_everything(args.seed)

        # 测试模型
        LogUtil.logger.info("Testing model...")
        self.bert_mention_process.test(self.bert_mention_classify_model, test_dataloader)

    def predict(self):
        """
        使用训练好的分类模型预测短语类型
        :return:
        """
        # 加载数据
        LogUtil.logger.info("Loading data...")
        pred_dataloader = self.mention_data_processor.load_dataset(self.args.pred_data_path, is_predict=True)
        LogUtil.logger.info("Finished loading data!!!")

        # 固定种子，保证每次运行结果一致
        self.model_util.seed_everything(args.seed)

        # 模型预测
        LogUtil.logger.info("Predicting model...")
        all_pred_label_list, all_score_list = self.bert_mention_process.predict(self.bert_mention_classify_model, pred_dataloader)

        all_mention_result_list = self.mention_data_processor.output_mention_type(
            self.args.pred_data_path, all_pred_label_list, all_score_list)

        # 只使用预测始终为同一类别的mention类别
        phrase_types_dict = {}
        for mention_type_score in all_mention_result_list:
            phrase_types_dict.setdefault(mention_type_score[0], []).append(mention_type_score[1])
        phrase_type_dict = {}
        for mention_form, mention_type_list in phrase_types_dict.items():
            if len(set(mention_type_list)) == 1:
                phrase_type_dict[mention_form] = mention_type_list[0]
                if self.args.task_name == "ncbi":
                    phrase_type_dict[mention_form] = "disease"

        # 存储预测中间结果
        FileUtil.save_mention_score(all_mention_result_list, self.args.phrase_type_score_path)
        # 存储mention的类别
        FileUtil.save_entity_type(phrase_type_dict, self.args.phrase_label_path)

        LogUtil.logger.info("End!!!")

    def eval_phrase_label(self):
        """
        评测短语打标正确率
        :return:
        """
        # 只使用预测始终为同一类别的mention类别
        phrase_type_dict = FileUtil.read_entity_type_dict(self.args.phrase_label_path)
        gold_entity_dict = FileUtil.read_entity_type_dict(self.args.gold_entity_path)

        right_count = 0
        all_count = 0
        for phrase, phrase_type in phrase_type_dict.items():
            if phrase in gold_entity_dict and phrase_type.lower() == gold_entity_dict[phrase]:
                right_count += 1
                all_count += 1
                print("right:", "####".join([phrase, phrase_type]))
            elif phrase in gold_entity_dict:
                all_count += 1
                print("wrong:", "####".join([phrase, phrase_type]))

        LogUtil.logger.info("短语打标正确数:{0}, 存在于标注集的总短语数: {1}, 短语打标正确率为: {2}".format(
            right_count, all_count, right_count / all_count))


if __name__ == "__main__":
    mention_classify = MentionClassify(args)

    # 模型训练
    if args.do_train:
        mention_classify.train()

    # 模型测试，有真实标签
    if args.do_test:
        mention_classify.test()

    # 模型预测，无真实标签
    if args.do_predict:
        mention_classify.predict()

    # 结果评测
    if args.do_eval:
        mention_classify.eval_phrase_label()



