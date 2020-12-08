# encoding: utf-8


import sys
sys.path.append("../BERTAutoNER")

import os
from util.arg_util import ArgparseUtil
args = ArgparseUtil().bert_mention_classify_argparse()
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_devices

from util.log_util import LogUtil
from util.model_util import ModelUtil
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
        self.mention_data_processor = BERTMentionDataProcessor(args)
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
        pred_dataloader = self.mention_data_processor.load_dataset(self.args.pred_data_path)
        LogUtil.logger.info("Finished loading data!!!")

        # 固定种子，保证每次运行结果一致
        self.model_util.seed_everything(args.seed)

        # 模型预测
        LogUtil.logger.info("Testing model...")
        all_seq_score_list, all_seq_tag_list = self.bert_mention_process.predict(self.bert_mention_classify_model, pred_dataloader)

        # 输出结果
        LogUtil.logger.info("Save to File...")
        data_processor.output_entity(all_seq_entity_list, self.args.pred_data_path, self.args.output_path)

        LogUtil.logger.info("End!!!")


if __name__ == "__main__":
    mention_classify = MentionClassify(args)

    # 模型训练
    if args.do_train:
        mention_classify.train()

    # 模型测试，有真实标签
    if args.do_test:
        mention_classify.test()


