# encoding: utf-8

import sys
sys.path.append("../BERTAutoNER")

import os

from util.log_util import LogUtil
from util.model_util import ModelUtil
from util.arg_util import ArgparseUtil
args = ArgparseUtil().bert_sent_argparse()
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_devices

from model.model_data_process.bert_sent_data_processor import BERTSentDataProcessor
from model.model_define.bert_sent_model import BertSentModel
from model.model_process.bert_sent_process import BERTSentProcess
from model.model_config.bert_sent_config import BERTSentConfig

class BERTSentRun(object):
    """
    运行BERT+Softmax NER模型
    """
    def __init__(self, args):
        self.args = args
        self.model_util = ModelUtil()

        self.bert_sent_config = BERTSentConfig(self.args)
        self.bert_data_processor = BERTSentDataProcessor(self.bert_sent_config)
        self.bert_sent_model = BertSentModel(self.bert_sent_config).to(self.bert_sent_config.device)
        self.bert_sent_process = BERTSentProcess(self.bert_sent_config)

    def train(self):
        """
        训练 BERT 序列标注模型
        :return:
        """
        # 加载数据
        LogUtil.logger.info("Loading data...")
        train_dataloader = self.bert_data_processor.load_dataset(self.args.train_data_path, is_train=True,
                                                                 is_supervised=self.args.do_supervised)
        dev_dataloader = self.bert_data_processor.load_dataset(self.args.dev_data_path, is_dev=True,
                                                               is_supervised=self.args.do_supervised)
        LogUtil.logger.info("Finished loading data ...")

        # 固定种子，保证每次运行结果一致
        self.model_util.seed_everything(self.args.seed)

        # 训练模型
        LogUtil.logger.info("Training model...")
        self.bert_sent_process.train(self.bert_sent_model, train_dataloader, dev_dataloader)
        LogUtil.logger.info("Finished Training model!!!")

    def test(self):
        """
        测试模型
        :return:
        """
        # 加载数据
        LogUtil.logger.info("Loading data...")
        test_dataloader = self.bert_data_processor.load_dataset(self.args.test_data_path, is_test=True,
                                                                is_supervised=self.args.do_supervised)
        LogUtil.logger.info("Finished loading data!!!")

        # 固定种子，保证每次运行结果一致
        self.model_util.seed_everything(args.seed)

        # 测试模型
        LogUtil.logger.info("Testing model...")
        self.bert_sent_process.test(self.bert_sent_model, test_dataloader)

    def predict(self):
        """
        使用训练好的NER模型预测数据
        :return:
        """
        # 加载数据
        LogUtil.logger.info("Loading data...")
        pred_dataloader = self.bert_data_processor.load_dataset(self.args.pred_data_path)
        LogUtil.logger.info("Finished loading data!!!")

        # 固定种子，保证每次运行结果一致
        self.model_util.seed_everything(args.seed)

        # 模型预测
        LogUtil.logger.info("Testing model...")
        all_seq_score_list, all_seq_tag_list = self.bert_sent_process.predict(self.bert_sent_model, pred_dataloader)

        # 实体挖掘, tag序列中包含[CLS]
        LogUtil.logger.info("Extract Entity...")
        all_seq_entity_list = self.bert_data_processor.extract_entity(all_seq_score_list, all_seq_tag_list)

        # 输出结果
        LogUtil.logger.info("Save to File...")
        self.bert_data_processor.output_entity(all_seq_entity_list, self.args.pred_data_path, self.args.output_path)

        LogUtil.logger.info("End!!!")

if __name__ == "__main__":
    bert_sent_run = BERTSentRun(args)

    # 模型训练
    if args.do_train:
        bert_sent_run.train()

    # 模型测试，有真实标签
    if args.do_test:
        bert_sent_run.test()

    # 模型预测, 实体挖掘
    if args.do_predict:
        bert_sent_run.predict()