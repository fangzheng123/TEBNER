# encoding: utf-8

import sys
sys.path.append("../BERTAutoNER")

import os
from util.arg_util import ArgparseUtil
args = ArgparseUtil().bert_word_argparse()
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_devices

from model.model_config.bert_word_config import BERTWordConfig
from model.model_define.bert_word_model import BertWordModel
from model.model_process.bert_word_process import BERTWordProcess
from model.model_data_process.bert_word_data_processor import BERTWordProcessor
from util.model_util import ModelUtil
from util.log_util import LogUtil

class BERTWordRun(object):
    """
    运行BERT AutoNER模型
    """

    def __init__(self, args):
        self.args = args
        self.model_util = ModelUtil()

        self.bert_word_config = BERTWordConfig(self.args)
        self.bert_data_processor = BERTWordProcessor(self.bert_word_config)
        self.bert_word_model = BertWordModel(self.bert_word_config).to(self.bert_word_config.device)
        self.bert_word_process = BERTWordProcess(self.bert_word_config)

    def train(self):
        """
        训练 BERT AutoNER模型
        :return:
        """
        # 加载数据
        LogUtil.logger.info("Loading data...")

        train_dataloader = self.bert_data_processor.load_dataset(
            self.args.train_data_path, is_train=True, is_supervised=self.args.do_supervised,
            is_only_boundary=self.args.do_only_boundary)
        dev_dataloader, sent_entity_dict = self.bert_data_processor.load_dataset(
            self.args.dev_data_path, is_dev=True, is_supervised=self.args.do_supervised,
            is_only_boundary=self.args.do_only_boundary)
        LogUtil.logger.info("Finished loading data!!!")

        # 固定种子，保证每次运行结果一致
        self.model_util.seed_everything(args.seed)

        # 训练模型
        LogUtil.logger.info("Training model...")
        # 边界模型
        if args.do_only_boundary:
            self.bert_word_process.train_boundary(self.bert_word_model, train_dataloader, dev_dataloader)
        # 联合模型
        else:
            self.bert_word_process.train_joint(self.bert_word_model, train_dataloader, dev_dataloader, sent_entity_dict)

        LogUtil.logger.info("Finished Training model!!!")

    def test(self):
        """
        测试模型
        :return:
        """
        # 加载数据
        LogUtil.logger.info("Loading data...")
        test_dataloader, sent_entity_dict = self.bert_data_processor.load_dataset(
            self.args.test_data_path, is_test=True, is_only_boundary=self.args.do_only_boundary)
        LogUtil.logger.info("Finished loading data!!!")

        # 固定种子，保证每次运行结果一致
        self.model_util.seed_everything(args.seed)

        # 测试模型
        LogUtil.logger.info("Testing model...")
        if args.do_only_boundary:
            pass
        else:
            self.bert_word_process.test_joint(self.bert_word_model, test_dataloader, sent_entity_dict)

        LogUtil.logger.info("Finished Testing model!!!")
        
    def predict(self):
        """
        使用训练好的NER模型预测数据
        :return:
        """
        # 加载数据
        LogUtil.logger.info("Loading data...")
        pred_dataloader = self.bert_data_processor.load_dataset(
            self.args.pred_data_path, is_only_boundary=self.args.do_only_boundary)
        LogUtil.logger.info("Finished loading data!!!")

        # 固定种子，保证每次运行结果一致
        self.model_util.seed_everything(args.seed)

        # 模型预测
        LogUtil.logger.info("Predicting model...")

        if args.do_only_boundary:
            all_seq_score_list, all_seq_tag_list, all_seq_sent_index_list = \
                self.bert_word_process.predict_boundary(self.bert_word_model, pred_dataloader)
        else:
            all_seq_entity_dict = self.bert_word_process.predict_joint(self.bert_word_model, pred_dataloader)
            # 输出结果
            LogUtil.logger.info("Output Entity...")
            self.bert_data_processor.output_entity(all_seq_entity_dict, self.args.pred_data_path, self.args.output_path)

        LogUtil.logger.info("End!!!")

if __name__ == "__main__":
    bert_word_run = BERTWordRun(args)

    # 模型训练
    if args.do_train:
        bert_word_run.train()

    # 模型测试，有真实标签
    if args.do_test:
        bert_word_run.test()

    # 模型预测, 实体挖掘
    if args.do_predict:
        bert_word_run.predict()