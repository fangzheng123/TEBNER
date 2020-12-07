# encoding: utf-8

import sys
sys.path.append("../user_label_miningg")

import os
from model.model_config.bert_autoner_config import BERTAutoNERConfig
from model.model_define.bert_autoner import BertAutoNERModel
from model.model_define.bert_autoner_no_seg import BertAutoNERNoSegModel
from model.model_process.bert_autoner_process import BERTAutoNERProcess
from model.model_process.bert_autoner_no_seg_process import BERTAutoNERNoSegProcess
from model.model_data_process.bert_autoner_data_processor import BERTAutoNERDataProcessor
from model.model_data_process.bert_autoner_no_seg_data_processor import BERTAutoNERNoSegDataProcessor
from util.model_util import ModelUtil
from util.arg_util import ArgparseUtil
from util.log_util import LogUtil

class BERTWordRun(object):
    """
    运行BERT AutoNER模型
    """

    def __init__(self, args):
        self.args = args
        self.model_util = ModelUtil()

        self.bert_autoner_config = BERTAutoNERConfig(self.args)
        self.bert_autoner_model = BertAutoNERModel(self.bert_autoner_config).to(self.bert_autoner_config.device)
        self.bert_autoner_process = BERTAutoNERProcess(self.bert_autoner_config)

        # 无分词
        self.bert_autoner_no_seg_model = BertAutoNERNoSegModel(self.bert_autoner_config).to(self.bert_autoner_config.device)
        self.bert_autoner_no_seg_process = BERTAutoNERNoSegProcess(self.bert_autoner_config)

    def train(self):
        """
        训练 BERT AutoNER模型
        :return:
        """
        # 加载数据
        LogUtil.logger.info("Loading data...")
        # 无分词
        if self.args.do_no_seg:
            data_processor = BERTAutoNERNoSegDataProcessor(self.args)
        else:
            data_processor = BERTAutoNERDataProcessor(self.args)
        train_dataloader = data_processor.load_dataset(self.args.train_data_path, is_train=True)
        dev_dataloader = data_processor.load_dataset(self.args.dev_data_path, is_dev=True)
        LogUtil.logger.info("Finished loading data!!!")

        # 固定种子，保证每次运行结果一致
        self.model_util.seed_everything(args.seed)

        # 训练模型
        LogUtil.logger.info("Training model...")
        if self.args.do_no_seg:
            self.bert_autoner_no_seg_process.train(self.bert_autoner_no_seg_model, train_dataloader, dev_dataloader)
        else:
            self.bert_autoner_process.train(self.bert_autoner_model, train_dataloader, dev_dataloader)
        LogUtil.logger.info("Finished Training model!!!")

    def test(self):
        """
        测试模型
        :return:
        """
        # 加载数据
        LogUtil.logger.info("Loading data...")
        # 无分词
        if self.args.do_no_seg:
            data_processor = BERTAutoNERNoSegDataProcessor(self.args)
        else:
            data_processor = BERTAutoNERDataProcessor(self.args)
        test_dataloader = data_processor.load_dataset(self.args.test_data_path, is_test=True)
        LogUtil.logger.info("Finished loading data!!!")

        # 固定种子，保证每次运行结果一致
        self.model_util.seed_everything(args.seed)

        # 测试模型
        LogUtil.logger.info("Testing model...")
        if self.args.do_no_seg:
            self.bert_autoner_no_seg_process.test(self.bert_autoner_no_seg_model, test_dataloader)
        else:
            self.bert_autoner_process.test(self.bert_autoner_model, test_dataloader)
        LogUtil.logger.info("Finished Testing model!!!")
        
    def predict(self):
        """
        使用训练好的NER模型预测数据
        :return:
        """
        # 加载数据
        LogUtil.logger.info("Loading data...")
        # 无分词
        if self.args.do_no_seg:
            data_processor = BERTAutoNERNoSegDataProcessor(self.args)
        else:
            data_processor = BERTAutoNERDataProcessor(self.args)
        pred_dataloader = data_processor.load_dataset(self.args.pred_data_path)
        LogUtil.logger.info("Finished loading data!!!")

        # 固定种子，保证每次运行结果一致
        self.model_util.seed_everything(args.seed)

        # 模型预测
        LogUtil.logger.info("Testing model...")
        if self.args.do_no_seg:
            all_seq_entity_dict = self.bert_autoner_no_seg_process.predict(self.bert_autoner_no_seg_model, pred_dataloader)
        else:
            all_seq_entity_dict = self.bert_autoner_process.predict(self.bert_autoner_model, pred_dataloader)

        # 输出结果
        LogUtil.logger.info("Output Entity...")
        data_processor.output_entity(all_seq_entity_dict, self.args.pred_data_path, self.args.output_path)

        LogUtil.logger.info("End!!!")

if __name__ == "__main__":
    args = ArgparseUtil().bert_autoner_argparse()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_devices

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