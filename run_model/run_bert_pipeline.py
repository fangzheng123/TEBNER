# encoding: utf-8


import sys
sys.path.append("../BERTAutoNER")

import os
from util.arg_util import ArgparseUtil
args = ArgparseUtil().bert_sent_pipline_argparse()
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_devices

from util.log_util import LogUtil
from util.model_util import ModelUtil
from util.file_util import FileUtil
from model.model_config.bert_word_config import BERTWordConfig
from model.model_config.bert_sent_config import BERTSentConfig
from model.model_config.bert_mention_config import BERTMentionConfig

from model.model_data_process.bert_word_data_processor import BERTWordProcessor
from model.model_data_process.bert_sent_data_processor import BERTSentDataProcessor
from model.model_data_process.bert_mention_data_processor import BERTMentionDataProcessor

from model.model_define.bert_word_model import BertWordModel
from model.model_define.bert_sent_model import BertSentModel
from model.model_define.bert_mention_classify_model import BERTMentionClassifyModel

from model.model_process.bert_word_process import BERTWordProcess
from model.model_process.bert_sent_process import BERTSentProcess
from model.model_process.bert_mention_process import BERTMentionProcess

class BERTSentPiplineRun(object):
    """
    Sent Pipline模型
    """
    def __init__(self, args):
        self.args = args
        self.model_util = ModelUtil()

        # 实体边界(Word 级别)
        self.bert_word_config = BERTWordConfig(self.args)
        self.bert_word_processor = BERTWordProcessor(self.bert_word_config)
        self.bert_word_model = BertWordModel(self.bert_sent_config).to(self.bert_sent_config.device)
        self.bert_word_process = BERTWordProcess(self.bert_word_config)

        # 实体边界(Sent 级别)
        self.bert_sent_config = BERTSentConfig(self.args)
        self.bert_sent_processor = BERTSentDataProcessor(self.bert_sent_config)
        self.bert_sent_model = BertSentModel(self.bert_sent_config).to(self.bert_sent_config.device)
        self.bert_sent_process = BERTSentProcess(self.bert_sent_config)

        # 实体边界(Corpus 级别)，为保证实验公平性，仅加载在训练集中出现的短语
        phrase_type_dict = FileUtil.read_entity_type_dict(self.args.phrase_label_path)

        # 实体分类模型
        self.bert_mention_config = BERTMentionConfig(self.args)
        self.mention_data_processor = BERTMentionDataProcessor(self.bert_mention_config)
        self.bert_mention_classify_model = BERTMentionClassifyModel(self.bert_mention_config)\
            .to(self.bert_mention_config.device)
        self.bert_mention_process = BERTMentionProcess(self.bert_mention_config)

    def pred_boundary_in_sent(self, pred_dataloader):
        """
        在句子级别预测实体边界
        :param pred_dataloader:
        :return:
        """
        # 预测实体边界
        LogUtil.logger.info("Testing Bert Sent Model...")
        all_seq_score_list, all_seq_tag_list, all_seq_sent_index_list = \
            self.bert_sent_process.predict(self.bert_sent_model, pred_dataloader)

        # 实体挖掘, tag序列中包含[CLS]
        LogUtil.logger.info("Extract Entity...")
        pred_sent_entity_dict = self.bert_sent_processor.extract_entity(
            all_seq_score_list, all_seq_tag_list, all_seq_sent_index_list)

        # 加载分类数据
        test_dataloader, all_sent_label_dict = self.mention_data_processor. \
            load_data_from_sent_model(self.args.test_data_path, pred_sent_entity_dict)

        # 预测实体类别
        LogUtil.logger.info("Testing model...")
        self.bert_mention_process.test_by_connect_model(
            self.bert_mention_classify_model, test_dataloader, pred_sent_entity_dict, all_sent_label_dict)

    def test(self):
        """
        测试模型
        :return:
        """
        # 加载边界识别数据
        LogUtil.logger.info("Loading data...")
        pred_dataloader = self.bert_sent_processor.load_dataset(self.args.test_data_path)
        LogUtil.logger.info("Finished loading data!!!")

        # 固定种子，保证每次运行结果一致
        self.model_util.seed_everything(args.seed)




if __name__ == "__main__":
    bert_sent_pipline = BERTSentPiplineRun(args)

    bert_sent_pipline.test()




