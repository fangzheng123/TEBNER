# encoding: utf-8


import sys
sys.path.append("../BERTAutoNER")

import os
from util.arg_util import ArgparseUtil
args = ArgparseUtil().bert_pipline_argparse()
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_devices

from util.log_util import LogUtil
from util.model_util import ModelUtil
from util.file_util import FileUtil
from model.model_metric.base_metric import BaseMetric
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
    Sent Pipeline模型
    """
    def __init__(self, args):
        self.args = args
        self.model_util = ModelUtil()

        # 实体边界(Word 级别)
        self.bert_word_config = BERTWordConfig(self.args)
        self.bert_word_processor = BERTWordProcessor(self.bert_word_config)
        self.bert_word_model = BertWordModel(self.bert_word_config).to(self.bert_word_config.device)
        self.bert_word_process = BERTWordProcess(self.bert_word_config)

        # 实体边界(Sent 级别)
        self.bert_sent_config = BERTSentConfig(self.args)
        self.bert_sent_processor = BERTSentDataProcessor(self.bert_sent_config)
        self.bert_sent_model = BertSentModel(self.bert_sent_config).to(self.bert_sent_config.device)
        self.bert_sent_process = BERTSentProcess(self.bert_sent_config)

        # 实体分类模型(由于NCBI及Laptop均只有一个类型实体，无需预测类型，坑数据！！！)
        if self.args.task_name == "bc5cdr":
            self.bert_mention_config = BERTMentionConfig(self.args)
            self.mention_data_processor = BERTMentionDataProcessor(self.bert_mention_config)
            self.bert_mention_classify_model = BERTMentionClassifyModel(self.bert_mention_config)\
                .to(self.bert_mention_config.device)
            self.bert_mention_process = BERTMentionProcess(self.bert_mention_config)

    def pred_boundary_in_word(self):
        """
        在词级别预测实体边界
        :return:
        """
        # 加载边界识别数据
        LogUtil.logger.info("Loading data...")
        pred_dataloader = self.bert_word_processor.load_dataset(self.args.test_data_path)
        LogUtil.logger.info("Finished loading data!!!")

        # 预测实体边界
        LogUtil.logger.info("Testing Bert Word Model...")
        all_seq_score_list, all_seq_tag_list, all_seq_sent_index_list = \
            self.bert_word_process.predict_boundary(self.bert_word_model, pred_dataloader)

        # 实体挖掘, tag序列中包含[CLS]
        LogUtil.logger.info("Extract Entity...")
        pred_word_entity_dict = self.bert_word_processor.extract_entity(
            all_seq_score_list, all_seq_tag_list, all_seq_sent_index_list)

        return pred_word_entity_dict

    def pred_boundary_in_sent(self):
        """
        在句子级别预测实体边界
        :return:
        """
        # 加载边界识别数据
        LogUtil.logger.info("Loading data...")
        pred_dataloader = self.bert_sent_processor.load_dataset(self.args.test_data_path)
        LogUtil.logger.info("Finished loading data!!!")

        # 预测实体边界
        LogUtil.logger.info("Testing Bert Sent Model...")
        all_seq_score_list, all_seq_tag_list, all_seq_sent_index_list = \
            self.bert_sent_process.predict(self.bert_sent_model, pred_dataloader)

        # 实体挖掘, tag序列中包含[CLS]
        LogUtil.logger.info("Extract Entity...")
        pred_sent_entity_dict = self.bert_sent_processor.extract_entity(
            all_seq_score_list, all_seq_tag_list, all_seq_sent_index_list)

        return pred_sent_entity_dict

    def combine_boundary_result(self, pred_in_word_list, pred_in_sent_list):
        """
        融合边界结果
        :param pred_in_word_list:
        :param pred_in_sent_list:
        :return:
        """
        pred_word_begin_end_dict = {entity[1]: entity[2] for entity in pred_in_word_list}
        pred_sent_begin_end_dict = {entity[1]: entity[2] for entity in pred_in_sent_list}

        # 结果融合
        all_begin_end_dict = {}
        for begin, end in pred_word_begin_end_dict.items():
            all_begin_end_dict.setdefault(begin, {}).setdefault(end, []).append(1)
        for begin, end in pred_sent_begin_end_dict.items():
            all_begin_end_dict.setdefault(begin, {}).setdefault(end, []).append(1)

        filter_begin_end_dict = {}
        for begin, end_dict in all_begin_end_dict.items():
            sort_end_list = sorted(end_dict.items(), key=lambda x:sum(x[1]), reverse=True)
            if sum(sort_end_list[0][1]) > 1:
                filter_begin_end_dict[begin] = sort_end_list[0][0]

        pred_all_list = [("", begin, end, 0) for begin, end in filter_begin_end_dict.items()]

        return pred_all_list

    def get_boundary(self):
        """
        获取所有实体边界
        :return:
        """
        # 预测边界结果
        pred_word_entity_dict = self.pred_boundary_in_word()
        pred_sent_entity_dict = self.pred_boundary_in_sent()

        # 标注边界结果
        label_index_entity_dict = self.bert_sent_processor.load_label_dataset(self.args.test_data_path)

        all_result_obj_list = []
        pred_index_entity_dict = {}
        for sent_index, split_text_obj in label_index_entity_dict.items():
            split_text_obj["sent_index"] = sent_index
            split_text_obj["pred_in_word"] = pred_word_entity_dict.get(sent_index, [])
            split_text_obj["pred_in_sent"] = pred_sent_entity_dict.get(sent_index, [])
            pred_all_list = self.combine_boundary_result(split_text_obj["pred_in_word"], split_text_obj["pred_in_sent"])
            split_text_obj["pred_all"] = pred_all_list
            all_result_obj_list.append(split_text_obj)
            pred_index_entity_dict[sent_index] = pred_all_list

        # 将边界结果存储到文件中
        FileUtil.save_text_obj_data(all_result_obj_list, self.args.pred_boundary_path)

        return pred_index_entity_dict

    def classify_entity(self, pred_index_entity_dict):
        """
        预测实体类别
        :param pred_index_entity_dict:
        :return:
        """
        # 加载分类数据(同时加入远程监督边界信息)
        test_dataloader, all_sent_label_dict = self.mention_data_processor. \
            load_data_from_sent_model(self.args.test_data_path, pred_index_entity_dict)

        # 预测实体类别
        LogUtil.logger.info("Testing model...")
        self.bert_mention_process.test_by_connect_model(
            self.bert_mention_classify_model, test_dataloader, pred_index_entity_dict, all_sent_label_dict)

    def eval_result(self, pred_sent_entity_dict, label_sent_entity_dict):
        """
        评测边界结果
        :param pred_sent_entity_dict:
        :param label_sent_entity_dict:
        :return:
        """
        pred_num = 0
        pred_right_num = 0
        label_num = 0

        for sent_index, label_entity_list in label_sent_entity_dict.items():
            label_list = [[ele[0], ele[1]] for ele in label_entity_list]
            label_num += len(label_list)

        for sent_index, pred_entity_list in pred_sent_entity_dict.items():
            pred_list = [[ele[0], ele[1]] for ele in pred_entity_list]
            pred_num += len(pred_list)
            label_list = [[ele[0], ele[1]] for ele in label_sent_entity_dict.get(sent_index, [])]
            pred_right_num += len([pre_entity for pre_entity in pred_list if pre_entity in label_list])

        base_metric = BaseMetric()
        precision, recall, f1 = base_metric.compute_metric(label_num, pred_num, pred_right_num)
        metric_result_dict = {"precision": precision, "recall": recall, "f1": f1}
        LogUtil.logger.info(metric_result_dict)

    def classify_single_type_entity(self, pred_index_entity_dict):
        """
        检测单类别实体
        :param pred_index_entity_dict:
        :return:
        """
        bert_mention_config = BERTMentionConfig(self.args)
        mention_data_processor = BERTMentionDataProcessor(bert_mention_config)
        pred_sent_entity_dict, label_sent_entity_dict = mention_data_processor.get_single_type_entity(
            self.args.test_data_path, pred_index_entity_dict)
        self.eval_result(pred_sent_entity_dict, label_sent_entity_dict)

    def pipeline(self):
        """
        测试模型
        :return:
        """
        # 固定种子，保证每次运行结果一致
        self.model_util.seed_everything(args.seed)

        # 预测实体边界
        pred_index_entity_dict = self.get_boundary()

        # 预测实体类型
        if self.args.task_name == "bc5cdr":
            self.classify_entity(pred_index_entity_dict)
        else:
            # NCBI 及 Laptop均只有一个类型实体，无需预测类型
            self.classify_single_type_entity(pred_index_entity_dict)


if __name__ == "__main__":
    bert_sent_pipline = BERTSentPiplineRun(args)

    bert_sent_pipline.pipeline()




