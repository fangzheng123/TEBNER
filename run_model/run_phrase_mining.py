# encoding: utf-8

import sys
sys.path.append("../TEBNER")

import nltk
from nltk import TreebankWordTokenizer

from util.arg_util import ArgparseUtil
from util.file_util import FileUtil
from util.log_util import LogUtil
from phrase_mining.phrase_config import PhraseConfig
from phrase_mining.phrase_controller import PhraseController

class PhraseMining(object):
    """
    短语挖掘
    """
    def __init__(self, args):
        # 脚本加入的参数
        self.args = args

    def load_train_data(self):
        """
        加载实体及其所在的文本
        :return:
        """
        # 读取原始纯文本数据
        all_raw_text_list = FileUtil.read_raw_data(self.args.phrase_train_raw_text_path)

        # 读取种子实体数据
        seed_entity_type_dict = FileUtil.read_entity_type_dict(self.args.phrase_train_entity_path)

        # 格式化数据
        text_obj_list = []
        for index, text in enumerate(all_raw_text_list):
            # 切词
            token_list = TreebankWordTokenizer().tokenize(text)
            # 词性标注
            token_pos = nltk.pos_tag(token_list)
            text_json = {"text_id": str(index), "text": text, "text_cut": token_list, "text_pos": token_pos}
            text_obj_list.append(text_json)

            if index % 10000 == 0:
                LogUtil.logger.info("已处理原始数据: {0}".format(index))

        return seed_entity_type_dict, text_obj_list

    def load_test_data(self):
        """
        加载实体及其所在的文本
        :return:
        """
        # 读取原始纯文本数据
        text_obj_list = FileUtil.read_text_obj_data(self.args.source_data_path)
        # 读取种子实体数据
        seed_entity_type_dict = FileUtil.read_entity_type_dict(self.args.seed_entity_path)

        # 对原始文本进行切词与词性标注
        for index, text_obj in enumerate(text_obj_list):
            token_list = TreebankWordTokenizer().tokenize(text_obj["text"])
            token_pos = nltk.pos_tag(token_list)

            text_obj["text_cut"] = token_list
            text_obj["text_pos"] = token_pos

            if index % 100 == 0:
                LogUtil.logger.info("已处理原始数据: {0}".format(index))

        return seed_entity_type_dict, text_obj_list

    def train_phrase_mining_model(self):
        """
        训练短语挖掘模型
        :return:
        """
        # 加载实体及其所在的文本
        LogUtil.logger.info("数据加载...")
        seed_entity_type_dict, cut_pos_obj_list = self.load_train_data()

        # 保存切词与词性标注后的文本数据，避免后续重复处理
        FileUtil.save_text_obj_data(cut_pos_obj_list, args.text_format_path)

        # 挖掘高质量短语
        LogUtil.logger.info("领域短语抽取...")
        phrase_config = PhraseConfig(self.args)
        phrase_controller = PhraseController(phrase_config)
        phrase_score_list, phrase_fea_dict = phrase_controller.mining_phrase_from_text(seed_entity_type_dict,
                                                                                       cut_pos_obj_list)
        # 将预测短语存入文件
        phrase_controller.phrase_processor.save_pred_data(phrase_score_list, phrase_fea_dict,
                                                          self.args.pred_result_path)

    def pred_phrase(self):
        """
        使用训练好的模型挖掘短语
        :return:
        """
        # 加载实体及其所在的文本
        LogUtil.logger.info("数据加载...")
        seed_entity_type_dict, cut_pos_obj_list = self.load_test_data()

        # 保存切词与词性标注后的文本数据，避免后续重复处理
        FileUtil.save_text_obj_data(cut_pos_obj_list, args.text_format_path)

        # 挖掘高质量短语
        LogUtil.logger.info("领域短语抽取...")
        phrase_config = PhraseConfig(self.args)
        phrase_controller = PhraseController(phrase_config)
        phrase_fea_dict = phrase_controller.extract_phrase_fea(cut_pos_obj_list)
        phrase_score_list = phrase_controller.pred_phrase_score(phrase_fea_dict)

        # 将预测短语存入文件
        phrase_controller.phrase_processor.save_pred_data(phrase_score_list, phrase_fea_dict,
                                                          self.args.pred_result_path)

if __name__ == "__main__":
    # 解析参数
    args = ArgparseUtil().phrase_argparse()

    entity_expand_run = PhraseMining(args)

    if args.do_train:
        entity_expand_run.train_phrase_mining_model()
    elif args.do_test:
        entity_expand_run.pred_phrase()
