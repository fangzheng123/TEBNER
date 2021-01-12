# encoding: utf-8

import sys
sys.path.append("../BERTAutoNER")

import os
from model.model_data_process.phrase_data_processor import PhraseProcessor
from model.model_process.phrase_label_process import PhraseLabelProcess
from util.file_util import FileUtil
from util.arg_util import ArgparseUtil
from util.log_util import LogUtil

class PhraseLabel(object):
    """
    短语标注
    """
    def __init__(self, args):
        # 脚本加入的参数
        self.args = args
        self.phrase_processor = PhraseProcessor()
        self.label_process = PhraseLabelProcess()

    def load_entity_phrase_vec(self, seed_entity_dict, all_phrase_list):
        """
        加载实体及短语向量
        :param seed_entity_dict:
        :param all_phrase_list:
        :return:
        """
        # 获取所有mention包含词的词向量
        if not os.path.exists(self.args.part_word_vec_path):
            all_mention_name_list = list(set(list(seed_entity_dict.keys()) + all_phrase_list))
            mention_word_vec_dict = self.phrase_processor.get_mention_word_vec(all_mention_name_list, self.args.word_vec_path)
            FileUtil.save_word_vec(mention_word_vec_dict, self.args.part_word_vec_path)
        else:
            mention_word_vec_dict = FileUtil.read_word_vec(self.args.part_word_vec_path)

        # 构建实体向量和短语向量
        all_entity_vec_dict, all_phrase_vec_dict = self.phrase_processor.build_entity_phrase_word_vec(
            list(seed_entity_dict.keys()), all_phrase_list, mention_word_vec_dict)

        return all_entity_vec_dict, all_phrase_vec_dict

    def get_phrase_label(self, phrase_entity_dict, seed_entity_dict):
        """
        评测短语打标正确率
        :param phrase_entity_dict:
        :param seed_entity_dict:
        :return:
        """
        gold_entity_dict = FileUtil.read_entity_type_dict(self.args.gold_entity_path)

        right_count = 0
        all_count = 0
        phrase_type_dict = {}
        for phrase, entity_tuple in phrase_entity_dict.items():
            phrase_type = seed_entity_dict[entity_tuple[0]]
            if phrase in gold_entity_dict and phrase_type.lower() == gold_entity_dict[phrase]:
                right_count += 1
                all_count += 1
            elif phrase in gold_entity_dict:
                all_count += 1

            phrase_type_dict[phrase] = phrase_type

        LogUtil.logger.info("短语打标正确数:{0}, 存在于标注集的总短语数: {1}, 短语打标正确率为: {2}".format(
            right_count, all_count, right_count / all_count))

        return phrase_type_dict

    def main(self):
        """
        主流程
        :return:
        """
        # 加载种子字典
        seed_entity_dict = FileUtil.read_entity_type_dict(self.args.seed_entity_path)
        # 加载挖掘出的所有短语列表
        all_phrase_list = FileUtil.read_raw_data(self.args.phrase_path)

        # 加载实体向量和短语向量
        all_entity_vec_dict, all_phrase_vec_dict = self.load_entity_phrase_vec(seed_entity_dict, all_phrase_list)

        # 使用KNN模型对短语进行标注
        knn_phrase_entity_dict = self.label_process.label_phrase_by_knn(
            seed_entity_dict, all_phrase_list, all_entity_vec_dict, all_phrase_vec_dict)

        # 获取短语类别
        phrase_type_dict = self.get_phrase_label(knn_phrase_entity_dict, seed_entity_dict)

        # 存储mention的类别
        FileUtil.save_entity_type(phrase_type_dict, self.args.phrase_label_path)

if __name__ == "__main__":
    args = ArgparseUtil().phrase_label_argparse()

    phrase_label = PhraseLabel(args)

    phrase_label.main()

