# encoding: utf-8

import sys
sys.path.append("../BERTAutoNER")

import os
import numpy as np
from sklearn import neighbors

from model.model_data_process.phrase_data_processor import PhraseProcessor
from util.file_util import FileUtil
from util.arg_util import ArgparseUtil

class PhraseLabel(object):
    """
    短语标注
    """

    def __init__(self, args):
        # 脚本加入的参数
        self.args = args
        self.phrase_processor = PhraseProcessor()

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

    def label_phrase(self, seed_entity_dict, all_phrase_list, all_entity_vec_dict, all_phrase_vec_dict):
        """
        根据词向量打标短语类型
        :param seed_entity_dict:
        :param all_phrase_list:
        :param all_entity_vec_dict:
        :param all_phrase_vec_dict:
        :return:
        """
        entity_vec_list = []
        entity_type_list = []
        for entity_name, entity_vec in all_entity_vec_dict.items():
            entity_vec_list.append(entity_vec)
            entity_type_list.append(seed_entity_dict[entity_name])
        entity_type_index_dict = {entity_type: index for index, entity_type in enumerate(list(set(entity_type_list)))}
        entity_index_type_dict = {index: entity_type for entity_type, index in entity_type_index_dict.items()}
        type_index_list = [entity_type_index_dict[entity_type] for entity_type in entity_type_list]

        knn_clf = neighbors.KNeighborsClassifier(5, weights='distance', metric="euclidean")\
            .fit(np.array(entity_vec_list), np.array(type_index_list))

        phrase_type_dict = {}
        for phrase in all_phrase_list:
            if phrase in seed_entity_dict:
                phrase_type_dict[phrase] = seed_entity_dict[phrase]
            elif phrase in all_phrase_vec_dict:
                print(phrase, entity_index_type_dict[knn_clf.predict(np.array([all_phrase_vec_dict[phrase]]))[0]])

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

        # 打标短语类型
        self.label_phrase(seed_entity_dict, all_phrase_list, all_entity_vec_dict, all_phrase_vec_dict)

if __name__ == "__main__":
    args = ArgparseUtil().phrase_label_argparse()

    phrase_label = PhraseLabel(args)

    phrase_label.main()

