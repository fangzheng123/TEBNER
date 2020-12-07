# encoding: utf-8

import numpy as np
from gensim.models import KeyedVectors

from util.log_util import LogUtil

class PhraseProcessor(object):
    """
    短语数据处理
    """

    def get_mention_word_vec(self, all_mention_list, all_vec_path):
        """
        从全量词向量中获取mention中出现词的词向量
        :param all_mention_list: mention名称
        :param all_vec_path: 全量词向量文件
        :return:
        """
        LogUtil.logger.info("开始获取mention中所含词的词向量")

        # 候选短语词
        all_word_set = set()
        for mention in all_mention_list:
            for word in mention.split():
                all_word_set.add(word)
                all_word_set.add(word.lower())
                all_word_set.add(word.upper())

        # 过滤词向量
        mention_word_vec_dict = {}

        # 词向量为word2vec二进制文件
        if ".bin" in all_vec_path:
            wv_from_bin = KeyedVectors.load_word2vec_format(all_vec_path, binary=True)
            for word in all_word_set:
                if word in wv_from_bin:
                    mention_word_vec_dict[word] = wv_from_bin[word]
        else:
            with open(all_vec_path, "r", encoding="utf-8") as all_vec_file:
                # 跳过第一行
                next(all_vec_file)
                for item in all_vec_file:
                    ele_list = item.strip().split(" ")
                    if ele_list[0] in all_word_set:
                        mention_word_vec_dict[ele_list[0]] = [float(val) for val in ele_list[1:]]

        LogUtil.logger.info("获取短语词向量完成")

        return mention_word_vec_dict

    def get_name_vec(self, name, all_vec_dict):
        """
        对词语平均获取向量表示
        :param name:
        :param all_vec_dict:
        :return:
        """
        _vec_list = []
        for word in name.split():
            if word in all_vec_dict:
                _vec_list.append(all_vec_dict[word])
            elif word.upper() in all_vec_dict:
                _vec_list.append(all_vec_dict[word.upper()])
            elif word.lower() in all_vec_dict:
                _vec_list.append(all_vec_dict[word.lower()])

        if len(_vec_list) > 0:
            _vec_list = np.mean(np.array(_vec_list), axis=0).tolist()

        return _vec_list

    def build_entity_phrase_word_vec(self, entity_name_list, phrase_name_list, mention_word_vec_dict):
        """
        构建实体及短语对应的词向量
        :param entity_name_list:
        :param phrase_name_list:
        :param all_phrase_list:
        :return:
        """
        entity_name_set = set(entity_name_list)

        # 根据词平均获取实体向量
        all_entity_vec_dict = {}
        for entity_name in entity_name_list:
            _vec_list = self.get_name_vec(entity_name, mention_word_vec_dict)
            if len(_vec_list) > 0:
                all_entity_vec_dict[entity_name] = _vec_list

        # 根据词平均获取短语向量
        all_phrase_vec_dict = {}
        for phrase in phrase_name_list:
            if phrase not in entity_name_set:
                _vec_list = self.get_name_vec(phrase, mention_word_vec_dict)
                if len(_vec_list) > 0:
                    all_phrase_vec_dict[phrase] = _vec_list

        return all_entity_vec_dict, all_phrase_vec_dict