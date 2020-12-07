# encoding: utf-8

import json

class DataUtil(object):
    """
    数据工具类
    """

    @classmethod
    def read_symbol(cls, symbol_path):
        """
        读取标点符号
        :param symbol_path:
        :return:
        """
        symbol_set = set()
        with open(symbol_path, "r", encoding="utf-8") as symbol_file:
            for item in symbol_file:
                item = item.strip()
                symbol_set.add(item)

        return symbol_set

    @classmethod
    def read_pos_label(cls, pos_label_path):
        """
        读取pos标签
        :param pos_label_path:
        :return:
        """
        pos_label_dict = {}
        with open(pos_label_path, "r", encoding="utf-8") as pos_label_file:
            for i, item in enumerate(pos_label_file):
                item = item.strip()

                pos_label_dict[item.split("\t")[0]] = i

        return pos_label_dict

    @classmethod
    def read_stopwords(cls, stopword_path):
        """
        读取停用词
        :param stopword_path:
        :return:
        """
        stopword_set = set()
        with open(stopword_path, "r", encoding="utf-8") as stopword_file:
            for item in stopword_file:
                item = item.strip()

                stopword_set.add(item)

        return stopword_set

    @classmethod
    def read_seed_entity(cls, entity_path):
        """
        读取种子实体文件
        :param entity_path: 实体文件
        :return:
        """
        entity_set = set()

        with open(entity_path, "r", encoding="utf-8") as entity_file:
            for item in entity_file:
                entity_type, entity_name = item.strip().split("\t")
                entity_set.add(entity_name)

        return entity_set

    ##############################读取中间结果文件##############################
    @classmethod
    def read_cut_pos_data(cls, cut_pos_path):
        """
        读取分词及词性标注后的中间文件
        :param cut_pos_path:
        :return:
        """
        cut_data_list = []
        with open(cut_pos_path, "r", encoding="utf-8") as cut_pos_file:
            for item in cut_pos_file:
                item = item.strip()

                text_obj = json.loads(item)
                cut_data_list.append(text_obj)

        return cut_data_list

    @classmethod
    def read_phrase_feature(cls, candidate_phrase_feature_path):
        """
        读取候选短语特征文件
        :param candidate_phrase_feature_path:
        :return:
        """
        phrase_fea_dict = {}
        with open(candidate_phrase_feature_path, "r", encoding="utf-8") as candidate_phrase_feature_file:
            for item in candidate_phrase_feature_file:
                item = item.strip()
                if len(item.split("\t")) != 2:
                    continue
                phrase, fea_dict_str = item.split("\t")
                fea_dict = json.loads(fea_dict_str)
                phrase_fea_dict[phrase] = fea_dict

        return phrase_fea_dict

    @classmethod
    def read_candidate_phrase_data(cls, candidate_phrase_path):
        """
        读取候选短语数据
        :param candidate_phrase_path:
        :return:
        """
        all_phrase_dict = {}
        with open(candidate_phrase_path, "r", encoding="utf-8") as candidate_phrase_file:
            for item in candidate_phrase_file:
                item = item.strip()
                if len(item.split("\t")) != 2:
                    continue

                phrase, context_dict_str = item.split("\t")
                all_phrase_dict[phrase] = json.loads(context_dict_str)

        return all_phrase_dict

    @classmethod
    def read_word_vec(cls, phrase_vec_path):
        """
        读取短语中词语的词向量
        :param phrase_vec_path:
        :return:
        """
        word_vec_dict = {}
        with open(phrase_vec_path, "r", encoding="utf-8") as phrase_vec_file:
            for item in phrase_vec_file:
                ele_list = item.strip().split(" ")
                word_vec_dict[ele_list[0]] = [float(ele) for ele in ele_list[1:]]

        return word_vec_dict

    @classmethod
    def read_label_phrase_data(cls, phrase_label_path):
        """
        读取短语打标数据
        :param phrase_label_path:
        :return:
        """
        positive_phrase_dict = {}
        negative_phrase_dict = {}
        with open(phrase_label_path, "r", encoding="utf-8") as phrase_label_file:
            for item in phrase_label_file:
                item = item.strip()

                label_str, phrase, context_dict_str = item.split("\t")
                if int(label_str) == 1:
                    positive_phrase_dict[phrase] = json.loads(context_dict_str)
                else:
                    negative_phrase_dict[phrase] = json.loads(context_dict_str)

        return positive_phrase_dict, negative_phrase_dict