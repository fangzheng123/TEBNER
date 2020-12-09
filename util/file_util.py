# encoding: utf-8

import json


class FileUtil(object):
    """
    文件工具类
    """

    @classmethod
    def save_entity_type(cls, entity_type_dict, entity_type_path):
        """
        保存实体字典
        :param entity_type_dict:
        :param entity_type_path:
        :return:
        """
        with open(entity_type_path, "w", encoding="utf-8") as entity_type_file:
            for name, entity_type in entity_type_dict.items():
                entity_type_file.write(entity_type + "\t" + name + "\n")

    @classmethod
    def read_entity_type_dict(cls, entity_type_path):
        """
        读取实体及对应类别文件
        :param entity_type_path: 实体类别文件
        :return:
        """
        entity_type_dict = {}
        with open(entity_type_path, "r", encoding="utf-8") as entity_type_file:
            for item in entity_type_file:
                entity_type, name = item.strip().split("\t")
                entity_type_dict[name] = entity_type

        return entity_type_dict

    @classmethod
    def save_text_obj_data(cls, text_obj_list, text_format_path):
        """
        存储格式化的文本数据
        :param text_obj_list: 格式化文本
        :param text_format_path: 格式化文本存储路径
        :return:
        """
        with open(text_format_path, "w", encoding="utf-8") as text_format_file:
            for text_obj in text_obj_list:
                text_format_file.write(json.dumps(text_obj, ensure_ascii=False) + "\n")

    @classmethod
    def read_text_obj_data(cls, text_format_path) -> list:
        """
        读取格式化的文本数据
        :param text_format_path: 格式化文本路径
        :return:
        """
        text_obj_list = []
        with open(text_format_path, "r", encoding="utf-8") as text_format_file:
            for item in text_format_file:
                item = item.strip()
                text_obj = json.loads(item)
                text_obj_list.append(text_obj)

        return text_obj_list

    @classmethod
    def read_raw_data(cls, raw_text_path):
        """
        读取原始文本数据，每行均为纯文本
        :param raw_text_path:
        :return:
        """
        all_raw_text_list = []
        with open(raw_text_path, "r", encoding="utf-8") as raw_text_file:
            for item in raw_text_file:
                item = item.strip()
                all_raw_text_list.append(item)

        return all_raw_text_list

    @classmethod
    def save_word_vec(cls, word_vec_dict, part_word_vec_path):
        """
        存储词向量数据
        :param word_vec_dict:
        :param part_word_vec_path:
        :return:
        """
        with open(part_word_vec_path, "w", encoding="utf-8") as part_word_vec_file:
            for word, vec_list in word_vec_dict.items():
                part_word_vec_file.write(word + " " + " ".join([str(val) for val in vec_list]) + "\n")

    @classmethod
    def read_word_vec(cls, part_word_vec_path) -> dict:
        """
        读取词向量数据
        :param part_word_vec_path:
        :return:
        """
        word_vec_dict = {}
        with open(part_word_vec_path, "r", encoding="utf-8") as part_vec_file:
            for item in part_vec_file:
                ele_list = item.strip().split(" ")
                word_vec_dict[ele_list[0]] = [float(val) for val in ele_list[1:]]

        return word_vec_dict

    @classmethod
    def save_mention_score(cls, mention_type_score_list, mention_type_score_path):
        """
        存储mention分类结果
        :param mention_type_score_path:
        :return:
        """
        with open(mention_type_score_path, "w", encoding="utf-8") as mention_type_score_file:
            for mention_type_score in mention_type_score_list:
                mention_type_score_file.write("\t".join(mention_type_score) + "\n")

    @classmethod
    def read_mention_score(cls, mention_type_score_path):
        """
        读取mention分类结果
        :param mention_type_score_path:
        :return:
        """
        mention_type_score_list = []
        with open(mention_type_score_path, "r", encoding="utf-8") as mention_type_score_file:
            for item in mention_type_score_file:
                item = item.strip()
                mention_type_score_list.append(item.split("\t"))

        return mention_type_score_list

    @classmethod
    def save_user_cut_dict(cls, entity_type_dict, cut_user_dict_path):
        """
        构建自定义词典，分词用
        :param entity_type_dict: 实体类别字典
        :param user_dict_path: 原始分词字典
        :return:
        """
        # 种子实体词，词频设置为2000
        with open(cut_user_dict_path, "a+", encoding="utf-8") as user_dict_file:
            for name, entity_type in entity_type_dict.items():
                user_dict_file.write(name + "\t" + str(2000) + "\n")

    @classmethod
    def save_entity_score(cls, entity_combine_dict, entity_score_path):
        """
        存储多种模型对新实体的打分
        :param entity_combine_dict: 融合实体字典
        :param entity_score_path: 存储路径
        :return:
        """
        with open(entity_score_path, "w", encoding="utf-8") as entity_score_file:
            for name, combine_entity_dict in entity_combine_dict.items():
                entity_type = combine_entity_dict["entity_list"][0]["type"]
                entity_score_file.write(name + "\t" + entity_type + "\t" +
                                        json.dumps(combine_entity_dict, ensure_ascii=False) + "\n")

    @classmethod
    def read_rank_score_file(cls, rank_entity_score_path) -> list:
        """
        读取排序后的实体文件
        :param rank_entity_score_path:
        :return:
        """
        entity_combine_rank_list = []
        with open(rank_entity_score_path, "r", encoding="utf-8") as rank_entity_score_file:
            for item in rank_entity_score_file:
                item = item.strip()
                name, entity_type, entity_str = item.split("\t")
                entity_combine_rank_list.append((name, json.loads(entity_str)))

        return entity_combine_rank_list

    @classmethod
    def read_mutual_entity_file(cls, mutual_entity_path) -> dict:
        """
        读取人工筛选出的头部负例实体
        :param mutual_entity_path:
        :return:
        """
        entity_combine_dict = {}
        with open(mutual_entity_path, "r", encoding="utf-8") as entity_score_file:
            for item in entity_score_file:
                item = item.strip()
                name, entity_type = item.split("\t")[:2]
                entity_combine_dict[name] = entity_type

        return entity_combine_dict

    def save_manual_label_data(self, pos_label_entity_list, neg_label_entity_list, manual_label_path):
        """
        存储人工筛选标注的数据
        :param pos_label_entity_list: 正例数据
        :param neg_label_entity_list: 负例数据
        :param manual_label_path: 手工打标结果存储路径
        :return:
        """
        with open(manual_label_path, "w", encoding="utf-8") as manual_label_file:
            for entity_name, combine_entity in pos_label_entity_list:
                manual_label_file.write(
                    "\t".join([str(1), entity_name, json.dumps(combine_entity, ensure_ascii=False)]) + "\n")

            for entity_name, combine_entity in neg_label_entity_list:
                manual_label_file.write(
                    "\t".join([str(0), entity_name, json.dumps(combine_entity, ensure_ascii=False)]) + "\n")

    def save_label_data(self, label_data_list, label_path):
        """
        存储打标数据
        :param label_data_list:
        :param label_path:
        :param index_list: 需要存储数据的下标列表
        :return:
        """
        with open(label_path, "w", encoding="utf-8") as label_file:
            for text_obj in label_data_list:
                label_file.write(json.dumps(text_obj, ensure_ascii=False) + "\n")

    def read_label_data(self, label_path):
        """
        存储打标数据
        :param label_path:
        :return:
        """
        label_list = []
        with open(label_path, "r", encoding="utf-8") as label_file:
            for item in label_file:
                item = item.strip()
                text_obj = json.loads(item)
                label_list.append(text_obj)

        return label_list
