# encoding: utf-8

import random
from util.log_util import LogUtil
from util.trie_en import Trie

class EntityLabel(object):
    """
    实体数据标注类
    """

    def __init__(self, entity_type_dict, phrase_list):
        self.entity_type_dict = entity_type_dict
        self.process_phrase_list = self.process_phrase(phrase_list)

        # 构建字典树进行模式串匹配
        self.trie = Trie()
        self.trie.build_trie(list(entity_type_dict.keys()) + self.process_phrase_list)

    def process_phrase(self, all_phrase_list):
        """
        处理短语
        :param phrase_list:
        :return:
        """
        # 对phrase中的相关符号前后的空格去电
        all_phrase_list = [phrase.replace(" - ", "-").replace("( ", "(").replace(" )", ")").replace(" ,", ",")
                           for phrase in all_phrase_list]
        lower_phrase_list = [phrase.lower() for phrase in all_phrase_list if len(phrase.split()) > 1]
        upper_phrase_list = [phrase.upper() for phrase in all_phrase_list]
        all_phrase_list = all_phrase_list + lower_phrase_list + upper_phrase_list

        return all_phrase_list

    def evaluate_distance_label(self, text_obj_list):
        """
        评测远程打标准确率及召回率
        :param text_obj_list:
        :return:
        """
        label_right_num = 0
        all_golden_num = 0
        all_distance_num = 0
        for text_obj in text_obj_list:
            golden_entity_list = text_obj.get("entity_list", [])
            distance_entity_list = text_obj.get("distance_entity_list", [])
            golden_entity_dict = {entity_obj["offset"]: entity_obj["form"] for entity_obj in golden_entity_list}
            distance_entity_dict = {entity_obj["offset"]: entity_obj["form"] for entity_obj in distance_entity_list
                                    if entity_obj["type"] != "unknown"}

            for golden_entity in golden_entity_list:
                if golden_entity["offset"] in distance_entity_dict \
                        and golden_entity["form"] == distance_entity_dict[golden_entity["offset"]]:
                    label_right_num += 1
                # else:
                #     print("gold: " + golden_entity["form"])
                #     if golden_entity["offset"] in distance_entity_dict:
                #         print("distance: " + distance_entity_dict[golden_entity["offset"]])
                #     print("#################################################")

            all_golden_num += len(golden_entity_list)
            all_distance_num += len(distance_entity_dict)

        LogUtil.logger.info("标注正确数: {0}, 总标签数: {1}, 远程标注标签数: {2}, 准确率: {3}, 召回率: {4}"
                            .format(label_right_num, all_golden_num, all_distance_num,
                                    label_right_num/all_distance_num, label_right_num/all_golden_num))

    def generate_distance_label_data(self, text_obj_list) -> list:
        """
        给定实体词典及自由文本，生成ner远程监督数据
        :param text_obj_list: 格式化文本
        :return:
        """
        for text_obj in text_obj_list:
            sent = text_obj["text"]

            # 远程标注
            distance_label_list = []
            for entity_obj in self.trie.search_entity(sent):
                entity_obj["type"] = self.entity_type_dict.get(entity_obj["form"], "unknown").lower()
                distance_label_list.append(entity_obj)

            text_obj["distance_entity_list"] = distance_label_list

        # 评价远程标注结果
        self.evaluate_distance_label(text_obj_list)

        return text_obj_list

    def get_unknown_entity(self, entity_type_dict, phrase_score_list, min_phrase_threshold=0.7):
        """
        获取类别为unknown的短语实体
        :param entity_type_dict: 实体字典
        :param phrase_score_list: 挖掘出的所有短语
        :param min_phrase_threshold: 将短语标注为unknown的实体的最小置信度
        :return:
        """
        # 构建所有实体的字典树
        trie = Trie()
        trie.build_trie(entity_type_dict.keys())

        unknown_entity_dict = {}
        for phrase, score in phrase_score_list:
            phrase = phrase.replace(" ", "")

            if float(score) < min_phrase_threshold:
                break

            # 从短语中去除包含种子实体的短语 (为了尽可能多打标种子实体)
            if len(trie.search_entity(phrase)) == 0:
                unknown_entity_dict[phrase] = "unknown"

        return unknown_entity_dict

    def get_manual_rule_entity(self, rank_entity_list, head_pos_entity_dict, label_head_num=3000, neg_tail_num=3000):
        """
        加入手工标注(正负例)和规则标注(负例)的结果
        :param rank_entity_list: 排序后的所有新实体
        :param head_pos_entity_dict: 手工标注的头部正例实体
        :param label_head_num: 手工标注的头部实体数量
        :param neg_tail_num: 从新实体排序尾部筛选负例的数量
        :return:
        """
        # 打标正例数据
        pos_label_entity_list = [(entity_name, combine_dict) for entity_name, combine_dict in
                                 rank_entity_list[:label_head_num] if entity_name in head_pos_entity_dict]

        # 打标负例数据
        neg_label_entity_list = [(entity_name, combine_dict) for entity_name, combine_dict in
                                 rank_entity_list[:label_head_num] if entity_name not in head_pos_entity_dict]

        # 直接从排序实体尾部筛选的负例
        if 1 < neg_tail_num < len(rank_entity_list):
            neg_rule_entity_list = rank_entity_list[-1 * neg_tail_num:]
            neg_label_entity_list.extend(neg_rule_entity_list)

        return pos_label_entity_list, neg_label_entity_list

    def generate_pos_neg_label_data(self, pos_label_entity_dict, neg_label_entity_dict, text_obj_list) -> list:
        """
        根据正负例实体远程打标数据
        :param pos_label_entity_dict: 打标的正例数据
        :param neg_label_entity_dict: 打标的负例数据
        :param text_obj_list: 格式化文本
        :return: 打标数据集
        """
        # 构建字典树
        pos_trie = Trie()
        neg_trie = Trie()
        pos_trie.build_trie(pos_label_entity_dict.keys())
        neg_trie.build_trie(neg_label_entity_dict.keys())

        text_id = 0
        label_data_list = []
        neg_label_data_list = []
        for text_obj in text_obj_list:
            sent = text_obj["text"]
            # 打标正例
            pos_entity_list = pos_trie.search_entity(sent)
            if len(pos_entity_list) > 0:
                for entity_obj in pos_entity_list:
                    entity_obj["type"] = pos_label_entity_dict[entity_obj["form"]]
                text_obj["entity_list"] = pos_entity_list
                label_data_list.append(text_obj)
            else:
                # 识别包含负例的句子
                neg_entity_list = neg_trie.search_entity(sent)
                if len(neg_entity_list) > 0:
                    text_obj["entity_list"] = []
                    text_obj["neg_entity_list"] = neg_entity_list
                    neg_label_data_list.append(text_obj)

        # 如果负例数据过多，则从打标的负例数据中随机选取部分数据
        if len(neg_label_data_list) > len(label_data_list) / 2:
            neg_label_data_list = random.sample(neg_label_data_list, int(len(label_data_list) / 2))

        label_data_list.extend(neg_label_data_list)

        return label_data_list


if __name__ == "__main__":
    pass