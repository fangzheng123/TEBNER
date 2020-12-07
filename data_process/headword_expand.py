# encoding: utf-8

class HeadwordExpand(object):
    """
    根据中心词扩充种子实体
    """
    def extract_headword(self, cut_entity_type_dict, min_freq_threshold=8):
        """
        从种子词典中抽取中心词, 此处中心词定义为出现次数大于设定阈值的词语
        :param cut_entity_type_dict: 切词后的实体字典
        :param min_freq_threshold: 最小频率
        :return: headword_dict[word] = entity_type
        """
        # 获取每个实体类别对应的词语
        type_words_dict = {}
        for entity_name, entity_type in cut_entity_type_dict.items():
            word_list = entity_name.split(" ")
            word_num = len(word_list)
            for position, word in enumerate(word_list):
                if position == word_num - 1:
                    position = -1
                type_words_dict[entity_type][word][position] = \
                    type_words_dict.setdefault(entity_type, {}).setdefault(word, {}).setdefault(position, 0) + 1

        # 根据阈值获取种子实体中心词，同时保存中心词的位置
        entity_headword_dict = {}
        for entity_type, word_dict in type_words_dict.items():
            for word, pos_dict in word_dict.items():
                for pos, freq in pos_dict.items():
                    if freq > min_freq_threshold:
                        entity_headword_dict.setdefault(word, {}).setdefault("pos", set()).add(pos)
                        entity_headword_dict[word]["type"] = entity_type

        return entity_headword_dict

    def extract_candidate_entity(self, headword_dict, candidate_phrase_list, head_phrase_num=5000):
        """
        根据中心词从候选短语中挖掘实体，即是根据中心词对候选短语进行分类
        :param headword_dict: 中心词字典
        :param candidate_phrase_list: 候选短语列表
        :param head_phrase_num: 选取头部的短语数
        :return:
        """
        # 过滤候选短语
        candidate_entity_dict = {}
        for candidate_phrase, prob in candidate_phrase_list[:head_phrase_num]:
            word_list = candidate_phrase.split(" ")
            word_num = len(word_list)
            for position, word in enumerate(word_list):
                if position == word_num - 1:
                    position = -1
                if word in headword_dict and position in headword_dict[word]["pos"]:
                    candidate_entity_dict[candidate_phrase] = headword_dict[word]["type"]
                    break

        return candidate_entity_dict

if __name__ == "__main__":
    pass
