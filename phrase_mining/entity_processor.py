# encoding: utf-8

import ahocorasick
from phrase_mining.data_util import DataUtil
from util.log_util import LogUtil

class EntityProcessor(object):
    """
    处理实体
    """
    def __init__(self, config):
        self.config = config
        self.symbol_set = DataUtil.read_symbol(self.config.symbol_path)

    def build_ac_automation(self, entity_cut_name_list):
        """
        构建AC自动机
        :param entity_cut_name_list:
        :return:
        """
        ac_automation = ahocorasick.Automaton()
        # 构建Trie树
        for entity_cut_name in entity_cut_name_list:
            ac_automation.add_word(entity_cut_name.replace(" ", ""), entity_cut_name)
        # 构建自动机
        ac_automation.make_automaton()

        return ac_automation

    def get_cut_offsets(self, cut_word_list):
        """
        获取每个词语的起始位置
        :param cut_word_list:
        :return:
        """
        offset = 0
        cut_offset_list = []
        for word in cut_word_list:
            cut_offset_list.append(offset)
            offset += len(word)

        return cut_offset_list

    def get_text_phrase(self, cut_word_list, ac_automation):
        """
        使用AC自动机获取文本中的短语
        :param cut_word_list:
        :param ac_automation:
        :return:
        """
        # 获取每个词语的起始位置
        cut_offset_list = self.get_cut_offsets(cut_word_list)
        cut_offset_set = set(cut_offset_list)

        text_phrase_list = []
        content = "".join(cut_word_list)
        content_len = len(content)

        for end_index, entity in ac_automation.iter(content):
            start_index = end_index - len(entity.replace(" ", "")) + 1

            # 短语边界错误
            if start_index not in cut_offset_set or (end_index+1 < content_len and end_index+1 not in cut_offset_set):
                continue

            # 获取短语在分词序列中的位置
            cut_start_index = cut_offset_list.index(start_index)
            if end_index + 1 == content_len:
                cut_end_index = len(cut_offset_list) - 1
            else:
                cut_end_index = cut_offset_list.index(end_index + 1) - 1

            text_phrase_list.append((entity, cut_start_index, cut_end_index))

        return text_phrase_list

    def extract_entity_context_info(self, entity_cut_name_list, cut_pos_data_list):
        """
        挖掘实体上下文特征
        :param entity_cut_name_list: 实体名称列表
        :param cut_pos_data_list: 分词及词性标注后的文本obj
        :return:
        """
        LogUtil.logger.info("开始识别文本中实体")

        # 构建AC自动机进行模式串匹配
        ac_automation = self.build_ac_automation(entity_cut_name_list)

        # 统计文本中的实体上下文信息
        all_text_num = 0
        entity_context_dict = {}

        for data_id, data_obj in enumerate(cut_pos_data_list):
            all_text_num += 1
            word_cut_list = data_obj["cut_text"]
            word_pos_list = [ele.split("#")[-1] for ele in data_obj["pos_text"]]

            # 获取文本中的短语及其位置
            text_entity_list = self.get_text_phrase(word_cut_list, ac_automation)

            for entity, cut_start_index, cut_end_index in text_entity_list:
                # 统计短语前面词的相关信息
                if cut_start_index != 0:
                    # 统计短语前的词性信息
                    entity_context_dict[entity]["pre_pos"][word_pos_list[cut_start_index - 1]] = \
                        entity_context_dict.setdefault(entity, {}).setdefault("pre_pos", {}). \
                            setdefault(word_pos_list[cut_start_index - 1], 0) + 1

                    # 统计短语前的符号信息
                    if word_cut_list[cut_start_index - 1] in self.symbol_set:
                        entity_context_dict[entity]["pre_symbol"][word_cut_list[cut_start_index - 1]] = \
                            entity_context_dict.setdefault(entity, {}).setdefault("pre_symbol", {}). \
                                setdefault(word_cut_list[cut_start_index - 1], 0) + 1

                # 统计短语后面词的相关信息
                if cut_end_index != len(word_cut_list) - 1:
                    # 统计短语后的词性信息
                    entity_context_dict[entity]["end_pos"][word_pos_list[cut_end_index + 1]] = \
                        entity_context_dict.setdefault(entity, {}).setdefault("end_pos", {}). \
                            setdefault(word_pos_list[cut_end_index + 1], 0) + 1

                    # 统计短语后的符号信息
                    if word_cut_list[cut_end_index + 1] in self.symbol_set:
                        entity_context_dict[entity]["end_symbol"][word_cut_list[cut_end_index + 1]] = \
                            entity_context_dict.setdefault(entity, {}).setdefault("end_symbol", {}). \
                                setdefault(word_cut_list[cut_end_index + 1], 0) + 1

                # 统计当前短语的词性信息
                current_phrase_pos = " ".join(word_pos_list[cut_start_index: cut_end_index + 1]).strip()
                entity_context_dict[entity]["current_pos"][current_phrase_pos] = \
                    entity_context_dict.setdefault(entity, {}).setdefault("current_pos", {}). \
                        setdefault(current_phrase_pos, 0) + 1

                # 文档频率DF
                entity_context_dict.setdefault(entity, {}).setdefault("df", set()).add(data_id)

                # 出现频次
                entity_context_dict[entity]["freq"] = entity_context_dict.setdefault(entity, {}) \
                                                          .setdefault("freq", 0) + 1

            if data_id % 5000 == 0:
                LogUtil.logger.info(str(data_id))

        # 计算每个实体的df值
        for entity, context_info_dict in entity_context_dict.items():
            context_info_dict["df"] = len(context_info_dict["df"])

        # 保存全量文本数
        entity_context_dict.setdefault("all_text_num", {}).setdefault("all_text_num", all_text_num)

        LogUtil.logger.info("文本中实体识别完毕")

        return entity_context_dict