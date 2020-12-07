# encoding: utf-8

import json
import itertools
from gensim.models import KeyedVectors

from phrase_mining.data_util import DataUtil
from util.log_util import LogUtil

class PhraseProcessor(object):
    """
    短语数据处理
    """
    def __init__(self, config):
        self.config = config

        self.symbol_set = DataUtil.read_symbol(self.config.symbol_path)

    def generate_candidate_phrase(self, cut_pos_data_list):
        """
        生成n-gram候选短语, 并统计相关上下文特征
        :param cut_pos_data_list: 分词及词性标注后的json数据
        :return:
        """
        LogUtil.logger.info("开始生成候选短语")

        all_phrase_dict = {}
        # 按顺序统计1-gram, 2-gram, ..., n-gram
        for phrase_len in range(1, self.config.MAX_PHRASE_LEN+1):
            gram_phrase_dict = {}
            for count_ind, text_obj in enumerate(cut_pos_data_list):
                text_id = text_obj["text_id"]

                # 切词后的列表
                word_cut_list = text_obj["text_cut"]
                word_pos_list = [ele[1] for ele in text_obj["text_pos"]]
                sent_word_num = len(word_cut_list)

                begin_index = 0
                while begin_index <= sent_word_num - phrase_len:
                    # 开头遇到标点符号则跳过
                    if word_cut_list[begin_index] in self.symbol_set:
                        begin_index = begin_index + 1
                        continue
                    # # 结尾遇到标点则跳过
                    # if word_cut_list[begin_index+phrase_len-1] in self.symbol_set:
                    #     begin_index = begin_index + phrase_len
                    #     continue

                    # 当n-1gram的阈值不满足要求时，则不考虑其对应的n-gram短语
                    if phrase_len > 1:
                        pre_original_phrase = " ".join(word_cut_list[begin_index:begin_index + phrase_len - 1]).strip()
                        if pre_original_phrase.lower() not in all_phrase_dict:
                            begin_index = begin_index + 1
                            continue

                    # 获取短语
                    original_phrase = " ".join(word_cut_list[begin_index:begin_index + phrase_len]).strip()
                    phrase = original_phrase.lower()

                    if phrase == "":
                        begin_index = begin_index + 1
                        continue

                    # 存储phrase原始形式
                    gram_phrase_dict[phrase]["original_phrase"] = \
                        gram_phrase_dict.setdefault(phrase, {}).setdefault("original_phrase", original_phrase)

                    # 统计当前短语的词性信息
                    current_phrase_pos = " ".join(word_pos_list[begin_index:begin_index + phrase_len]).strip()
                    gram_phrase_dict[phrase]["current_pos"][current_phrase_pos] = \
                        gram_phrase_dict.setdefault(phrase, {}).setdefault("current_pos", {}). \
                            setdefault(current_phrase_pos, 0) + 1

                    # 统计短语前面词的相关信息
                    if begin_index != 0:
                        # 统计短语前的词性信息
                        gram_phrase_dict[phrase]["pre_pos"][word_pos_list[begin_index - 1]] = \
                            gram_phrase_dict.setdefault(phrase, {}).setdefault("pre_pos", {}). \
                                setdefault(word_pos_list[begin_index - 1], 0) + 1

                        # 统计短语前的符号信息
                        if word_cut_list[begin_index - 1] in self.symbol_set:
                            gram_phrase_dict[phrase]["pre_symbol"][word_cut_list[begin_index - 1]] = \
                                gram_phrase_dict.setdefault(phrase, {}).setdefault("pre_symbol", {}). \
                                    setdefault(word_cut_list[begin_index - 1], 0) + 1

                    # 统计短语后面词的相关信息
                    if begin_index + phrase_len < sent_word_num:
                        # 统计短语后的词性信息
                        gram_phrase_dict[phrase]["end_pos"][word_pos_list[begin_index + phrase_len]] = \
                            gram_phrase_dict.setdefault(phrase, {}).setdefault("end_pos", {}). \
                                setdefault(word_pos_list[begin_index + phrase_len], 0) + 1

                        # 统计短语后的符号信息
                        if word_cut_list[begin_index + phrase_len] in self.symbol_set:
                            gram_phrase_dict[phrase]["end_symbol"][word_cut_list[begin_index + phrase_len]] = \
                                gram_phrase_dict.setdefault(phrase, {}).setdefault("end_symbol", {}). \
                                    setdefault(word_cut_list[begin_index + phrase_len], 0) + 1

                    # 文档频率DF
                    gram_phrase_dict.setdefault(phrase, {}).setdefault("df", set()).add(text_id)

                    # 出现频次
                    gram_phrase_dict[phrase]["freq"] = gram_phrase_dict.setdefault(phrase, {}).setdefault("freq", 0) + 1

                    begin_index += 1

                if count_ind % 10000 == 0:
                    LogUtil.logger.info(str(count_ind))

            # 选择高于设定阈值的短语
            for phrase, context_info_dict in gram_phrase_dict.items():
                if context_info_dict.get("freq", 0) > self.config.MIN_PHRASE_FREQ:
                    all_phrase_dict[phrase] = context_info_dict
            LogUtil.logger.info("生成{0}-gram短语".format(phrase_len))

        for phrase, context_info_dict in all_phrase_dict.items():
            context_info_dict["df"] = len(context_info_dict["df"])

        # 保存全量文本数
        all_phrase_dict.setdefault("all_text_num", {}).setdefault("all_text_num", len(cut_pos_data_list))

        LogUtil.logger.info("生成全量候选短语")

        if self.config.IS_WRITE_INTER_RESULT:
            with open(self.config.candidate_phrase_path, "w", encoding="utf-8") as phrase_data_file:
                sort_phrase_list = sorted(all_phrase_dict.items(), key=lambda x: x[1].get("freq", 0), reverse=True)
                for phrase, context_info_dict in sort_phrase_list:
                    phrase_data_file.write(phrase + "\t" + json.dumps(context_info_dict, ensure_ascii=False) + "\n")

            LogUtil.logger.info("写入候选短语中间结果到文件")

        return all_phrase_dict

    def get_phrase_word_vec(self, all_phrase_dict, all_vec_path):
        """
        从全量词向量中获取短语中出现词的词向量
        :param all_phrase_dict: 候选短语
        :param all_vec_path: 全量词向量文件
        :return:
        """
        LogUtil.logger.info("开始获取短语词向量")

        # 候选短语词
        phrase_word_set = set()
        for phrase in all_phrase_dict.keys():
            for word in phrase.split():
                phrase_word_set.add(word)

        # 过滤词向量
        phrase_word_vec_dict = {}

        # 词向量为word2vec二进制文件
        if ".bin" in all_vec_path:
            wv_from_bin = KeyedVectors.load_word2vec_format(all_vec_path, binary=True)
            for word in phrase_word_set:
                if word in wv_from_bin:
                    phrase_word_vec_dict[word] = wv_from_bin[word]
        else:
            with open(all_vec_path, "r", encoding="utf-8") as all_vec_file:
                # 跳过第一行
                for item in itertools.islice(all_vec_file, 1, None):
                    ele_list = item.strip().split(" ")

                    if ele_list[0] in phrase_word_set:
                        phrase_word_vec_dict[ele_list[0]] = [float(val) for val in ele_list[1:]]

        if self.config.IS_WRITE_INTER_RESULT:
            with open(self.config.candidate_phrase_word_vec_path, "w", encoding="utf-8") as phrase_word_vec_file:
                for word, vec_list in phrase_word_vec_dict.items():
                    phrase_word_vec_file.write(word + " " + " ".join([str(val) for val in vec_list]) + "\n")

        LogUtil.logger.info("获取短语词向量完成")

        return phrase_word_vec_dict

    def label_data(self, phrase_fea_dict, entity_type_dict):
        """
        打标候选短语
        :param phrase_fea_dict: 候选短语及对应特征
        :param entity_type_dict: 种子实体字典
        :return:
        """
        LogUtil.logger.info("开始远程打标数据")

        # 根据种子实体对候选短语进行打标
        positive_phrase_dict = {}
        negative_phrase_dict = {}
        for phrase, fea_dict in phrase_fea_dict.items():
            if phrase.replace(" ", "") in entity_type_dict:
                positive_phrase_dict[phrase] = fea_dict
            else:
                negative_phrase_dict[phrase] = fea_dict

        LogUtil.logger.info("远程打标数据完成")

        # 打标结果写入文件
        if self.config.IS_WRITE_INTER_RESULT:
            with open(self.config.candidate_phrase_label_path, "w", encoding="utf-8") as phrase_feature_label_file:
                for phrase, fea_dict in positive_phrase_dict.items():
                    phrase_feature_label_file.write("\t".join([str(1), phrase, json.dumps(fea_dict, ensure_ascii=False)]) + "\n")

                for phrase, fea_dict in negative_phrase_dict.items():
                    phrase_feature_label_file.write("\t".join([str(0), phrase, json.dumps(fea_dict, ensure_ascii=False)]) + "\n")

            LogUtil.logger.debug("打标数据存入文件")

        return positive_phrase_dict, negative_phrase_dict

    def save_pred_data(self, phrase_pred_list, phrase_fea_dict, pred_result_path):
        """
        将预测结果写入文件
        :param phrase_pred_list: 预测结果列表
        :param phrase_fea_dict: 短语特征字典
        :param pred_result_path: 预测结果存储文件
        :return:
        """
        # 按预测概率从高到低排序
        phrase_pred_list = sorted(phrase_pred_list, key=lambda x: x[1], reverse=True)

        with open(pred_result_path, "w", encoding="utf-8") as pred_result_file:
            for phrase, prob in phrase_pred_list:
                fea_dict = phrase_fea_dict.get(phrase, {})
                fea_dict["mean_phrase_vec"] = []
                item = [phrase, str(prob), json.dumps(fea_dict, ensure_ascii=False)]
                pred_result_file.write("\t".join(item) + "\n")

        LogUtil.logger.info("预测并保存完结果")

if __name__ == "__main__":

    # from phrase_config import PhraseConfig
    # config = PhraseConfig()
    # phrase_processor = PhraseProcessor(config)
    #
    # data_util = DataUtil()
    # cut_pos_data_list = phrase_processor.cut_pos_word(config.source_data_path)
    # # cut_data_list = data_util.read_cut_pos_data(config.cut_pos_path)
    # phrase_processor.generate_candidate_phrase(cut_pos_data_list)

    # all_phrase_dict = data_util.read_candidate_phrase_data(config.candidate_phrase_path)
    # phrase_processor.get_phrase_word_vec(all_phrase_dict, config.word_vec_path)

    # phrase_fea_dict = data_util.read_phrase_feature(config.candidate_phrase_feature_path)
    # phrase_processor.label_data(phrase_fea_dict)
    pass