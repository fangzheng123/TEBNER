# encoding: utf-8

import json
import math
import numpy as np
from phrase_mining.data_util import DataUtil
from util.log_util import LogUtil

class PhraseFeature(object):
    """
    短语特征挖掘
    """

    def __init__(self, phrase_config):
        self.phrase_config = phrase_config

        self.pos_label_dict = DataUtil.read_pos_label(self.phrase_config.pos_label_path)
        self.symbol_set = DataUtil.read_symbol(self.phrase_config.symbol_path)
        self.stopword_set = DataUtil.read_stopwords(self.phrase_config.stopword_path)

    def cal_pmi_pkl(self, phrase, candidate_phrase_dict, all_phrase_freq_sum):
        """
        将短语根据概率计算切分为左右两部分, 并计算短语PMI 与 PKL
        :param phrase:
        :param candidate_phrase_dict:
        :param all_phrase_freq_sum: 所有短语频次总和
        :return:
        """
        pmi, pkl = 0.0, 0.0
        word_list = phrase.split()
        if len(word_list) < 2:
            return pmi, pkl

        phrase_freq = candidate_phrase_dict.get(phrase, {}).get("freq", 0)
        split_freq_list = []
        for i in range(1, len(word_list)):
            left_phrase = " ".join(word_list[:i])
            right_phrase = " ".join(word_list[i:])
            split_freq_list.append(candidate_phrase_dict.get(left_phrase, {}).get("freq", 0) *
                                   candidate_phrase_dict.get(right_phrase, {}).get("freq", 0))

        min_split_freq = min(split_freq_list) + 1
        pmi = math.log2(phrase_freq*all_phrase_freq_sum/min_split_freq + 1)
        pkl = pmi * phrase_freq / all_phrase_freq_sum
        return pmi, pkl

    def cal_feature(self, candidate_phrase_dict, filed_vec_dict):
        """
        计算特征
        :param candidate_phrase_dict: 候选短语及上下文信息
        :param filed_vec_dict: 短语包含词的词向量
        :return:
        """
        # 所有短语频次之和
        all_phrase_freq_sum = sum([ele.get("freq", 0) for ele in candidate_phrase_dict.values()])

        # 总文档数
        all_text_num = candidate_phrase_dict.get("all_text_num", {}).get("all_text_num", 10000)
        LogUtil.logger.info("all text num: {0}".format(all_text_num))

        phrase_fea_dict = {}
        for candidate_phrase, context_dict in candidate_phrase_dict.items():
            if candidate_phrase == "all_text_num":
                continue

            phrase_word_list = candidate_phrase.split()

            # 短语长度
            phrase_word_len = len(phrase_word_list)

            # 短语字符数量
            phrase_char_len = len(" ".join(phrase_word_list))

            # 频次
            phrase_freq = math.log2(context_dict.get("freq", 0) + 1)

            # DF
            phrase_df = math.log2(context_dict.get("df", 0) + 1)

            # IDF
            phrase_idf = math.log2(all_text_num / (phrase_df + 1) + 1)

            # PMI, PKL
            phrase_pmi, phrase_kl = self.cal_pmi_pkl(candidate_phrase, candidate_phrase_dict, all_phrase_freq_sum)

            # 短语词性特征
            pre_pos_dict = context_dict.get("pre_pos", {})
            end_pos_dict = context_dict.get("end_pos", {})
            current_pos_dict = context_dict.get("current_pos", {})

            _, _max_current_pos = max(zip(current_pos_dict.values(), current_pos_dict.keys()))
            if len(_max_current_pos.split()) == 0:
                print(current_pos_dict, candidate_phrase)
            # 当前短语首位占比最大的词性
            max_current_start_pos = self.pos_label_dict.get(_max_current_pos.split()[0], -1)
            # 当前短语末尾占比最大的词性
            max_current_end_pos = self.pos_label_dict.get(_max_current_pos.split()[-1], -1)

            # 当前短语前占比最大的词性
            max_pre_pos = -1
            if len(pre_pos_dict) > 0:
                _, _max_pre_pos = max(zip(pre_pos_dict.values(), pre_pos_dict.keys()))
                max_pre_pos = self.pos_label_dict.get(_max_pre_pos, -1)

            # 当前短语后占比最大的词性
            max_end_pos = -1
            if len(end_pos_dict) > 0:
                _, _max_end_pos = max(zip(end_pos_dict.values(), end_pos_dict.keys()))
                max_end_pos = self.pos_label_dict.get(_max_end_pos, -1)

            # 短语中是否有重复词
            is_contain_duplicate_word = 0
            if len(phrase_word_list) != len(set(phrase_word_list)):
                is_contain_duplicate_word = 1

            # 短语起始位置是否包含停用词
            is_phrase_start_contain_stopword = 0
            if phrase_word_list[0] in self.stopword_set:
                is_phrase_start_contain_stopword = 1

            # 短语结束位置是否包含停用词
            is_phrase_end_contain_stopword = 0
            if phrase_word_list[-1] in self.stopword_set:
                is_phrase_end_contain_stopword = 1

            # 候选短语起始位置是否有数字
            is_phrase_start_contain_digit = 0
            if phrase_word_list[0].isdigit():
                is_phrase_start_contain_digit = 1

            # 候选短语中间位置是否有数字
            is_phrase_middle_contain_digit = 0
            for word in phrase_word_list[1:-1]:
                if word.isdigit():
                    is_phrase_middle_contain_digit = 1
                    break

            # 候选短语后面位置是否有数字
            is_phrase_end_contain_digit = 0
            if phrase_word_list[-1].isdigit():
                is_phrase_end_contain_digit = 1

            pre_symbol_dict = context_dict.get("pre_symbol", {})
            end_symbol_dict = context_dict.get("end_symbol", {})
            pre_symbol_num = sum([val for _, val in pre_symbol_dict.items()]) + 1
            end_symbol_num = sum([val for _, val in end_symbol_dict.items()]) + 1

            # 候选短语前是否有逗号
            is_pre_contain_comma = 0
            if "," in pre_symbol_dict or "，" in pre_symbol_dict:
                is_pre_contain_comma = 1

            # 候选短语前逗号占比
            pre_contain_comma_ratio = (pre_symbol_dict.get(",", 0) + pre_symbol_dict.get("，", 0)) / pre_symbol_num

            # 候选短语后是否有逗号
            is_end_contain_comma = 0
            if "," in end_symbol_dict or "，" in end_symbol_dict:
                is_end_contain_comma = 1

            # 候选短语后逗号占比
            end_contain_comma_ratio = (end_symbol_dict.get(",", 0) + end_symbol_dict.get("，", 0)) / end_symbol_num

            # 候选短语前是否有顿号
            is_pre_contain_don = 0
            if "、" in pre_symbol_dict:
                is_pre_contain_don = 1

            # 候选短语前顿号占比
            pre_contain_don_ratio = pre_symbol_dict.get("、", 0) / pre_symbol_num

            # 候选短语后是否有顿号
            is_end_contain_don = 0
            if "、" in end_symbol_dict:
                is_end_contain_don = 1

            # 候选短语后顿号占比
            end_contain_don_ratio = end_symbol_dict.get("、", 0) / end_symbol_num

            # 候选短语前是否有引号
            is_pre_contain_quotation = 0
            if "\"" in pre_symbol_dict or "\'" in pre_symbol_dict:
                is_pre_contain_quotation = 1

            # 候选短语前引号占比
            pre_contain_quotation_ratio = (pre_symbol_dict.get("\"", 0) + pre_symbol_dict.get("\'", 0)) / pre_symbol_num

            # 候选短语后是否有引号
            is_end_contain_quotation = 0
            if "\"" in end_symbol_dict or "\'" in end_symbol_dict:
                is_end_contain_quotation = 1

            # 候选短语后引号占比
            end_contain_quotation_ratio = (end_symbol_dict.get("\"", 0) + end_symbol_dict.get("\'", 0)) / end_symbol_num

            # 候选短语前是否有正括号
            is_pre_contain_parentheses = 0
            if "(" in pre_symbol_dict or "（" in pre_symbol_dict:
                is_pre_contain_parentheses = 1

            # 候选短语前括号占比
            pre_contain_parentheses_ratio = (pre_symbol_dict.get("(", 0) + pre_symbol_dict.get("（", 0)) / pre_symbol_num

            # 候选短语后是否有反括号
            is_end_contain_parentheses = 0
            if ")" in end_symbol_dict or "）" in end_symbol_dict:
                is_end_contain_parentheses = 1

            # 候选短语后括号占比
            end_contain_parentheses_ratio = (end_symbol_dict.get(")", 0) + end_symbol_dict.get("）", 0)) / end_symbol_num

            # 候选短语前是否有书名号
            is_pre_contain_book_mark = 0
            if "<" in pre_symbol_dict or "《" in pre_symbol_dict:
                is_pre_contain_book_mark = 1

            # 候选短语前书名号占比
            pre_contain_book_mark_ratio = (pre_symbol_dict.get("<", 0) + pre_symbol_dict.get("《", 0)) / pre_symbol_num

            # 候选短语后是否有书名号
            is_end_contain_book_mark = 0
            if ">" in end_symbol_dict or "》" in end_symbol_dict:
                is_end_contain_book_mark = 1

            # 候选短语后书名号占比
            end_contain_book_mark_ratio = (end_symbol_dict.get(">", 0) + end_symbol_dict.get("》", 0)) / end_symbol_num

            # 候选短语前是否有中划线
            is_pre_contain_mid_line = 0
            if "-" in pre_symbol_dict:
                is_pre_contain_mid_line = 1

            # 候选短语前中划线占比
            pre_contain_mid_line_ratio = pre_symbol_dict.get("-", 0) / pre_symbol_num

            # 候选短语后是否有中划线
            is_end_contain_mid_line = 0
            if "-" in end_symbol_dict:
                is_end_contain_mid_line = 1

            # 候选短语后中划线占比
            end_contain_mid_line_ratio = end_symbol_dict.get("-", 0) / end_symbol_num

            # 候选短语前是否有井号
            is_pre_contain_hash_sign = 0
            if "#" in pre_symbol_dict:
                is_pre_contain_hash_sign = 1

            # 候选短语前井号占比
            pre_contain_hash_sign_ratio = pre_symbol_dict.get("#", 0) / pre_symbol_num

            # 候选短语后是否有井号
            is_end_contain_hash_sign = 0
            if "#" in end_symbol_dict:
                is_end_contain_hash_sign = 1

            # 候选短语后井号占比
            end_contain_hash_sign_ratio = end_symbol_dict.get("#", 0) / end_symbol_num

            # 候选短语是否包含中划线
            is_phrase_contain_mid_line = 0
            if "-" in candidate_phrase:
                is_phrase_contain_mid_line = 1

            # 候选短语是否包含逗号
            is_phrase_contain_comma = 0
            if "," in candidate_phrase:
                is_phrase_contain_comma = 1

            # 候选短语是否包含括号
            is_phrase_contain_parentheses = 0
            if "(" in candidate_phrase:
                is_phrase_contain_parentheses = 1

            # 短语原始形式均为大写
            is_phrase_upper = 0
            original_phrase = context_dict["original_phrase"]
            if original_phrase.isupper():
                is_phrase_upper = 1

            # 短语长度小于4且非停用词
            is_phrase_special = 0
            if len(candidate_phrase) < 4 and candidate_phrase not in self.stopword_set:
                is_phrase_special = 1

            # 候选短语的平均词向量
            phrase_vec_list = [filed_vec_dict.get(word, [0.0 for _ in range(200)]) for word in phrase_word_list]
            mean_phrase_vec = np.mean(np.array(phrase_vec_list), axis=0).tolist()

            phrase_fea_dict[candidate_phrase] = {
                "phrase_word_len": phrase_word_len,
                "phrase_char_len": phrase_char_len,
                "phrase_freq": round(phrase_freq, 6),
                "phrase_df": phrase_df,
                "phrase_idf": round(phrase_idf, 6),
                "phrase_pmi": round(phrase_pmi, 6),
                "phrase_kl": round(phrase_kl, 6),
                "max_current_start_pos": max_current_start_pos,
                "max_current_end_pos": max_current_end_pos,
                "max_pre_pos": max_pre_pos,
                "max_end_pos": max_end_pos,
                "is_contain_duplicate_word": is_contain_duplicate_word,
                "is_phrase_start_contain_stopword": is_phrase_start_contain_stopword,
                "is_phrase_end_contain_stopword": is_phrase_end_contain_stopword,
                "is_phrase_start_contain_digit": is_phrase_start_contain_digit,
                "is_phrase_middle_contain_digit": is_phrase_middle_contain_digit,
                "is_phrase_end_contain_digit": is_phrase_end_contain_digit,
                "is_pre_contain_comma": is_pre_contain_comma,
                "pre_contain_comma_ratio": round(pre_contain_comma_ratio, 6),
                "is_end_contain_comma": is_end_contain_comma,
                "end_contain_comma_ratio": round(end_contain_comma_ratio, 6),
                "is_pre_contain_don": is_pre_contain_don,
                "pre_contain_don_ratio": round(pre_contain_don_ratio, 6),
                "is_end_contain_don": is_end_contain_don,
                "end_contain_don_ratio": round(end_contain_don_ratio, 6),
                "is_pre_contain_quotation": is_pre_contain_quotation,
                "pre_contain_quotation_ratio": round(pre_contain_quotation_ratio, 6),
                "is_end_contain_quotation": is_end_contain_quotation,
                "end_contain_quotation_ratio": round(end_contain_quotation_ratio, 6),
                "is_pre_contain_parentheses": is_pre_contain_parentheses,
                "pre_contain_parentheses_ratio": round(pre_contain_parentheses_ratio, 6),
                "is_end_contain_parentheses": is_end_contain_parentheses,
                "end_contain_parentheses_ratio": round(end_contain_parentheses_ratio, 6),
                "is_pre_contain_book_mark": is_pre_contain_book_mark,
                "pre_contain_book_mark_ratio": round(pre_contain_book_mark_ratio, 6),
                "is_end_contain_book_mark": is_end_contain_book_mark,
                "end_contain_book_mark_ratio": round(end_contain_book_mark_ratio, 6),
                "is_pre_contain_mid_line": is_pre_contain_mid_line,
                "pre_contain_mid_line_ratio": round(pre_contain_mid_line_ratio, 6),
                "is_end_contain_mid_line": is_end_contain_mid_line,
                "end_contain_mid_line_ratio": round(end_contain_mid_line_ratio, 6),
                "is_pre_contain_hash_sign": is_pre_contain_hash_sign,
                "pre_contain_hash_sign_ratio": round(pre_contain_hash_sign_ratio, 6),
                "is_end_contain_hash_sign": is_end_contain_hash_sign,
                "end_contain_hash_sign_ratio": round(end_contain_hash_sign_ratio, 6),
                "is_phrase_contain_mid_line": is_phrase_contain_mid_line,
                "is_phrase_contain_comma": is_phrase_contain_comma,
                "is_phrase_contain_parentheses": is_phrase_contain_parentheses,
                "is_phrase_upper": is_phrase_upper,
                "is_phrase_special": is_phrase_special,
                "mean_phrase_vec": mean_phrase_vec,
            }

            if len(phrase_fea_dict) % 1000 == 0:
                LogUtil.logger.info("phrase fea num: {0}".format(len(phrase_fea_dict)))

        return phrase_fea_dict

    def extract_feature(self, all_phrase_dict, filed_vec_dict) -> dict:
        """
        挖掘候选短语的特征分数
        :param all_phrase_dict: 候选短语及上下文信息
        :param filed_vec_dict: 候选短语包含词的词向量
        :return: dict[phrase_name] = fea_dict
        """
        LogUtil.logger.info("计算特征值")

        # 计算特征值
        phrase_fea_dict = self.cal_feature(all_phrase_dict, filed_vec_dict)

        # # 写入特征到文件
        # if self.phrase_config.IS_WRITE_FEATURE:
        #     with open(self.phrase_config.candidate_phrase_feature_path, "w", encoding="utf-8") as candidate_phrase_feature_file:
        #         for phrase, fea_dict in phrase_fea_dict.items():
        #             candidate_phrase_feature_file.write(phrase + "\t" + json.dumps(fea_dict, ensure_ascii=False) + "\n")

        LogUtil.logger.info("特征值计算完毕")

        return phrase_fea_dict

if __name__ == "__main__":
    pass
