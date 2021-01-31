# encoding: utf-8

import sys
sys.path.append("../../TEBNER")

from collections import Counter

import ahocorasick
from nltk import TreebankWordTokenizer
from util.file_util import FileUtil
from util.log_util import LogUtil

class PubmedProcess(object):

    def __init__(self):
        pass

    def build_ac_automation(self, entity_cut_name_list):
        """
        构建AC自动机
        :param entity_cut_name_list:
        :return:
        """
        LogUtil.logger.info("开始构建AC自动机...")
        ac_automation = ahocorasick.Automaton()
        # 构建Trie树
        for i, entity_cut_name in enumerate(entity_cut_name_list):
            if len(entity_cut_name) < 4:
                continue
            ac_automation.add_word(entity_cut_name, (i, entity_cut_name))
        # 构建自动机
        ac_automation.make_automaton()

        LogUtil.logger.info("AC自动机构架完毕...")

        return ac_automation

    def extract_pubmed_sent(self, entity_name_path, pubmed_data_path, extract_pubmed_data_path):
        """
        从全量pubmed数据中挖掘包含给定实体名的句子
        :param entity_name_path:
        :param pubmed_data_path:
        :param extract_pubmed_data_path:
        :return:
        """
        entity_name_list = FileUtil.read_raw_data(entity_name_path)
        entity_name_set = set([name.lower() for name in entity_name_list])

        # 对实体名称切词
        entity_cut_name_list = [" ".join(TreebankWordTokenizer().tokenize(name)) for name in entity_name_set]
        # 构建AC自动机进行模式串匹配
        ac_automation = self.build_ac_automation(entity_cut_name_list)

        count = 0
        match_mention_dict = {}
        with open(pubmed_data_path, "r", encoding="utf-8") as pubmed_data_file:
            with open(extract_pubmed_data_path, "w", encoding="utf-8") as extract_pubmed_file:
                for i, item in enumerate(pubmed_data_file):
                    item = item.strip()

                    is_save = False
                    extract_phrase_list = list(ac_automation.iter(item))
                    if len(extract_phrase_list) > 0:
                        for end_index, (insert_order, original_value) in extract_phrase_list:
                            if match_mention_dict.get(original_value, 0) < 50:
                                is_save = True
                                match_mention_dict[original_value] = match_mention_dict.get(original_value, 0) + 1
                                break

                        if is_save:
                            extract_pubmed_file.write(item + "\n")
                            count += 1

                    if i % 500000 == 0:
                        LogUtil.logger.info("已解析文本: {0}".format(i))
                        LogUtil.logger.info("已抽取文本: {0}".format(count))

                    # if i > 200000:
                    #     break

        print(len(match_mention_dict), match_mention_dict)

if __name__ == "__main__":

    pubmed_process = PubmedProcess()
    pubmed_process.extract_pubmed_sent("", "", "")