# encoding: utf-8

import sys
sys.path.append("../../TEBNER")

from gensim.models import KeyedVectors
from nltk import TreebankWordTokenizer


from util.file_util import FileUtil
from util.log_util import LogUtil


class DictAnalyse(object):
    """
    字典分析
    """

    def __init__(self):
        pass

    def read_entity_name(self):
        data_path = "/ncbi/source_data/all.json"
        text_obj_list = FileUtil.read_text_obj_data(data_path)

        all_name_list = []
        for text_obj in text_obj_list:
            entity_name_list = [entity_obj["form"] for entity_obj in text_obj["entity_list"]]
            all_name_list.extend(entity_name_list)

        with open("/ncbi/source_data/ncbi_entity_name.txt", "w",
                  encoding="utf-8") as name_file:
            for name in set(all_name_list):
                name_file.write(name + "\n")

    def analyse_dict_data_recall(self, raw_data_path, data_dict_path, autophrase_dict_path):
        """
        分析字典
        :param raw_data_path:
        :param data_dict_path:
        :return:
        """
        data_dict = FileUtil.read_entity_type_dict(data_dict_path)
        data_dict = {form.lower(): form_type for form, form_type in data_dict.items()}
        autophrase_list = FileUtil.read_raw_data(autophrase_dict_path)
        autophrase_set = set([phrase.lower() for phrase in autophrase_list])
        text_obj_list = FileUtil.read_text_obj_data(raw_data_path)

        all_num = 0
        recall_num = 0
        phrase_recall_num = 0
        for text_obj in text_obj_list:
            all_num += len(text_obj["entity_list"])
            for entity_obj in text_obj["entity_list"]:
                if entity_obj["form"].lower() in data_dict:
                    recall_num += 1
                if entity_obj["form"].lower() in autophrase_set:
                    phrase_recall_num += 1

        LogUtil.logger.info("所有实体数量: {0}, 字典召回的实体数量: {1}, 实体召回率: {2}".format(all_num, recall_num, recall_num/all_num))
        LogUtil.logger.info("所有实体数量: {0}, 短语字典召回的实体数量: {1}, 实体召回率: {2}".format(all_num, phrase_recall_num, phrase_recall_num/all_num))

        # 所有实体数量: 9809, 字典召回的实体数量: 5996, 实体召回率: 0.61
        # 所有实体数量: 960, 字典召回的实体数量: 503, 实体召回率: 0.52
        # 所有实体数量: 654, 字典召回的实体数量: 330, 实体召回率: 0.50

    def compare_dict(self, our_dict_path, other_dict_path, mesh_dict_path):
        """
        比较字典性能
        :param our_dict_path:
        :param other_dict_path:
        :param mesh_dict_path:
        :return:
        """
        our_data_dict = FileUtil.read_entity_type_dict(our_dict_path)
        our_data_dict = {form.lower(): form_type for form, form_type in our_data_dict.items()}
        other_data_dict = FileUtil.read_entity_type_dict(other_dict_path)
        other_data_dict = {form.lower(): form_type for form, form_type in other_data_dict.items()}
        mesh_dict = FileUtil.read_entity_type_dict(other_dict_path)
        mesh_dict = {form.lower(): form_type for form, form_type in mesh_dict.items()}

        in_our_count = 0
        in_mesh_count = 0
        for form, form_type in other_data_dict.items():
            if form in our_data_dict:
                in_our_count += 1
            elif form not in our_data_dict and form in mesh_dict:
                in_mesh_count += 1

        print(in_our_count, in_mesh_count)

    def dict_vec_analyse(self):
        """
        分析字典词向量
        :return:
        """
        all_vec_path = "/word2vec/wikipedia-pubmed-and-PMC-w2v.bin"

        entity_name_list = FileUtil.read_raw_data("/bc5cdr/source_data/bc5cdr_entity_name.txt")
        entity_name_set = set([name for name in entity_name_list])

        all_token_list = []
        for name in entity_name_set:
            token_list = TreebankWordTokenizer().tokenize(name)
            all_token_list.extend(token_list)
        all_token_set = set(all_token_list)

        LogUtil.logger.info("开始加载词向量...")
        wv_from_bin = KeyedVectors.load_word2vec_format(all_vec_path, binary=True)
        LogUtil.logger.info("加载词向量完毕")

        count = 0
        for token in all_token_set:
            if token not in wv_from_bin:
                print(token)
                count += 1

        print(count, len(all_token_set))

if __name__ == "__main__":
    dict_analyse = DictAnalyse()

    root_dir = "bc5cdr/source_data/"
    dict_analyse.analyse_dict_data_recall(root_dir + "all.json",
                                          root_dir + "bc5cdr_dict.txt",
                                          root_dir + "bc5cdr_autophrase.txt")

