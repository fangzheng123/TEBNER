# encoding: utf-8

import sys
sys.path.append("../../BERTAutoNER")

from util.file_util import FileUtil

class DictProcess(object):
    """
    字典数据处理
    """

    def get_gold_entity(self, source_data_path, gold_entity_path):
        """
        获取标注实体
        :param source_data_path:
        :param gold_entity_path:
        :return:
        """
        all_gold_entity_dict = {}
        text_obj_list = FileUtil.read_text_obj_data(source_data_path)
        for text_obj in text_obj_list:
            for entity_obj in text_obj["entity_list"]:
                all_gold_entity_dict[entity_obj["form"]] = entity_obj["type"]

        FileUtil.save_entity_type(all_gold_entity_dict, gold_entity_path)


if __name__ == "__main__":

    source_data_path = "/data/fangzheng/bert_autoner/bc5cdr/source_data/bc5cdr_all.json"
    gold_entity_path = "/data/fangzheng/bert_autoner/bc5cdr/source_data/bc5cdr_gold_entity.txt"

    dict_process = DictProcess()
    dict_process.get_gold_entity(source_data_path, gold_entity_path)