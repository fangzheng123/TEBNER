# encoding: utf-8

import csv
import nltk
from xml.dom import minidom
from util.file_util import FileUtil
from util.log_util import LogUtil

class BioMedicalDataProcess(object):
    """
    生物医学领域数据处理
    """
    def process_mesh_dict(self, mesh_xml_path, mesh_dict_path):
        """
        解析医疗领域字典
        :param mesh_xml_path:
        :param mesh_dict_path:
        :return:
        """
        medical_name_dict = {}
        dom_tree = minidom.parse(mesh_xml_path)
        # 文档根元素
        root_node = dom_tree.documentElement

        medical_objs = root_node.getElementsByTagName("DescriptorRecord")
        for medical_obj in medical_objs:
            name = medical_obj.getElementsByTagName("DescriptorName")[0].getElementsByTagName("String")[0]
            medical_name_dict[name.childNodes[0].data] = "medical"

        # 存储字典
        FileUtil.save_entity_type(medical_name_dict, mesh_dict_path)

    def process_chemical_and_disease_dict(self, entity_type, dict_csv_path, dict_path):
        """
        加载化学及疾病字典
        :param entity_type: 实体类型
        :param dict_csv_path:
        :param dict_path:
        :return:
        """
        all_lines = csv.reader(open(dict_csv_path, "r"))
        name_dict = {line[0]: entity_type for line in list(all_lines)[30:]}

        # 存储字典
        FileUtil.save_entity_type(name_dict, dict_path)

    def process_txt_to_json(self, data_path, json_path):
        """
        将原文本数据转化为json格式
        :param data_path:
        :param json_path:
        :return:
        """
        all_line_list = open(data_path, "r", encoding="utf-8").readlines()
        if all_line_list[-1] != "\n":
            all_line_list += ["\n"]

        split_index_list = [index for index, line in enumerate(all_line_list) if line.strip() == ""]

        text_obj_list = []
        last_split_index = 0
        for split_index in split_index_list:
            item_list = all_line_list[last_split_index:split_index]

            text_obj = {}
            entity_list = []
            for item in item_list:
                item = item.strip()
                if item.__contains__("|t|"):
                    doc_id, doc_title = item.split("|t|")
                    text_obj["text_id"] = doc_id
                    text_obj["text_title"] = doc_title
                elif item.__contains__("|a|"):
                    doc_id, doc_abstract = item.split("|a|")
                    text_obj["text_abstract"] = doc_abstract
                else:
                    if len(item.split("\t")) < 5:
                        continue
                    doc_id, entity_offset, end, entity_name, entity_type = item.split("\t")[:5]
                    entity_length = len(entity_name)
                    if data_path.__contains__("NCBI"):
                        entity_type = "disease"
                    entity_obj = {
                        "type": entity_type.lower(),
                        "form": entity_name,
                        "offset": int(entity_offset),
                        "length": entity_length,
                    }
                    entity_list.append(entity_obj)

            text_obj["text"] = text_obj["text_title"] + " " + text_obj["text_abstract"]
            text_obj["entity_list"] = entity_list
            text_obj_list.append(text_obj)
            text_obj_list.append(text_obj)
            last_split_index = split_index + 1

            for entity_obj in entity_list:
                print(entity_obj["form"], text_obj["text"][entity_obj["offset"]: entity_obj["offset"]+entity_obj["length"]])

        LogUtil.logger.info("文档数量: {0}, Mention数量: {1}".format(
            len(text_obj_list), sum([len(text_obj["entity_list"]) for text_obj in text_obj_list])))

        FileUtil.save_text_obj_data(text_obj_list, json_path)

    def cut_pos_data(self, data_path):
        """
        对数据进行切词与词性标注
        :param data_path:
        :return:
        """
        text_obj_list = FileUtil.read_text_obj_data(data_path)

        for index, text_obj in enumerate(text_obj_list):
            token_list = nltk.word_tokenize(text_obj["text"])
            token_pos = nltk.pos_tag(token_list)

            text_obj["text_cut"] = token_list
            text_obj["text_pos"] = token_pos

        FileUtil.save_text_obj_data(text_obj_list, data_path)


if __name__ == "__main__":
    bio_data_process = BioMedicalDataProcess()
    bio_data_process.process_txt_to_json("/Users/bytedance/ner_data/NCBI/NCBIdevelopset_corpus.txt",
                                         "/Users/bytedance/ner_data/NCBI/ncbi_dev.json")
