# encoding: utf-8

from xml.dom import minidom
from util.file_util import FileUtil

class LaptopDataProcess(object):
    """
    笔记本领域词典处理run_seed_expand.py
    """
    def process_laptop_to_json(self, laptop_xml_path, laptop_json_path):
        """
        将Laptop数据转化为json格式
        :param laptop_xml_path:
        :param laptop_json_path:
        :return:
        """
        dom_tree = minidom.parse(laptop_xml_path)
        # 文档根元素
        root_node = dom_tree.documentElement

        text_obj_list = []
        sent_objs = root_node.getElementsByTagName("sentence")
        for sent_obj in sent_objs:
            text_id = sent_obj.getAttribute("id")
            text = sent_obj.getElementsByTagName("text")[0].childNodes[0].data

            entity_list = []
            if len(sent_obj.getElementsByTagName("aspectTerms")) > 0:
                term_objs = sent_obj.getElementsByTagName("aspectTerms")[0].getElementsByTagName("aspectTerm")
                for term_obj in term_objs:
                    entity_form = term_obj.getAttribute("term")
                    entity_offset = term_obj.getAttribute("from")
                    entity_length = len(entity_form)

                    entity_obj = {
                        "form": entity_form,
                        "offset": entity_offset,
                        "length": entity_length,
                        "type": "laptop"
                    }
                    entity_list.append(entity_obj)

            text_obj = {
                "text_id": text_id,
                "text": text,
                "entity_list": entity_list
            }
            text_obj_list.append(text_obj)

        FileUtil.save_text_obj_data(text_obj_list, laptop_json_path)


if __name__ == "__main__":
    laptop_data_process = LaptopDataProcess()
    laptop_data_process.process_laptop_to_json("/Users/bytedance/ner_data/LaptopReview/Laptop_Train_v2.xml",
                                               "/Users/bytedance/ner_data/LaptopReview/laptop_train.json")






