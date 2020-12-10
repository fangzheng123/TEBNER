# encoding: utf-8

from util.file_util import FileUtil
from util.entity_util import EntityUtil

class BaseDataProcessor(object):
    """
    数据处理基类
    """
    def __init__(self, model_config):
        self.model_config = model_config
        self.tokenizer = self.model_config.tokenizer

    def split_text_obj(self, text_obj) -> list:
        """
        将长文本进行划分
        :param text_obj:
        :return:
        """
        split_text_obj_list = []
        all_content = text_obj["text"]
        split_content_list = all_content.split(". ")

        current_content_offset = 0
        for split_index, split_content in enumerate(split_content_list):
            split_content = split_content + ". "
            next_content_offset = current_content_offset + len(split_content)

            label_entity_list = []
            for entity_obj in text_obj["entity_list"]:
                if current_content_offset <= entity_obj["offset"] < next_content_offset:
                    entity_obj["offset"] = entity_obj["offset"] - current_content_offset
                    label_entity_list.append(entity_obj)

            distance_entity_list = []
            for entity_obj in text_obj["distance_entity_list"]:
                if current_content_offset <= entity_obj["offset"] < next_content_offset:
                    entity_obj["offset"] = entity_obj["offset"] - current_content_offset
                    distance_entity_list.append(entity_obj)

            current_content_offset = next_content_offset

            split_text_obj = {
                "text_id": text_obj["text_id"] + "_" + str(split_index),
                "text": split_content,
                "entity_list": label_entity_list,
                "distance_entity_list": distance_entity_list
            }
            split_text_obj_list.append(split_text_obj)

        all_label_num = sum([len(ele["entity_list"]) for ele in split_text_obj_list])
        all_distance_num = sum([len(ele["distance_entity_list"]) for ele in split_text_obj_list])

        assert all_label_num == len(text_obj["entity_list"])
        assert all_distance_num == len(text_obj["distance_entity_list"])

        return split_text_obj_list

    def get_split_text_obj(self, data_path):
        """
        获取切分后的文本对象
        :param data_path:
        :return:
        """
        all_text_obj_list = FileUtil.read_text_obj_data(data_path)

        all_split_text_obj_list = []
        for text_obj in all_text_obj_list:
            # 对长文本按句号进行划分
            split_text_obj_list = self.split_text_obj(text_obj)
            all_split_text_obj_list.extend(split_text_obj_list)

        return all_split_text_obj_list

    def get_entity_token_pos(self, entity_obj, content):
        """
        获取实体在bert token中的位置（在某些情况下单词位置和token后的位置不一致）
        :param entity_obj:
        :param content:
        :param tokenizer:
        :return: 实体位置为: [token_begin ... token_end], token_end即为实体最后一位
        """
        offset = entity_obj["offset"]
        end = offset + len(entity_obj["form"])

        mask_content = content[:offset] + "[MASK]" + content[offset:end] + "[MASK]" + content[end:]
        mask_token_list = self.tokenizer.tokenize(mask_content)

        mask_pos_list = [i for i, token in enumerate(mask_token_list) if token == "[MASK]"]

        token_begin = mask_pos_list[0]
        token_end = mask_pos_list[1] - 2

        return token_begin, token_end

    def save_token_label(self, all_seq_token_list, token_label_path):
        """
        将数据存储为BIOS格式
        :param all_seq_token_list: 包含(token_list, seq_connect_label, seq_type_label)的列表
        :param token_label_path: 写入标注结果的文件路径
        :return:
        """
        all_sent_list = []
        for token_list, seq_connect_label, seq_type_label in all_seq_token_list:
            sent_list = []
            for i in range(min(len(token_list), len(seq_connect_label))):
                sent_list.append((token_list[i], seq_connect_label[i], seq_type_label[i]))

            all_sent_list.append(sent_list)

        with open(token_label_path, "w", encoding="utf-8") as token_label_file:
            for sent_list in all_sent_list:
                for item in sent_list:
                    token_label_file.write(" ".join(item) + "\n")

                token_label_file.write("\n")
