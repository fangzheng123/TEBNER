# encoding: utf-8

import os
import json
import torch
from torch.utils.data import DataLoader, TensorDataset, RandomSampler, SequentialSampler

from util.file_util import FileUtil
from model.model_data_process.base_data_processor import BaseDataProcessor

class BERTWordProcessor(BaseDataProcessor):
    """
    加载BERT Word模型的训练、验证、测试数据
    """
    def __init__(self, model_config):
        super().__init__(model_config)

    def get_token_label(self, entity_list, encoded_dict):
        """
        对序列中每个token进行打标
        两种标注: 一种是tie or break; 一种是entity_type
        当类别为"unknown"时, 连接关系标注为S(skip)
        :param entity_list:
        :return: seq_connect_label(B 表示Break, T 表示Tie, S 表示Skip), seq_type_label
        """
        seq_connect_label = ["B"] * (self.model_config.max_seq_len - 1)
        seq_type_label = ["None"] * self.model_config.max_seq_len

        # token连接关系mask列表
        token_connect_mask = encoded_dict["attention_mask"][:-1]

        for entity_obj in entity_list:
            # 实体所在位置超过序列最大长度则当前实体不打标
            if "bert_token_pos" not in entity_obj:
                continue
            
            # 加 [CLS]
            entity_token_begin, entity_token_end = entity_obj["bert_token_pos"]

            # 当前token与下一个token的连接关系打标
            if entity_obj["type"] == "unknown":
                seq_connect_label[entity_token_begin: entity_token_end] = ["S"] * (entity_token_end - entity_token_begin)
                token_connect_mask[entity_token_begin: entity_token_end] = [0] * (entity_token_end - entity_token_begin)
            else:
                seq_connect_label[entity_token_begin: entity_token_end] = ["T"] * (entity_token_end - entity_token_begin)

            seq_type_label[entity_token_begin: entity_token_end+1] = [entity_obj["type"]] * (entity_token_end+1-entity_token_begin)

        return seq_connect_label, seq_type_label, token_connect_mask

    def load_dataset(self, data_path, is_train=False, is_dev=False, is_test=False):
        """
        加载模型所需数据，包括训练集，验证集，测试集（有标签） 及预测集合（无标签）
        """
        all_split_text_obj_list = self.get_split_text_obj(data_path)

        # token之间连接关系数据
        all_token_connect_mask = []
        all_token_connect_labels = []

        # 实体相关数据
        all_entity_begins = []
        all_entity_ends = []
        all_entity_type_labels = []
        all_sent_index_list = []

        # 用于bios格式
        all_bios_word_list = []
        # 用于bert模型输入
        all_token_encode_list = []
        # 用于评测
        sent_entity_dict = {}
        for sent_index, split_text_obj in enumerate(all_split_text_obj_list):
            content = split_text_obj["text"]
            encoded_dict = self.tokenizer.encode_plus(content, truncation=True, padding="max_length",
                                                      max_length=self.model_config.max_seq_len)
            # 打标数据
            if is_train or is_dev or is_test:
                if is_train:
                    entity_list = split_text_obj["distance_entity_list"]
                else:
                    entity_list = split_text_obj["entity_list"]
                seq_entity_num = 0
                for entity_obj in entity_list:
                    # 获取实体在token列表中的首尾位置
                    entity_token_begin, entity_token_end = self.get_entity_token_pos(entity_obj, content)

                    # 实体所在位置超过序列最大长度则当前实体不打标(-2是考虑了[CLS]和[SEP]的位置)
                    if entity_token_end >= self.model_config.max_seq_len - 2:
                        continue

                    # 加 [CLS]
                    entity_obj["bert_token_pos"] = (entity_token_begin + 1, entity_token_end + 1)

                    # 训练时不考虑unknown类别(即提前挖掘的高质量短语)
                    if entity_obj["type"] != "unknown":
                        if is_train:
                            # 加 [CLS]
                            all_entity_begins.append(entity_token_begin + 1)
                            all_entity_ends.append(entity_token_end + 1)
                            all_entity_type_labels.append(self.model_config.type_label_id_dict[entity_obj["type"]])
                            seq_entity_num += 1
                        else:
                            sent_entity_dict.setdefault(sent_index, [])\
                                .append((entity_token_begin + 1, entity_token_end + 1,
                                         self.model_config.type_label_id_dict[entity_obj["type"]]))

                # 对序列中每个词语打标
                seq_connect_label, seq_type_label, token_connect_mask = self.get_token_label(entity_list, encoded_dict)
                # 训练时每个实体将单独预测，因此一个句子中含有n个实体时，将被复制n次
                if is_train:
                    all_token_encode_list.extend([encoded_dict for _ in range(seq_entity_num)])
                    all_token_connect_mask.extend([token_connect_mask for _ in range(seq_entity_num)])
                    all_token_connect_labels.extend([[self.model_config.connect_label_id_dict[ele] for ele
                                                      in seq_connect_label] for _ in range(seq_entity_num)])
                    all_sent_index_list.extend([sent_index for _ in range(seq_entity_num)])
                else:
                    all_token_encode_list.append(encoded_dict)
                    all_token_connect_mask.append(token_connect_mask)
                    all_token_connect_labels.append([self.model_config.connect_label_id_dict[ele] for ele
                                                      in seq_connect_label])
                    all_sent_index_list.append(sent_index)
                    all_entity_begins.append(0)
                    all_entity_ends.append(0)
                    all_entity_type_labels.append(self.model_config.type_label_id_dict["None"])
            # 非打标数据
            else:
                all_token_encode_list.append(encoded_dict)
                all_token_connect_mask.append(encoded_dict["attention_mask"][:-1])
                all_token_connect_labels.append([self.model_config.connect_label_id_dict["B"]]
                                                * (self.model_config.max_seq_len - 1))
                all_entity_begins.append(0)
                all_entity_ends.append(0)
                all_entity_type_labels.append(self.model_config.type_label_id_dict["None"])
                all_sent_index_list.append(sent_index)
                seq_connect_label = ["B"] * (self.model_config.max_seq_len - 1)
                seq_type_label = ["None"] * self.model_config.max_seq_len

            # 保存bios数据格式用
            all_bios_word_list.append(
                (self.tokenizer.tokenize("[CLS]" + content + "[SEP]"), seq_connect_label, seq_type_label))

        # 将数据存储为BIOS格式, 方便人为检查和查看
        token_label_path = data_path + "_bios"
        if not os.path.exists(token_label_path):
            self.save_token_label(all_bios_word_list, token_label_path)

        assert len(all_token_encode_list) == len(all_token_connect_mask) == len(all_token_connect_labels)\
               == len(all_entity_begins) == len(all_entity_ends) == len(all_entity_type_labels)

        # 模型输入数据
        all_input_ids = torch.LongTensor([encoded_dict["input_ids"] for encoded_dict in all_token_encode_list])
        all_input_mask = torch.LongTensor([encoded_dict["attention_mask"] for encoded_dict in all_token_encode_list])
        all_token_type_ids = torch.LongTensor([encoded_dict["token_type_ids"] for encoded_dict in all_token_encode_list])
        all_token_connect_masks = torch.LongTensor(all_token_connect_mask)
        all_token_connect_labels = torch.LongTensor(all_token_connect_labels)
        all_entity_begins = torch.LongTensor(all_entity_begins)
        all_entity_ends = torch.LongTensor(all_entity_ends)
        all_entity_type_labels = torch.LongTensor(all_entity_type_labels)
        all_sent_indexs = torch.LongTensor(all_sent_index_list)

        # for i in range(4):
        #     print(all_input_ids[i])
        #     print(all_input_mask[i])
        #     print(all_token_type_ids[i])
        #     print(all_token_connect_masks[i])
        #     print(all_token_connect_labels[i])
        #     print(all_entity_begins[i])
        #     print(all_entity_ends[i])
        #     print(all_entity_type_labels[i])

        tensor_dataset = TensorDataset(all_input_ids, all_input_mask, all_token_type_ids,
                                       all_token_connect_masks, all_token_connect_labels,
                                       all_entity_begins, all_entity_ends, all_entity_type_labels, all_sent_indexs)

        if is_train:
            batch_size = self.model_config.train_batch_size
            data_sampler = RandomSampler(tensor_dataset)
        elif is_dev:
            batch_size = self.model_config.dev_batch_size
            data_sampler = SequentialSampler(tensor_dataset)
        else:
            batch_size = self.model_config.test_batch_size
            data_sampler = SequentialSampler(tensor_dataset)

        dataloader = DataLoader(tensor_dataset, sampler=data_sampler, batch_size=batch_size)

        if is_dev or is_test:
            return dataloader, sent_entity_dict
        else:
            return dataloader

    def output_entity(self, all_seq_entity_dict, data_path, output_path):
        """
        输出挖掘实体结果
        :param all_seq_entity_list:
        :param data_path:
        :param output_path:
        :return:
        """
        all_text_obj_list = FileUtil.read_text_obj_data(data_path)
        with open(output_path, "w", encoding="utf-8") as output_file:
            for index, text_obj in enumerate(all_text_obj_list):
                content = text_obj["text"]
                seq_token = self.tokenizer.tokenize("[CLS]" + content + "[SEP]")

                if index not in all_seq_entity_dict:
                    continue

                new_entity_list = []
                entity_obj_list = all_seq_entity_dict[index]
                for entity_obj in entity_obj_list:
                    entity_begin, entity_end = entity_obj["token_pos"]
                    entity_obj["offset"] = len("".join(seq_token[1:entity_begin]).replace("##", ""))
                    if entity_obj["offset"] >= len(text_obj["text"]):
                        continue

                    entity_obj["form"] = "".join(seq_token[entity_begin: entity_end + 1]).replace("##", "")
                    entity_obj["length"] = len(entity_obj["form"])
                    del entity_obj["token_pos"]

                    new_entity_list.append(entity_obj)

                text_obj["entity_list"] = new_entity_list
                output_file.write(json.dumps(text_obj, ensure_ascii=False) + "\n")
