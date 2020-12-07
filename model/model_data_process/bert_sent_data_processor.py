# encoding: utf-8

import json
import torch
from torch.utils.data import DataLoader, TensorDataset, RandomSampler, SequentialSampler

from util.log_util import LogUtil
from util.entity_util import EntityUtil
from util.file_util import FileUtil
from model.model_config.bert_sent_config import BERTSentConfig

class BERTSentDataProcessor(object):
    """
    加载BERT 序列标注模型的训练、验证、测试数据
    """

    def __init__(self, args):
        self.model_config = BERTSentConfig(args)
        self.tokenizer = self.model_config.tokenizer
        self.entity_util = EntityUtil()

    def get_entity_token_pos(self, entity_obj, content):
        """
        获取实体在bert token中的位置（在某些情况下单词位置和token后的位置不一致）
        :param entity_obj:
        :param content:
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

    def get_seq_label(self, entity_list):
        """
        获取序列标注结果
        :param entity_list:
        :return:
        """
        seq_label = ["O"] * self.model_config.max_seq_len

        for entity_obj in entity_list:
            # 实体所在位置超过序列最大长度则当前实体不打标
            if "bert_token_pos" not in entity_obj:
                continue

            token_begin, token_end = entity_obj["bert_token_pos"]

            # 英文实体打标(word tokenize后的首位token打标，其他标注为"X")
            # seq_label[token_begin] = "B-" + entity_obj["type"]
            # seq_label[token_begin+1: token_end+1] = ["X"] * (token_end - token_begin)
            # if len(entity_obj["form"].split()) > 1:
            #     inter_word_offset_list = []
            #     for word in entity_obj["form"].split()[:-1]:
            #         inter_word_offset_list.append(token_begin+len(self.tokenizer.tokenize(word)))
            #     for inter_offset in inter_word_offset_list:
            #         seq_label[inter_offset] = "I-" + entity_obj["type"]

            seq_label[token_begin] = "B-" + entity_obj["type"]
            seq_label[token_begin + 1:token_end + 1] = ['I-' + entity_obj["type"]] * (token_end - token_begin)

        return seq_label

    def save_token_label(self, all_seq_token_list, token_label_path):
        """
        将数据存储为BIOS格式
        :param all_seq_token_list: 包含(token_list及token标注)的列表
        :param token_label_path: 写入标注结果的文件路径
        :return:
        """
        all_sent_list = []
        for token_list, seq_label in all_seq_token_list:
            sent_list = []
            for i in range(min(len(token_list), len(seq_label))):
                sent_list.append((token_list[i], seq_label[i]))

            all_sent_list.append(sent_list)

        with open(token_label_path, "w", encoding="utf-8") as token_label_file:
            for sent_list in all_sent_list:
                for token, label in sent_list:
                    token_label_file.write(token + " " + label + "\n")

                token_label_file.write("\n")

    def load_dataset(self, data_path, is_train=False, is_dev=False, is_test=False):
        """
        加载模型所需数据，包括训练集，验证集，测试集（有标签） 及预测集合（无标签）
        :param data_path:
        :param is_train: 是否为训练集
        :param is_dev: 是否为验证集
        :param is_test: 是否为测试集
        :return:
        """
        all_data_list = []
        all_seq_token_list = []

        token_len_list = []
        all_text_obj_list = []
        with open(data_path, "r", encoding="utf-8") as data_file:
            for item in data_file:
                item = item.strip()
                text_obj = json.loads(item)

                # 对长文本按句号进行划分
                split_text_obj_list = EntityUtil.split_text_obj(text_obj)
                all_text_obj_list.extend(split_text_obj_list)

            # 处理数据
            for split_text_obj in all_text_obj_list:
                content = split_text_obj["text"]

                # 加载序列标签（预测数据无标签）
                seq_label = ["O"] * self.model_config.max_seq_len
                if is_train or is_dev or is_test:
                    if is_train:
                        entity_list = split_text_obj["distance_entity_list"]
                    else:
                        entity_list = split_text_obj["entity_list"]

                    for entity_obj in entity_list:
                        if entity_obj["type"] == "unknown":
                            continue
                        # 获取实体在bert分词后的位置
                        token_begin, token_end = self.get_entity_token_pos(entity_obj, content)
                        # 实体所在位置超过序列最大长度则当前实体不打标
                        if token_end >= self.model_config.max_seq_len - 2:
                            continue
                        # 加 [CLS]
                        entity_obj["bert_token_pos"] = (token_begin + 1, token_end + 1)

                    # 获取序列中每个token的标签
                    seq_label = self.get_seq_label(entity_list)

                encoded_dict = self.tokenizer.encode_plus(content, truncation=True, padding="max_length",
                                                          max_length=self.model_config.max_seq_len)

                all_data_list.append((encoded_dict["input_ids"], encoded_dict["attention_mask"],
                                      encoded_dict["token_type_ids"], seq_label))

                # 保存bios数据格式用
                token_list = self.tokenizer.tokenize("[CLS]" + content + "[SEP]")
                all_seq_token_list.append((token_list, seq_label))
                token_len_list.append(len(token_list))

            for i in range(3):
                print(all_data_list[i][0])
                print(all_data_list[i][1])
                print(all_data_list[i][2])
                print(all_data_list[i][3])

        LogUtil.logger.info("token切分后最大长度为: {}".format(max(token_len_list)))
        # 将数据存储为BIOS格式, 方便人为检查和查看
        token_label_path = data_path + "_bios"

        # if not os.path.exists(token_label_path):
        #     self.save_token_label(all_seq_token_list, token_label_path)
        self.save_token_label(all_seq_token_list, token_label_path)

        all_input_ids = torch.LongTensor([_[0] for _ in all_data_list])
        all_input_mask = torch.LongTensor([_[1] for _ in all_data_list])
        all_type_ids = torch.LongTensor([_[2] for _ in all_data_list])
        all_label_ids = torch.LongTensor([[self.model_config.label_id_dict[label]
                                           for label in _[3]] for _ in all_data_list])
        tensor_dataset = TensorDataset(all_input_ids, all_input_mask, all_type_ids, all_label_ids)

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
        return dataloader

    def extract_entity(self, all_seq_score_list, all_seq_tag_list):
        """
        从序列中挖掘实体
        :param all_seq_score_list: 所有序列中每个token类别的预测分数
        :param all_seq_tag_list: 所有序列中每个token预测类别
        :return:
        """
        all_entity_list = []
        for seq_score_list, seq_tag_list in zip(all_seq_score_list, all_seq_tag_list):
            pre_entities = self.entity_util.get_seq_entity(seq_tag_list)
            for entity in pre_entities:
                token_num = entity[2] - entity[1] + 1
                if entity[2] - entity[1] + 1 == 0:
                    token_num = max(1, len(seq_score_list))
                entity_scores = round(sum(seq_score_list[entity[1]:entity[2] + 1]) / token_num, 2)
                entity.append(entity_scores)

            all_entity_list.append(pre_entities)

        return all_entity_list

    def output_entity(self, all_seq_entity_list, data_path, output_path):
        """
        输出挖掘实体结果
        :param all_seq_entity_list:
        :param data_path:
        :param output_path:
        :return:
        """
        all_text_obj_list = FileUtil.read_text_obj_data(data_path)
        with open(output_path, "w", encoding="utf-8") as output_file:
            for text_obj, seq_entity_list in zip(all_text_obj_list, all_seq_entity_list):
                content = text_obj["text"]
                seq_token = self.tokenizer.tokenize("[CLS]" + content + "[SEP]")

                entity_obj_list = []
                for i in range(len(seq_entity_list)):
                    entity_obj = {}
                    entity_type, entity_begin, entity_end, entity_score = seq_entity_list[i]
                    entity_obj["form"] = "".join(seq_token[entity_begin: entity_end + 1]).replace("##", "")
                    if entity_obj["form"] == "":
                        continue
                    entity_obj["token_score"] = entity_score
                    entity_obj["type"] = entity_type
                    entity_obj["offset"] = len("".join(seq_token[1:entity_begin]).replace("##", ""))
                    entity_obj["length"] = len(entity_obj["form"])
                    entity_obj_list.append(entity_obj)

                text_obj["entity_list"] = entity_obj_list
                output_file.write(json.dumps(text_obj, ensure_ascii=False) + "\n")
