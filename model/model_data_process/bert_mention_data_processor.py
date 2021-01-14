# encoding: utf-8

import torch
from torch.utils.data import DataLoader, TensorDataset, RandomSampler, SequentialSampler

from model.model_data_process.base_data_processor import BaseDataProcessor

class BERTMentionDataProcessor(BaseDataProcessor):
    """
    BERT Mention分类模型处理
    """
    def __init__(self, model_config):
        super().__init__(model_config)

    def load_dataset(self, data_path, is_train=False, is_dev=False, is_test=False, is_predict=False):
        """
        加载模型所需数据，包括训练集，验证集，测试集（有标签） 及预测集合（无标签）
        :param data_path:
        :param is_train: 是否为训练集
        :param is_dev: 是否为验证集
        :param is_test: 是否为测试集
        :param is_test: 是否为预测集
        :return:
        """
        all_text_obj_list = self.get_split_text_obj(data_path)

        # 处理数据
        all_data_list = []
        for split_text_obj in all_text_obj_list:
            content = split_text_obj["text"]
            # mention位置
            mention_loc_list = []
            # mention类别标签（预测数据无标签）
            mention_label_list = []

            # 测试集加载正确标签，其他情况加载远程标注标签
            if is_test or is_dev:
                entity_list = split_text_obj["entity_list"]
            else:
                entity_list = split_text_obj["distance_entity_list"]

            for entity_obj in entity_list:
                # 获取实体在bert分词后的位置
                token_begin, token_end = self.get_entity_token_pos(entity_obj, content)
                # 实体所在位置超过序列最大长度则当前实体不打标
                if token_end >= self.model_config.max_seq_len - 2:
                    continue

                if (is_train or is_dev or is_test) and entity_obj["type"] != "unknown":
                    # 加 [CLS]
                    mention_loc_list.append((token_begin + 1, token_end + 1))
                    mention_label_list.append(entity_obj["type"])

                # 预测时专门对unknown标注mention进行类型预测
                if is_predict and entity_obj["type"] == "unknown":
                    # 加 [CLS]
                    mention_loc_list.append((token_begin + 1, token_end + 1))
                    # 预测时随机选择1个标签用于占位
                    mention_label_list.append(self.model_config.label_list[0])

            encoded_dict = self.tokenizer.encode_plus(content, truncation=True, padding="max_length",
                                                      max_length=self.model_config.max_seq_len)

            all_data_list.extend([(encoded_dict["input_ids"], encoded_dict["attention_mask"],
                                   encoded_dict["token_type_ids"], mention_beg, mention_end, mention_label)
                                  for (mention_beg, mention_end), mention_label in
                                  zip(mention_loc_list, mention_label_list)])

        all_input_ids = torch.LongTensor([_[0] for _ in all_data_list])
        all_input_mask = torch.LongTensor([_[1] for _ in all_data_list])
        all_type_ids = torch.LongTensor([_[2] for _ in all_data_list])
        all_mention_begs = torch.LongTensor([_[3] for _ in all_data_list])
        all_mention_ends = torch.LongTensor([_[4] for _ in all_data_list])
        all_label_ids = torch.LongTensor([self.model_config.label_id_dict[_[5]] for _ in all_data_list])
        tensor_dataset = TensorDataset(all_input_ids, all_input_mask, all_type_ids,
                                       all_mention_begs, all_mention_ends, all_label_ids)

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

    def load_data_from_distance(self, data_path):
        """
        根据远程标注结果加载分类数据
        :param data_path:
        :return:
        """
        all_text_obj_list = self.get_split_text_obj(data_path)

        # 处理数据
        all_data_list = []
        all_entity_sent_index_list = []
        all_sent_label_dict = {}
        for sent_index, split_text_obj in enumerate(all_text_obj_list):
            content = split_text_obj["text"]
            # mention位置
            mention_loc_list = []

            label_entity_list = split_text_obj["entity_list"]
            distance_entity_list = split_text_obj["distance_entity_list"]

            # 获取远程标注的实体位置
            for distance_entity_obj in distance_entity_list:
                # 获取实体在bert分词后的位置
                token_begin, token_end = self.get_entity_token_pos(distance_entity_obj, content)
                # 实体所在位置超过序列最大长度则当前实体不打标
                if token_end >= self.model_config.max_seq_len - 2:
                    continue
                # 加 [CLS]
                mention_loc_list.append((token_begin + 1, token_end + 1))
                all_entity_sent_index_list.append(sent_index)

            # 获取真实标注的实体位置及类型
            for label_entity_obj in label_entity_list:
                # 获取实体在bert分词后的位置
                token_begin, token_end = self.get_entity_token_pos(label_entity_obj, content)
                # 实体所在位置超过序列最大长度则当前实体不打标
                if token_end >= self.model_config.max_seq_len - 2:
                    continue
                # 加 [CLS]
                all_sent_label_dict.setdefault(sent_index, []).append((
                    token_begin + 1, token_end + 1, self.model_config.label_id_dict[label_entity_obj["type"]]))

            encoded_dict = self.tokenizer.encode_plus(content, truncation=True, padding="max_length",
                                                      max_length=self.model_config.max_seq_len)

            all_data_list.extend([(encoded_dict["input_ids"], encoded_dict["attention_mask"],
                                   encoded_dict["token_type_ids"], mention_beg, mention_end)
                                  for (mention_beg, mention_end) in mention_loc_list])

        all_input_ids = torch.LongTensor([_[0] for _ in all_data_list])
        all_input_mask = torch.LongTensor([_[1] for _ in all_data_list])
        all_type_ids = torch.LongTensor([_[2] for _ in all_data_list])
        all_mention_begs = torch.LongTensor([_[3] for _ in all_data_list])
        all_mention_ends = torch.LongTensor([_[4] for _ in all_data_list])
        all_sent_indexs = torch.LongTensor(all_entity_sent_index_list)
        tensor_dataset = TensorDataset(all_input_ids, all_input_mask, all_type_ids,
                                       all_mention_begs, all_mention_ends, all_sent_indexs)

        batch_size = self.model_config.test_batch_size
        data_sampler = SequentialSampler(tensor_dataset)

        dataloader = DataLoader(tensor_dataset, sampler=data_sampler, batch_size=batch_size)

        return dataloader, all_sent_label_dict

    def load_data_from_sent_model(self, data_path, all_sent_entity_dict):
        """
        从边界识别结果中构造数据
        :param data_path:
        :param all_sent_entity_dict:
        :return:
        """
        all_text_obj_list = self.get_split_text_obj(data_path)

        # 处理数据
        all_data_list = []
        all_entity_sent_index_list = []
        all_sent_label_dict = {}
        for sent_index, split_text_obj in enumerate(all_text_obj_list):
            content = split_text_obj["text"]

            pred_entity_list = all_sent_entity_dict.get(sent_index, [])
            label_entity_list = split_text_obj["entity_list"]

            # mention位置
            mention_loc_list = []

            # 获取远程监督位置
            distance_token_begin_dict = {}
            distance_entity_list = split_text_obj["distance_entity_list"]
            for dis_entity_obj in distance_entity_list:
                # 获取实体在bert分词后的位置
                token_begin, token_end = self.get_entity_token_pos(dis_entity_obj, content)
                # 加[CLS]
                distance_token_begin_dict[token_begin+1] = (token_begin+1, token_end+1)
                mention_loc_list.append((token_begin+1, token_end+1))
                all_entity_sent_index_list.append(sent_index)

            # 获取模型预测的实体位置
            for pred_entity_tuple in pred_entity_list:
                _, token_begin, token_end, token_score = pred_entity_tuple
                if token_end >= self.model_config.max_seq_len - 2:
                    continue
                # 当起始位置相同时，以远程监督位置为准
                if token_begin not in distance_token_begin_dict:
                    # 边界模型预测结果时偏移已经计算[CLS],此处无需加1
                    mention_loc_list.append((token_begin, token_end))
                    all_entity_sent_index_list.append(sent_index)

            # 获取真实标注的实体位置及类型
            for label_entity_obj in label_entity_list:
                # 获取实体在bert分词后的位置
                token_begin, token_end = self.get_entity_token_pos(label_entity_obj, content)
                # 实体所在位置超过序列最大长度则当前实体不打标
                if token_end >= self.model_config.max_seq_len - 2:
                    continue
                # 加 [CLS]
                all_sent_label_dict.setdefault(sent_index, []).append((
                    token_begin + 1, token_end + 1, self.model_config.label_id_dict[label_entity_obj["type"]]))

            encoded_dict = self.tokenizer.encode_plus(content, truncation=True, padding="max_length",
                                                      max_length=self.model_config.max_seq_len)

            all_data_list.extend([(encoded_dict["input_ids"], encoded_dict["attention_mask"],
                                   encoded_dict["token_type_ids"], mention_beg, mention_end)
                                  for (mention_beg, mention_end) in mention_loc_list])

        all_input_ids = torch.LongTensor([_[0] for _ in all_data_list])
        all_input_mask = torch.LongTensor([_[1] for _ in all_data_list])
        all_type_ids = torch.LongTensor([_[2] for _ in all_data_list])
        all_mention_begs = torch.LongTensor([_[3] for _ in all_data_list])
        all_mention_ends = torch.LongTensor([_[4] for _ in all_data_list])
        all_sent_indexs = torch.LongTensor(all_entity_sent_index_list)
        tensor_dataset = TensorDataset(all_input_ids, all_input_mask, all_type_ids,
                                       all_mention_begs, all_mention_ends, all_sent_indexs)

        batch_size = self.model_config.test_batch_size
        data_sampler = SequentialSampler(tensor_dataset)

        dataloader = DataLoader(tensor_dataset, sampler=data_sampler, batch_size=batch_size)

        return dataloader, all_sent_label_dict

    def output_mention_type(self, data_path, all_mention_type_list, all_mention_score_list):
        """
        输出mention类型
        :param data_path:
        :param all_mention_type_list:
        :param all_mention_score_list:
        :return:
        """
        all_text_obj_list = self.get_split_text_obj(data_path)

        all_mention_form_list = []
        all_mention_loc_list = []
        all_mention_content_list = []
        for split_text_obj in all_text_obj_list:
            content = split_text_obj["text"]
            entity_list = split_text_obj["distance_entity_list"]

            for entity_obj in entity_list:
                # 获取实体在bert分词后的位置
                token_begin, token_end = self.get_entity_token_pos(entity_obj, content)
                # 实体所在位置超过序列最大长度则当前实体不打标
                if token_end >= self.model_config.max_seq_len - 2:
                    continue

                # 预测时专门对unknown标注mention进行类型预测
                if entity_obj["type"] == "unknown":
                    # 加 [CLS]
                    all_mention_loc_list.append((token_begin + 1, token_end + 1))
                    all_mention_content_list.append(content)
                    all_mention_form_list.append(entity_obj["form"])

        assert len(all_mention_loc_list) == len(all_mention_type_list)

        all_mention_result_list = []
        for mention_loc, mention_form, mention_type, mention_score, mention_content in \
                zip(all_mention_loc_list, all_mention_form_list, all_mention_type_list,
                    all_mention_score_list, all_mention_content_list):
            content_token_list = self.tokenizer.tokenize("[CLS]" + mention_content)
            token_mention_form = "".join([ele for ele in content_token_list[mention_loc[0]: mention_loc[1]+1]])
            all_mention_result_list.append((mention_form, mention_type, str(mention_score), token_mention_form))

        return all_mention_result_list



