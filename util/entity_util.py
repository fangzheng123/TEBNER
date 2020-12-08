# encoding:utf-8

class EntityUtil(object):
    """
    实体工具类
    """
    @classmethod
    def split_text_obj(cls, text_obj) -> list:
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

    @classmethod
    def get_entity_token_pos(cls, entity_obj, content, tokenizer):
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
        mask_token_list = tokenizer.tokenize(mask_content)

        mask_pos_list = [i for i, token in enumerate(mask_token_list) if token == "[MASK]"]

        token_begin = mask_pos_list[0]
        token_end = mask_pos_list[1] - 2

        return token_begin, token_end

    @classmethod
    def get_entity_word_pos(cls, text_obj):
        """
        获取实体在text中的位置
        :param text_obj:
        :return:
        """
        content = text_obj["text"]
        for entity_obj in text_obj["entity_list"]:
            entity_obj["word_offset"] = len(content[:entity_obj["offset"]].split())
        for entity_obj in text_obj["distance_entity_list"]:
            entity_obj["word_offset"] = len(content[:entity_obj["offset"]].split())

        return text_obj

    def get_entity_boundary(self, connect_index_list, token_offset_list, max_len) -> list:
        """
        根据连接关系确定实体边界
        :param connect_index_list:
        :param token_offset_list:
        :return:
        """
        entity_boundary_list = []
        if len(connect_index_list) > 0:
            index = 1
            pre_word_index = connect_index_list[0]
            entity_begin = token_offset_list[pre_word_index]
            while index < len(connect_index_list):
                current_word_index = connect_index_list[index]
                if current_word_index != pre_word_index + 1:
                    if pre_word_index + 2 < len(token_offset_list):
                        entity_end = token_offset_list[pre_word_index + 2] - 1
                    else:
                        entity_end = token_offset_list[-1] - 1

                    if entity_begin < entity_end < max_len - 1:
                        entity_boundary_list.append((entity_begin, entity_end))

                    entity_begin = token_offset_list[current_word_index]

                pre_word_index = current_word_index
                index += 1

            if pre_word_index + 2 < len(token_offset_list):
                entity_end = token_offset_list[pre_word_index + 2] - 1
            else:
                entity_end = token_offset_list[-1] - 1

            if entity_begin < entity_end < max_len - 1:
                entity_boundary_list.append((entity_begin, entity_end))

        return entity_boundary_list
    
    def get_entity_boundary_no_seg(self, connect_index_list, max_len) -> list:
        """
        根据连接关系确定实体边界(无分词)
        :param connect_index_list:
        :return:
        """
        entity_boundary_list = []
        if len(connect_index_list) > 0:
            index = 1
            pre_token_index = connect_index_list[0]
            entity_begin = pre_token_index
            while index < len(connect_index_list):
                current_token_index = connect_index_list[index]
                if current_token_index != pre_token_index + 1:
                    entity_end = pre_token_index + 1
                    if entity_begin < entity_end < max_len - 1:
                        entity_boundary_list.append((entity_begin, entity_end))

                    entity_begin = current_token_index

                pre_token_index = current_token_index
                index += 1

            if pre_token_index + 2 < max_len:
                entity_end = pre_token_index + 1
            else:
                entity_end = max_len - 2

            if entity_begin < entity_end < max_len - 1:
                entity_boundary_list.append((entity_begin, entity_end))

        return entity_boundary_list

    @classmethod
    def get_seq_entity(cls, token_labels):
        """
        获取序列中的实体
        :param token_labels: 序列标注 BIOS
        :return: ['B-PER', 'I-PER', 'O', 'S-LOC'] -> [['PER', 0, 1], ['LOC', 3, 3]]
        """
        entity_list = []
        entity = [-1, -1, -1]
        for index, tag in enumerate(token_labels):
            if tag.startswith("S-"):
                # 先前识别的实体
                if entity[2] != -1:
                    entity_list.append(entity)
                entity = [-1, -1, -1]
                entity[0] = tag.split("-")[1]
                entity[1] = index
                entity[2] = index
                entity_list.append(entity)
                entity = (-1, -1, -1)
            elif tag.startswith("B-"):
                # 先前识别的实体
                if entity[2] != -1:
                    entity_list.append(entity)
                entity = [-1, -1, -1]
                entity[0] = tag.split("-")[1]
                entity[1] = index
            elif tag.startswith('I-') and entity[1] != -1:
                _type = tag.split('-')[1]
                if _type == entity[0]:
                    entity[2] = index
                if index == len(token_labels) - 1:
                    entity_list.append(entity)
            else:
                # 先前识别的实体
                if entity[2] != -1:
                    entity_list.append(entity)
                entity = [-1, -1, -1]

        return entity_list
