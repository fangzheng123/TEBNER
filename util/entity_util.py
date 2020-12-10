# encoding:utf-8

class EntityUtil(object):
    """
    实体工具类
    """
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

    @classmethod
    def get_entity_boundary(cls, connect_index_list, token_offset_list, max_len) -> list:
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

    @classmethod
    def get_entity_boundary_no_seg(cls, connect_index_list, max_len) -> list:
        """
        根据连接关系确定实体边界
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
