# encoding: utf-8


class BERTAutoNERMetric(object):
    """
    BERT AutoNER 模型评价
    """

    def __init__(self):

        self.reset()

    def reset(self):
        """
        重置列表
        :return:
        """
        # 标注实体列表
        self.label_entity_list = []
        # 预测正确实体列表
        self.pred_right_entity_list = []

    def update(self, word_pred_ids, type_pred_ids, word_offsets, entity_begins, entity_ends, entity_type_labels):
        """
        更新相关实体列表
        :param word_pred_ids: shape=(B,W-1)
        :param type_pred_ids: shape=(B)
        :param word_offsets: shape=(B, W)
        :param entity_begins: shape=(B)
        :param entity_ends: shape=(B)
        :param entity_type_labels: shape=(B)
        :return:
        """
        for index in range(len(word_pred_ids)):
            # 标注结果
            entity_begin = entity_begins[index]
            entity_end = entity_ends[index]
            label_type = entity_type_labels[index]
            self.label_entity_list.append((label_type, entity_begin, entity_end))

            # 预测结果
            pred_right_flag = True
            pred_word_connect = word_pred_ids[index]
            word_offset = word_offsets[index]
            pred_type = type_pred_ids[index]
            tie_word_offset_set = set([i for i, val in enumerate(pred_word_connect) if val == 1])
            tie_offset_begin = word_offset.index(entity_begin)
            if entity_end+1 in word_offset:
                tie_offset_end = word_offset.index(entity_end + 1) - 1
            else:
                tie_offset_end = len(word_offset) - 2

            # 判断边界预测是否正确
            for i in range(tie_offset_begin, tie_offset_end):
                if i not in tie_word_offset_set:
                    pred_right_flag = False
                    break
            # 判断类别预测是否正确
            if pred_type != label_type:
                pred_right_flag = False

            # 预测正确
            if pred_right_flag:
                self.pred_right_entity_list.append((pred_type, entity_begin, entity_end))
    
    def update_no_seg(self, token_pred_ids, type_pred_ids, entity_begins, entity_ends, entity_type_labels):
        """
        更新相关实体列表(无分词)
        :param token_pred_ids: shape=(B,S-1)
        :param type_pred_ids: shape=(B)
        :param entity_begins: shape=(B)
        :param entity_ends: shape=(B)
        :param entity_type_labels: shape=(B)
        :return:
        """
        for index in range(len(token_pred_ids)):
            # 标注结果
            entity_begin = entity_begins[index]
            entity_end = entity_ends[index]
            label_type = entity_type_labels[index]
            self.label_entity_list.append((label_type, entity_begin, entity_end))

            # 预测结果
            pred_right_flag = True
            pred_token_connect = token_pred_ids[index]
            pred_type = type_pred_ids[index]
            tie_token_offset_set = set([i for i, val in enumerate(pred_token_connect) if val == 1])

            # 判断边界预测是否正确
            for i in range(entity_begin, entity_end):
                if i not in tie_token_offset_set:
                    pred_right_flag = False
                    break
            # 判断类别预测是否正确
            if pred_type != label_type:
                pred_right_flag = False

            # 预测正确
            if pred_right_flag:
                self.pred_right_entity_list.append((pred_type, entity_begin, entity_end)) 

    def get_metric_result(self):
        """
        计算各实体类别相关评测结果
        :return:
        """
        acc = len(self.pred_right_entity_list) / len(self.label_entity_list)

        return acc