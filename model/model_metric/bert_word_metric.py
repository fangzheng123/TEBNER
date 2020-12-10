# encoding: utf-8

from model.model_metric.base_metric import BaseMetric

class BERTWordMetric(BaseMetric):
    """
    BERT AutoNER 模型评价
    """

    def __init__(self):
        super().__init__()

    def get_pred_entity_pos(self, connect_index_list):
        """
        获取实体边界
        :param connect_index_list:
        :return:
        """
        pred_entity_list = []
        if len(connect_index_list) > 0:
            index = 1
            entity_begin = connect_index_list[0]
            while index < len(connect_index_list):
                if connect_index_list[index] == entity_begin + 1:
                    index += 1
                else:
                    entity_end = connect_index_list[index - 1] + 1
                    pred_entity_list.append((entity_begin, entity_end))
                    entity_begin = connect_index_list[index]
                    index += 1

            entity_end = connect_index_list[index - 1] + 1
            pred_entity_list.append((entity_begin, entity_end))

        return pred_entity_list

    def update_batch_result(self, token_pred_ids, type_pred_ids, entity_begins, entity_ends, entity_type_labels):
        """
        更新相关实体列表
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
            tie_token_offset_set = set([i for i, val in enumerate(pred_token_connect) if val == 1])
            pred_type = type_pred_ids[index]

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

    def get_acc_result(self):
        """
        计算预测正确率
        :return:
        """
        acc = len(self.pred_right_entity_list) / len(self.label_entity_list)
        return acc

    def get_metric_result(self, all_connect_pred_id_list, all_type_pred_id_list,
                          all_entity_label_list, all_sent_index_list):
        """
        计算测试准确率，召回率，F1
        :param all_connect_pred_id_list:
        :param all_type_pred_id_list:
        :param all_entity_label_list:
        :param all_sent_index_list:
        :return:
        """
        all_sent_connect_dict = {}
        all_sent_type_dict = {}
        all_sent_label_dict = {}

        for connect_pred_ids, type_pred_id, entity_label, sent_index in \
                zip(all_connect_pred_id_list, all_type_pred_id_list, all_entity_label_list, all_sent_index_list):
            all_sent_connect_dict[sent_index] = connect_pred_ids
            all_sent_type_dict.setdefault(sent_index, []).append(type_pred_id)
            all_sent_label_dict.setdefault(sent_index, []).append(entity_label)

        pred_num = 0
        pred_right_num = 0
        all_label_num = sum([len(_) for _ in all_sent_label_dict.values()])
        for sent_index, connect_pred_ids in all_sent_connect_dict.items():
            pred_type_list = all_sent_type_dict.get(sent_index, [])
            label_entity_list = all_sent_label_dict.get(sent_index, [])
            label_entity_begin_set = set([ele[0] for ele in label_entity_list])
            label_entity_end_set = set([ele[1] for ele in label_entity_list])

            connect_index_list = [i for i, val in enumerate(connect_pred_ids) if val == 1]
            pred_entity_list = self.get_pred_entity_pos(connect_index_list)
            pred_num += len(pred_entity_list)
            for entity_begin, entity_end in pred_entity_list:
                if entity_begin in label_entity_begin_set and entity_end in label_entity_end_set:
                    pred_right_num += 1

            for pred_type, label_entity in zip(pred_type_list, label_entity_list):
                if pred_type != label_entity[-1] and pred_num == len(label_entity_list):
                    pred_right_num -= 1

        precision, recall, f1 = self.compute_metric(all_label_num, pred_num, pred_right_num)
        metric_result_dict = {"precision": precision, "recall": recall, "f1": f1}
        return metric_result_dict












