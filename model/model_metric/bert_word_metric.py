# encoding: utf-8

from util.entity_util import EntityUtil
from model.model_metric.base_metric import BaseMetric

class BERTWordMetric(BaseMetric):
    """
    BERT AutoNER 模型评价
    """

    def __init__(self, model_config):
        super().__init__()
        self.model_config = model_config

    def reset(self):
        super().reset()
        self.pred_sent_entity_dict = {}

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

    def update_boundary_batch_result(self, pred_seq_connect_ids, label_seq_connect_ids):
        """
        更新实体列表
        :param pred_seq_connect_ids:
        :param label_seq_connect_ids:
        :return:
        """
        for pre_seq, label_seq in zip(pred_seq_connect_ids, label_seq_connect_ids):
            pred_connect_index_list = [i for i, connect in enumerate(pre_seq) if connect == 1]
            pred_boundary_list = EntityUtil.get_entity_boundary_no_seg(pred_connect_index_list, self.model_config.max_seq_len)
            pred_entity_list = [[entity_begin, entity_end] for entity_begin, entity_end in pred_boundary_list]

            label_connect_index_list = [i for i, connect in enumerate(label_seq) if connect == 1]
            label_boundary_list = EntityUtil.get_entity_boundary_no_seg(label_connect_index_list, self.model_config.max_seq_len)
            label_entity_list = [[entity_begin, entity_end] for entity_begin, entity_end in label_boundary_list]

            self.label_entity_list.extend(label_entity_list)
            self.pred_entity_list.extend(pred_entity_list)
            self.pred_right_entity_list.extend([pre_entity for pre_entity in pred_entity_list if pre_entity in label_entity_list])

    def update_joint_batch_result(self, token_pred_ids, type_pred_ids, entity_begins, entity_ends, entity_type_labels):
        """
        更新实体列表
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

    def update_eval_result(self, entity_types, entity_type_scores, entity_begins, entity_ends, entity_sent_index_dict):
        """
        更新预测结果
        :param entity_types:
        :param entity_type_scores:
        :param entity_begins:
        :param entity_ends:
        :param entity_sent_index_dict:
        :return:
        """
        for index in range(len(entity_begins)):
            sent_index = entity_sent_index_dict[index]
            entity_begin = entity_begins[index]
            entity_end = entity_ends[index]
            entity_type = entity_types[index]
            entity_type_score = entity_type_scores[index]
            self.pred_sent_entity_dict.setdefault(sent_index, []).append((entity_begin, entity_end, entity_type, entity_type_score))

    def get_acc_result(self):
        """
        计算预测正确率
        :return:
        """
        acc = len(self.pred_right_entity_list) / len(self.label_entity_list)
        return acc

    def get_boundary_metric_result(self):
        """
        获取边界模型评测结果
        :return:
        """
        precision, recall, f1 = self.compute_metric(len(self.label_entity_list), len(self.pred_entity_list),
                                                    len(self.pred_right_entity_list))
        metric_result_dict = {"precision": precision, "recall": recall, "f1": f1}
        return metric_result_dict

    def get_joint_metric_result(self, label_sent_entity_dict):
        """
        计算测试准确率，召回率，F1
        :param label_sent_entity_dict:
        :return:
        """
        pred_num = 0
        pred_right_num = 0
        label_entity_num = 0

        for sent_index, pred_entity_list in self.pred_sent_entity_dict.items():
            label_entity_list = label_sent_entity_dict.get(sent_index, [])

            pred_num += len(pred_entity_list)
            label_entity_num += len(label_entity_list)
            pred_right_num += len([pred_entity for pred_entity in pred_entity_list if
                                   (pred_entity[0], pred_entity[1], pred_entity[2]) in label_entity_list])

        precision, recall, f1 = self.compute_metric(label_entity_num, pred_num, pred_right_num)
        metric_result_dict = {"precision": precision, "recall": recall, "f1": f1}
        return metric_result_dict












