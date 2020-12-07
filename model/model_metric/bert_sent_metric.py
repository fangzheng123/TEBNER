# encoding: utf-8

from collections import Counter
from seqeval.metrics import f1_score, precision_score, recall_score

from util.entity_util import EntityUtil

class SeqEntityMetric(object):
    """
    序列NER模型评价
    """
    def __init__(self):
        self.entity_util = EntityUtil()
        self.reset()

    def reset(self):
        """
        重置列表
        :return:
        """
        # 标注实体列表
        self.label_entity_list = []
        # 预测实体列表
        self.pred_entity_list = []
        # 预测正确实体列表
        self.pred_right_entity_list = []

    def update(self, label_seqs, pred_seqs, id2label):
        """
        更新相关实体列表
        Example:
        label_seq = [['O', 'O', 'O', 'B-MISC', 'I-MISC', 'I-MISC', 'O'], ['B-PER', 'I-PER', 'O']]
        pred_seq = [['O', 'O', 'B-MISC', 'I-MISC', 'I-MISC', 'I-MISC', 'O'], ['B-PER', 'I-PER', 'O']]
        :param label_seqs: [[],[],[],....]
        :param pred_seqs: [[],[],[],.....]
        :id2label
        :return:
        """
        for label_seq, pre_seq in zip(label_seqs, pred_seqs):
            label_entities = self.entity_util.get_seq_entity([id2label[i] for i in label_seq])
            pre_entities = self.entity_util.get_seq_entity([id2label[i] for i in pre_seq])
            self.label_entity_list.extend(label_entities)
            self.pred_entity_list.extend(pre_entities)
            self.pred_right_entity_list.extend([pre_entity for pre_entity in pre_entities if pre_entity in label_entities])

    def compute_metric(self, label_num, pred_num, pred_right_num):
        """
        计算P, R, F1
        :param origin:
        :param found:
        :param right:
        :return:
        """
        precision = 0 if pred_num == 0 else (pred_right_num / pred_num)
        recall = 0 if label_num == 0 else (pred_right_num / label_num)
        f1 = 0 if recall + precision == 0 else (2 * precision * recall) / (precision + recall)
        return round(precision, 4), round(recall, 4), round(f1, 4)

    def metric_result(self):
        """
        计算各实体类别相关评测结果
        :return:
        """
        metric_result_dict = {}
        label_counter = Counter([x[0] for x in self.label_entity_list])
        pred_counter = Counter([x[0] for x in self.pred_entity_list])
        pred_right_counter = Counter([x[0] for x in self.pred_right_entity_list])
        for type_, label_num in label_counter.items():
            pred_num = pred_counter.get(type_, 0)
            pred_right_num = pred_right_counter.get(type_, 0)
            precision, recall, f1 = self.compute_metric(label_num, pred_num, pred_right_num)
            metric_result_dict[type_] = {"precision": precision, "recall": recall, "f1": f1}

        precision, recall, f1 = self.compute_metric(len(self.label_entity_list), len(self.pred_entity_list), len(self.pred_right_entity_list))
        metric_result_dict["all_type"] = {"precision": precision, "recall": recall, "f1": f1}

        return metric_result_dict

    def get_metric_by_seqeval(self, pred_seqs, label_seqs, id2label):
        """get_metric_by_seqeval
        通过seqeval来评估
        :param pred_seqs:
        :param label_seqs:
        :param id2label:
        :return:
        """
        all_label_seq_list = []
        all_pred_seq_list = []
        for label_seq, pre_seq in zip(label_seqs, pred_seqs):
            all_label_seq_list.append([id2label[i] for i in label_seq])
            all_pred_seq_list.append([id2label[i] for i in pre_seq])

        result_dict = {
            "precision": precision_score(all_label_seq_list, all_pred_seq_list),
            "recall": recall_score(all_label_seq_list, all_pred_seq_list),
            "f1": f1_score(all_label_seq_list, all_pred_seq_list),
        }

        return result_dict

