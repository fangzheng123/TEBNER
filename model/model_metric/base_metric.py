# encoding: utf-8

class BaseMetric(object):
    """
    评测基类
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
        # 预测实体列表
        self.pred_entity_list = []
        # 预测正确实体列表
        self.pred_right_entity_list = []

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