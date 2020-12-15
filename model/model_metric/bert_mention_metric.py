# encoding: utf-8

from model.model_metric.base_metric import BaseMetric

class BERTMentionMetric(BaseMetric):
    """
    BERT Mention 模型评价
    """

    def __init__(self):
        super().__init__()

    def reset(self):
        super().reset()
        self.pred_sent_entity_dict = {}

    def update_eval_result(self, entity_types, entity_type_scores, entity_begins, entity_ends,
                           sent_indexs, all_pred_sent_entity_dict, model_config):
        """
        更新预测结果
        :param entity_types:
        :param entity_type_scores:
        :param entity_begins:
        :param entity_ends:
        :param sent_indexs:
        :return:
        """
        for index in range(len(entity_begins)):
            sent_index = sent_indexs[index]
            entity_begin = entity_begins[index]
            entity_end = entity_ends[index]
            entity_type = entity_types[index]
            entity_type_score = entity_type_scores[index]

            # 序列标注模型预测结果
            # sent_model_pred_entity_list = all_pred_sent_entity_dict.get(sent_index, [])
            # sent_model_pred_entity_dict = {str(ele[1])+"_"+str(ele[2]): ele[0] for ele in sent_model_pred_entity_list}
            # if str(entity_begin)+"_"+str(entity_end) in sent_model_pred_entity_dict:
            #     entity_type = model_config.label_id_dict[sent_model_pred_entity_dict[str(entity_begin)+"_"+str(entity_end)]]

            self.pred_sent_entity_dict.setdefault(sent_index, []).append((entity_begin, entity_end, entity_type, entity_type_score))

    def get_metric_result(self, label_sent_entity_dict):
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












