# encoding: utf-8

import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from transformers import AdamW
from transformers.optimization import get_linear_schedule_with_warmup

from util.model_util import ModelUtil
from util.entity_util import EntityUtil
from util.log_util import LogUtil
from model.model_metric.bert_word_metric import BERTWordMetric

class BERTWordProcess(object):
    """
    训练、验证、测试BERT AutoNER(无分词)模型
    """
    def __init__(self, model_config):
        self.model_config = model_config
        self.args = self.model_config.args
        self.model_util = ModelUtil()
        self.entity_util = EntityUtil()
        self.model_metric = BERTWordMetric()

    def cal_connect_loss(self, word_connect_outputs, word_mask, word_connect_labels):
        """
        计算句子中词语连接关系损失
        :param word_connect_outputs: shape=(B,S-1,2)
        :param word_mask: shape=(B,S-1)
        :param word_connect_labels: shape=(B, S-1)
        :return:
        """
        # 计算有效loss，去除mask部分
        active_loss = word_mask.view(-1) == 1
        active_logits = word_connect_outputs.view(-1, self.model_config.connect_label_num)[active_loss]
        active_labels = word_connect_labels.view(-1)[active_loss]
        loss_func = CrossEntropyLoss()
        loss = loss_func(active_logits, active_labels)

        # loss_func = CrossEntropyLoss()
        # loss = loss_func(word_connect_outputs.view(-1, self.model_config.connect_label_num), word_connect_labels.view(-1))

        return loss

    def cal_type_loss(self, entity_type_outputs, entity_type_labels):
        """
        计算句子中实体类别损失
        :param entity_type_outputs:
        :param entity_type_labels:
        :return:
        """
        loss_func = CrossEntropyLoss()
        loss = loss_func(entity_type_outputs, entity_type_labels)
        return loss

    def train(self, model, train_loader, dev_loader, sent_entity_dict):
        """
        训练模型
        :param model:
        :param train_loader:
        :param dev_loader:
        :return:
        """
        model.train()
        # Prepare optimizer and schedule (linear warmup and decay)
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {"params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
             "weight_decay": self.args.weight_decay},
            {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
             "weight_decay": 0.0},
        ]
        t_total = len(train_loader) * self.model_config.num_epochs
        optimizer = AdamW(optimizer_grouped_parameters, lr=self.args.learning_rate, eps=self.args.adam_epsilon)
        scheduler = get_linear_schedule_with_warmup(optimizer,
                                                    num_warmup_steps=int(t_total * self.args.warmup_proportion),
                                                    num_training_steps=t_total)
        if torch.cuda.device_count() > 1:
            model = torch.nn.DataParallel(model)

        # 记录进行到多少batch
        total_batch = 0
        dev_best_result = 0
        # 记录上次验证集loss下降的batch数
        last_improve = 0
        # 记录是否很久没有效果提升
        no_improve_flag = False

        LogUtil.logger.info("Batch Num: {0}".format(len(train_loader)))
        for epoch in range(self.model_config.num_epochs):
            LogUtil.logger.info("Epoch [{}/{}]".format(epoch + 1, self.model_config.num_epochs))
            for i, batch_data in enumerate(train_loader):
                # 将数据加载到gpu
                batch_data = tuple(ele.to(self.model_config.device) for ele in batch_data)
                input_ids, input_mask, token_type_ids, token_connect_masks, token_connect_labels,\
                entity_begins, entity_ends, entity_type_labels, sent_indexs = batch_data

                sequence_output = model((input_ids, input_mask, token_type_ids))

                if torch.cuda.device_count() > 1:
                    token_connect_output = model.module.token_connecting(sequence_output)
                    entity_type_output = model.module.entity_typing(sequence_output, entity_begins, entity_ends)
                else:
                    token_connect_output = model.token_connecting(sequence_output)
                    entity_type_output = model.entity_typing(sequence_output, entity_begins, entity_ends)

                # 连接关系损失计算
                token_connect_loss = self.cal_connect_loss(token_connect_output, token_connect_masks, token_connect_labels)
                # 实体类别损失计算
                entity_type_loss = self.cal_type_loss(entity_type_output, entity_type_labels)
                loss = token_connect_loss + entity_type_loss

                model.zero_grad()
                loss.backward()
                # 对norm大于1的梯度进行修剪
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()

                # 每多少轮输出在训练集和验证集上的效果
                LogUtil.logger.info("total_batch: {0}".format(total_batch))
                if total_batch % self.model_config.per_eval_batch_step == 0:
                    # torch.max返回一个元组（最大值列表, 最大值对应的index列表）
                    token_pred_ids = torch.max(token_connect_output.data, axis=2)[1]
                    type_pred_ids = torch.max(entity_type_output.data, axis=1)[1]

                    # 计算当前train_batch acc
                    self.model_metric.reset()
                    metric_arg_list = [token_pred_ids, type_pred_ids, entity_begins, entity_ends, entity_type_labels]
                    metric_arg_list = [ele.cpu().numpy().tolist() for ele in metric_arg_list]
                    self.model_metric.update_batch_result(*metric_arg_list)
                    train_acc = self.model_metric.get_acc_result()
                    # 验证集上验证
                    # dev_loss, dev_acc = self.evaluate(model, dev_loader)
                    dev_metric_dict = self.evaluate_connect_type(model, dev_loader, sent_entity_dict)
                    if dev_metric_dict["f1"] > dev_best_result:
                        dev_best_result = dev_metric_dict["f1"]
                        torch.save(model.state_dict(), self.model_config.model_save_path)
                        improve = "*"
                        last_improve = total_batch
                    else:
                        improve = ""

                    msg = "Iter: {0:>6},  Train Loss: {1:>5.2},  Train Acc: {2:>6.2%}, Dev Precision: {3:>5.2%},  " \
                          "Dev Recall: {4:>6.2%}, Dev F1: {5:>6.2%}, {6}"
                    LogUtil.logger.info(msg.format(total_batch, loss.item(), train_acc,
                                                   dev_metric_dict["precision"], dev_metric_dict["recall"],
                                                   dev_metric_dict["f1"], improve))
                    model.train()
                total_batch += 1
                if total_batch - last_improve > self.model_config.require_improvement:
                    # 验证集loss超过require_improvement没下降，结束训练
                    LogUtil.logger.info("No optimization for a long time, auto-stopping...")
                    no_improve_flag = True
                    break
            if no_improve_flag:
                break

    def evaluate(self, model, data_loader):
        """
        验证模型
        :param model:
        :param data_loader:
        :return:
        """
        model.eval()
        loss_total = 0

        self.model_metric.reset()
        with torch.no_grad():
            for i, batch_data in enumerate(data_loader):
                # 将数据加载到gpu
                batch_data = tuple(ele.to(self.model_config.device) for ele in batch_data)
                input_ids, input_mask, token_type_ids, token_connect_masks, token_connect_labels, \
                entity_begins, entity_ends, entity_type_labels, sent_indexs = batch_data

                sequence_output = model((input_ids, input_mask, token_type_ids))

                if torch.cuda.device_count() > 1:
                    token_connect_output = model.module.token_connecting(sequence_output)
                    entity_type_output = model.module.entity_typing(sequence_output, entity_begins, entity_ends)
                else:
                    token_connect_output = model.token_connecting(sequence_output)
                    entity_type_output = model.entity_typing(sequence_output, entity_begins, entity_ends)

                # 连接关系损失计算
                token_connect_loss = self.cal_connect_loss(token_connect_output, token_connect_masks, token_connect_labels)
                # 实体类别损失计算
                entity_type_loss = self.cal_type_loss(entity_type_output, entity_type_labels)
                loss = token_connect_loss + entity_type_loss
                loss_total += loss

                # torch.max返回一个元组（最大值列表, 最大值对应的index列表）
                token_pred_ids = torch.max(token_connect_output.data, axis=2)[1]
                type_pred_ids = torch.max(entity_type_output.data, axis=1)[1]

                # 计算当前batch acc
                metric_arg_list = [token_pred_ids, type_pred_ids, entity_begins, entity_ends, entity_type_labels]
                metric_arg_list = [ele.cpu().numpy().tolist() for ele in metric_arg_list]
                self.model_metric.update_batch_result(*metric_arg_list)

        dev_loss = loss_total / len(data_loader)

        dev_acc = self.model_metric.get_acc_result()

        return dev_loss, dev_acc

    def evaluate_connect_type(self, model, data_loader, sent_entity_dict):
        """
        验证模型
        :param model:
        :param data_loader:
        :return:
        """
        model.eval()

        self.model_metric.reset()
        with torch.no_grad():
            for i, batch_data in enumerate(data_loader):
                # 将数据加载到gpu
                batch_data = tuple(ele.to(self.model_config.device) for ele in batch_data)
                input_ids, input_mask, token_type_ids, token_connect_masks, token_connect_labels, \
                _entity_begins, _entity_ends, _entity_type_labels, sent_indexs = batch_data
                sent_indexs = sent_indexs.cpu().numpy().tolist()

                sequence_output = model((input_ids, input_mask, token_type_ids))
                if torch.cuda.device_count() > 1:
                    # shape=(B,W-1,2)
                    token_connect_output = model.module.token_connecting(sequence_output)
                else:
                    token_connect_output = model.token_connecting(sequence_output)

                # torch.max返回一个元组（最大值列表, 最大值对应的index列表）
                connect_scores, connect_outputs = torch.max(token_connect_output.data, axis=2)

                # 根据连接结果获取实体边界
                entity_batch_index_list, entity_sent_index_dict, seq_connect_score_dict, \
                entity_begin_list, entity_end_list = self.infer_entity_boundary(connect_scores.cpu().numpy().tolist(),
                                               connect_outputs.cpu().numpy().tolist(), sent_indexs)

                # 构建实体分类数据
                entity_begins, entity_ends, all_outputs = self.build_entity_typing_data(entity_batch_index_list,
                                                                                        entity_begin_list,
                                                                                        entity_end_list,
                                                                                        sequence_output.cpu().numpy().tolist())
                if len(entity_begins) == 0:
                    continue

                if torch.cuda.device_count() > 1:
                    # shape=(B,L)
                    entity_type_output = model.module.entity_typing(all_outputs, entity_begins, entity_ends)
                else:
                    entity_type_output = model.entity_typing(all_outputs, entity_begins, entity_ends)

                entity_type_scores, entity_types = torch.max(entity_type_output.data, axis=1)
                self.model_metric.update_eval_result(entity_types.cpu().numpy().tolist(),
                                                     entity_type_scores.cpu().numpy().tolist(),
                                                     entity_begin_list, entity_end_list, entity_sent_index_dict)

        metric_result_dict = self.model_metric.get_metric_result(sent_entity_dict)
        return metric_result_dict

    def test(self, model, test_loader, sent_entity_dict):
        """
        测试模型
        :param model:
        :param test_loader:
        :param sent_entity_dict:
        :return:
        """
        # 加载模型
        self.model_util.load_model(model, self.model_config.model_save_path, self.model_config.device)
        model.eval()
        # 多块卡并行测试
        if torch.cuda.device_count() > 1:
            model = torch.nn.DataParallel(model)

        test_metric_dict = self.evaluate_connect_type(model, test_loader, sent_entity_dict)
        LogUtil.logger.info("Test Precision: {0}, Test Recall: {1}, Test F1: {2}".format(
            test_metric_dict["precision"], test_metric_dict["recall"], test_metric_dict["f1"]))

    def infer_entity_boundary(self, connect_scores, connect_outputs, sent_indexs):
        """
        根据连接关系推测实体边界
        :param connect_scores: 连接关系分数, shape=(B,S-1)
        :param connect_outputs: 连接关系, shape=(B,S-1)
        :param sent_indexs: 句子编号, shape=(B)
        :return:
        """
        # 实体在当前batch中句子序号
        entity_batch_index_list = []
        # 实体在所有数据中对应的序列号
        entity_sent_index_dict = {}
        # 实体对应的连接分数
        seq_entity_connect_score_dict = {}
        entity_begin_list = []
        entity_end_list = []
        for batch_sent_index in range(len(connect_outputs)):
            sent_index = sent_indexs[batch_sent_index]
            connect_list = connect_outputs[batch_sent_index]
            score_list = connect_scores[batch_sent_index]
            connect_index_list = [i for i, connect in enumerate(connect_list) if connect == 1]
            entity_boundary_list = EntityUtil.get_entity_boundary_no_seg(connect_index_list, self.model_config.max_seq_len)
            if len(entity_boundary_list) > 0:
                for entity_begin, entity_end in entity_boundary_list:
                    entity_begin_list.append(entity_begin)
                    entity_end_list.append(entity_end)
                    entity_batch_index_list.append(batch_sent_index)
                    entity_sent_index_dict[len(entity_begin_list)-1] = sent_index
                    seq_entity_connect_score_dict[len(entity_begin_list)-1] = sum(score_list[entity_begin:entity_end+1]) / \
                                                                      max(1, entity_end+1-entity_begin)

        return entity_batch_index_list, entity_sent_index_dict, seq_entity_connect_score_dict, entity_begin_list, entity_end_list

    def build_entity_typing_data(self, *args):
        """
        构建实体分类数据
        :param args:
        :return:
        """
        entity_batch_index_list, entity_begin_list, entity_end_list, outputs = args
        all_outputs = [outputs[index] for index in entity_batch_index_list]

        entity_begins = torch.LongTensor(entity_begin_list).to(self.model_config.device)
        entity_ends = torch.LongTensor(entity_end_list).to(self.model_config.device)
        all_outputs = torch.FloatTensor(all_outputs).to(self.model_config.device)

        return entity_begins, entity_ends, all_outputs

    def get_entity_type_scoce(self, *args):
        """
        获取序列实体及对应分数
        :param args:
        :return:
        """
        entity_type, entity_type_scores, entity_begins, entity_ends, \
        seq_entity_index_dict, seq_connect_score_dict = args

        assert len(entity_type) == len(seq_entity_index_dict)

        batch_seq_entity_dict = {}
        for entity_index, entity_type_id in enumerate(entity_type):
            entity_dict = {
                "type": self.model_config.type_id_label_dict[entity_type_id],
                "token_pos": (entity_begins[entity_index], entity_ends[entity_index]),
                "connect_score": seq_connect_score_dict[entity_index],
                "type_score": entity_type_scores[entity_index]
            }
            if seq_entity_index_dict[entity_index] not in batch_seq_entity_dict:
                batch_seq_entity_dict[seq_entity_index_dict[entity_index]] = [entity_dict]
            else:
                batch_seq_entity_dict[seq_entity_index_dict[entity_index]].append(entity_dict)

        return batch_seq_entity_dict

    def predict(self, model, data_loader):
        """
        预测句子中的实体
        :param model:
        :param data_loader:
        :return:
        """
        # 加载模型
        self.model_util.load_model(model, self.model_config.model_save_path, self.model_config.device)
        model.eval()
        # 多块卡并行测试
        if torch.cuda.device_count() > 1:
            model = torch.nn.DataParallel(model)

        batch_num = 0
        all_seq_entity_dict = {}
        LogUtil.logger.info("Batch Num: {0}".format(len(data_loader)))
        with torch.no_grad():
            for i, batch_data in enumerate(data_loader):
                # 将数据加载到gpu
                batch_data = tuple(ele.to(self.model_config.device) for ele in batch_data)
                input_ids, input_mask, token_type_ids, token_connect_masks, token_connect_labels, \
                entity_begins, entity_ends, entity_type_labels = batch_data

                sequence_output = model((input_ids, input_mask, token_type_ids))
                if torch.cuda.device_count() > 1:
                    # shape=(B,W-1,2)
                    token_connect_output = model.module.token_connecting(sequence_output)
                else:
                    token_connect_output = model.token_connecting(sequence_output)

                # torch.max返回一个元组（最大值列表, 最大值对应的index列表）
                connect_scores, connect_outputs = torch.max(token_connect_output.data, axis=2)

                # 根据连接结果获取实体边界
                seq_index_list, seq_entity_index_dict, seq_connect_score_dict, \
                entity_begin_list, entity_end_list = self.infer_entity_boundary(connect_scores.cpu().numpy().tolist(),
                                                                                connect_outputs.cpu().numpy().tolist())

                # 构建实体分类数据
                entity_begins, entity_ends, all_outputs = self.build_entity_typing_data(seq_index_list,
                                                                                        entity_begin_list,
                                                                                        entity_end_list,
                                                                                        sequence_output.cpu().numpy().tolist())
                if torch.cuda.device_count() > 1:
                    # shape=(B,L)
                    entity_type_output = model.module.entity_typing(all_outputs, entity_begins, entity_ends)
                else:
                    entity_type_output = model.entity_typing(all_outputs, entity_begins, entity_ends)
                
                entity_type_scores, entity_type = torch.max(entity_type_output.data, axis=1)

                # 获取每个序列中包含的实体及对应的分数
                batch_seq_entity_dict = self.get_entity_type_scoce(entity_type.cpu().numpy().tolist(),
                                                                   entity_type_scores.cpu().numpy().tolist(),
                                                                   entity_begins.cpu().numpy().tolist(),
                                                                   entity_ends.cpu().numpy().tolist(),
                                                                   seq_entity_index_dict, seq_connect_score_dict)

                for seq_index, entity_list in batch_seq_entity_dict.items():
                    all_seq_entity_dict[seq_index + batch_num] = entity_list

                batch_num += len(input_ids)

                LogUtil.logger.info("Total batch: {0}".format(i))

        return all_seq_entity_dict