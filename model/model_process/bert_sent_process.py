# encoding: utf-8

import numpy as np
import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from transformers import AdamW
from transformers.optimization import get_linear_schedule_with_warmup

from model.model_metric.bert_sent_metric import SeqEntityMetric
from util.model_util import ModelUtil
from util.log_util import LogUtil

class BERTSentProcess(object):
    """
    训练、验证、测试BERT序列标注模型
    """

    def __init__(self, model_config):
        self.model_config = model_config
        self.args = self.model_config.args
        self.model_util = ModelUtil()
        self.seq_model_metric = SeqEntityMetric()

    def cal_loss(self, output, batch_data):
        """
        计算损失
        :param output:
        :param batch_data:
        :return:
        """
        # 获取loss函数
        loss_func = CrossEntropyLoss()

        # 计算有效loss，去除mask部分
        # attention_mask = batch_data[1]
        # label = batch_data[3]
        # active_loss = attention_mask.view(-1) == 1
        # active_logits = output.view(-1, self.model_config.label_num)[active_loss]
        # active_labels = label.view(-1)[active_loss]
        # loss = loss_func(active_logits, active_labels)

        loss = loss_func(output.view(-1, self.model_config.label_num), batch_data[3].view(-1))

        return loss

    def train(self, model, train_loader, dev_loader):
        """
        训练模型
        :param model: 模型
        :param train_loader: 训练数据
        :param dev_loader: 验证数据
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
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(t_total * self.args.warmup_proportion),
                                                    num_training_steps=t_total)

        # 多GPU训练
        if torch.cuda.device_count() > 1:
            model = torch.nn.DataParallel(model)

        # 记录进行到多少batch
        total_batch = 0
        dev_best_f1 = 0
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
                input_ids, input_mask, type_ids, label_ids = batch_data
                outputs = model((input_ids, input_mask, type_ids))
                model.zero_grad()
                loss = self.cal_loss(outputs, batch_data)
                loss.backward()
                # 对norm大于1的梯度进行修剪
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()

                # 每多少轮输出在训练集和验证集上的效果
                if total_batch % self.model_config.per_eval_batch_step == 0:
                    # torch.max返回一个元组（最大值列表, 最大值对应的index列表）
                    pred_ids = torch.max(outputs.data, axis=2)[1]
                    train_result_dict = self.seq_model_metric.get_metric_by_seqeval(
                        pred_ids.cpu().numpy().tolist(), label_ids.cpu().numpy().tolist(), self.model_config.id_label_dict)
                    dev_loss, dev_result_dict = self.evaluate(model, dev_loader)
                    if dev_result_dict["f1"] > dev_best_f1:
                        dev_best_f1 = dev_result_dict["f1"]
                        torch.save(model.state_dict(), self.model_config.model_save_path)
                        improve = "*"
                        last_improve = total_batch
                    else:
                        improve = ""
                    msg = "Iter: {0:>6},  Train Loss: {1:>5.2},  Train Prec: {2:>6.2%},  Train Recall: {3:>6.2%},  " \
                          "Train F1: {4:>6.2%},  Dev Loss: {5:>5.2},  Dev Prec: {6:>6.2%},  Dev Recall: {7:>6.2%},  " \
                          "Dev F1: {8:>6.2%} {9}"
                    LogUtil.logger.info(msg.format(total_batch, loss.item(), train_result_dict["precision"],
                                     train_result_dict["recall"], train_result_dict["f1"],
                                     dev_loss, dev_result_dict["precision"], dev_result_dict["recall"],
                                                   dev_result_dict["f1"], improve))
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
        :param is_test: 是否为测试集
        :return:
        """
        model.eval()
        loss_total = 0

        predict_all_list = []
        labels_all_list = []
        with torch.no_grad():
            for i, batch_data in enumerate(data_loader):
                # 将数据加载到gpu
                batch_data = tuple(ele.to(self.model_config.device) for ele in batch_data)
                input_ids, input_mask, type_ids, label_ids = batch_data
                outputs = model((input_ids, input_mask, type_ids))
                loss = self.cal_loss(outputs, batch_data)
                loss_total += loss
                pred_ids = torch.max(outputs.data, axis=2)[1]
                predict_all_list.extend(pred_ids.cpu().numpy().tolist())
                labels_all_list.extend(label_ids.cpu().numpy().tolist())

        dev_loss = loss_total / len(data_loader)
        dev_result_dict = self.seq_model_metric.get_metric_by_seqeval(
            predict_all_list, labels_all_list, self.model_config.id_label_dict)

        return dev_loss, dev_result_dict

    def test(self, model, test_loader):
        """
        测试模型
        :param model:
        :param test_loader:
        :return:
        """
        # 加载模型
        self.model_util.load_model(model, self.model_config.model_save_path, self.model_config.device)

        model.eval()
        if torch.cuda.device_count() > 1:
            model = torch.nn.DataParallel(model)

        test_loss, test_metric_result = self.evaluate(model, test_loader)
        LogUtil.logger.info("Test Loss: {0}".format(test_loss))
        LogUtil.logger.info(test_metric_result)

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
        if torch.cuda.device_count() > 1:
            model = torch.nn.DataParallel(model)

        all_seq_score_list = []
        all_seq_tag_list = []
        with torch.no_grad():
            LogUtil.logger.info("Batch Num: {0}".format(len(data_loader)))
            for i, batch_data in enumerate(data_loader):
                # 将数据加载到gpu
                batch_data = tuple(ele.to(self.model_config.device) for ele in batch_data)
                input_ids, input_mask, type_ids, label_ids = batch_data
                outputs = model((input_ids, input_mask, type_ids))
                # torch.max返回一个元组（最大值列表, 最大值对应的index列表）
                scores, preds = torch.max(outputs.data, axis=2)
                scores = scores.cpu().numpy().tolist()
                preds = preds.cpu().numpy().tolist()
                all_seq_score_list.extend(scores)
                for seq in preds:
                    tag_list = [self.model_config.id_label_dict[ele] for ele in seq]
                    all_seq_tag_list.append(tag_list)
                
                LogUtil.logger.info("Batch Num: {0}".format(i))

        LogUtil.logger.info("Finished Predicting!!!")

        # shape=(B,S)
        return all_seq_score_list, all_seq_tag_list