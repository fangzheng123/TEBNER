# encoding: utf-8

import sys
sys.path.append("../../BERTAutoNER")

import random
import numpy as np
import xgboost as xgb
from util.log_util import LogUtil

class XgbForest(object):
    """
    以Xgboost作为基分类器的随机森林模型
    """

    def __init__(self, phrase_config):
        self.phrase_config = phrase_config
        self.model_path = self.phrase_config.model_path
        self.param = {"max_depth": 5, "eta": 0.01, "objective": "binary:logistic"}
        self.num_round = 1000

    def read_feature(self, data_dict, label):
        """
        读取特征数据
        :param data_dict:
        :param label:
        :return:
        """
        label_data_list = []
        filter_fea_set = set()
        # filter_fea_set = {"phrase_freq", "phrase_df", "phrase_idf"}
        for phrase, fea_dict in data_dict.items():
            fea_list = []
            for fea_name, fea_val in fea_dict.items():
                if type(fea_val) == list:
                    fea_list.extend(fea_val)
                else:
                    if fea_name not in filter_fea_set:
                        fea_list.append(fea_val)
            label_data_list.append((phrase, fea_list, label))

        return label_data_list

    def sample_data(self, positive_data_list, negative_data_list):
        """
        采样训练数据
        :param positive_data_list: 正例数据
        :param negative_data_list: 负例数据
        :return:
        """
        all_data_list = []
        all_data_list.extend(positive_data_list)

        # 负例取N倍正例数
        negative_num = len(positive_data_list) * self.phrase_config.NEG_POS_TIMES

        # 随机筛选负例
        select_negative_list = random.sample(negative_data_list, negative_num)
        all_data_list.extend(select_negative_list)

        random.shuffle(all_data_list)

        x_data = np.array([_[1] for _ in all_data_list])
        y_data = np.array([_[2] for _ in all_data_list])

        return x_data, y_data

    def train_model(self, positive_phrase_dict, negative_phrase_dict):
        """
        训练模型
        :param positive_phrase_dict: 正例数据
        :param negative_phrase_dict: 负例数据
        :return:
        """
        LogUtil.logger.info("开始训练XGB模型")

        positive_data_list = self.read_feature(positive_phrase_dict, 1)
        negative_data_list = self.read_feature(negative_phrase_dict, 0)

        LogUtil.logger.debug("pos num:{0}, neg num:{1}".format(len(positive_data_list), len(negative_data_list)))

        xgb_model_list = []
        for model_index in range(self.phrase_config.BASE_MODEL_NUM):
            LogUtil.logger.info("训练第{0}个XGB模型".format(model_index+1))
            # 随机采样负例
            x_data, y_data = self.sample_data(positive_data_list, negative_data_list)

            dtrain = xgb.DMatrix(x_data, label=y_data)
            xgb_model = xgb.train(self.param, dtrain, self.num_round)

            base_model_path = self.model_path + "_" + str(model_index)
            xgb_model.save_model(base_model_path)
            xgb_model_list.append(xgb_model)

        LogUtil.logger.info("训练XGB模型结束")

        return xgb_model_list

    def pred_model(self, xgb_model_list, phrase_fea_dict) -> list:
        """
        预测模型
        :param xgb_model_list: 多个基分类器
        :param phrase_fea_dict: 短语特征字典
        :return:
        """
        LogUtil.logger.info("预测数据")

        fea_data_list = self.read_feature(phrase_fea_dict, 1)
        x_data = np.array([_[1] for _ in fea_data_list])

        all_pred_list = []
        for xgb_model in xgb_model_list:
            dtest = xgb.DMatrix(x_data)
            xgb_preds = xgb_model.predict(dtest)
            all_pred_list.append(xgb_preds)

        # 取所有基分类器平均值
        pred_prob_list = np.mean(np.array(all_pred_list), axis=0).tolist()

        # 短语预测结果
        phrase_pred_list = [(data[0], prob) for data, prob in zip(fea_data_list, pred_prob_list)]

        return phrase_pred_list

    def load_models(self):
        """
        加载训练好的模型
        :return:
        """
        model_list = []
        for model_index in range(self.phrase_config.BASE_MODEL_NUM):
            model_path = self.phrase_config.model_path + "_" + str(model_index)
            xgb_model = xgb.Booster()
            xgb_model.load_model(model_path)
            model_list.append(xgb_model)

        return model_list

    def get_importance(self, xgb_model_list, negative_phrase_dict):
        """
        获取各维度重要性
        :return:
        """
        fea_name_list = []
        for phrase, fea_dict in negative_phrase_dict.items():
            for fea_name, val in fea_dict.items():
                if type(val) == list:
                    fea_name_list.extend([fea_name for _ in range(200)])
                else:
                    fea_name_list.append(fea_name)
            break

        for xgb_model in xgb_model_list:
            importance_dict = xgb_model.get_fscore()

            all_fea_score_list = []
            for fea, val in importance_dict.items():
                index = int(fea.replace("f", ""))
                all_fea_score_list.append((fea_name_list[index], val))

            fea_score_list = [(fea_name, fea_score) for fea_name, fea_score in all_fea_score_list if fea_name != "mean_phrase_vec"]
            fea_score_list.append(("mean_phrase_vec", sum([val for fea_name, val in all_fea_score_list if fea_name == "mean_phrase_vec"])))
            fea_score_list = sorted(fea_score_list, key=lambda x:x[1], reverse=True)
            print(fea_score_list)
            print("\n")



if __name__ == "__main__":

    pass