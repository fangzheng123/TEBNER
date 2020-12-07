# encoding: utf-8

from phrase_mining.phrase_processor import PhraseProcessor
from phrase_mining.phrase_feature import PhraseFeature
from phrase_mining.xgb_forest import XgbForest
from phrase_mining.entity_processor import EntityProcessor

class PhraseController(object):
    """
    短语相关操作控制类
    """

    def __init__(self, phrase_config):
        self.phrase_config = phrase_config

        self.phrase_processor = PhraseProcessor(phrase_config)
        self.phrase_feature = PhraseFeature(phrase_config)
        self.xgb_forest = XgbForest(phrase_config)

        self.entity_processor = EntityProcessor(phrase_config)

    def extract_phrase_fea(self, cut_pos_text_list):
        """
        挖掘短语及其特征
        :param cut_pos_text_list:
        :return:
        """
        # 生成候选短语
        all_phrase_dict = self.phrase_processor.generate_candidate_phrase(cut_pos_text_list)

        # 从全量词向量中抽取候选短语所含词语及对应向量
        filed_vec_dict = self.phrase_processor.get_phrase_word_vec(all_phrase_dict, self.phrase_config.word_vec_path)

        # 计算短语相关特征
        phrase_fea_dict = self.phrase_feature.extract_feature(all_phrase_dict, filed_vec_dict)

        return phrase_fea_dict

    def mining_phrase_from_text(self, seed_entity_dict, cut_pos_text_list):
        """
        给定切词及词性标注后的数据，挖掘高质量短语
        :param seed_entity_dict:  种子字典
        :param cut_pos_text_list: 切词词性标注后的文本对象列表
        :return: 短语得分列表，短语特征字典
        """
        # 挖掘短语特征
        phrase_fea_dict = self.extract_phrase_fea(cut_pos_text_list)

        # 根据种子词典打标数据
        positive_phrase_dict, negative_phrase_dict = self.phrase_processor.label_data(phrase_fea_dict, seed_entity_dict)

        # 随机森林分类
        xgb_model_list = self.xgb_forest.train_model(positive_phrase_dict, negative_phrase_dict)
        phrase_pred_list = self.xgb_forest.pred_model(xgb_model_list, negative_phrase_dict)

        return phrase_pred_list, negative_phrase_dict

    def mining_entity_feature(self, entity_cut_name_list, cut_pos_data_list) -> dict:
        """
        给定实体名列表及其所在的文本，挖掘实体相关特征
        :param entity_cut_name_list: 实体名称列表（切词后）
        :param cut_pos_data_list: 切词词性标注后的列表
        :return: dict[phrase_name] = fea_dict
        """
        # 匹配文本中的实体，计算上下文特征
        entity_context_dict = self.entity_processor.extract_entity_context_info(entity_cut_name_list, cut_pos_data_list)

        # 从全量词向量中抽取候选短语所含词语及对应向量
        filed_vec_dict = self.phrase_processor.get_phrase_word_vec(entity_context_dict, self.phrase_config.word_vec_path)

        # 计算实体相关特征
        entity_fea_dict = self.phrase_feature.extract_feature(entity_context_dict, filed_vec_dict)

        return entity_fea_dict

    def pred_phrase_score(self, phrase_fea_dict):
        """
        根据特征预测短语分数
        :param phrase_fea_dict: 短语特征字典
        :return:
        """
        # 随机森林分类
        xgb_model_list = self.xgb_forest.load_models()
        phrase_pred_list = self.xgb_forest.pred_model(xgb_model_list, phrase_fea_dict)
        return phrase_pred_list

if __name__ == "__main__":
    pass