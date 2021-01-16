# encoding: utf-8

import sys
sys.path.append("../BERTAutoNER")

from util.file_util import FileUtil
from util.arg_util import ArgparseUtil
from util.log_util import LogUtil
from data_process.entity_label import EntityLabel

class DataLabelRun(object):
    """
    打标ner数据
    """
    def __init__(self, args):
        self.args = args

    def distance_label(self):
        """
        使用原始实体字典+短语进行远程标注
        :return:
        """
        # 加载种子字典
        seed_entity_dict = FileUtil.read_entity_type_dict(self.args.seed_entity_path)
        # 加载挖掘出的所有短语列表
        all_phrase_list = FileUtil.read_raw_data(self.args.phrase_path)
        # 加载格式化文本
        train_obj_list = FileUtil.read_text_obj_data(self.args.train_data_path)
        dev_obj_list = FileUtil.read_text_obj_data(self.args.dev_data_path)
        test_obj_list = FileUtil.read_text_obj_data(self.args.test_data_path)

        # 远程标注数据
        entity_label = EntityLabel(seed_entity_dict, all_phrase_list)
        train_distance_obj_list = entity_label.generate_distance_label_data(train_obj_list)
        dev_distance_obj_list = entity_label.generate_distance_label_data(dev_obj_list)
        test_distance_obj_list = entity_label.generate_distance_label_data(test_obj_list, is_test=True)

        # 存储训练集、验证集和测试集
        FileUtil.save_text_obj_data(train_distance_obj_list, self.args.train_distance_data_path)
        FileUtil.save_text_obj_data(dev_distance_obj_list, self.args.dev_distance_data_path)
        FileUtil.save_text_obj_data(test_distance_obj_list, self.args.test_distance_data_path)

    def add_entity_distance_label(self):
        """
        增加分类的短语实体进行远程标注
        :return:
        """
        # 加载种子字典
        seed_entity_dict = FileUtil.read_entity_type_dict(self.args.seed_entity_path)
        # 加载挖掘出的所有短语列表
        all_phrase_list = FileUtil.read_raw_data(self.args.phrase_path)
        # 加载分类后的短语
        add_entity_dict = FileUtil.read_entity_type_dict(self.args.phrase_label_path)

        # 获得新的种子字典及短语列表
        old_entity_num = len(seed_entity_dict)
        for entity_name, entity_type in add_entity_dict.items():
            if entity_name not in seed_entity_dict:
                seed_entity_dict[entity_name] = entity_type
        new_phrase_list = [phrase for phrase in all_phrase_list if phrase not in add_entity_dict]
        LogUtil.logger.info("原种子实体数量为: {0}, 新种子实体数量为: {1}, 原短语数量: {2}, 新短语数量为: {3}"
                            .format(old_entity_num, len(seed_entity_dict), len(all_phrase_list), len(new_phrase_list)))

        # 加载格式化文本
        train_obj_list = FileUtil.read_text_obj_data(self.args.train_data_path)
        dev_obj_list = FileUtil.read_text_obj_data(self.args.dev_data_path)
        test_obj_list = FileUtil.read_text_obj_data(self.args.test_data_path)

        # 远程标注数据
        entity_label = EntityLabel(seed_entity_dict, new_phrase_list)
        train_distance_obj_list = entity_label.generate_distance_label_data(train_obj_list)
        dev_distance_obj_list = entity_label.generate_distance_label_data(dev_obj_list)
        test_distance_obj_list = entity_label.generate_distance_label_data(test_obj_list, is_test=True)

        # 存储训练集、验证集和测试集
        FileUtil.save_text_obj_data(train_distance_obj_list, self.args.add_train_distance_data_path)
        FileUtil.save_text_obj_data(dev_distance_obj_list, self.args.add_dev_distance_data_path)
        FileUtil.save_text_obj_data(test_distance_obj_list, self.args.add_test_distance_data_path)


if __name__ == "__main__":
    args = ArgparseUtil().distance_label_argparse()

    data_label_run = DataLabelRun(args)

    # 原始数据进行远程监督
    if args.do_source_distance:
        data_label_run.distance_label()

    # 增加分类的短语实体进行远程标注
    if args.do_add_distance:
        data_label_run.add_entity_distance_label()



