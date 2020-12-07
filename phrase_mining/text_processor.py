# encoding: utf-8

import json
from tool.pos.pos import CRF as POS
from tool.libcut.python.cut import Cutter
from util.log_util import LogUtil

class TextProcessor(object):
    """
    文本处理类
    """
    def __init__(self, config):
        self.config = config

    def load_cutter(self):
        """
        加载切词器
        :return:
        """
        cutter = Cutter("LM_CRF", "../tool/libcut/data")
        return cutter

    def load_postagger(self):
        """
        加载词性标注器
        :return:
        """
        postagger = POS("../tool/pos/data/chinese-model.txt", "../tool/pos/data/symbol.txt")
        return postagger

    def cut_pos_word(self, data_path):
        """
        分词及词性标注
        :param data_path:
        :return:
        """
        LogUtil.logger.info("开始文本分词及词性标注")

        # 加载切词器和磁性标注器
        cutter = self.load_cutter()
        postagger = self.load_postagger()

        cut_pos_data_list = []
        with open(data_path, "r", encoding="utf-8") as data_file:
            for text_id, item in enumerate(data_file):
                item = item.strip()
                # 细粒度，粗粒度分词结果
                fine_words, coarse_words = cutter.segment(item)

                # 词性标注
                pos_result_list = postagger.process(" ".join(fine_words)).decode("utf-8").split()

                assert len(fine_words) == len(pos_result_list)

                text_obj = {"text_id": text_id, "text": item, "cut_text": fine_words, "pos_text": pos_result_list}
                cut_pos_data_list.append(text_obj)

                if text_id % 5000 == 0:
                    LogUtil.logger.info(str(text_id))

        if self.config.IS_WRITE_INTER_RESULT:
            with open(self.config.cut_pos_path, "w", encoding="utf-8") as cut_pos_file:
                for text_obj in cut_pos_data_list:
                    cut_pos_file.write(json.dumps(text_obj, ensure_ascii=False) + "\n")

            LogUtil.logger.info("写入文本分词及标注中间结果到文件")

        LogUtil.logger.info("完成文本分词及词性标注")

        return cut_pos_data_list

