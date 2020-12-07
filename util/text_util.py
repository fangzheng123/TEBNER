# encoding: utf-8

from textblob import TextBlob

class TextUtil(object):
    """
    文本工具类
    """
    @classmethod
    def tagger_data(cls, text) -> list:
        """
        对数据进行词性标注
        :param data_path:
        :return:
        """
        blob = TextBlob(text)
        pos_tag_list = blob.tags

        return pos_tag_list
