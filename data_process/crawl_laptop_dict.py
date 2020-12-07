# encoding: utf-8

import requests
from lxml import etree

from util.log_util import LogUtil
from util.file_util import FileUtil

class CrawlLaptopDict(object):
    """
    爬取笔记本领域字典数据
    """
    def crawl_computer_html(self, crawl_url):
        """
        爬取电脑页面
        :param crawl_url:
        :return:
        """
        form_list = []
        response = requests.get(url=crawl_url)
        response.encoding = "utf-8"
        try:
            html = etree.HTML(response.text)
            form_list = html.xpath("//*[@id='main-content']/article/table/tr/td/p/a/text()")
        except requests.RequestException as e:
            LogUtil.logger.info(e)

        return form_list

    def crawl_control(self):
        """

        :return:
        """
        suffix_list = ["num"] + [chr(i) for i in range(ord("a"), ord("z") + 1)]
        url_list = ["https://www.computerhope.com/jargon/j" + suffix + ".htm" for suffix in suffix_list]

        entity_type_dict = {}
        for url in url_list:
            mention_list = crawler.crawl_computer_html(url)
            for mention in mention_list:
                entity_type_dict[mention] = "computer"
        LogUtil.logger.info("爬取mention数量:{0}".format(len(entity_type_dict)))

        # 存储字典
        FileUtil.save_entity_type(entity_type_dict, "computer_dict")

if __name__ == "__main__":
    crawler = CrawlLaptopDict()
    crawler.crawl_control()



