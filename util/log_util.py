# encoding: utf-8

import logging

class LogUtil(object):
    """
    日志工具类
    """

    # 日志配置
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(filename)s line:%(lineno)d] %(levelname)s: %(message)s",
                        datefmt="%Y-%m-%d %H:%M:%S")

    logger = logging.getLogger()

