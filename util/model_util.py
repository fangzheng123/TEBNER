# encoding: utf-8

import numpy as np
import torch
from collections import OrderedDict

class ModelUtil(object):
    """
    模型工具类
    """
    def seed_everything(self, seed=42):
        '''
        设置整个开发环境的seed
        :param seed:
        :return:
        '''
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # some cudnn methods can be random even after fixing the seed, unless you tell it to be deterministic
        torch.backends.cudnn.deterministic = True

    def load_model(self, model, model_save_path, device):
        """
        加载模型
        :param model: 模型对象
        :param model_save_path: 模型存储路径
        :param model_save_path: device
        :return:
        """
        # 当使用DataParallel训练时，key值会多出"module."
        state_dict = torch.load(model_save_path, map_location=device)
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            # 移除 "module."
            if k.startswith("module."):
                k = k[7:]
            new_state_dict[k] = v

        model.load_state_dict(new_state_dict)

