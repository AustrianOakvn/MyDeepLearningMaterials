import torch
import torch.nn as nn


cfg_res_18 = {'conv1':[7, 3, 64],
        'M':[],
        'conv2':[[3, 64, 64], [3, 64, 64]],
        'conv3':[[3, 64, 128], [3, 128, 128]],
        'conv4':[[3, 128, 256], [3, 256, 256]],
        'conv5':[[3, 256, 512], [3, 512, 512]]
        }


class ResNet18(nn.Module):
    def __init__(self, cfg):
        super(ResNet18, self).__init__()

        self.cfg = cfg

        self.layers = []

        for k,v in self.cfg:
            if 



