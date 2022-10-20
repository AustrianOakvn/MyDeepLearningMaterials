import torch
import torch.nn as nn


class InceptionBlock(nn.Module):
    def __init__(self, in_channels):
        super(InceptionBlock, self).__init__()

        self.in_channels = in_channels
        
        self.first_direction = nn.Conv2d()
