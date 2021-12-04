import torch

from model.base import BaseModule

"""
Fill some basic layers that you want with specification
"""

class Conv1dWithInitialization(BaseModule):
    def __init__(self, **kwargs):
        super(Conv1dWithInitialization, self).__init__()
        self.conv1d = torch.nn.Conv1d(**kwargs)
        torch.nn.init.orthogonal_(self.conv1d.weight.data, gain=1)

    def forward(self, x):
        return self.conv1d(x)


class DepthwiseConv(BaseModule):
    def __init__(self, nin, kernels_per_layer, kernel_size):
        super(DepthwiseConv, self).__init__()
        self.depthwise = torch.nn.Conv2d(nin, nin*kernels_per_layer, kernel_size, padding=1, groups=nin)

    def forward(self, x):
        out = self.depthwise(x)
        return out
