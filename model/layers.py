import torch

from model.base import BaseModule

"""
Fill some basic layers that you want with specification
"""


class DepthwiseConv(BaseModule):
    def __init__(self, nin, kernels_per_layer, kernel_size):
        super(DepthwiseConv, self).__init__()
        self.depthwise = torch.nn.Conv2d(nin, nin*kernels_per_layer, kernel_size, padding=1, groups=nin)

    def forward(self, x):
        out = self.depthwise(x)
        return out


class Conv2dBlock(BaseModule):
    def __init__(self, activation=None, **kwargs):
        super(Conv2dBlock, self).__init__()
        self.conv2d = torch.nn.Conv2d(**kwargs)

        if activation == 'relu':
            self.activation = torch.nn.ReLU()
        elif activation == 'elu':
            self.activation = torch.nn.ELU()
        elif activation == 'lrelu':
            self.activation = torch.nn.LeakyReLU(negative_slope=0.2)
        elif activation == 'none':
            self.activation = None
        else:
            raise Exception(f"Unsupported activation: {activation}")

    def forward(self, x):
        output = x
        if self.activation is None:
            output = self.conv2d(output)
        else:
            output = self.activation(self.conv2d(output))
        return output
