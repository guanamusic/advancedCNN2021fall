import torch

from model.base import BaseModule
from model.layers import Conv2dBlock


class LocalDiscriminator(BaseModule):
    def __init__(self, config):
        super(LocalDiscriminator, self).__init__()
        self.input_dim = 1
        self.cnum = config.model_config.discriminator_channel_factor

        self.dis_conv_module = DisConvModule(self.input_dim, self.cnum)
        self.linear = torch.nn.Linear(self.cnum*4*8*8, 1)

    def forward(self, x):
        x = self.dis_conv_module(x)
        x = x.view(x.size()[0], -1)
        x = self.linear(x)

        return x


class GlobalDiscriminator(BaseModule):
    def __init__(self, config):
        super(GlobalDiscriminator, self).__init__()
        self.input_dim = 1
        self.cnum = config.model_config.discriminator_channel_factor

        self.dis_conv_module = DisConvModule(self.input_dim, self.cnum)
        self.linear = torch.nn.Linear(self.cnum*4*16*16, 1)

    def forward(self, x):
        x = self.dis_conv_module(x)
        x = x.view(x.size()[0], -1)
        x = self.linear(x)

        return x


class DisConvModule(BaseModule):
    def __init__(self, input_dim, cnum):
        super(DisConvModule, self).__init__()
        input_sizes = [input_dim, cnum, cnum*2, cnum*4]
        output_sizes = input_sizes[1:] + [cnum*4]

        self.blocks = torch.nn.ModuleList([
            Conv2dBlock(
                activation='lrelu',
                in_channels=input_size,
                out_channels=output_size,
                kernel_size=5,
                stride=2,
                padding=2,
                dilation=1
            ) for input_size, output_size in zip(input_sizes, output_sizes)
        ])

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return x

