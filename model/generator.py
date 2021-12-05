import torch

from model.base import BaseModule
from model.layers import Conv2dBlock
from model.contextual_attention import ContextualAttention

from tools import torch_nearest_upsampler


class Generator(BaseModule):
    def __init__(self, config):
        super(Generator, self).__init__()
        self.input_dim = 1
        self.cnum = config.model_config.generator_channel_factor

        self.coarse_generator = CoarseGenerator(input_dim=self.input_dim, cnum=self.cnum)
        self.fine_generator = FineGenerator(input_dim=self.input_dim, cnum=self.cnum)

    def forward(self, x, mask):
        x_stage1 = self.coarse_generator(x, mask)
        x_stage2, offset_flow = self.fine_generator(x, x_stage1, mask)
        return x_stage1, x_stage2, offset_flow


class CoarseGenerator(BaseModule):
    def __init__(self, input_dim, cnum):
        super(CoarseGenerator, self).__init__()
        input_sizes = [input_dim+2, cnum, cnum*2, cnum*2, cnum*4, cnum*4,
                       cnum*4, cnum*4, cnum*4, cnum*4, cnum*4, cnum*4,
                       cnum*4, cnum*2, cnum*2, cnum, cnum//2]
        output_sizes = input_sizes[1:] + [input_dim]
        activations = ['elu' for _ in input_sizes[:-1]] + ['none']
        kernel_sizes = [5] + [3 for _ in input_sizes[1:]]
        strides = [2 if (idx==1 or idx==3) else 1 for idx in range(len(input_sizes))]
        dilations = [1, 1, 1, 1, 1, 1,
                     2, 4, 8, 16, 1, 1,
                     1, 1, 1, 1, 1]
        paddings = [2] + dilations[1:]
        assert len(input_sizes)==len(dilations)

        self.blocks = torch.nn.ModuleList([
            Conv2dBlock(
                activation=activation,
                in_channels=input_size,
                out_channels=output_size,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation
            ) for activation, input_size, output_size, kernel_size, stride, padding, dilation in zip(
                activations, input_sizes, output_sizes, kernel_sizes, strides, paddings, dilations
            )
        ])

    def forward(self, x, mask):
        assert x.device == mask.device
        # For indicating the boundaries of images
        ones = torch.ones(x.size(0), 1, x.size(2), x.size(3)).to(x.device)

        # initialize output, (B, 3, w*f, h*t) maybe
        output = torch.cat([x, ones, mask], dim=1)
        for idx, block in enumerate(self.blocks):
            output = block(output)
            if idx == 11 or idx == 13:
                output = torch_nearest_upsampler(output, 2)

        x_stage1 = torch.clamp(output, -1., 1.)

        return x_stage1


class FineGenerator(BaseModule):
    def __init__(self, input_dim, cnum):
        super(FineGenerator, self).__init__()
        # conv_branch configurations
        conv_branch_input_sizes = [input_dim+2, cnum, cnum, cnum*2, cnum*2, cnum*4,
                                   cnum*4, cnum*4, cnum*4, cnum*4]
        conv_branch_output_sizes = conv_branch_input_sizes[1:] + [cnum*4]
        conv_branch_activations = ['elu' for _ in conv_branch_input_sizes]
        conv_branch_kernel_sizes = [5] + [3 for _ in conv_branch_input_sizes[1:]]
        conv_branch_strides = [2 if (idx==1 or idx==3) else 1 for idx in range(len(conv_branch_input_sizes))]
        conv_branch_dilations = [1, 1, 1, 1, 1, 1,
                                 2, 4, 8, 16]
        conv_branch_paddings = [2] + conv_branch_dilations[1:]
        assert len(conv_branch_input_sizes)==len(conv_branch_dilations)

        self.conv_branch_blocks = torch.nn.ModuleList([
            Conv2dBlock(
                activation=activation,
                in_channels=input_size,
                out_channels=output_size,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation
            ) for activation, input_size, output_size, kernel_size, stride, padding, dilation in zip(
                conv_branch_activations,
                conv_branch_input_sizes,
                conv_branch_output_sizes,
                conv_branch_kernel_sizes,
                conv_branch_strides,
                conv_branch_paddings,
                conv_branch_dilations
            )
        ])

        # attention_branch configurations
        attention_branch_input_sizes = [input_dim+2, cnum, cnum, cnum*2, cnum*4, cnum*4, cnum*4, cnum*4]
        attention_branch_output_sizes = attention_branch_input_sizes[1:] + [cnum * 4]
        attention_branch_activations = ['relu' if idx==5 else 'elu'
                                        for idx in range(len(attention_branch_input_sizes))]
        attention_branch_kernel_sizes = [5] + [3 for _ in attention_branch_input_sizes[1:]]
        attention_branch_strides = [2 if (idx == 1 or idx == 3) else 1
                                    for idx in range(len(attention_branch_input_sizes))]
        attention_branch_dilations = [1 for _ in attention_branch_input_sizes]
        attention_branch_paddings = [2] + attention_branch_dilations[1:]

        self.contextual_attention = ContextualAttention(ksize=3, stride=1, rate=2, fuse_k=3, softmax_scale=10,
                                                       fuse=True, use_cuda=True)

        self.attention_branch_blocks = torch.nn.ModuleList([
            Conv2dBlock(
                activation=activation,
                in_channels=input_size,
                out_channels=output_size,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation
            ) for activation, input_size, output_size, kernel_size, stride, padding, dilation in zip(
                attention_branch_activations,
                attention_branch_input_sizes,
                attention_branch_output_sizes,
                attention_branch_kernel_sizes,
                attention_branch_strides,
                attention_branch_paddings,
                attention_branch_dilations
            )
        ])

        # concat_branch configurations
        concat_branch_input_sizes = [cnum*8, cnum*4, cnum*4, cnum*2, cnum*2, cnum, cnum//2]
        concat_branch_output_sizes = concat_branch_input_sizes[1:] + [input_dim]
        concat_branch_activations = ['elu' for _ in concat_branch_input_sizes[:-1]] + ['none']
        concat_branch_kernel_sizes = [3 for _ in concat_branch_input_sizes]
        concat_branch_strides = [1 for _ in concat_branch_input_sizes]
        concat_branch_dilations = [1 for _ in concat_branch_input_sizes]
        concat_branch_paddings = concat_branch_dilations

        self.concat_branch_blocks = torch.nn.ModuleList([
            Conv2dBlock(
                activation=activation,
                in_channels=input_size,
                out_channels=output_size,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation
            ) for activation, input_size, output_size, kernel_size, stride, padding, dilation in zip(
                concat_branch_activations,
                concat_branch_input_sizes,
                concat_branch_output_sizes,
                concat_branch_kernel_sizes,
                concat_branch_strides,
                concat_branch_paddings,
                concat_branch_dilations
            )
        ])

    def forward(self, x_in, x_stage1, mask):
        assert x_in.device == mask.device
        x1_inpaint = x_stage1 * mask + x_in * (1. - mask)

        # For indicating the boundaries of images
        ones = torch.ones(x_in.size(0), 1, x_in.size(2), x_in.size(3)).to(x_in.device)

        # initialize output, (B, 3, w*f, h*t) maybe
        xnow = torch.cat([x1_inpaint, ones, mask], dim=1)

        output_conv = xnow
        output_attention = xnow

        # conv branch
        for idx, conv_branch_block in enumerate(self.conv_branch_blocks):
            output_conv = conv_branch_block(output_conv)

        # attention branch
        for idx, attention_branch_block in enumerate(self.attention_branch_blocks):
            output_attention = attention_branch_block(output_attention)
            if idx == 5:
                output_attention, offset_flow = self.contextual_attention(output_attention, output_attention, mask)

        # merge two branches
        output = torch.cat([output_conv, output_attention], dim=1)

        for idx, concat_branch_block in enumerate(self.concat_branch_blocks):
            output = concat_branch_block(output)
            if idx == 1 or idx == 3:
                output = torch_nearest_upsampler(output, 2)

        x_stage2 = torch.clamp(output, -1., 1.)

        return x_stage2, offset_flow