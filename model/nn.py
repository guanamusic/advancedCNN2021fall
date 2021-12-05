import torch
# import numpy as np
from model.base import BaseModule
# from model.layers import Conv1dWithInitialization
import torch.nn.functional as F
from model.layers import gen_conv, ContextualAttention

#Let's see if it's pushed directly
class NeuralNetwork(BaseModule):
    def __init__(self, config, use_cuda, device_ids):
        super(NeuralNetwork, self).__init__()
        self.input_dim = config['input_dim']
        self.cnum = config['ngf']
        self.use_cuda = use_cuda
        self.device_ids = device_ids

        self.coarse_generator = CoarseGenerator(self.input_dim, self.cnum, self.use_cuda, self.device_ids)
        self.fine_generator = FineGenerator(self.input_dim, self.cnum, self.use_cuda, self.device_ids)

    def forward(self, x, mask):
        x_stage1 = self.coarse_generator(x, mask)
        x_stage2, offset_flow = self.fine_generator(x, x_stage1, mask)
        return x_stage1, x_stage2, offset_flow

