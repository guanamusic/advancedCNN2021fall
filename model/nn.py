import torch
import numpy as np

from model.base import BaseModule
from model.layers import Conv2dBlock


class NeuralNetwork(BaseModule):
    """
    Actually, the neural network works here lmao
    """
    def __init__(self, config):
        """
        Fill in some neural network stuff initialization
        """
        super(NeuralNetwork, self).__init__()

    def forward(self, masked_input):
        """
        Compute forward pass of neural network
        """
