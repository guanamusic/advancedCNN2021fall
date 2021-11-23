import torch

from model.base import BaseModule
from model.nn import NeuralNetwork


class Inpainter(BaseModule):
    """
    Model (including neural network) works here
    """
    def __init__(self, config):
        super(Inpainter, self).__init__()
        self.nn = NeuralNetwork(config)

    def compute_loss(self, masked_input, ground_truth):
        """
        Compute the loss with just MSE lmao
        """
        network_output = self.nn(masked_input)
        loss = torch.nn.MSELoss()(network_output, ground_truth)
        return loss

    def forward(self, masked_input):
        """
        Compute forward pass of neural network
        """
        inpainted = self.nn(masked_input)
        return inpainted