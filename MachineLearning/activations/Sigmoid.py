import numpy as np

from MachineLearning.activations.BaseActivation import BaseActivation


class Sigmoid(BaseActivation):
    def __init__(self):
        self.output = None
    def forward(self, x):
        self.output = 1 / (1 + np.exp(-x))
        return self.output

    def backward(self, grad):
        out = self.output
        return grad * out * (1 - out)
