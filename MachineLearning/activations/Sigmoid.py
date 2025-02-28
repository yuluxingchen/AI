import numpy as np

from MachineLearning.activations.BaseActivation import BaseActivation


class Sigmoid(BaseActivation):
    def forward(self, x):
        return 1 / (1 + np.exp(-x))

    def backward(self, out):
        return out * (1 - out)
