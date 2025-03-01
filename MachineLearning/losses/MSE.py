import numpy as np

from MachineLearning.losses.BaseLoss import BaseLoss


class MSE(BaseLoss):
    def forward(self, output, Y):
        return np.mean((output - Y) ** 2)

    def backward(self, output, Y):
        return (output - Y) / Y.shape[0]
