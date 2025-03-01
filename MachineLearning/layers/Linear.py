import numpy as np

from MachineLearning.layers.BaseLayer import BaseLayer


class Linear(BaseLayer):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.w = np.random.randn(output_size, input_size)
        self.b = np.zeros((output_size, 1))
        self.input = None

    def forward(self, X):
        self.input = X
        result = np.dot(self.w, X) + self.b
        return result

    def backward(self, grad):
        dw = np.dot(grad, self.input.T)
        db = np.sum(grad, axis=1, keepdims=True)
        grad = np.dot(self.w.T, grad)
        return dw, db, grad

    def update(self, dw, db, learning_rate):
        self.w -= dw * learning_rate
        self.b -= db * learning_rate
