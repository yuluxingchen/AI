from abc import ABC, abstractmethod

import numpy as np


class ActivationBase(ABC):
    def __init__(self, **kwargs):
        super().__init__()

    def __call__(self, x):
        if x.ndim == 1:
            x = x.reshape(1, -1)
        return self.fn(x)

    @abstractmethod
    def fn(self, x):
        """ 对输入进行激活 """
        raise NotImplementedError  # 必须实现这个函数

    @abstractmethod
    def grad(self, x, **kwargs):
        """ 对输入进行梯度计算 """
        raise NotImplementedError  # 必须实现这个函数


class Sigmoid(ActivationBase):
    def __init__(self):
        super().__init__()

    def __str__(self):
        return "Sigmoid"

    def fn(self, x):
        """ f(x) = 1 / 1 + e^(-x) """
        return 1 / (1 + np.exp(-x))

    def grad(self, x, **kwargs):
        """ f'(x) = f(x)(1 - f(x)) """
        fn_x = self.fn(x)
        return fn_x * (1 - fn_x)

    def grad2(self, x):
        """ f''(x) = f(x)(1 - f(x))(1 - 2f(x)) """
        fn_x = self.fn(x)
        return fn_x * (1 - fn_x) * (1 - 2 * fn_x)


class ReLU(ActivationBase):
    def __init__(self):
        super().__init__()

    def fn(self, x):
        """
        f(x) = max(x, 0)
        """
        return np.clip(x, 0, np.inf)  # 将函数限制在 [0, inf] 之间，超过则会等于限制值

    def grad(self, x, **kwargs):
        """
        f'(x) = {
                1, x>0
                0, x<=0
                }
        """
        return (x > 0).astype(int)

    def grad2(self, x):
        """
        f''(x) = 0
        """
        return np.zeros_like(x)  # 输出一个零矩阵


class LeakyReLU(ActivationBase):
    def __init__(self, alpha=0.01):
        super().__init__()
        self.alpha = alpha

    def fn(self, x):
        """
            f(x) = {
                    x, x>0
                    alpha * x, x<=0 (alpha < 1)
                    }
        """
        return np.where(x > 0, x, x * self.alpha)

    def grad(self, x, **kwargs):
        """
            f'(x) = {
                    1, x>0
                    alpha, x<=0
                    }
         """
        return np.where(x > 0, 1, self.alpha)

    def grad2(self, x):
        """
        f''(x) = 0
        """
        return np.zeros_like(x)  # 输出一个零矩阵


class Affine(ActivationBase):
    def __init__(self, slope=1, intercept=0):
        super().__init__()
        self.slope = slope
        self.intercept = intercept

    def __str__(self):
        return "Affine(slope={}, intercept={})".format(self.slope, self.intercept)

    def fn(self, x):
        return self.slope * x + self.intercept

    def grad(self, x, **kwargs):
        return self.slope * np.ones_like(x)

    def grad2(self, x):
        return np.zeros_like(x)


class Identity(Affine):
    def __init__(self):
        super().__init__(slope=1, intercept=0)

    def __str__(self):
        return "Identity(slope={}, intercept={})".format(self.slope, self.intercept)
