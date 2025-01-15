from abc import ABC, abstractmethod
from copy import deepcopy

import numpy as np
from numpy.linalg import norm


class OptimizerBase(ABC):
    def __init__(self, lr, scheduler=None):
        self.cache = {}
        self.cur_step = 0
        self.hyperparameters = {}
        self.lr_scheduler = SchedulerInitializer(scheduler, lr)

    def __call__(self, param, param_grad, param_name, cur_loss=None):
        return self.update(param, param_grad, param_name, cur_loss)

    def step(self):
        self.cur_step += 1

    def reset_step(self):
        self.cur_step = 0

    def copy(self):
        return deepcopy(self)

    @abstractmethod
    def update(self, param, param_grad, param_name, cur_loss):
        raise NotImplementedError


class SGD(OptimizerBase):
    def __init__(self, lr=0.01, momentum=0.0, clip_norm=None, lr_scheduler=None):
        """
        一个随机梯度下降优化器
        :param lr: 浮点数类型，表示优化器的学习率
        :param momentum: 浮点数类型，取值范围为[0,1]，如果上次的下降方向与本次的下降方向相同，就加速梯度更新；反之则减缓梯度更新
        :param clip_norm: 浮点数类型，限制梯度的最大值，防止出现梯度爆炸现象
        :param lr_scheduler: 自定义的学习率调度器
        :param kwargs: 其他参数
        """
        super().__init__(lr, lr_scheduler)
        self.hyperparameters = {
            "id": "SGD",
            "lr": lr,
            "momentum": momentum,
            "clip_norm": clip_norm,
            "lr_scheduler": str(self.lr_scheduler)
        }

    def __str__(self):
        hp = self.hyperparameters
        id, lr, mm, clip, sche = hp["id"], hp["lr"], hp["momentum"], hp["clip_norm"], hp["lr_scheduler"]
        return "id={}(lr={}, momentum={}, clip_norm={}, lr_scheduler={})".format(
            id, lr, mm, clip, sche
        )

    def update(self, param, param_grad, param_name, cur_loss):
        """
        训练时更新参数
        :param param: ndarray类型，大小为 (n, m)，代表了参与训练的模型参数
        :param param_grad: ndarray类型，代表了参数的梯度
        :param param_name: 字符串类型，代表了参数的名字
        :param cur_loss: 浮点数类型，代表了当前的损失
        """
        C = self.cache
        H = self.hyperparameters
        momentum, clip_norm = H["momentum"], H["clip_norm"]
        lr = self.lr_scheduler(self.cur_step, cur_loss)

        if param_name not in C:
            C[param_name] = np.zeros_like(param_grad)

        t = np.inf if clip_norm is None else clip_norm
        if norm(param_grad) > t:
            param_grad = param_grad * t / norm(param_grad)

        update_param = momentum * C[param_name] + lr * param_grad
        self.cache[param_name] = update_param
        return param - update_param
