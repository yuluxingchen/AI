# 最小绝对收缩和选择算子
import numpy as np


def sign(x):
    """
    符号函数
    :param x: 浮点数值
    :return: 整数符号
    """
    if x > 0:
        return 1
    elif x < 0:
        return -1
    else:
        return 0


def initialize_params(dims):
    """
    初始化模型参数
    :param dims: 数据变量维数
    :return:
    w: 权重系数值
    b: 偏置参数值
    """
    w = np.zeros((dims, 1))
    b = 0
    return w, b


def l1_loss(X, y, w, b, alpha):
    """
    LASSO回归损失函数
    :param X: 变量矩阵
    :param y: 标签矩阵
    :param w: 权重矩阵
    :param b: 偏置
    :param alpha: 正则化系数
    :return:
    y_hat: 线性模型预测输出<br>
    loss: 均方损失函数
    dw: 权重系数一阶偏导
    db: 偏置一阶偏导
    """
    vec_sign = np.vectorize(sign)
    # 训练样本量
    num_train = X.shape[0]
    # 训练特征数
    num_feature = X.shape[1]
    # 回归模型预测输出
    y_hat = np.dot(X, w) + b
    # L1 损失函数
    loss = np.sum((y_hat - y) ** 2) / num_train + np.sum(alpha * abs(w))
    # 基于向量化符号函数的参数梯度计算
    dw = np.dot(X.T, (y_hat - y)) / num_train + alpha * vec_sign(w)
    db = np.sum((y_hat - y)) / num_train
    return y_hat, loss, dw, db


def lasso_train(X, y, learning_rate=0.01, epochs=1000):
    """
    训练过程
    :param X: 输入变量矩阵
    :param y: 输出标签向量
    :param learning_rate: 学习率
    :param epochs: 训练迭代次数
    :return:
    loss_his: 每次迭代的均方损失 <br>
    params: 优化后的参数字典 <br>
    grads: 优化后的参数梯度字典
    """
    # 记录训练损失的空列表
    loss_his = []
    params = []
    grads = []
    # 初始化模型参数
    w, b = initialize_params(X.shape[1])
    # 迭代训练
    for i in range(epochs):
        y_hat, loss, dw, db = l1_loss(X, y, w, b, 0.1)
        w = w - learning_rate * dw
        b = b - learning_rate * db
        loss_his.append(loss)
        if i % 50 == 0:
            print("epoch %d loss %f" % (i, loss))
        # 将当前迭代步优化后的参数保存到字典中
        params = {
            'w': w,
            'b': b
        }
        # 将当前迭代步的梯度保存到字典中
        grads = {
            'dw': dw,
            'db': db
        }
    return loss_his, params, grads
