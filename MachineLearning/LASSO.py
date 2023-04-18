# 最小绝对收缩和选择算子
import numpy as np
from matplotlib import pyplot as plt
from sklearn.datasets import load_diabetes
from sklearn.utils import shuffle


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
    w: 权重系数值 <br>
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
    # 回归模型预测输出
    y_hat = np.dot(X, w) + b
    # L1 损失函数
    loss = np.sum((y_hat - y) ** 2) / num_train + alpha * np.sum(abs(w))
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
    loss_list: 每次迭代的均方损失 <br>
    params: 优化后的参数字典 <br>
    grads: 优化后的参数梯度字典
    """
    # 记录训练损失的空列表
    loss_list = []
    params = []
    grads = []
    # 初始化模型参数
    w, b = initialize_params(X.shape[1])
    # 迭代训练
    for i in range(epochs):
        y_hat, loss, dw, db = l1_loss(X, y, w, b, 0.1)
        w = w - learning_rate * dw
        b = b - learning_rate * db
        loss_list.append(loss)
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
    return loss_list, params, grads


if __name__ == '__main__':
    # 获取数据集
    diabetes = load_diabetes()
    data, target = diabetes.data, diabetes.target
    # 将数据随机打乱
    X, y = shuffle(data, target, random_state=13)
    offset = int(X.shape[0] * 0.01)
    # 划分训练集和验证集
    X_train, y_train = X[:offset], y[:offset]
    X_test, y_test = X[offset:], y[offset:]
    y_train = y_train.reshape((-1, 1))
    y_test = y_test.reshape((-1, 1))
    print('X_train = ', X_train.shape)
    print('X_test = ', X_test.shape)
    print('y_train = ', y_train.shape)
    print('y_test = ', y_test.shape)

    loss_list, params, grads = lasso_train(X_train, y_train, 0.1, 3000)
    # 获取损失函数值作为 y 轴
    y_loss = loss_list
    # 获取损失函数个数作为 x 轴
    x_loss = range(len(loss_list))
    # 设置轴标签
    plt.xlabel("epoch")
    plt.ylabel("loss")
    # 绘制损失函数曲线
    plt.plot(x_loss, y_loss, linewidth=1, linestyle="solid", label="train loss")
    plt.show()