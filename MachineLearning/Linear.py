# 线性回归模型
# author: 羽路星尘
import numpy as np
from matplotlib import pyplot as plt
from sklearn.datasets import make_regression
from sklearn.utils import shuffle


def initializeParams(dims):
    """
    初始化
    :param dims: 变量的维度
    :return:
    w: 初始化权重参数 <br>
    b: 初始化偏置参数
    """
    w = np.zeros((dims, 1))
    b = 0
    return w, b


def linearLoss(X, y, w, b):
    """
    线性回归模型
    :param X: 变量矩阵
    :param y: 标签向量
    :param w: 变量参数权重矩阵
    :param b: 偏置
    :return:
    y_hat: 预测值 <br>
    loss: 均方损失 <br>
    dw: 权重系数一阶偏导 <br>
    db: 偏置一阶偏导
    """

    # 训练样本量
    num_train = X.shape[0]
    # 线性回归预测
    y_hat = np.dot(X, w) + b
    # 计算均方损失
    loss = np.sum((y_hat - y) ** 2) / num_train
    # 基于均方损失对权重系数求一阶导数
    dw = np.dot(X.T, (y_hat - y)) / num_train
    # 基于均方损失对偏置求一阶导数
    db = np.sum(y_hat - y)
    return y_hat, loss, dw, db


def linearTrain(X, y, learning_rate=0.1, epochs=10000):
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
    w, b = initializeParams(X.shape[1])
    # 迭代训练
    for i in range(epochs):
        y_hat, loss, dw, db = linearLoss(X, y, w, b)
        w += -learning_rate * dw
        b += -learning_rate * db
        loss_list.append(loss)
        if i % 1000 == 0:
            print("epoch %d loss %f" % (i, loss))
        # 将当前迭代优化后的参数保存到字典中
        params = {
            'w': w,
            'b': b
        }
        # 将当前迭代的梯度保存到字典中
        grads = {
            'dw': dw,
            'db': db
        }
    return loss_list, params, grads


def predict(X, params):
    """
    预测函数
    :param X: 测试集
    :param params: 模型训练参数
    :return: y_pred: 模型预测结果
    """
    # 获取模型参数
    w = params['w']
    b = params['b']
    # 预测
    y_pred = np.dot(X, w) + b
    return y_pred


def r2_score(y_test, y_pred):
    """
    R2系数函数，R2系数表示了因变量能通过回归关系被自变量解释的比例
    :param y_test: 测试集标签
    :param y_pred: 测试集预测值
    :return: r2: R2系数
    """
    # 标签均值
    y_avg = np.mean(y_test)
    # 估计值与平均值的误差
    ss_tot = np.sum((y_test - y_avg) ** 2)
    # 估计值与真实值的误差

    ss_res = np.sum((y_test - y_pred) ** 2)
    # R2计算
    r2 = 1 - (ss_res / ss_tot)
    return r2


if __name__ == '__main__':
    # 获取数据集
    X, y = make_regression(n_features=1, noise=10, random_state=8)
    plt.xlabel("X")
    plt.ylabel("Y", rotation=0)
    plt.title("Dataset")
    plt.scatter(X, y)
    plt.show()

    X, y = shuffle(X, y, random_state=13)
    offset = int(X.shape[0] * 0.8)
    # 划分训练集和验证集
    X_train, y_train = X[:offset], y[:offset]
    X_test, y_test = X[offset:], y[offset:]
    y_train = y_train.reshape((-1, 1))
    y_test = y_test.reshape((-1, 1))
    print("X_train's shape: ", X_train.shape)
    print("X_test's shape: ", X_test.shape)
    print("y_train's shape: ", y_train.shape)
    print("Y_test's shape: ", y_test.shape)

    # 训练模型
    loss_list, params, grads = linearTrain(X_train, y_train, 0.001, 15000)
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

    # 预测训练集中的数据的值
    y_pred0 = predict(X_train, params)
    # 预测验证集中数据的值
    y_pred = predict(X_test, params)
    # R2系数计算
    r2 = r2_score(y_test, y_pred)
    print(r2)

    plt.xlabel("X")
    plt.ylabel("Y", rotation=0)
    plt.title("Train Dataset")
    plt.scatter(X_train, y_train)
    plt.plot(X_train, y_pred0)
    plt.show()

    plt.xlabel("X")
    plt.ylabel("Y", rotation=0)
    plt.title("Predict Dataset")
    plt.scatter(X_test, y_test)
    plt.plot(X_test, y_pred)
    plt.show()
