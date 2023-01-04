import numpy as np
from sklearn.datasets import load_diabetes
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
    # 训练的特征数
    num_feature = X.shape[1]
    # 线性回归预测
    y_hat = np.dot(X, w) + b
    # 计算均方损失
    loss = np.sum((y_hat - y) ** 2) / num_train
    # 基于均方损失对权重系数的一阶导数
    dw = 2 * np.dot(X.T, (y_hat - y)) / num_train
    # 基于均方损失对偏置的一阶导数
    db = 2 * np.sum(y_hat - y) / num_train
    return y_hat, loss, dw, db


def linearTrain(X, y, learning_rate=0.01, epochs=10000):
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
    w, b = initializeParams(X.shape[1])
    # 迭代训练
    for i in range(epochs):
        y_hat, loss, dw, db = linearLoss(X, y, w, b)
        w += -learning_rate * dw
        b += -learning_rate * db
        loss_his.append(loss)
        if i % 10000 == 0:
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
    R2系数函数
    :param y_test: 测试集标签
    :param y_pred: 测试集预测值
    :return: r2: R2系数
    """
    # 标签均值
    y_avg = np.mean(y_test)
    # 总离差平方和
    ss_tot = np.sum((y_test - y_avg) ** 2)
    # 残差平方和
    ss_res = np.sum((y_test - y_pred) ** 2)
    # R2计算
    r2 = 1 - (ss_res / ss_tot)
    return r2


if __name__ == '__main__':
    diabetes = load_diabetes()
    data, target = diabetes.data, diabetes.target
    X, y = shuffle(data, target, random_state=13)
    offset = int(X.shape[0] * 0.8)
    X_train, y_train = X[:offset], y[:offset]
    X_test, y_test = X[offset:], y[offset:]
    y_train = y_train.reshape((-1, 1))
    y_test = y_test.reshape((-1, 1))
    print("X_train's shape: ", X_train.shape)
    print("X_test's shape: ", X_test.shape)
    print("y_train's shape: ", y_train.shape)
    print("Y_test's shape: ", y_test.shape)

    loss_his, params, grads = linearTrain(X_train, y_train, 0.001, 200000)
    print(params)
    y_pred = predict(X_test, params)
    r2 = r2_score(y_test, y_pred)
    print(r2)
