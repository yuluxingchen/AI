from collections import Counter

import numpy as np
from matplotlib import pyplot as plt
from sklearn import datasets
from sklearn.utils import shuffle


def distances(X_test, X_train):
    """
    定义欧氏距离
    :param X_test: 测试样本实例矩阵
    :param X_train: 训练样本实例矩阵
    :return: dists: 欧氏距离
    """
    # 测试样本与训练样本的矩阵点乘
    M = np.dot(X_test, X_train.T)
    # 测试样本矩阵平方和
    te = np.square(X_test).sum(axis=1)
    # 训练样本矩阵平方和
    tr = np.square(X_train).sum(axis=1)
    # 计算欧式距离
    dists = np.sqrt(-2 * M + tr + np.matrix(te).T)
    return dists


def predict_labels(y_train, dists, k=1):
    # 测试样本量
    num_test = dists.shape[0]
    # 初始化测试集预测结果
    y_pred = np.zeros(num_test)
    # 遍历
    for i in range(num_test):
        # 按欧氏距离排序后驱索引，并用训练集标签按排序后的索引取值
        labels = y_train[np.argsort(dists[i, :])].flatten()
        # 取最近的k个值
        closest_y = labels[0:k]
        # 对最近的k个值进行计数统计
        c = Counter(closest_y)
        # 取类别计数最多的
        y_pred[i] = c.most_common(1)[0][0]
    return y_pred

if __name__ == '__main__':
    # 导入iris数据集
    iris = datasets.load_iris()
    # 打乱数据和标签
    X, y = shuffle(iris.data, iris.target, random_state=13)
    # 数据转换成float32格式
    X = X.astype(np.float32)
    # 简单划分训练集和测试集
    offset = int(X.shape[0] * 0.7)
    X_train, y_train = X[:offset], y[:offset]
    X_test, y_test = X[offset:], y[offset:]
    # 将标签转换为竖向量
    y_train = y_train.reshape((-1, 1))
    y_test = y_test.reshape((-1, 1))
    # 打印训练集和测试集的大小
    print('X_train=', X_train.shape)
    print('X_test=', X_test.shape)
    print('y_train=', y_train.shape)
    print('y_test=', y_test.shape)

    dists = distances(X_test, X_train)

    k = 1
    # 测试集预测结果
    y_test_pred = predict_labels(y_train, dists, k)
    y_test_pred = y_test_pred.reshape((-1, 1))
    # 计算预测正确的数量
    num_correct = np.sum(y_test_pred == y_test)
    # 计算分类准确率
    accuracy = float(num_correct) / X_test.shape[0]
    print('KNN Accuracy: ', accuracy)

