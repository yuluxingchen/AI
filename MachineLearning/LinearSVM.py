# 导入相关库
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


class LinearSVM:
    def __init__(self, C=1.0, learning_rate=0.01, epochs=1000):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.C = C
        self.w = None
        self.b = 0.0

    def train(self, X, y):
        np.random.seed(3)
        self.w = np.random.normal(size=X.shape[1])
        for _ in range(self.epochs):
            last_w, last_b = self.w, self.b
            for index in range(X.shape[0]):
                if y[index] * (np.dot(X[index], self.w) + self.b) <= 0:
                    self.w -= self.learning_rate * (self.w - self.C * X[index] * y[index])
                    self.b -= self.learning_rate * y[index]
            if self.w.all() == last_w.all() and self.b == last_b:
                break

    def predict(self, X):
        return np.sign(np.dot(X, self.w) + self.b)


if __name__ == '__main__':
    # 生成模拟二分类数据集
    X, y = make_blobs(n_samples=400, n_features=2, centers=2, cluster_std=1.2, random_state=40)
    y[y == 0] = -1

    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=43)
    # 创建线性可分支持向量机模型
    svm = LinearSVM(0.5, 0.01, 1000)
    svm.train(X_train, y_train)
    y_predict = svm.predict(X_test)
    print(svm.w, svm.b)
    # 计算测试集上的分类准确率
    print("Accuracy of linear svm: ", accuracy_score(y_test, y_predict))
    # 设置颜色参数
    colors = {-1: 'r', 1: 'g'}
    # 绘制二分类数据集的散点图
    plt.scatter(X_test[:, 0], X_test[:, 1], marker='o', c=pd.Series(y_predict).map(colors))
    plt.show()

    # 绘制二分类数据集的散点图
    plt.scatter(X_test[:, 0], X_test[:, 1], marker='o', c=pd.Series(y_test).map(colors))
    plt.show()

