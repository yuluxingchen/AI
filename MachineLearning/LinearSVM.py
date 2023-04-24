# 导入相关库
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


# 线性可分支持向量机
class LinearSVM:
    def __init__(self, learning_rate=0.01, lambda_param=1, epochs=1000):
        # 学习率
        self.learning_rate = learning_rate
        # α系数
        self.lambda_param = lambda_param
        # 训练轮数
        self.epochs = epochs
        # 权重
        self.w = None
        # 偏置
        self.b = 0.0
        # 每一轮的损失函数
        self.loss = [1]

    # 训练函数
    def train(self, X, y):
        # 设置随机数种子，便于复现
        np.random.seed(3)
        # 使用正态分布函数随机初始化权重
        self.w = np.random.normal(size=X.shape[1])
        # 训练模型
        for _ in range(self.epochs):
            last_w, last_b = self.w, self.b
            # 依次对每一个X中的向量分别计算 w, b 是否需要更新
            for index in range(X.shape[0]):
                # 如果分类不正确则更新 w，b
                if y[index] * (np.dot(X[index], self.w) + self.b) <= 0:
                    self.w -= self.learning_rate * (self.lambda_param * self.w - X[index] * y[index])
                    self.b -= self.learning_rate * self.lambda_param * y[index]
            loss = y[y != self.predict(X)].shape[0]
            # 存储损失函数
            self.loss.append(loss)
            # 如果参数未变化则提前结束训练
            if self.w.all() == last_w.all() and self.b == last_b:
                break

    # 预测函数
    def predict(self, X):
        return np.sign(np.dot(X, self.w) + self.b)


if __name__ == '__main__':
    # 生成模拟二分类数据集
    X, y = make_blobs(n_samples=150, n_features=2, centers=2, cluster_std=1.2, random_state=40)
    y[y == 0] = -1

    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=43)
    # 创建线性可分支持向量机模型
    svm = LinearSVM(0.01, 1, 1000)
    # 对训练集进行拟合
    svm.train(X_train, y_train)
    y_predict = svm.predict(X_test)
    print(svm.loss)
    # 计算测试集上的分类准确率
    print("Accuracy of linear svm: ", accuracy_score(y_test, y_predict))

    # 绘制损失函数图像
    plt.plot(range(len(svm.loss)), svm.loss)
    plt.xlabel("epochs")
    plt.ylabel("loss")
    plt.show()

    # 设置颜色参数
    colors = {-1: 'r', 1: 'g'}
    # 绘制二分类测试集的预测结果的散点图
    plt.scatter(X_test[:, 0], X_test[:, 1], marker='o', c=pd.Series(y_predict).map(colors))

    # 获取测试集最大值和最小值坐标
    x_min, x_max = X_test[:, 0].min() - 0.5, X_test[:, 0].max() + 0.5
    y_min, y_max = X_test[:, 1].min() - 0.5, X_test[:, 1].max() + 0.5
    # 生成边界点坐标
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                         np.arange(y_min, y_max, 0.02))
    # 预测类别
    Z = svm.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    # 绘制边界线
    plt.contour(xx, yy, Z, colors='blue', levels=[-1, 0, 1], alpha=0.5,
                linestyles=['--', '-', '--'])
    plt.show()

    # 绘制二分类测试集的真实结果的散点图
    plt.scatter(X_test[:, 0], X_test[:, 1], marker='o', c=pd.Series(y_test).map(colors))
    plt.show()


