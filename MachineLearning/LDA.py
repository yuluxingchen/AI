# 线性判别分析算法
import numpy as np
from sklearn import datasets
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


class LDA:
    def __init__(self):
        # 初始化权重矩阵
        self.w = None

    # 协方差矩阵计算方法
    def calc_cov(self, X, Y=None):
        m = X.shape[0]
        # 数据标准化
        X = (X - np.mean(X, axis=0)) / np.std(X, axis=0)
        # 如果 Y 存在则计算 Y
        Y = X if Y is None else (Y - np.mean(Y, axis=0)) / np.std(Y, axis=0)
        return 1 / m * np.matmul(X.T, Y)

    # 数据投影方法
    def project(self, X, y):
        # LDA拟合获取模型权重
        self.fit(X, y)
        X_projection = X.dot(self.w)
        return X_projection

    # fit拟合方法
    def fit(self, X, y):
        # 按类分组
        X0 = X[y == 0]
        X1 = X[y == 1]
        # 分别计算两类数据自变量的协方差矩阵
        sigma0 = self.calc_cov(X0)
        sigma1 = self.calc_cov(X1)
        # 计算类内散度矩阵
        Sw = sigma0 + sigma1
        # 计算两类数据自变量的均值和差
        u0, u1 = np.mean(X0, axis=0), np.mean(X1, axis=0)
        mean_diff = np.atleast_1d(u0 - u1)
        # 对类内散度矩阵进行奇异值分解
        U, S, V = np.linalg.svd(Sw)
        # 计算类内散度矩阵的逆
        Sw_ = np.dot(np.dot(V.T, np.linalg.pinv(np.diag(S))), U.T)
        # 计算 w
        self.w = Sw_.dot(mean_diff)

    def predict(self, X):
        # 初始化预测结果为空列表
        y_pred = []
        # 遍历待预测样本
        for x_i in X:
            # 模型预测
            h = x_i.dot(self.w)
            y = 1 * (h < 0)
            y_pred.append(y)
        return y_pred


if __name__ == '__main__':
    # 导入 iris 数据集
    data = datasets.load_iris()
    # 数据与标签
    X, y = data.data, data.target
    # 取标签不为 2 的数据
    X = X[y != 2]
    y = y[y != 2]
    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=41)
    # 创建LDA模型实例
    lda = LDA()
    # LDA模型拟合
    lda.fit(X_train, y_train)
    # LDA模型预测
    y_pred = lda.predict(X_test)
    # 测试集上的分类准确率
    acc = accuracy_score(y_test, y_pred)
    print("Accuracy of Numpy LDA:", acc)
