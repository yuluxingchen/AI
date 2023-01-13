# PCA算法类
import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets


class PCA:
    # 定义协方差矩阵计算方法
    def calc_cov(self, X):
        # 样本量
        m = X.shape[0]
        # 数据标准化
        X = (X - np.mean(X, axis=0)) / np.var(X, axis=0)
        return 1 / m * np.matmul(X.T, X)

    def pca(self, X, n_components):
        # 计算协方差矩阵
        cov_martix = self.calc_cov(X)
        # 计算协方差矩阵的特征值和对应特征向量
        eigenvalues, eigenvectors = np.linalg.eig(cov_martix)
        # 对特征值进行排序
        idx = eigenvalues.argsort()[::-1]
        # 取最大的前 n_component组
        eigenvectors = eigenvectors[:, idx]
        eigenvectors = eigenvectors[:, :n_components]
        return np.matmul(X, eigenvectors)


if __name__ == '__main__':
    iris = datasets.load_iris()
    X, y = iris.data, iris.target
    # 将数据降维到三个主成分
    X_trans = PCA().pca(X, 3)
    # 颜色列表
    colors = ['navy', 'turquoise', 'darkorange']
    # 绘制不同的类别
    for c, i, target_name in zip(colors, [0, 1, 2], iris.target_names):
        plt.scatter(X_trans[y == i, 0], X_trans[y == i, 1], color=c, lw=2, label=target_name)
    plt.legend()
    plt.show()
