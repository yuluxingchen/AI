# 分类回归树
import numpy as np
from matplotlib import pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split


class TreeNode:
    def __init__(self, feature=None, value=None, result=None):
        # 特征编号
        self.feature = feature
        # 特征取值
        self.value = value
        # 如果是叶子节点，代表分类结果
        self.result = result
        # 子节点
        self.son = {}


class CARTTree:
    def __init__(self):
        self.root = None

    # 计算数据集的基尼指数
    def gini_index(self, y):
        count = len(y)
        if count == 0:
            return 1.0
        gini = 1.0
        for label in set(y):
            # 特征概率
            prob = float(y.count(label)) / count
            gini -= prob ** 2
        return gini

    def empirical_gini_index(self, X, y, feature, value):
        m, n = X.shape
        count = len(y)
        left_y = []
        right_y = []
        # 根据划分值将样本标签分为左右两类
        for i in range(m):
            if X[i][feature] <= value:
                left_y.append(y[i])
            else:
                right_y.append(y[i])
        new_gini = float(len(left_y)) / count * self.gini_index(left_y) + float(len(right_y)) / count * self.gini_index(right_y)
        return new_gini

    def choose_best_feature(self, X, y):
        n, m = X.shape

        # 初始化最大基尼指数、最优特征和最优特征下的划分值
        best_gini, best_feature, best_value = 1.0, -1, None
        for i in range(m):
            # 获取此特征下的所有取值
            feature_values = set(X[:, i])
            # 对每一个取值依次进行条件信息熵计算
            for value in feature_values:
                gini = self.empirical_gini_index(X, y, i, value)
                # 存储可获得最大基尼指数的特征
                if gini < best_gini:
                    best_gini = gini
                    best_feature = i
                    best_value = value
        return best_feature, best_value, best_gini

    def construct(self, X, y):
        """
        生成特征树结构
        :param X: 样本集
        :param y: 样本集的标签
        :return:
        """
        # 如果样本中所有类别相同，则直接返回
        if len(set(y)) == 1:
            return TreeNode(result=y[0])
        # 如果没有特征可用，则选择出现最多的结果作为分类结果
        if len(X[0]) == 0:
            return TreeNode(result=max(y, key=y.count))
        # 选择基尼指数最大的特征作为分割点
        feature, value, gini = self.choose_best_feature(X, y)
        # 以此特征作为特征根建立特征树
        root = TreeNode(feature=feature, value=value)
        # 建立左子树
        index = np.where(X[:, feature] <= value)[0]
        root.son["left"] = self.construct(X[index], y[index])
        # 建立右子树
        index = np.where(X[:, feature] > value)[0]
        root.son["right"] = self.construct(X[index], y[index])
        return root

    def fit(self, X, y):
        self.root = self.construct(X, y)

    def predict_one(self, root, X):
        """
        预测单个样本类别
        :param root: 特征根节点
        :param X: 待预测样本
        :return: 样本类别
        """
        # 如果此特征节点存在结果，则直接返回
        if root.result != None:
            return root.result
        # 如果特征值小于划分值从左子树中找，否则从右子树中找
        if X[root.feature] <= root.value:
            son = root.son["left"]
        else:
            son = root.son["right"]
        return self.predict_one(son, X)

    def predict(self, X):
        """
        预测样本集中所有样本类别
        :param X: 待预测样本集
        :return: 样本类别
        """
        # 返回结果
        result = []
        # 对样本集中的每一个样本单独进行预测
        for i in range(len(X)):
            result.append(self.predict_one(self.root, X[i]))
        return np.array(result)


if __name__ == '__main__':
    data = load_iris()
    # 将数据集分为训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.3, random_state=42)
    # 创建CART树
    cart = CARTTree()
    # 拟合数据
    cart.fit(X_train, y_train)
    # 预测数据
    y_pred = cart.predict(X_test)
    # 精确率
    accuracy = sum([y_pred[i] == y_test[i] for i in range(len(y_test))]) / len(y_test)
    print(accuracy)

    # 不同类别颜色设置
    colors = ['navy', 'turquoise', 'darkorange']
    # 绘制预测分类
    for c, i, target_name in zip(colors, [0, 1, 2], data.target_names):
        plt.scatter(X_test[y_pred == i, 0], X_test[y_pred == i, 1], color=c, lw=2, label=target_name)
    plt.legend()
    plt.show()

    # 绘制测试集分类
    for c, i, target_name in zip(colors, [0, 1, 2], data.target_names):
        plt.scatter(X_test[y_test == i, 0], X_test[y_test == i, 1], color=c, lw=2, label=target_name)
    plt.legend()
    plt.show()
