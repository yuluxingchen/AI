# 3代迭代二叉树
from math import log

import numpy as np
from matplotlib import pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split


class ID3Tree:
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

    def __init__(self):
        self.root = None

    def choose_best_feature(self, X, y):
        n, m = X.shape

        # 计算标签的信息熵
        base_entropy = self.entropy(y)
        # 初始化最大信息增益值、最优特征和划分后的数据集
        max_info_gain, best_feature = -999, -1

        for i in range(m):
            # 获取此特征下的所有取值
            feature_values = set(X[:, i])
            new_entropy = 0.0
            # 对每一个取值依次进行条件信息熵计算
            for value in feature_values:
                # 获取取值的行坐标
                index = np.where(X[:, i] == value)[0]
                sub_y = y[index]
                # 计算取值占比
                prob = float(len(index)) / n
                # 获得信息熵
                new_entropy += prob * self.entropy(sub_y)
            # 计算信息增益
            info_gain = base_entropy - new_entropy
            # 保留信息增益最大的特征
            if info_gain > max_info_gain:
                max_info_gain = info_gain
                best_feature = i
        return best_feature

    def construct(self, X, y):
        """
        生成特征树结构
        :param X: 样本集
        :param y: 样本集的标签
        :return:
        """
        # 如果样本中所有类别相同，则直接返回
        if len(set(y)) == 1:
            return self.TreeNode(result=y[0])
        # 如果没有特征可用，则选择出现最多的结果作为分类结果
        if len(X[0]) == 0:
            return self.TreeNode(result=max(y, key=y.count))
        # 选择信息增益最大的特征作为分割点
        best_feature = self.choose_best_feature(X, y)
        # 以此特征作为特征根建立特征树
        root = self.TreeNode(feature=best_feature)
        # 此特征下的特征值集
        values = set(X[:, best_feature])
        for val in values:
            # 获取满足某一个特征值的样本集行坐标
            idx = np.where(X[:, best_feature] == val)[0]
            # 对这些样本进行进一步划分
            root.son[val] = self.construct(X[idx], y[idx])
        return root

    def entropy(self, y):
        """
        信息熵计算函数
        :param target: 标签列表
        :param labels: 标签类别列表
        :return: 信息熵值
        """
        # 特征集
        labels = np.unique(y)
        # 计算各个标签的概率分布
        probs = [np.sum(y == i) / len(y) for i in labels]
        # 计算信息熵
        entropy_probs = - sum([prob * log(prob, 2) for prob in probs])
        return entropy_probs

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
        if root.result is not None:
            return root.result
        # 获取此特征节点的特征编号
        feature = X[root.feature]
        # 如果此特征不属于子特征节点则返回未知
        if feature not in root.son:
            return -1
        # 从子特征节点中继续寻找
        son = root.son[feature]
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
    # 创建ID3树
    id3 = ID3Tree()
    # 拟合数据
    id3.fit(X_train, y_train)
    # 预测数据
    y_pred = id3.predict(X_test)
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
