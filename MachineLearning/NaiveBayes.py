import numpy as np
import pandas as pd


def nb_fit(X, y):
    """
    定义朴素贝叶斯模型训练过程
    :param X: 训练样本输入
    :param y: 训练样本标签
    :return:
    classes: 标签类别 <br>
    class_prior: 类先验概率分布 <br>
    class_condition: 类条件概率分布
    """
    # 标签类别
    classes = y[y.columns[0]].unique()
    # 标签类别数量统计
    class_count = y[y.columns[0]].value_counts()
    # 极大似然估计: 类先验概率
    class_prior = class_count / len(y)
    # 类条件概率: 字典初始化
    prior_condition_prod = dict()
    # 遍历计算类条件概率
    # 遍历特征
    for col in X.columns:
        # 遍历类别
        for j in classes:
            # 统计当前类别下特征的不同取值
            p_x_y = X[(y == j).values][col].value_counts()
            # 遍历计算类条件概率
            for i in p_x_y.index:
                prior_condition_prod[(col, i, j)] = p_x_y[i] / class_count[j]
    return classes, class_prior, prior_condition_prod


def nb_predict(X_test, classes, class_prior):
    """
    定义朴素贝叶斯预测函数
    :param X_test: 测试输入
    :return: 类别
    """
    # 初始化结果列表
    res = []
    # 遍历样本类别
    for c in classes:
        # 获取当前类的先验概率
        p_y = class_prior[c]
        # 初始化类条件概率
        p_x_y = 1
        # 遍历字典每个元素
        for i in X_test.items():
            # 似然函数：类条件概率连乘
            p_x_y *= class_prior[tuple(list(i) + [c])]
        # 类先验概率与类条件概率乘积
        res.append(p_y * p_x_y)
    return classes[np.argmax(res)]


if __name__ == '__main__':
    X1 = [1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3]
    X2 = ['S', 'M', 'M', 'S', 'S', 'S', 'M', 'M', 'L', 'L', 'L', 'M', 'M', 'L', 'L']
    y = [-1, -1, 1, 1, -1, -1, -1, 1, 1, 1, 1, 1, 1, 1, -1]
    df = pd.DataFrame({'x1': X1, 'x2': X2, 'y': y})
    # 获取训练输入和输出
    X, y = df[['x1', 'x2']], df[['y']]
    # 朴素贝叶斯模型训练
    classes, class_prior, prior_condition_prob = nb_fit(X, y)
    # print(classes, class_prior, prior_condition_prob)
    X_test = {'x1': 2, 'x2': 'S'}
    # print('测试数据预测类别为：', nb_predict(X_test, classes, class_prior))
