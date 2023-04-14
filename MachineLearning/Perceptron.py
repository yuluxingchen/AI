import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris


def initialize_params(dims):
    """
    初始化
    :param dims: 变量的维度
    :return:
    w: 初始化权重参数 <br>
    b: 初始化偏置参数
    """
    w = np.zeros(dims)
    b = 0
    return w, b


def train(X, y, learning_rate=0.01):
    w, b = initialize_params(dims=X.shape[1])
    # 记录训练损失的空列表
    loss_his = []
    params = []
    wrong = 1
    while wrong != 0:
        wrong = 0
        for j in range(X.shape[0]):
            Xj = X[j]
            yj = y[j]
            if (np.dot(w.T, Xj) + b) * yj <= 0:
                w += learning_rate * np.dot(Xj, yj)
                b += learning_rate * yj
                wrong += 1
        loss_his.append(wrong)
        # 将当前迭代步优化后的参数保存到字典中
        params = {
            'w': w,
            'b': b
        }
    return loss_his, params


if __name__ == '__main__':
    iris = load_iris()
    # 转换为 pandas 数据框
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    # 取标签
    df['label'] = iris.target
    # 重命名
    df.columns = ['sepal length', 'sepal width', 'petal length', 'petal width', 'label']
    # 取前 100 行数据
    data = np.array(df.iloc[:100, [0, 1, -1]])
    # 定义训练输入和输出
    X, y = data[:, :-1], data[:, -1]
    y = np.array([1 if i == 1 else -1 for i in y])
    print(X.shape)
    print(y.shape)

    loss, params = train(X, y, 0.01)
    print(params)

    x_points = np.linspace(4, 7, 10)
    # 线性分割超平面
    y_hat = - (params['w'][0] * x_points + params['b']) / params['w'][1]
    plt.plot(x_points, y_hat)
    plt.scatter(data[:50, 0], data[:50, 1], color='red', label='0')
    plt.scatter(data[50:100, 0], data[50:100, 1], color='green', label='1')
    plt.xlabel('sepal length')
    plt.ylabel('sepal width')
    plt.legend()
    plt.show()