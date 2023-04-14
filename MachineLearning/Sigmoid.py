import numpy as np
from matplotlib import pyplot as plt
from sklearn.datasets import load_diabetes, make_classification
from sklearn.metrics import classification_report
from sklearn.utils import shuffle


def sigmoid(x):
    return 1.0 / (1 + np.exp(-x))


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


def logisticLoss(X, y, w, b):
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
    y_hat = sigmoid(np.dot(X, w) + b)
    # 计算交叉熵损失函数
    loss = -1 / num_train * np.sum(y * np.log(y_hat) + (1 - y) * np.log(1 - y_hat))
    # 基于均方损失对权重系数的一阶导数
    dw = np.dot(X.T, (y_hat - y)) / num_train
    # 基于均方损失对偏置的一阶导数
    db = np.sum(y_hat - y) / num_train
    return y_hat, loss, dw, db


def logisticTrain(X, y, learning_rate=0.01, epochs=10000):
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
        y_hat, loss, dw, db = logisticLoss(X, y, w, b)
        w = w - learning_rate * dw
        b = b - learning_rate * db
        loss_his.append(loss)
        if i % 100 == 0:
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
    y_pred = sigmoid(np.dot(X, w) + b)
    for i in range(len(y_pred)):
        if y_pred[i] > 0.5:
            y_pred[i] = 1
        else:
            y_pred[i] = 0
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


def plot_decision_boundary(X_train, y_train, params):
    """
    绘制对数几率回归分类决策边界
    :param X_train: 训练集输入
    :param y_train: 训练集标签
    :param params: 训练好的模型参数
    :return: 分类决策边界图
    """
    # 训练样本量
    n = X_train.shape[0]
    # 初始化类别坐标点列表
    xcord1 = []
    ycord1 = []
    xcord2 = []
    ycord2 = []
    # 获取两类坐标点并并入列表
    for i in range(n):
        if y_train[i] == 1:
            xcord1.append(X_train[i][0])
            ycord1.append(X_train[i][1])
        else:
            xcord2.append(X_train[i][0])
            ycord2.append(X_train[i][1])
    # 创建绘画
    fig = plt.figure()
    ax = fig.add_subplot(111)
    # 绘制两类散点，以不同颜色表示
    ax.scatter(xcord1, ycord1, s=32, c='red')
    ax.scatter(xcord2, ycord2, s=32, c='green')
    # 取值范围
    x = np.arange(-1.5, 3, 0.1)
    # 分类决策边界公式
    b = params['b']
    w = params['w']
    y = (-b - w[0] * x) / w[1]
    ax.plot(x, y)
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.show()


if __name__ == '__main__':
    X, labels = make_classification(
        n_samples=100,
        n_features=2,
        n_redundant=0,
        n_informative=2,
        random_state=1,
        n_clusters_per_class=2
    )
    # 设置随机数种子
    rng = np.random.RandomState(2)
    # 对生成的特征数据添加一组均匀分布噪声
    X += 2 * rng.uniform(size=X.shape)
    # 标签类别数
    unique_labels = set(labels)
    # 根据标签类别数设置颜色
    colors = plt.cm.Spectral(np.linspace(0, 1, len(unique_labels)))
    for k, col in zip(unique_labels, colors):
        x_k = X[labels == k]
        plt.plot(x_k[:, 0], x_k[:, 1], 'o',
                 markerfacecolor=col,
                 markeredgecolor='k',
                 markersize=14)
    plt.title('Simulated binary data set')
    plt.show()

    offset = int(X.shape[0] * 0.9)
    X_train, y_train = X[:offset], labels[:offset]
    X_test, y_test = X[offset:], labels[offset:]
    y_train = y_train.reshape((-1, 1))
    y_test = y_test.reshape((-1, 1))
    print('X_train = ', X_train.shape)
    print('X_test = ', X_test.shape)
    print('y_train = ', y_train.shape)
    print('y_test = ', y_test.shape)

    cost_list, params, grads = logisticTrain(X_train, y_train, 0.01, 1000)
    y_pred = predict(X_test, params)
    print(classification_report(y_test, y_pred))
    plot_decision_boundary(X_train, y_train, params)
