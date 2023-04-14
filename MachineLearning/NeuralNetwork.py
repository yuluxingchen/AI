import numpy as np

from Sigmoid import sigmoid


def layer_sizes(X, Y):
    """
    定义网络结构
    :param X: 训练输入
    :param Y: 训练输出
    :return:
    n_x: 输入层大小 <br>
    n_h: 隐藏层大小 <br>
    n_y: 输出层大小 <br>
    """
    # 输入层大小
    n_x = X.shape[0]
    n_h = 4
    n_y = Y.shape[0]
    return n_x, n_h, n_y


def initialize_parameters(n_x, n_h, n_y):
    """
    模型参数初始化
    :param n_x: 输入层神经元个数
    :param n_h: 隐藏层神经元个数
    :param n_y: 输出层神经元个数
    :return: 初始化后的模型参数
    """
    # 权重系数随机初始化
    W1 = np.random.randn(n_h, n_x) * 0.01
    # 偏置参数以 0 为初始化值
    b1 = np.zeros((n_h, 1))
    W2 = np.random.randn(n_y, n_h) * 0.01
    b2 = np.zeros((n_y, 1))
    # 封装为字典
    parameters = {
        "W1": W1,
        "b1": b1,
        "W2": W2,
        "b2": b2
    }
    return parameters


def forward_propagation(X, parameters):
    """
    前向传播过程
    :param X: 训练输入
    :param parameters: 初始化的模型参数
    :return:
    A2: 模型输出
    caches: 前向传播过程计算的中间值缓存
    """
    # 获取各参数的初始值
    W1 = parameters['W1']
    b1 = parameters['b1']
    W2 = parameters['W2']
    b2 = parameters['b2']
    # 执行前向计算
    Z1 = np.dot(W1, X) + b1
    A1 = np.tanh(Z1)
    Z2 = np.dot(W2, A1) + b2
    A2 = sigmoid(Z2)
    cache = {
        "Z1": Z1,
        "A1": A1,
        "Z2": Z2,
        "A2": A2
    }
    return A2, cache


def compute_cost(A2, Y):
    """
    定义损失函数
    :param A2: 前向计算输出
    :param Y: 训练标签
    :return: cost: 当前损失
    """
    # 训练样本量
    m = Y.shape[1]
    # 计算交叉熵损失
    logprobs = np.multiply(np.log(A2), Y) + np.multiply(np.log(1 - A2), 1 - Y)
    cost = -1 / m * np.sum(logprobs)
    # 维度压缩
    cost = np.squeeze(cost)
    return cost


def backward_propagation(parameters, cache, X, Y):
    """
    定义反向传播过程
    :param parameters: 神经网络参数字典
    :param cache: 神经网络前向计算中间缓存字典
    :param X: 训练输入
    :param Y: 训练输出
    :return: grads: 权重梯度字典
    """

    # 样本量
    m = X.shape[1]
    W1 = parameters['W1']
    W2 = parameters['W2']
    A1 = cache['A1']
    A2 = cache['A2']

    # 执行反向传播
    dZ2 = A2 - Y
    dW2 = 1 / m * np.dot(dZ2, A1.T)
    db2 = 1 / m * np.sum(dZ2, axis=1, keepdims=True)
    dZ1 = np.dot(W2.T, dZ2) * (1 - np.power(A1, 2))
    dW1 = 1 / m * np.dot(dZ1, X.T)
    db1 = 1 / m * np.sum(dZ1, axis=1, keepdims=True)

    # 将权重梯度封装为字典
    grads = {
        "dW1": dW1,
        "db1": db1,
        "dW2": dW2,
        "db2": db2
    }
    return grads


def update_parameters(parameters, grads, learning_rate=1.2):
    """
    定义权重更新过程
    :param parameters: 神经网络参数字典
    :param grads: 权重梯度字典
    :param learning_rate: 学习率
    :return: parameters: 更新后的权重字典
    """
    # 获取参数
    W1 = parameters['W1']
    b1 = parameters['b1']
    W2 = parameters['W2']
    b2 = parameters['b2']
    # 获取梯度
    dW1 = grads['dW1']
    db1 = grads['db1']
    dW2 = grads['dW2']
    db2 = grads['db2']
    # 参数更新
    W1 -= dW1 * learning_rate
    b1 -= db1 * learning_rate
    W2 -= dW2 * learning_rate
    b2 -= db2 * learning_rate
    # 将更新后的权重封装为字典
    parameters = {
        "W1": W1,
        "b1": b1,
        "W2": W2,
        "b2": b2
    }
    return parameters


def nn_model(X, Y, n_h, num_iterations=10000, print_cost=False):
    """
    神经网络模型封装
    :param X: 训练输入
    :param Y: 训练输出
    :param n_h: 隐藏层结点数
    :param num_iterations: 迭代次数
    :param print_cost: 训练过程中是否打印损失
    :return: parameters: 神经网络训练优化后的权重系数
    """
    # 设置随机数种子
    np.random.seed(3)
    # 输入和输出节点数
    n_x = layer_sizes(X, Y)[0]
    n_y = layer_sizes(X, Y)[2]
    # 初始化模型参数
    parameters = initialize_parameters(n_x, n_h, n_y)
    for i in range(0, num_iterations):
        A2, cache = forward_propagation(X, parameters)
        # 计算当前损失
        cost = compute_cost(A2, Y)
        # 反向传播
        grads = backward_propagation(parameters, cache, X, Y)
        # 参数更新
        parameters = update_parameters(parameters, grads, learning_rate=1.2)
        # 打印损失
        if print_cost and i % 1000 == 0:
            print("Cost after iteration %i: %f" % (i, cost))
    return parameters


def create_dataset():
    """
    生成非线性可分数据集
    :return:
    X: 模拟数据集输入 <br>
    Y: 模拟数据集输出
    """
    # 设置随机数种子
    np.random.seed(1)
    # 数据量
    m = 400
    # 每个标签的实例数
    N = int(m / 2)
    # 数据维度
    D = 2
    # 数据矩阵
    X = np.zeros((m, D))
    # 标签维度
    Y = np.zeros((m, 1), dtype='uint8')
    a = 4
    # 遍历生成数据

    for j in range(2):
        ix = range(N * j, N * (j + 1))
        t = np.linspace(j * 3.12, (j + 1) * 3.12, N) + np.random.randn(N) * 0.2
        r = a * np.sin(4 * t) + np.random.randn(N) * 0.2
        X[ix] = np.c_[r * np.sin(t), r * np.cos(t)]
        Y[ix] = j
    X = X.T
    Y = Y.T
    return X, Y


if __name__ == '__main__':
    X, Y = create_dataset()
    parameters = nn_model(X, Y, n_h=4, num_iterations=10000, print_cost=True)
    print(parameters)
