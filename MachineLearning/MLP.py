import numpy as np


def sigmoid(x):
    """
    定义sigmoid函数
    :param x: 输入数据
    :return: 计算后的数据
    """
    return 1.0 / (1 + np.exp(-x))


def initialize_parameters(input_dim, output_dim):
    np.random.seed(3)
    w = np.random.normal(size=(output_dim, input_dim))
    b = np.random.normal(size=(output_dim, 1))
    return w, b


def cost(A, Y):
    m = Y.shape[0]
    log_probs = np.multiply(np.log(A), Y) + np.multiply(np.log(1 - A), (1 - Y))
    cost = - 1 / m * np.sum(log_probs)
    return cost


class MLP:
    def __init__(self, input_dim, hidden_dim, output_dim, layer=2):
        self.layer_parameters = []
        self.cache = {}
        for i in range(layer):
            if i == 0:
                w, b = initialize_parameters(input_dim, hidden_dim)
            elif 0 < i < layer - 1:
                w, b = initialize_parameters(hidden_dim, hidden_dim)
            else:
                w, b = initialize_parameters(hidden_dim, output_dim)
            self.layer_parameters.append([w, b])

    def forward_propagation(self, X):
        for index in range(len(self.layer_parameters)):
            w, b = self.layer_parameters[index]
            if index != len(self.layer_parameters) - 1:
                A = np.tanh(np.dot(X, w.T) + b.T)
            else:
                A = sigmoid(np.dot(X, w.T) + b.T)
            self.cache[index] = [X, A]
            X = A
        return X

    def backward_propagation(self, X, Y):
        last_dw = 1
        grads = [[] for _ in range(len(self.cache))]
        for i in range(len(self.cache) - 1, 0, -1):
            m = X.shape[0]
            x, a = self.cache[i]
            w, b = self.layer_parameters[i]
            if i == len(self.cache) - 1:
                dw = 1/m * np.dot(x.T, (a - Y))
                db = 1/m * np.sum((a - Y), axis=0, keepdims=True)
                last_dw = 1/m * (a - Y) * w
            else:
                dw = np.dot(x.T, last_dw * a * (1 - a))
                db = np.sum(last_dw * a * (1 - a), axis=0, keepdims=True)
                last_dw = last_dw * w * a * (1 - a)
            grads[i] = [dw, db]
        return grads

    def update_parameters(self, grads, learning_rate=0.5):
        for index in range(len(self.layer_parameters) - 1, 0, -1):
            w, b = self.layer_parameters[index]
            dw, db = grads[index]
            w -= learning_rate * dw.T
            b -= learning_rate * db.T
            self.layer_parameters[index] = [w, b]

    def train(self, X, Y, learning_rate=0.5, epochs=100):
        for i in range(epochs):
            A = self.forward_propagation(X)
            loss = cost(A, Y)
            grads = self.backward_propagation(X, Y)
            self.update_parameters(grads, learning_rate)
            if (i + 1) % 100 == 0:
                print("Cost after iteration %i:%f" % ((i + 1), loss))


# 生成非线性可分数据集
def create_dataset():
    """
    :return:
    X：模拟数据集输入 <br>
    Y：模拟数据集输出
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
    return X, Y


if __name__ == '__main__':
    X, Y = create_dataset()
    mlp = MLP(X.shape[1], 4, Y.shape[1], layer=2)
    mlp.train(X, Y, learning_rate=1.2, epochs=1000)
