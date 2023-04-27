import numpy as np
from keras.datasets import mnist


# TODO: 程序需要进一步修改。现在存在训练速度慢、无法批量训练、训练时容易出现梯度爆炸或消失、loss无法收敛等问题

class Convolution:
    def __init__(self, filter_size=3, filters_num=5):
        # 卷积核大小
        self.filter_size = filter_size
        # 卷积核数量
        self.filter_num = filters_num
        # 卷结核矩阵
        self.filters = np.random.normal(size=(self.filter_num, self.filter_size, self.filter_size)) / (filter_size ** 2)
        # 保存输入结果
        self.last_input = None

    def forward(self, input):
        assert input.ndim == 2
        # 获取输入的高度、宽度
        height, width = input.shape
        # 初始化输出为0
        output_height = height - self.filter_size + 1
        output_width = width - self.filter_size + 1
        output = np.zeros((output_height, output_width, self.filter_num))
        # 卷积核层级移动
        for f in range(self.filter_num):
            # 卷积核垂直移动
            for h in range(output_height):
                # 卷积核水平移动
                for w in range(output_width):
                    # 对相应部分进行卷积计算
                    output[h, w, f] = np.sum(
                        input[h: h + self.filter_size, w: w + self.filter_size] * self.filters[f])

        # 保存输入以便于进行反向传播计算
        self.last_input = input
        return output

    def backward(self, output_grad, learning_rate):
        # 获得上一个输入的高度、宽度和通道数
        height, width = self.last_input.shape
        # 临时存储反向传播计算出的卷积核
        filter_grad = np.zeros(self.filters.shape)
        # 临时存储反向传播计算出的损失函数对此输入的导数，即 dl/dy * dy/dx，用作下一层反向传播的输入
        input_grad = np.zeros(self.last_input.shape)
        for f in range(self.filter_num):
            for h in range(height - self.filter_size + 1):
                for w in range(width - self.filter_size + 1):
                    # 提取输入每一次被卷积的部分
                    im_region = self.last_input[h:h + self.filter_size, w:w + self.filter_size]
                    # y = wx, dy/dw = x, dl/dy * dy/dw = dl/dy * x, 因此将输出的梯度乘以输入得到卷积核梯度
                    filter_grad[f] += output_grad[h, w, f] * im_region
                    # y = wx, dy/dx = w, dl/dy * dy/dx = dl/dy * w, 因此将输出的梯度乘以卷积核得到输入梯度
                    input_grad[h:h + self.filter_size, w:w + self.filter_size] += output_grad[h, w, f] * \
                                                                                  self.filters[f]

        # 根据学习率进行卷积核更新
        self.filters -= learning_rate * filter_grad
        return input_grad


class ReLU:
    def __init__(self):
        self.last_input = None

    def forward(self, input):
        # 小于等于零的部分都会被置为0
        self.last_input = input
        return np.maximum(0, input)

    def backward(self, output_grad, learning_rate):
        input_grad = output_grad.copy()
        # 输入小于0的部分梯度将被置为0
        input_grad[self.last_input < 0] = 0
        return input_grad


class Flattening:
    def __init__(self):
        self.last_input_shape = None

    def forward(self, input):
        # 保存输入的形状便于反向传播时进行转换
        self.last_input_shape = input.shape
        height, width, channel = input.shape
        # 将三维张量转换为一维向量
        return input.reshape(height * width * channel)

    def backward(self, output_grad, learning_rate):
        height, width, channel = self.last_input_shape
        # 将一维向量转换回三维张量
        return output_grad.reshape(height, width, channel)


class FullyConnected:
    def __init__(self, input_dim, output_dim):
        # 初始化 w, b
        self.w = np.random.normal(size=(input_dim, output_dim)) / input_dim
        self.b = np.zeros(output_dim)
        self.last_input = None

    def forward(self, input):
        self.last_input = input
        # 计算全连接层输出
        return np.dot(input, self.w) + self.b

    def backward(self, output_grad, learning_rate):
        # 计算输入的梯度
        input_grad = np.dot(output_grad, self.w.T)
        # 更新 w, b
        self.w -= learning_rate * np.dot(self.last_input.reshape(1, -1).T, output_grad.reshape(1, -1))
        self.b -= learning_rate * np.sum(output_grad, axis=0)
        return input_grad


class Softmax:
    def __init__(self):
        self.last_input = None

    def forward(self, input):
        self.last_input = input
        # 减去最大值防止数据溢出
        exp_scores = np.exp(input - np.max(input))
        # 计算 softmax 计算后的概率
        probs = exp_scores / np.sum(exp_scores)
        return probs

    def backward(self, output_grad, learning_rate):
        out = self.forward(self.last_input)
        # 计算输入的梯度
        input_grad = out - output_grad
        return input_grad


# 组合模型
class CNN:
    def __init__(self):
        self.layers = []

    # 添加层
    def add_layer(self, layer):
        self.layers.append(layer)

    # 前向传播计算
    def forward(self, input):
        output = input
        for layer in self.layers:
            output = layer.forward(output)
        return output

    # 反向传播计算
    def backward(self, output_grad, learning_rate):
        input_grad = output_grad
        for layer in reversed(self.layers):
            input_grad = layer.backward(input_grad, learning_rate)

    # 训练模型
    def train(self, X_train, Y_train, epochs, learning_rate):
        for epoch in range(epochs):
            loss = 0
            for i in range(len(X_train)):
                # 前向传播
                y_pred = self.forward(X_train[i])
                # 计算交叉熵损失
                if y_pred[Y_train[i]] != 0:
                    loss += -np.log(y_pred[Y_train[i]])
                # 反向传播
                output_grad = np.zeros(len(set(Y_train)))
                output_grad[Y_train[i]] = 1
                # 将交叉熵损失函数的导数传入反向传播
                self.backward(y_pred - output_grad, learning_rate)
            print("Epoch %d loss: %.4f" % (epoch + 1, loss / len(X_train)))

    # 预测标签
    def predict(self, input):
        output = np.zeros((input.shape[0], 10))
        for i in range(len(input)):
            output[i] = self.forward(input[i])
        output = np.argmax(output, axis=1)
        return output


if __name__ == '__main__':
    # 加载 MNIST 数据集
    (X_train, Y_train), (X_test, Y_test) = mnist.load_data()
    X_train = X_train.astype('float32') / 255
    X_test = X_test.astype('float32') / 255

    X_train = X_train[:50]
    Y_train = Y_train[:50]
    X_test = X_test[:15]
    Y_test = Y_test[:15]
    print(X_train.shape, X_test.shape)

    out_dim = len(set(Y_train))
    # 创建神经网络模型
    model = CNN()
    model.add_layer(Convolution(5, 5))
    model.add_layer(ReLU())
    model.add_layer(Flattening())
    model.add_layer(FullyConnected(24 * 24 * 5, 64))
    model.add_layer(ReLU())
    model.add_layer(FullyConnected(64, out_dim))
    model.add_layer(Softmax())

    # 在训练集上训练模型
    model.train(X_train, Y_train, epochs=1, learning_rate=0.001)

    # 在测试集上评估模型性能
    y_pred = model.predict(X_test)
    # print(y_pred)
    accuracy = np.mean(y_pred == Y_test)
    print("Test accuracy: %.4f" % accuracy)
