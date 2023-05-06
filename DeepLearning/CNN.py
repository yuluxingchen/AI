import numpy as np
from keras.datasets import mnist


# TODO: 程序需要进一步修改。现在存在训练速度慢、模型精度差等问题

class Convolution:
    def __init__(self, kernel_size=(3, 3), kernel_num=5):
        # 卷积核大小
        self.kernel_size = kernel_size
        # 卷积核数量
        self.kernel_num = kernel_num
        # 卷结核矩阵
        self.kernels = np.random.normal(size=(kernel_num, *kernel_size, 1))
        # 偏置矩阵
        self.bias = np.zeros((kernel_num,))
        # 保存输入结果
        self.last_input = None

    def forward(self, input):
        # 保存输入以便于进行反向传播计算
        self.last_input = input
        # 获取输入的批量大小、高度、宽度、通道数
        batch_size, height, width, channels = input.shape
        # 初始化输出为0
        output_height = height - self.kernel_size[0] + 1
        output_width = width - self.kernel_size[0] + 1
        output = np.zeros((batch_size, output_height, output_width, self.kernel_num))
        # 卷积核垂直移动
        for h in range(output_height):
            # 卷积核水平移动
            for w in range(output_width):
                # 卷积核层级移动
                for k in range(self.kernel_num):
                    # 对相应部分进行卷积计算
                    output[:, h, w, k] = np.sum(
                        input[:, h: h + self.kernel_size[0], w: w + self.kernel_size[1], :] * self.kernels[k],
                        axis=(1, 2, 3)) + self.bias[k]
        return output

    def backward(self, output_grad, learning_rate):
        # 获得上一个输入的批量大小、高度、宽度、通道数
        batch_size, height, width, channels = self.last_input.shape
        # 临时存储反向传播计算出的卷积核
        kernels_grad = np.zeros_like(self.kernels)
        # 临时存储反向传播计算出的偏置
        bias_grad = np.sum(output_grad, axis=(0, 1, 2))
        # 临时存储反向传播计算出的损失函数对此输入的导数，即 dl/dy * dy/dx，用作下一层反向传播的输入
        input_grad = np.zeros_like(self.last_input)
        output_height = height - self.kernel_size[0] + 1
        output_width = width - self.kernel_size[0] + 1

        for h in range(output_height):
            for w in range(output_width):
                for f in range(self.kernel_num):
                    # 提取输入每一次被卷积的部分
                    im_region = self.last_input[:, h:h + self.kernel_size[0], w:w + self.kernel_size[1], :]
                    # y = wx, dy/dw = x, dl/dy * dy/dw = dl/dy * x, 因此将输出的梯度乘以输入得到卷积核梯度
                    kernels_grad[f] += np.sum(
                        im_region * (output_grad[:, h, w, f])[:, np.newaxis, np.newaxis, np.newaxis])
                    # y = wx, dy/dx = w, dl/dy * dy/dx = dl/dy * w, 因此将输出的梯度乘以卷积核得到输入梯度
                    input_grad[:, h:h + self.kernel_size[0], w:w + self.kernel_size[1], :] += self.kernels[f] * (output_grad[:, h, w, f])[:, np.newaxis, np.newaxis, np.newaxis]

        # 根据学习率进行卷积核更新
        self.kernels -= learning_rate * kernels_grad
        self.bias -= learning_rate * bias_grad
        return input_grad

class ReLU:
    def __init__(self):
        self.last_input = None

    def forward(self, input):
        # 小于等于零的部分都会被置为0
        self.last_input = input
        return np.maximum(0, input)

    def backward(self, output_grad, learning_rate):
        input_grad = output_grad
        # 输入小于0的部分梯度将被置为0
        input_grad[self.last_input < 0] = 0
        return input_grad


class Flattening:
    def __init__(self):
        self.last_input_shape = None

    def forward(self, input):
        # 保存输入的形状便于反向传播时进行转换
        self.last_input_shape = input.shape
        # 将四维张量转换为二维向量
        return input.reshape(input.shape[0], -1)

    def backward(self, output_grad, learning_rate):
        # 将二维向量转换回四维张量
        return output_grad.reshape(self.last_input_shape)


class FullyConnected:
    def __init__(self, units):
        # 初始化 w, b
        self.w = None
        self.b = np.zeros(units)
        self.units = units
        self.last_input = None

    def forward(self, input):
        self.w = np.random.normal(size=(input.shape[-1], self.units)) / input.shape[-1]
        self.last_input = input
        # 计算全连接层输出
        return np.dot(input, self.w) + self.b

    def backward(self, output_grad, learning_rate):
        # 计算输入的梯度
        input_grad = np.dot(output_grad, self.w.T)
        # 更新 w, b
        self.w -= learning_rate * np.dot(self.last_input.T, output_grad)
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
    def train(self, X_train, y_train, epochs, batch_size, learning_rate):
        num_batches = len(X_train) // batch_size
        for epoch in range(epochs):
            for batch_idx in range(num_batches):
                batch_start = batch_idx * batch_size
                batch_end = (batch_idx + 1) * batch_size
                X_batch = X_train[batch_start:batch_end]
                y_batch = y_train[batch_start:batch_end]
                logits = self.forward(X_batch)
                loss, dout = self.softmax_cross_entropy_with_logits(logits, y_batch)
                acc = self.accuracy(logits, y_batch)
                self.backward(dout, learning_rate)
                if batch_idx % 100 == 0:
                    print("Epoch {}, batch {}: loss {:.4f}, accuracy {:.2f}%".format(epoch + 1, batch_idx + 1, loss, acc * 100))


    # 预测标签
    def predict(self, input):
        output = self.forward(input)
        output = np.argmax(output, axis=1)
        return output

    def evaluate(self, X, y):
        logits = self.forward(X)
        loss, _ = self.softmax_cross_entropy_with_logits(logits, y)
        acc = self.accuracy(logits, y)
        return loss, acc

    def accuracy(self, logits, y):
        preds = np.argmax(logits, axis=-1)
        return np.mean(preds == y)

    def softmax(self, x):
        exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

    def softmax_cross_entropy_with_logits(self, logits, labels):
        batch_size = logits.shape[0]
        softmax_output = self.softmax(logits)
        loss = -np.sum(np.log(softmax_output[np.arange(batch_size), labels])) / batch_size
        dout = softmax_output.copy()
        dout[np.arange(batch_size), labels] -= 1
        dout /= batch_size
        return loss, dout


if __name__ == '__main__':
    # 加载 MNIST 数据集
    (X_train, Y_train), (X_test, Y_test) = mnist.load_data()
    X_train = X_train.reshape(-1, 28, 28, 1)
    X_test = X_test.reshape(-1, 28, 28, 1)
    X_train = X_train.astype('float32') / 255
    X_test = X_test.astype('float32') / 255

    X_train = X_train[:1000]
    Y_train = Y_train[:1000]
    X_test = X_test[0:100]
    Y_test = Y_test[0:100]
    print(X_train.shape, X_test.shape)

    # 创建神经网络模型
    model = CNN()
    model.add_layer(Convolution(kernel_size=(5, 5), kernel_num=5))
    model.add_layer(ReLU())
    model.add_layer(Convolution(kernel_size=(3, 3), kernel_num=10))
    model.add_layer(ReLU())
    model.add_layer(Flattening())
    model.add_layer(FullyConnected(64))
    model.add_layer(ReLU())
    model.add_layer(FullyConnected(10))
    model.add_layer(Softmax())

    # 在训练集上训练模型
    model.train(X_train, Y_train, epochs=30, batch_size=32, learning_rate=0.001)

    # 在测试集上评估模型性能
    y_pred = model.predict(X_test)
    accuracy = np.mean(y_pred == Y_test)
    print("Test accuracy: %.4f" % accuracy)
