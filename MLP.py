import torch
import torchvision.datasets
from torch import nn
from torch.utils import data
from torchvision import transforms


def get_dataloader_workers():
    """
    获取读取数据的进程数
    :return: 读取数据的进程数
    """
    return 4


def load_data(batch_size, resize=None):
    """
    下载 Fashion-MNIST数据集， 然后将其加载到内存中
    :param batch_size: 批处理大小
    :param resize: 调整后图像大小
    :return: 加载后的数据集
    """
    # 声明一个操作列表。列表中包含 ToTensor 步骤
    # ToTensor 实例可将图像数据从 PIL 类型变换成 32 位浮点数
    trans = [transforms.ToTensor()]
    if resize:
        # Resize 用于调整图像大小
        # 将 Resize 步骤放在操作列表的第一位
        trans.insert(0, transforms.Resize(resize))
    # 用来组合多个 transforms 操作
    # 根据操作列表中放入的操作顺序依次执行
    trans = transforms.Compose(trans)
    mnist_train = torchvision.datasets.FashionMNIST(
        root="/data", train=True, transform=trans, download=True
    )
    mnist_test = torchvision.datasets.FashionMNIST(
        root="/data", train=False, transform=trans, download=True
    )
    return (data.DataLoader(mnist_train, batch_size, shuffle=True,
                            num_workers=get_dataloader_workers()),
            data.DataLoader(mnist_test, batch_size, shuffle=False,
                            num_workers=get_dataloader_workers()))


def init_params(num_inputs, num_outputs, num_hiddens):
    W1 = nn.Parameter(torch.randn(num_inputs, num_hiddens, requires_grad=True) * 0.01)
    b1 = nn.Parameter(torch.zeros(num_hiddens, requires_grad=True))
    W2 = nn.Parameter(torch.randn(num_hiddens, num_outputs, requires_grad=True) * 0.01)
    b2 = nn.Parameter(torch.zeros(num_outputs, requires_grad=True))
    params = [W1, b1, W2, b2]
    return params


def relu(X):
    a = torch.zeros_like(X)
    return torch.max(X, a)


def net(X, params, num_inputs):
    X = X.reshape((-1, num_inputs))
    H = relu(torch.dot(X, params.W1) + params.b1)
    return torch.dot(H, params.W2) + params.b2


def cross_entropy(y_hat, y):
    return - torch.log()


if __name__ == '__main__':
    # 单批量数据大小
    # batch_size = 256
    # train_iter, test_iter = load_data(batch_size)
    a = [1,2,2,3,4,5,6]
    b = a[range(len(a))]
    print(b)
