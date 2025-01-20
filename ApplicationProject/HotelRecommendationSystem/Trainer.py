import os

import numpy as np
import torch
from matplotlib import pyplot as plt
from torch import nn
from torch.optim import lr_scheduler
from torch.utils.data import TensorDataset, DataLoader

plt.rcParams['font.family'] = 'SimHei'
device = 'cuda' if torch.cuda.is_available() else 'cpu'


class Trainer:
    def __init__(self, model):
        self.model = model
        self.optimizer = None
        self.epochs = None
        self.loss_fn = None
        self.lr_scheduler = None
        self.acc_train = []
        self.acc_val = []
        self.loss_train = []
        self.loss_val = []

    def compile(self, optimizer='sgd', loss='binary_crossentropy', lr=0.01, momentum=0.9, step_size=10, gamma=0.5):
        optimizer = optimizer.lower()
        if optimizer == 'sgd':
            self.optimizer = torch.optim.SGD(self.model.parameters(), lr=lr, momentum=momentum)
        else:
            raise ValueError("{} is not exist!".format(optimizer))

        if loss == 'binary_crossentropy':
            self.loss_fn = nn.CrossEntropyLoss()
        else:
            raise ValueError("{} is not exist!".format(loss))
        self.lr_scheduler = lr_scheduler.StepLR(self.optimizer, step_size=step_size, gamma=gamma)

    def fit(self, X_train, y_train, val_size=0.4, batch_size=32, epochs=40):
        self.epochs = epochs
        # 划分训练集和验证集
        val_size = int(val_size * len(X_train))
        X_train, X_val = X_train[:val_size], X_train[val_size:]
        y_train, y_val = y_train[:val_size], y_train[val_size:]

        # 训练集加载为 Tensor 格式
        X_train_tensor = torch.tensor(X_train.to_numpy(), dtype=torch.float32)
        y_train_tensor = torch.tensor(y_train.to_numpy(), dtype=torch.long)

        # 验证集加载为 Tensor 格式
        X_val_tensor = torch.tensor(X_val.to_numpy(), dtype=torch.float32)
        y_val_tensor = torch.tensor(y_val.to_numpy(), dtype=torch.long)
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
        val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

        model = self.model.to(device)

        min_acc = 0
        for t in range(epochs):
            print(f"epoch{t + 1}\n---------------")
            train_loss, train_acc = self.train(train_dataloader, model, self.loss_fn, self.optimizer)
            val_loss, val_acc = self.val(val_dataloader, model, self.loss_fn, self.optimizer)
            self.lr_scheduler.step()

            self.loss_train.append(train_loss)
            self.acc_train.append(train_acc)
            self.loss_val.append(val_loss)
            self.acc_val.append(val_acc)

            if val_acc > min_acc:
                folder = 'save_model'
                if not os.path.exists(folder):
                    os.mkdir('save_model')
                min_acc = val_acc
                print(f"save the best model, epoch{t + 1}")
                torch.save(model.state_dict(), 'save_model/best_model.pth')
            if t == epochs - 1:
                torch.save(model.state_dict(), 'save_model/best_model.pth')

    def train(self, dataloader, model, loss_fn, optimizer):
        loss, current, n = 0.0, 0.0, 0
        for batch, (x, y) in enumerate(dataloader):
            x, y = x.to(device), y.to(device)
            output = model(x)
            cur_loss = loss_fn(output, y)
            _, pred = torch.max(output, dim=1)
            cur_acc = np.sum(y == pred) / output.shape[0]

            optimizer.zero_grad()
            cur_loss.backward()
            optimizer.step()
            loss += cur_loss.item()
            current += cur_acc.item()
            n += 1
        train_loss = loss / n
        train_acc = current / n
        print('train_loss: ' + str(train_loss))
        print('train_acc: ' + str(train_acc))
        return train_loss, train_acc

    def val(self, dataloader, model, loss_fn, optimizer):
        loss, current, n = 0.0, 0.0, 0
        for batch, (x, y) in enumerate(dataloader):
            image, y = x.to(device), y.to(device)
            output = model(image)
            cur_loss = loss_fn(output, y)
            _, pred = torch.max(output, dim=1)
            cur_acc = np.sum(y == pred) / output.shape[0]

            optimizer.zero_grad()
            cur_loss.backward()
            optimizer.step()
            loss += cur_loss.item()
            current += cur_acc.item()
            n += 1
        val_loss = loss / n
        val_acc = current / n
        print('val_loss: ' + str(val_loss))
        print('val_acc: ' + str(val_acc))
        return val_loss, val_acc

    def test(self, X_test, y_test, model_path, batch_size=512):
        X_test_tensor = torch.tensor(X_test.to_numpy(), dtype=torch.float32)
        y_test_tensor = torch.tensor(y_test.to_numpy(), dtype=torch.long)
        test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        self.model.load_state_dict(torch.load(model_path, weights_only=True))
        self.model.to(device)
        self.model.eval()
        predictions = []
        with torch.no_grad():
            for batch_idx, batch in enumerate(test_loader):
                inputs = batch[0]
                if isinstance(inputs, torch.Tensor):
                    inputs = inputs.to(device)
                else:
                    raise TypeError(f"Expected batch[0] to be a torch.Tensor, but got {type(inputs)} instead.")
                outputs = self.model(inputs)
                _, pred = torch.max(outputs, dim=1)
                predictions.append(pred.cpu())
                print(f"Batch {batch_idx + 1}: Predictions Finish!")
        return torch.cat(predictions, dim=0).squeeze().numpy()

    def predict(self, X_test, model_path, batch_size=128):
        # 将 X_test 转换为张量
        X_test_tensor = torch.tensor(X_test.to_numpy(), dtype=torch.float32)

        # 创建测试数据集
        test_dataset = TensorDataset(X_test_tensor)

        # 创建 DataLoader
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        # 加载模型权重
        self.model.load_state_dict(torch.load(model_path, weights_only=True))
        self.model.to(device)
        self.model.eval()

        # 存储预测结果
        predictions = []
        with torch.no_grad():
            for batch_idx, batch in enumerate(test_loader):
                inputs = batch[0]
                if isinstance(inputs, torch.Tensor):
                    inputs = inputs.to(device)
                else:
                    raise TypeError(f"Expected batch[0] to be a torch.Tensor, but got {type(inputs)} instead.")

                # 前向传播，获取预测结果
                outputs = self.model(inputs)
                _, pred = torch.max(outputs, dim=1)
                predictions.append(pred.cpu())

        # 将所有预测结果拼接成一个张量并返回
        return torch.cat(predictions, dim=0).squeeze().numpy()

    def show(self):
        plt.plot(self.loss_train, label='train_loss')
        plt.plot(self.loss_val, label='val_loss')
        plt.legend(loc='best')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.title('loss 比较图')
        plt.show()

        plt.plot(self.acc_train, label='train_acc')
        plt.plot(self.acc_val, label='val_acc')
        plt.legend(loc='best')
        plt.ylabel('acc')
        plt.xlabel('epoch')
        plt.title('acc 比较图')
        plt.show()
