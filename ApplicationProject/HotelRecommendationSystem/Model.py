from torch import nn


class TextNet(nn.Module):
    def __init__(self):
        super(TextNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv1d(1, 32, 3, 1, 1),
            nn.ReLU(),
            nn.Conv1d(32, 128, 3, 1, 1),
            nn.MaxPool1d(2, 2),
            nn.Flatten()
        )
        self.fc = nn.Sequential(
            nn.Linear(1664, 2048),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(2048, 1000),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1000, 2),
        )
        self.loss_fn = None
        self.optimizer = None
        self.lr_scheduler = None
        self.epochs = None
        self.model_history = None

    def forward(self, x):
        x = x.unsqueeze(1)
        feature = self.features(x)
        # print("Conv1 output size: ", feature.size())
        result = self.fc(feature)
        return result


if __name__ == '__main__':
    model = TextNet()
    print(model)
