import numpy as np

from MachineLearning.models.MLP import MLP

if __name__ == '__main__':
    # 示例数据：XOR 问题
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]]).T
    Y = np.array([[0], [1], [1], [0]]).T

    model = MLP()
    model.compiler(learning_rate=0.1)
    model.train(X, Y, epochs=1000)
    y_pred = model.forward(X)
    print("Predictions:")
    print((y_pred >= 0.5).astype(int))
