from MachineLearning.activations.Sigmoid import Sigmoid
from MachineLearning.losses.MSE import MSE
from MachineLearning.models.BaseModel import BaseModel
from MachineLearning.layers.Linear import Linear


class MLP(BaseModel):
    def __init__(self, activation=None):
        super().__init__()
        self.activation = activation
        self.hidden_size = 10
        self.layers = []
        self.inputs = None
        self.input = None
        self.optimizer = None
        self.loss_function = None
        self.learning_rate = 0.0

    def initLayers(self, n, input_size, output_size):
        self.layers.append(Linear(input_size, self.hidden_size))
        for i in range(1, n - 1):
            self.layers.append(Linear(self.hidden_size, self.hidden_size))
            self.layers.append(Sigmoid())
        self.layers.append(Linear(self.hidden_size, output_size))

    def forward(self, X):
        self.inputs = [X]
        x = X
        for layer in self.layers:
            x = layer.forward(x)
            self.inputs.append(x)
        return x

    def backward(self, output, Y):
        grad = self.loss_function.backward(output, Y)
        for layer in reversed(self.layers):
            if isinstance(layer, Linear):
                dw, db, grad = layer.backward(grad)
                layer.update(dw, db, self.learning_rate)
            elif isinstance(layer, Sigmoid):
                grad = layer.backward(grad)

    def compiler(self, optimizer='sgd', loss='mse', learning_rate=0.1):
        if loss == 'mse':
            self.loss_function = MSE()

        self.learning_rate = learning_rate

    def train(self, X, Y, epochs=100, n=1):
        print(X.shape, Y.shape)
        self.initLayers(n, X.shape[0], Y.shape[0])
        for epoch in range(epochs):
            y_pred = self.forward(X)
            loss = self.loss_function.forward(y_pred, Y)
            self.backward(y_pred, Y)
            if epoch % 10 == 0:
                print(f"Epoch {epoch}, Loss: {loss}")
