from abc import ABC, abstractmethod


class BaseLayer(ABC):
    @abstractmethod
    def forward(self, X):
        raise NotImplementedError

    @abstractmethod
    def backward(self, grad):
        raise NotImplementedError

    @abstractmethod
    def update(self, dw, db, learning_rate):
        raise NotImplementedError
