from abc import ABC, abstractmethod


class BaseModel(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def forward(self, X):
        raise NotImplementedError

    @abstractmethod
    def backward(self, output, Y):
        raise NotImplementedError
