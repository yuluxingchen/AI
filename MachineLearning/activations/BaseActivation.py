from abc import ABC, abstractmethod


class BaseActivation(ABC):
    @abstractmethod
    def forward(self, x):
        raise NotImplementedError

    @abstractmethod
    def backward(self, x):
        raise NotImplementedError
