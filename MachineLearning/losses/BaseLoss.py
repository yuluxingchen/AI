from abc import ABC, abstractmethod


class BaseLoss(ABC):
    @abstractmethod
    def forward(self, output, Y):
        raise NotImplementedError

    @abstractmethod
    def backward(self, output, Y):
        raise NotImplementedError
