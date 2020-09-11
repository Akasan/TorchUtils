from abc import ABCMeta, abstractmethod


class TrainerBase(metaclass=ABCMeta):
    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def reset_history(self):
        pass

    @abstractmethod
    def fit(self):
        pass

    @abstractmethod
    def predict(self):
        pass

    @abstractmethod
    def save(self):
        pass

    @abstractmethod
    def read(self):
        pass

    @abstractmethod
    def plot_result(self):
        pass
