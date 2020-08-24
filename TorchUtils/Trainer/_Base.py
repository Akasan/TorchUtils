from abc import abstractmethod, ABCMeta


class TrainerBase(metaclass=ABCMeta):
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


class OneCycleTrainerBase(metaclass=ABCMeta):
    @abstractmethod
    def fit_one_cycle(self):
        pass