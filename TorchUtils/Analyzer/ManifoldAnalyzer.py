from pprint import pprint

import matplotlib.pyplot as plt
import numpy as np
import torch
from mlxtend.plotting import plot_decision_regions
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.manifold import TSNE
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from abc import abstractmethod, ABCMeta
from typing import Any

import warnings
warnings.simplefilter("ignore")


def _get_standard_scaler(data: np.ndarray) -> StandardScaler:
    """ _get_standard_scaler

    Arguments:
    ----------
        data {np.ndarray} -- input data for adapting standard scaler

    Returns:
    --------
        {sklearn.preprocessing.StandardScaler} -- scaler
    """
    scaler = StandardScaler()
    scaler.fit(data)
    return scaler


class AnalyzerBase(metaclass=ABCMeta):
    def __init__(self, analyzer: Any, scaler: Any):
        self.scaler = scaler
        self.analyzer = analyzer
        self.stats = {}

    def fit(self, X: np.ndarray) -> None:
        if self.scaler is not None:
            X = self.scaler.transform(X)

        self.analyzer.fit(X)

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = X if self.scaler is None else self.scaler.transform(X)
        return self.analyzer.transform(X)

    def calculate_stats(self, X: np.ndarray, y: np.ndarray) -> None:
        unique_y = np.unique(y)
        self.stats = {k: {} for k in unique_y}

        for k in unique_y:
            idx = y[y==k]
            transformed = self.analyzer.transform(y[idx])
            self.stats[k] = {
                "center_x": transformed[:, 0].mean(),
                "center_y": transformed[:, 1].mean(),
                "std_x": transformed[:, 0].std(),
                "std_y": transformed[:, 1].std(),
                "num": transformed.shape[0]
            }

    def plot(self, X: np.ndarray, y: np.ndarray) -> None:
        pred = self.analyzer.transform(X)
        plt.scatter(pred[:, 0], pred[:, 1], c=y)

        for k in self.stats:
            plt.scatter(self.stats[k]["center_x"], self.stats[k]["center_y"], color="red", s=5)

        plt.colorbar()
        plt.show()


class TSNEAnalyzer:
    def __init__(self, *args, **kwargs):
        self.tsne = TSNE(*args, **kwargs)

    def fit_transform(self, inputs: np.ndarray, labels: np.ndarray) -> None:
        self.outputs = self.tsne.fit_transform(inputs)
        self.labels = labels

    def plot(self) -> None:
        plt.scatter(self.outputs[:, 0], self.outputs[:, 1], c=self.labels)
        plt.colorbar()
        plt.show()


class PCAAnalyzer(AnalyzerBase):
    def __init__(self, n_components: int = 2, scaler: Any = None):
        super(PCAAnalyzer, self).__init__(PCA(n_components=2), scaler)


# 次元削減
class TruncatedSVDAnalyzer(AnalyzerBase):
    def __init__(self, n_components: int = 2, n_iter: int = 7,
                 random_state: int = 42, scaler: Any = None):
        super(TruncatedSVDAnalyzer, self).__init__(
            TruncatedSVD(n_components=n_components, n_iter=n_iter,
                         random_state=random_state),
            scaler
        )


class LDAAnalyzer(AnalyzerBase):
    def __init__(self, scaler: Any = None):
        super(LDAAnalyzer, self).__init__(LinearDiscriminantAnalysis(), scaler)


class SVCAnalyzer(AnalyzerBase):
    def __init__(self, scaler: Any = None):
        super(SVCAnalyzer, self).__init__(SVC(), scaler)