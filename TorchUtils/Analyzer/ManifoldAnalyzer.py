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

import warnings
warnings.simplefilter("ignore")


def _get_standard_scaler(data):
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


class TSNEAnalyzer:
    def __init__(self, *args, **kwargs):
        self.tsne = TSNE(*args, **kwargs)

    def fit_transform(self, inputs, labels):
        self.outputs = self.tsne.fit_transform(inputs)
        self.labels = labels

    def plot(self):
        plt.scatter(self.outputs[:, 0], self.outputs[:, 1], c=self.labels)
        plt.colorbar()
        plt.show()


class PCAAnalyzer:
    def __init__(self, *args, **kwargs):
        self.pca = PCA(n_components=2)
        self.stats = {}
        self.scaler = None

    def fit(self, inputs, is_normalize=True):
        if is_normalize:
            self.scaler = _get_standard_scaler(inputs)
            inputs = self.scaler.transform(inputs)

        self.pca.fit(inputs)

    def plot(self, inputs, labels):
        outputs = self.pca.transform(inputs)
        plt.scatter(outputs[:, 0], outputs[:, 1], c=labels)
        for k in self.stats:
            plt.scatter(self.stats[k]["center_x"], self.stats[k]["center_y"], color="red", s=5)

        plt.colorbar()
        plt.show()

    def calculate_stats(self, inputs, labels):
        unique_labels = np.unique(labels)
        self.stats = {k: {} for k in unique_labels}

        for k in unique_labels:
            idx = labels[labels==k]
            transformed = self.pca.transform(inputs[idx])
            self.stats[k]["center_x"] = transformed[:, 0].mean()
            self.stats[k]["center_y"] = transformed[:, 1].mean()
            self.stats[k]["std_x"] = transformed[:, 0].std()
            self.stats[k]["std_y"] = transformed[:, 1].std()
            self.stats[k]["num"] = transformed.shape[0]

    def predict(self, inputs):
        inputs = inputs if self.scaler is None else self.scaler.transform(inputs)
        return self.pca.transform(inputs)


# 次元削減
class TruncatedSVDAnalyzer:
    def __init__(self, *args, **kwargs):
        self.svd = TruncatedSVD(n_components=2, n_iter=7, random_state=42)
        self.stats = {}
        self.scaler = None

    def fit(self, inputs, is_normalize=True):
        if is_normalize:
            self.scaler = _get_standard_scaler(inputs)
            inputs = self.scaler.transform(inputs)

        self.svd.fit(inputs)

    def plot(self, inputs, labels):
        outputs = self.svd.transform(inputs)
        plt.scatter(outputs[:, 0], outputs[:, 1], c=labels)
        for k in self.stats:
            plt.scatter(self.stats[k]["center_x"], self.stats[k]["center_y"], color="red", s=5)

        plt.colorbar()
        plt.show()

    def calculate_stats(self, inputs, labels):
        unique_labels = np.unique(labels)
        self.stats = {k: {} for k in unique_labels}

        for k in unique_labels:
            idx = labels[labels==k]
            transformed = self.pca.transform(inputs[idx])
            self.stats[k]["center_x"] = transformed[:, 0].mean()
            self.stats[k]["center_y"] = transformed[:, 1].mean()
            self.stats[k]["std_x"] = transformed[:, 0].std()
            self.stats[k]["std_y"] = transformed[:, 1].std()
            self.stats[k]["num"] = transformed.shape[0]

    def predict(self, inputs):
        inputs = inputs if self.scaler is None else self.scaler.transform(inputs)
        return self.svd.transform(inputs)


class LDAAnalyzer:
    def __init__(self, *args, **kwargs):
        self.lda = LinearDiscriminantAnalysis()
        self.stats = {}

    def fit(self, inputs, labels):
        self.scaler = _get_standard_scaler(inputs)
        inputs = self.scaler.transform(inputs)
        self.lda.fit(inputs, labels)

    def plot(self, inputs, labels):
        outputs = self.lda.transform(self.scaler.transform(inputs))
        plt.scatter(outputs[:, 0], outputs[:, 1], c=labels)
        for k in self.stats:
            plt.scatter(self.stats[k]["center_x"], self.stats[k]["center_y"], color="red", s=5)

        plt.colorbar()
        plt.show()

    def predict(self, inputs):
        inputs = self.scaler.transform(inputs)
        return self.lda.transform(inputs)


class SVCAnalyzer:
    def __init__(self):
        self.svc = SVC()
        self.scaler = None

    def fit(self, inputs, labels, is_normalize=True):
        """ fit

        Arguments:
        ----------
            inputs {np.ndarray} -- input data
            labels {np.ndarray} -- label
        """
        if is_normalize:
            self.scaler = _get_standard_scaler(inputs)
            inputs = self.scaler.transform(inputs)

        self.svc.fit(inputs, labels)

    def predict(self, inputs):
        """ predict

        Arguments:
        ----------
            inputs {np.ndarray} -- input data

        Returns:
        --------
            {np.ndarray} -- predicted label
        """
        inputs = inputs if self.scaler is None else self.scaler.transform(inputs)
        return self.svc.predict(inputs)

    def evaluate(self, inputs, labels, metric="accuracy"):
        """ evaluate

        Arguments:
        ----------
            inputs {np.ndarray} -- input data
            labels {np.ndarray} -- label

        Keyword Arguments:
        ------------------
            metric {str} -- evaluation metric (default: "accuracy")
        """
        predicted = self.predict(inputs)
        accuracy = accuracy_score(predicted, labels)
        print(accuracy)

    def plot(self, data, labels):
        plot_decision_regions(data, labels.reshape(-1), clf=self.svc, res=0.01, legend=2)
        plt.show()
