import numpy as np
import torch
from sklearn.manifold import TSNE
from sklearn.svm import SVC
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from mlxtend.plotting import plot_decision_regions
from pprint import pprint


class TSNEVizualizer:
    def __init__(self, *args, **kwargs):
        self.tsne = TSNE(*args, **kwargs)

    def fit_transform(self, inputs, labels):
        self.outputs = self.tsne.fit_transform(inputs)
        self.labels = labels

    def plot(self):
        plt.scatter(self.outputs[:, 0], self.outputs[:, 1], c=self.labels)
        plt.colorbar()
        plt.show()


class PCAVizualizer:
    def __init__(self, *args, **kwargs):
        self.pca = PCA(n_components=2)

    def fit(self, inputs):
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
        predicted = self.pca.transform(inputs)

class SVCVisualizer:
    def __init__(self, *args, **kwargs):
        self.svc = SVC(*args, **kwargs)

    def fit(self, inputs, labels):
        self.svc.fit(inputs, labels)

    def predict(self, inputs, is_return=False):
        return self.svc.predict(inputs)

    # 2次元データに落とし込む必要がある
    def plot(self, data, labels):
        # data_plot = np.vstack(data)
        # labels_plot = np.hstack(labels)
        print(data.shape, labels.shape)
        plot_decision_regions(data, labels, clf=self.svc, res=0.01, legend=2)
