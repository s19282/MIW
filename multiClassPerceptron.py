import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import datasets
from plotka import plot_decision_regions


class Perceptron(object):

    def __init__(self, eta=0.01, n_iter=10):
        self.eta = eta
        self.n_iter = n_iter

    def fit(self, X, y):
        self.w_ = np.zeros(1 + X.shape[1])

        for _ in range(self.n_iter):
            for xi, target in zip(X, y):
                update = self.eta * (target - self.predict(xi))
                self.w_[1:] += update * xi
                self.w_[0] += update
        return self

    def net_input(self, X):
        return np.dot(X, self.w_[1:]) + self.w_[0]

    def predict(self, X):
        return np.where(self.net_input(X) >= 0.0, 1, -1)


class MultiClassClassifier:
    def __init__(self, ppn1, ppn2, ppn3):
        self.ppn1 = ppn1
        self.ppn2 = ppn2
        self.ppn3 = ppn3

    def predict(self, X):
        return np.where(self.ppn1.predict(X) == 1, 0,
                        np.where(self.ppn3.predict(X) == 1, 2, 1))
        # def predict(self, X):
        # return np.where(self.ppn1.predict(X) == 1, 0,
        #                 np.where(self.ppn2.predict(X) == 1, 1,
        #                          np.where(self.ppn3.predict(X) == 1, 2, -1)))


def main():
    iris = datasets.load_iris()
    X = iris.data[:, [2, 3]]
    y = iris.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1, stratify=y)

    y_train_1 = np.copy(y_train)
    y_train_2 = np.copy(y_train)
    y_train_3 = np.copy(y_train)

    y_train_1[(y_train_1 != 0)] = -1
    ppn1 = Perceptron(eta=0.1, n_iter=10)
    ppn1.fit(X_train, y_train_1)

    y_train_2[(y_train_2 != 1)] = -1
    ppn2 = Perceptron(eta=0.1, n_iter=10)
    ppn2.fit(X_train, y_train_2)

    y_train_3[(y_train_3 != 2)] = -1
    ppn3 = Perceptron(eta=0.1, n_iter=10)
    ppn3.fit(X_train, y_train_3)

    multiClassClassifier = MultiClassClassifier(ppn1, ppn2, ppn3)

    plot_decision_regions(X=X_train, y=y_train, classifier=multiClassClassifier)
    plt.xlabel(r'$x_1$')
    plt.ylabel(r'$x_2$')
    plt.legend(loc='upper left')
    plt.show()


if __name__ == '__main__':
    main()
