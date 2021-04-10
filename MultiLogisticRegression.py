import numpy as np


class MultiLogisticRegression:
    def __init__(self, logisticRegression1, logisticRegression2, logisticRegression3):
        self.lrgd1 = logisticRegression1
        self.lrgd2 = logisticRegression2
        self.lrgd3 = logisticRegression3

    def predictClass(self, X):
        return np.where(self.lrgd1.predictClass(X) == 1, 0,
                        np.where(self.lrgd3.predictClass(X) == 1, 1,
                                 np.where(self.lrgd2.predictClass(X) == 1, 2, 2)))
