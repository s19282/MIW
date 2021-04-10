import numpy as np


class MultiPerceptron:
    def __init__(self, ppn1, ppn2, ppn3):
        self.ppn1 = ppn1
        self.ppn2 = ppn2
        self.ppn3 = ppn3

    def predictClass(self, X):
        return np.where(self.ppn1.predictClass(X) == 1, 0,
                        np.where(self.ppn3.predictClass(X) == 1, 1,
                                 np.where(self.ppn2.predictClass(X) == 1, 2, 2)))
