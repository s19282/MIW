import numpy as np


class Perceptron(object):

    def __init__(self, learningRate=0.01, iterationsToStop=20, numberOfAttributes=2):
        self.wages = np.zeros(1 + numberOfAttributes)
        self.learningRate = learningRate
        self.iterationsToStop = iterationsToStop

    def learn(self, irisAttributes, irisClass):
        for each in range(self.iterationsToStop):
            for attributes, class_ in zip(irisAttributes, irisClass):
                update = self.learningRate * (class_ - self.predictClass(attributes))
                self.wages[1:] += update * attributes
                self.wages[0] += update     # threshold
        return self

    def calculateValue(self, irisInput):
        return np.dot(irisInput, self.wages[1:]) + self.wages[0]

    def predictClass(self, irisInput):
        return np.where(self.calculateValue(irisInput) >= 0.0, 1, -1)
