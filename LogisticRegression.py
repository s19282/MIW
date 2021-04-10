import numpy as np


class LogisticRegression(object):
    def __init__(self, learningRate=0.05, iterationsToStop=100, random_state=1, numberOfAttributes=2):
        self.learningRate = learningRate
        self.iterationsToStop = iterationsToStop
        self.random_state = random_state
        random = np.random.RandomState(self.random_state)
        self.wages = random.normal(loc=0.0, scale=0.01, size=1 + numberOfAttributes)

    def learn(self, irisAttributes, irisClass):
        for each in range(self.iterationsToStop):
            value = self.calculateValue(irisAttributes)
            output = activation(value)
            errors = (irisClass - output)
            self.wages[1:] += self.learningRate * irisAttributes.T.dot(errors)
            self.wages[0] += self.learningRate * errors.sum()
        return self

    def calculateValue(self, irisInput):
        return np.dot(irisInput, self.wages[1:]) + self.wages[0]

    def predictClass(self, irisInput):
        return np.where(self.calculateValue(irisInput) >= 0.0, 1, 0)

    def printProbability(self, irisInput):
        print("Class 2 probability")
        print(activation(self.calculateValue(irisInput)))
        print()


def activation(z):
    return 1. / (1. + np.exp(-np.clip(z, -250, 250)))
