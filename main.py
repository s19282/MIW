import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import datasets

from LogisticRegression import LogisticRegression
from MultiLogisticRegression import MultiLogisticRegression
from plotka import plot_decision_regions
from MultiPerceptron import MultiPerceptron
from Perceptron import Perceptron
from LogisticRegression import activation


def main():
    iris = datasets.load_iris()
    irisData = iris.data[:, [2, 3]]
    irisClass = iris.target
    dataTrainingSet, dataTestSet, classTrainingSet, classTestSet = train_test_split(irisData, irisClass,
                                                                                    test_size=0.3,
                                                                                    random_state=1,
                                                                                    stratify=irisClass)
    #     =============== Perceptron ====================
    # Perceptron 1
    classTrainingSubset1 = np.copy(classTrainingSet)
    classTrainingSubset1 = classTrainingSubset1[(classTrainingSubset1 != 2)]
    dataTrainingSubset1 = np.copy(dataTrainingSet)
    dataTrainingSubset1 = dataTrainingSubset1[(classTrainingSet != 2)]

    classTrainingSubset1[(classTrainingSubset1 != 0)] = -1
    classTrainingSubset1[(classTrainingSubset1 != -1)] = 1
    perceptron1 = Perceptron(learningRate=0.1, iterationsToStop=10)
    perceptron1.learn(dataTrainingSubset1, classTrainingSubset1)

    # Perceptron 2
    classTrainingSubset2 = np.copy(classTrainingSet)
    classTrainingSubset2 = classTrainingSubset2[(classTrainingSubset2 != 1)]
    dataTrainingSubset2 = np.copy(dataTrainingSet)
    dataTrainingSubset2 = dataTrainingSubset2[(classTrainingSet != 1)]

    classTrainingSubset2[(classTrainingSubset2 != 2)] = -1
    classTrainingSubset2[(classTrainingSubset2 != -1)] = 1

    perceptron2 = Perceptron(learningRate=0.1, iterationsToStop=10)
    perceptron2.learn(dataTrainingSubset2, classTrainingSubset2)

    # Perceptron 3
    classTrainingSubset3 = np.copy(classTrainingSet)
    classTrainingSubset3 = classTrainingSubset3[(classTrainingSubset3 != 0)]
    dataTrainingSubset3 = np.copy(dataTrainingSet)
    dataTrainingSubset3 = dataTrainingSubset3[(classTrainingSet != 0)]

    classTrainingSubset3[(classTrainingSubset3 != 1)] = -1

    ppn3 = Perceptron(learningRate=0.35, iterationsToStop=850)
    ppn3.learn(dataTrainingSubset3, classTrainingSubset3)

    multiPerceptron = MultiPerceptron(perceptron1, perceptron2, ppn3)

    plot_decision_regions(X=dataTestSet, y=classTestSet, classifier=multiPerceptron)
    plt.xlabel(r'$x_1$')
    plt.ylabel(r'$x_2$')
    plt.legend(loc='upper left')
    plt.show()

    #     =============== Logistic regression ====================

    classTrainingSubset1[(classTrainingSubset1 != 1)] = 0
    logisticRegression1 = LogisticRegression(learningRate=0.05, iterationsToStop=1000, random_state=1)
    logisticRegression1.learn(dataTrainingSubset1, classTrainingSubset1)
    print("Class 2 probability")
    print(activation(logisticRegression1.calculateValue(dataTrainingSubset1)))
    print()

    classTrainingSubset2[(classTrainingSubset2 != 1)] = 0
    logisticRegression2 = LogisticRegression(learningRate=0.05, iterationsToStop=1000, random_state=1)
    logisticRegression2.learn(dataTrainingSubset2, classTrainingSubset2)
    print("Class 1 probability")
    print(activation(logisticRegression1.calculateValue(dataTrainingSubset1)))
    print()

    classTrainingSubset3[(classTrainingSubset3 != 1)] = 0
    logisticRegression3 = LogisticRegression(learningRate=0.15, iterationsToStop=1500, random_state=1)
    logisticRegression3.learn(dataTrainingSubset3, classTrainingSubset3)
    print("Class 0 probability")
    print(activation(logisticRegression1.calculateValue(dataTrainingSubset1)))
    print()

    multiLogisticRegression = MultiLogisticRegression(logisticRegression1, logisticRegression2,
                                                      logisticRegression3)

    plot_decision_regions(X=dataTestSet, y=classTestSet, classifier=multiLogisticRegression)
    plt.xlabel(r'$x_1$')
    plt.ylabel(r'$x_2$')
    plt.legend(loc='lower right')
    plt.show()


if __name__ == '__main__':
    main()
