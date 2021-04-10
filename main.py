import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import datasets
from plotka import plot_decision_regions
from MultiPerceptron import MultiPerceptron
from Perceptron import Perceptron


def main():
    iris = datasets.load_iris()
    irisAttributes = iris.data[:, [2, 3]]
    irisClass = iris.target
    attributesTrainingSet, attributesTestSet, classTrainSet, classTestSet = train_test_split(irisAttributes, irisClass,
                                                                                             test_size=0.3,
                                                                                             random_state=1,
                                                                                             stratify=irisClass)
    # Perceptron 1
    classTrainingSubset1 = np.copy(classTrainSet)
    classTrainingSubset1 = classTrainingSubset1[(classTrainingSubset1 != 2)]
    attributesTrainingSubset1 = np.copy(attributesTrainingSet)
    attributesTrainingSubset1 = attributesTrainingSubset1[(classTrainSet != 2)]

    classTrainingSubset1[(classTrainingSubset1 != 0)] = -1
    classTrainingSubset1[(classTrainingSubset1 == 0)] = 1
    perceptron1 = Perceptron(learningRate=0.1, iterationsToStop=10)
    perceptron1.learn(attributesTrainingSubset1, classTrainingSubset1)

    # Perceptron 2
    classTrainingSubset2 = np.copy(classTrainSet)
    classTrainingSubset2 = classTrainingSubset2[(classTrainingSubset2 != 1)]
    attributesTrainingSubset2 = np.copy(attributesTrainingSet)
    attributesTrainingSubset2 = attributesTrainingSubset2[(classTrainSet != 1)]

    classTrainingSubset2[(classTrainingSubset2 != 2)] = -1
    classTrainingSubset2[(classTrainingSubset2 != -1)] = 1

    perceptron2 = Perceptron(learningRate=0.1, iterationsToStop=10)
    perceptron2.learn(attributesTrainingSubset2, classTrainingSubset2)

    # Perceptron 3
    classTrainingSubset3 = np.copy(classTrainSet)
    classTrainingSubset3 = classTrainingSubset3[(classTrainingSubset3 != 0)]
    attributesTrainingSubset3 = np.copy(attributesTrainingSet)
    attributesTrainingSubset3 = attributesTrainingSubset3[(classTrainSet != 0)]

    classTrainingSubset3[(classTrainingSubset3 != 1)] = -1

    ppn3 = Perceptron(learningRate=0.35, iterationsToStop=850)
    ppn3.learn(attributesTrainingSubset3, classTrainingSubset3)

    multiPerceptron = MultiPerceptron(perceptron1, perceptron2, ppn3)

    plot_decision_regions(X=attributesTestSet, y=classTestSet, classifier=multiPerceptron)
    plt.xlabel(r'$x_1$')
    plt.ylabel(r'$x_2$')
    plt.legend(loc='upper left')
    plt.show()


if __name__ == '__main__':
    main()
