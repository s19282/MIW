import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

a = np.loadtxt('data/dane2.txt')

x = a[:, [0]]
y = a[:, [1]]

xTraining, xTest, yTraining, yTest = train_test_split(x, y, test_size=0.5)

c = np.hstack([xTraining, np.ones(xTraining.shape)])
c1 = np.hstack([xTraining ** 3, xTraining ** 2, xTraining, np.ones(xTraining.shape)])

v = np.linalg.pinv(c) @ yTraining
v1 = np.linalg.pinv(c1) @ yTraining
eTraining = sum((yTraining - (v1[0] * xTraining + v1[1])) ** 2) / xTraining.shape[0]
eTest = sum((yTest - (v1[0] * xTest + v1[1])) ** 2) / xTest.shape[0]
e1Training = sum((yTraining - (v1[0] * xTraining ** 3 + v1[1] * xTraining ** 2 + v1[2] * xTraining + v1[3])) ** 2) / xTraining.shape[0]
e1Test = sum((yTest - (v1[0] * xTest ** 3 + v1[1] * xTest ** 2 + v1[2] * xTest + v1[3])) ** 2) / xTest.shape[0]

print("x Training: ", eTraining, "x Test: ", eTest)
print("x^3 Training: ", e1Training, "x^3 Test: ", e1Test)

plt.plot(x, y, 'ro')
plt.plot(x, v1[0] * x ** 3 + v1[1] * x ** 2 + v1[2] * x + v1[3])
plt.plot(x, v[0] * x + v[1])
plt.show()
