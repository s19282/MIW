import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

a = np.loadtxt('data/dane2.txt')

x = a[:, [1]]
y = a[:, [0]]

xTraining, xTest, yTraining, yTest = train_test_split(x, y, test_size=0.3)

c = np.hstack([xTraining, np.ones(xTraining.shape)])
c1 = np.hstack([np.arctan(xTraining), np.ones(xTraining.shape)])
c2 = np.hstack([np.cbrt(xTraining), np.ones(xTraining.shape)])  # cube root

v = np.linalg.pinv(c) @ yTraining
v1 = np.linalg.pinv(c1) @ yTraining
v2 = np.linalg.pinv(c2) @ yTraining

e = sum((y - v1[0] * x + v1[1]) ** 2)
e1 = sum((y - v1[0] * np.arctan(x) + v1[1]) ** 2)
e2 = sum((y - v2[0] * np.cbrt(x) + v2[1]) ** 2)

print("x: ", e)
print("arcus tangens: ", e1)
print("cube root: ", e2)

plt.plot(x, y, 'ro')
plt.plot(x, v[0] * x + v[1])
plt.plot(x, v1[0] * np.arctan(x) + v1[1])
plt.plot(x, v2[0] * np.cbrt(x) + v2[1])
plt.show()
