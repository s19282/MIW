import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

a = np.loadtxt('data/dane2.txt')

x = a[:, [1]]
y = a[:, [0]]

c = np.hstack([np.cbrt(x), np.ones(x.shape)])  # cube root
c1 = np.hstack([np.arctan(x), np.ones(x.shape)])

v = np.linalg.pinv(c) @ y
v1 = np.linalg.pinv(c1) @ y

e = np.average((sum(y - v[0] * np.cbrt(x) + v[1]) ** 2))
e1 = np.average(sum((y - v1[0] * np.arctan(x) + v1[1]) ** 2))
print("cube root: ", e)
print("arcus tangens: ", e1)

plt.plot(x, y, 'ro')
plt.plot(x, v[0] * np.cbrt(x) + v[1])
plt.plot(x, v1[0] * np.arctan(x) + v1[1])
plt.show()
