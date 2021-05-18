import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
a = np.loadtxt('alior.txt')

y = a[:30,[0]]
c = a[:30,1:]
v = np.linalg.pinv(c) @ y
print(v)

plt.plot(y,'r-')
plt.plot(v[0]*c[:,[0]] + v[1]*c[:,[1]] + v[2]*c[:,[2]])
plt.show()

