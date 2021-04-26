from time import sleep
import numpy as np
import matplotlib.pyplot as plt

p = np.arange(-2, 2, 0.1)
t = p.T ** 2 + 1*(np.random.rand(p)-0.5)
# TODO: check t once more
s1 = 100
w1 = np.random.rand(s1, 1) - .5
b1 = np.random.rand(s1, 1) - .5
w2 = np.random.rand(1, s1) - .5
b2 = np.random.rand(1, 1) - .5

lr = 0.001

for x in range(1, 20):
    a1 = np.tanh(w1 @ p + b1 @ np.ones(p))
    a2 = w2 @ a1 + b2

    e2 = t - a2
    e1 = w2.T @ e2

    dw2 = lr @ e2 @ a1.T
    db2 = lr @ e2 @ np.ones(e2).T
    dw1 = lr @ (1 - a1 * a1) * e1 @ p.T
    db1 = lr @ (1 - a1 * a1) * e1 @ np.ones(p).T

    if x % 1 == 0:
        plt.clf()
        plt.plot(p, t, 'r*')
        plt.plot(p, a2)
        sleep(.25)
