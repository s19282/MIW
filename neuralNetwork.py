from random import random
from time import sleep
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# a = np.loadtxt('data/dane2.txt')
#
# x = a[:, [0]]
# y = a[:, [1]]
#
# xTraining, xTest, yTraining, yTest = train_test_split(x, y, test_size=0.7)

p = np.hstack(-2, 0.1, 2)
t = p ** 2 + 1*(np.random.rand(p) - .5)

s1 = 100
w1 = random.randint(s1, 1) - .5
b1 = random.randint(s1, 1) - .5
w2 = random.randint(1, s1) - .5
b2 = random.randint(1, 1) - .5

lr = 0.001

for x in range(1, 20):
    a1 = np.tanh(w1 * p, + b1 * np.ones(p))
    a2 = w2 * a1 + b2

    e2 = t - a2
    e1 = w2.T * e2

    dw2 = lr * e2 * a1.T  # TODO: .*
    db2 = lr * e2 * np.ones(e2).T
    dw1 = lr * (1 - a1 * a1) * e1 * p.T
    db1 = lr * (1 - a1 * a1) * e1 * np.ones(p).T

    if x % 1 == 0:
        plt.clf()
        plt.plot(p, t, 'r*')
        plt.plot(p, a2)
        sleep(.25)
