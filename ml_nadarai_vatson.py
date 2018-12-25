import numpy as np
import matplotlib
from scipy.spatial import distance
import numpy as np
import math
matplotlib.use('TkAgg')

import matplotlib.pyplot as pl

eps = 1e-5




def euclidean(x, y):
    return np.sqrt(np.sum((x - y) ** 2))


def Kgauss(r):  # gauss kernel
    return (((2 * math.pi) ** (-0.5)) * math.exp(-0.5 * r**2));


def Kquad(r):  # qwadratic kernel
    if (abs(r) <= 1):
        return (1 - r ** 2) ** 2
    else:
        return 0


def QuadraticError(Y, Yt):
    return ((Yt - Y) ** 2).sum()


def nadaray(X, Y, h, K, dist=euclidean):
    n = X.size
    Yt = np.zeros(n)
    for t in range(n):
        W = np.zeros(n)
        for i in range(n):
            W[i] = K(dist(X[t], X[i]) / h)
        Yt[t] = sum(W * Y) / sum(W)
    return Yt


def lowess(X, Y, MAX, h, K, dist=euclidean):
    n = X.size
    delta = np.ones(n)
    Yt = np.zeros(n)

    for step in range(MAX):
        for t in range(n):
            num = 0
            den = 0
            for i in range(n):
                num += Y[i] * delta[i] * K(dist(X[i], X[t]) / h)
                den += delta[i] * K(dist(X[i], X[t]) / h)
            Yt[t] = num / den

        Q = np.abs(Y - Yt)
        delta = [K(Q[j]) for j in range(n)]
        delta = np.array(delta, dtype=float)
    return Yt



def generate_wave_set(n_support=1000, n_train=250):
    data = {}
    data['support'] = np.linspace(0, 2 * np.pi, num=n_support)
    data['x_train'] = np.sort(np.random.choice(data['support'], size=n_train, replace=True))
    data['y_train'] = np.sin(data['x_train']).ravel()
    data['y_train'] += 0.5 * (0.55 - np.random.rand(data['y_train'].size))
    return data


data = generate_wave_set(100, 80)
x = data['x_train']
y = data['y_train']
x[79] = x[79]*1.1

yest_nadaray = nadaray(x, y, 0.3, Kgauss)
yest_nadaray2 = nadaray(x, y, 0.5, Kgauss)
yest_nadaray3 = nadaray(x, y, 0.7, Kgauss)
yest_nadaray4 = nadaray(x, y, 0.9, Kgauss)

print("Nadaray with gauss 0.3")
print(QuadraticError(y, yest_nadaray))
print("Nadaray with gauss 0.5")
print(QuadraticError(y, yest_nadaray2))
print("Nadaray with gauss 0.7")
print(QuadraticError(y, yest_nadaray3))
print("Nadaray with gauss 0.9")
print(QuadraticError(y, yest_nadaray4))


yest_nadarayq = nadaray(x, y, 0.3, Kquad)
yest_nadaray2q = nadaray(x, y, 0.5, Kquad)
yest_nadaray3q = nadaray(x, y, 0.7, Kquad)
yest_nadaray4q = nadaray(x, y, 0.9, Kquad)

print("---")
print("Nadaray with Kquad 0.3")
print(QuadraticError(y, yest_nadarayq))
print("Nadaray with Kquad 0.5")
print(QuadraticError(y, yest_nadaray2q))
print("Nadaray with Kquad 0.7")
print(QuadraticError(y, yest_nadaray3q))
print("Nadaray with Kquad 0.9")
print(QuadraticError(y, yest_nadaray4q))

pl.clf()
pl.subplot(1, 2, 1)
pl.scatter(x, y, label='data set', color="black")
pl.title('Nadaray with Gauss')
pl.plot(x, yest_nadaray, label='y nadaray-watson 0.3', color="red")
pl.plot(x, yest_nadaray2, label='y nadaray-watson 0.5', color="yellow")
pl.plot(x, yest_nadaray3, label='y nadaray-watson 0.7', color="blue")
pl.plot(x, yest_nadaray4, label='y nadaray-watson 0.9', color="green")
pl.subplot(1, 2, 2)


pl.legend()
pl.scatter(x, y, label='data set', color="black")
pl.title('Nadaray with Kquad')
pl.plot(x, yest_nadaray, label='y nadaray-watson 0.3', color="red")
pl.plot(x, yest_nadaray2, label='y nadaray-watson 0.5', color="yellow")
pl.plot(x, yest_nadaray3, label='y nadaray-watson 0.7', color="blue")
pl.plot(x, yest_nadaray4, label='y nadaray-watson 0.9', color="green")
pl.legend()
# pl.clf()

# yest_lowess = lowess(x, y, 5, 0.3, Kquad)
# yest_lowess2 = lowess(x, y, 5, 0.3, Kquad)
# yest_lowess3 = lowess(x, y, 5, 0.3, Kquad)
# yest_lowess4 = lowess(x, y, 5, 0.3, Kquad)
#
# pl.scatter(x, y, label='data set', color="black")
# pl.title('Lowess with Kquad')
# pl.plot(x, yest_lowess, label='y lowess 0.3', color="red")
# pl.plot(x, yest_lowess2, label='y lowess 0.3', color="red")
# pl.plot(x, yest_lowess3, label='y lowess 0.3', color="red")
# pl.plot(x, yest_lowess4, label='y lowess 0.3', color="red")
# print("Lowess with Kquad 0.3")
# print(QuadraticError(y, yest_lowess))
# print("Lowess with Kquad 0.5")
# print(QuadraticError(y, yest_lowess2))
# print("Lowess with Kquad 0.7")
# print(QuadraticError(y, yest_lowess3))
# print("Lowess with Kquad 0.9")
# print(QuadraticError(y, yest_lowess4))

pl.show()
