# TMO

# Lowess и Надарай-Ватсон
(алгоритмы для моделирования и сглаживания двумерных данных )

# метод Надарая-Ватсона

Реализована основная Надарая-Ватсона для восстановления регрессии:
![](https://raw.githubusercontent.com/okiochan/Lowess/master/formula/h1.gif)
<br/>
В качестве дата-сета был взят зашумленный синус

```python
def nadaray(X, Y, h, K, dist=euclidean):
    n = X.size
    Yt = np.zeros(n)
    for t in range(n):
        W = np.zeros(n)
        for i in range(n):
            W[i] = K(dist(X[t], X[i]) / h)
        Yt[t] = sum(W * Y) / sum(W)
    return Yt
```
![alt text](https://github.com/AJereli/TMO/blob/master/nadar.png)



Nadaray with gauss 0.3
1.7035138268200636
Nadaray with gauss 0.5
2.5207086720141128
Nadaray with gauss 0.7
3.9312477100883125
Nadaray with gauss 0.9
5.7525997288336
---
Nadaray with Kquad 0.3
1.3374668090970618
Nadaray with Kquad 0.5
1.5397288887782348
Nadaray with Kquad 0.7
1.6545228257136637
Nadaray with Kquad 0.9
1.859198005528924

# Loweless
![alt text](https://github.com/AJereli/TMO/blob/master/lt.png)

```python
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
```
![alt text](https://github.com/AJereli/TMO/blob/master/lowess.png)

Lowess with Kquad 0.3
1.033975993884266
Lowess with Kquad 0.5
1.033975993884266
Lowess with Kquad 0.7
1.033975993884266
Lowess with Kquad 0.9
1.033975993884266


