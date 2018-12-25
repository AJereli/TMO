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



Nadaray with gauss 0.3 <br/>
1.7035138268200636 <br/>
Nadaray with gauss 0.5 <br/>
2.5207086720141128 <br/>
Nadaray with gauss 0.7 <br/>
3.9312477100883125 <br/>
Nadaray with gauss 0.9 <br/>
5.7525997288336 <br/>
--- <br/>
Nadaray with Kquad 0.3 <br/>
1.3374668090970618 <br/>
Nadaray with Kquad 0.5 <br/>
1.5397288887782348 <br/>
Nadaray with Kquad 0.7 <br/>
1.6545228257136637 <br/>
Nadaray with Kquad 0.9 <br/>
1.859198005528924 <br/>
<br/><br/>
# Loweless <br/>
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
<br/><br/>
![alt text](https://github.com/AJereli/TMO/blob/master/lowess.png)

Lowess with Kquad 0.3 <br/>
1.033975993884266 <br/>
Lowess with Kquad 0.5 <br/>
1.033975993884266 <br/>
Lowess with Kquad 0.7 <br/>
1.033975993884266 <br/>
Lowess with Kquad 0.9 <br/>
1.033975993884266 <br/>

<br/>

#Lowess с выбросом
![alt text](https://github.com/AJereli/TMO/blob/master/v1.png)

<br/>

# метод Надарая-Ватсона c выбросом

![alt text](https://github.com/AJereli/TMO/blob/master/v2.png)

