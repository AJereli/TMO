# TMO

### Постановка задачи
Пусть задано пространство объектов ![](https://raw.githubusercontent.com/IsmailovMukhammed/MLT/master/img's/1.PNG) и множество возможных ответов ![](https://raw.githubusercontent.com/IsmailovMukhammed/MLT/master/img's/2.PNG). Существует неизвестная зависимость ![](https://raw.githubusercontent.com/IsmailovMukhammed/MLT/master/img's/3.PNG) значения которой известны только на объектах обучающией выборки ![](https://raw.githubusercontent.com/IsmailovMukhammed/MLT/master/img's/4.PNG),требуется построить алгоритм ![](https://raw.githubusercontent.com/IsmailovMukhammed/MLT/master/img's/5.PNG), аппроксимирующий неизвестную зависимость ![](https://raw.githubusercontent.com/IsmailovMukhammed/MLT/master/img's/6.PNG).

# Lowess и Надарай-Ватсон
(алгоритмы для моделирования и сглаживания двумерных данных )

## Метод Надарая-Ватсона

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
## Выбор ядра и ширины окна
Выбор ядра мало влияет на точность аппроксимации, но определяющим образом влияет на степень гладкости функции ![](https://raw.githubusercontent.com/IsmailovMukhammed/MLT/master/img's/18.PNG). Для ядерного сглаживания чаще всего берут гауссовское ядро 
![](https://raw.githubusercontent.com/IsmailovMukhammed/MLT/master/img's/19.PNG) или квартическое ![](https://raw.githubusercontent.com/IsmailovMukhammed/MLT/master/img's/20.PNG).

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

# Lowess с выбросом

![alt text](https://github.com/AJereli/TMO/blob/master/v1.png)

<br/>

# метод Надарая-Ватсона c выбросом

![alt text](https://github.com/AJereli/TMO/blob/master/v2.png)



## Линейная регрессия

Пусть каждый объект описывается *n* числовыми признаками ![alt text](https://latex.codecogs.com/gif.latex?f_j%28x%29%2C%20f_j%3A%20X%20%5Crightarrow%20%5Cmathbb%7BR%7D%2C%20j%20%3D%201%2C%20...%2C%20n). *Линейной моделью регрессии* называется линейная комбинация признаков с коэффициентами ![alt text](https://latex.codecogs.com/gif.latex?%5Calpha%20%5Cepsilon%20%5Cmathbb%7BR%7D%5En):

![alt text](https://latex.codecogs.com/gif.latex?%5Cphi%28x%2C%20%5Calpha%29%20%3D%20%5Csum_%7Bj%20%3D%201%7D%5E%7Bn%7D%5Calpha_jf_j%28x%29)

Условие минимума функционала ![alt text](https://latex.codecogs.com/gif.latex?Q%28%5Calpha%29%20%3D%20%7C%7CF%5Calpha%20-%20y%7C%7C%5E2)
Решением нормальной системы уравнений является вектор:

![alt text](https://latex.codecogs.com/gif.latex?%5Calpha%5E*%20%3D%20%28F%5ETF%29%5E%7B-1%7D%20F%5ETy)

```python
def predict(w, p):
    return np.dot(w, np.transpose(p))
w = np.dot(np.dot(np.linalg.inv(np.dot(np.transpose(X), X)), np.transpose(X)), Y)
```

Тестовая выборка - Бостон, 80 элементов. 0-1 признаки, 5-6 признаки
![alt text](https://github.com/AJereli/TMO/blob/master/imgs/lr_0_1.png) <br/> <br/>
![alt text](https://github.com/AJereli/TMO/blob/master/imgs/lr_boston_5_6.png)


