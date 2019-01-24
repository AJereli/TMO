# SVM Doc
Support vecot machines (SVM) - множество алгоритмов обучения с учителем который применяются для классификации регрессии. Является линейным классификатором.

Плюсы SVM:
+ Эффективен в больших пространствах.
+ Эффективен в тех случаях, когда размер пространства количество образцов.
+ Использует подмножество обучающих точек в функции принятия решений (опорные вектора), по этому эффективно использует память.
+ Универсальность: различные функции ядра могут быть указаны для решающей функции. Предоставляются общие ядра, но также можно указывать собственные ядра.

### sklearn.svm.SVC
Support Vector Classification.
Интерфейс конструктора выглядит следующим образом:
```python
SVC(C=1.0, kernel=’rbf’, degree=3, gamma=’auto_deprecated’, coef0=0.0, shrinking=True, probability=False, tol=0.001, cache_size=200, class_weight=None, verbose=False, max_iter=-1, decision_function_shape=’ovr’, random_state=None)
```
Каждый из параметров является опциональным.

#### C : float, default = 1.0
Штрафной параметр.

#### kernel : string, default=’rbf’
Определяет тип ядра, который будет использоваться в алгоритме. Тип ядра должен быть ‘linear’, ‘poly’, ‘rbf’, ‘sigmoid’ или вызываемый. Если ничего не указано, будет использовано «rbf». Если дан вызываемый объект, он используется для предварительного вычисления матрицы ядра из матриц данных; эта матрица должна быть массивом формы (n_samples, n_samples).

#### degree : int, default=3
Степень функции ядра полинома («poly»). Игнорируется всеми другими ядрами.

#### gamma : float, default=’auto’
Коэффициент для ядер ‘rbf’, ‘poly’ и ‘sigmoid’.

#### coef0 : float, default=0.0
Независимый член в функции ядра. Используется в «poly» и «sigmoid».

#### shrinking : boolean, default=True
Следуют ли использовать сжимающие эвристики.

#### probability : boolean, default=False
Включить ли вычисление оценки вероятности. Это должно быть включено до вызова fit и замедлит этот метод.

#### tol : float, default=1e-3
Допуск для критерия остановки.

#### cache_size : float
Укажите размер кеша ядра (в МБ).

#### class_weight : {dict, ‘balanced’}
Устанавливает для параметра C класса i значение class_weight [i] * C для SVC. Если не дано, все классы должны иметь вес один. «Сбалансированный» режим использует значения y для автоматической регулировки весов, обратно пропорциональных частотам классов во входных данных как n_samples / (n_classes * np.bincount (y))

#### verbose : bool, default: False
Включить подробный вывод. Обратите внимание, что этот параметр использует параметр времени выполнения для каждого процесса в libsvm, который, если он включен, может не работать должным образом в многопоточном контексте.

#### max_iter : int, default=-1
Строгий лимит на итерации в решателе, или -1 для безлимитного.

#### decision_function_shape : ‘ovo’, ‘ovr’, default=’ovr’
Выбор стратегии многоклассовой классификации. Один-против-одного или один-против-всех.

#### random_state : int, RandomState instance or None, default=None
"Источник" генератора псевдослучайных чисел, используемого при перетасовке данных для оценки вероятности. Если int, random_state - начальное число, используемое генератором случайных чисел; Если экземпляр RandomState, random_state является генератором случайных чисел; Если None, генератор случайных чисел является экземпляром RandomState, используемым np.random.

#### Аттрибуты:	

#### support_ : array-like, shape = [n_SV]
Индексы опорных векторов

#### support_vectors_ : array-like, shape = [n_SV, n_features]
Опорные вектора

#### n_support_ : array-like, dtype=int32, shape = [n_class]
Кол-во опорных векторов для каждого класса

#### dual_coef_ : array, shape = [n_class-1, n_SV]
Коэффициенты опорных векторов в решающей функции

#### coef_ : array, shape = [n_class * (n_class-1) / 2, n_features]
Веса, присвоенные признакам (коэффициенты в основной задаче). Это доступно только в случае линейного ядра.

#### fit_status_ : int
0 если правильно обученно, 1 в противном случае (вызовет предупреждение)

#### Методы:

#### fit(X, y[, sample_weight])
Обучить алгоритм на выборке


#### get_params([deep])
Получить параметры классификатора

#### predict(X)
Выполнить классификацию X

#### Score (X, y [, sample_weight])
Возвращает среднюю точность (accuracy) данных и данных теста.

#### set_params(**params)
Установить параметры классификатора
# TMO

## _Задача обучения по прецедентам_ *(основная задача ТМО)*:

Задано множество объектов *X* и множество допустимых ответов *Y*.
Существует целевая функция ![alt text](https://latex.codecogs.com/gif.latex?y^*:&space;X\rightarrow&space;Y), значения которой 
![alt text](https://latex.codecogs.com/gif.latex?y_i&space;=&space;y^*(x_i)) известны только на конечном подмножестве объектов 
![alt text](https://latex.codecogs.com/gif.latex?\left&space;\{&space;x_1,&space;...,&space;x_l&space;\right&space;\}&space;\subset&space;X). Пары "объект-ответ" ![alt-text](https://latex.codecogs.com/gif.latex?\left&space;(&space;x_i,&space;y_i&space;\right&space;)) называются прецедентами. Совокупность пар ![alt text](https://latex.codecogs.com/gif.latex?X^l&space;=&space;\left&space;(&space;x_i,&space;y_i&space;\right&space;),&space;i&space;=&space;1,&space;...,&space;l) называется обучающей выборкой. 

Задача обучения по прецедентам заключается в том, чтобы по выборке ![alt text](https://latex.codecogs.com/gif.latex?X^l) восстановить зависимость ![alt text](https://latex.codecogs.com/gif.latex?y^*), т.е. построить решающую функцию 
![alt text](https://latex.codecogs.com/gif.latex?a:&space;X&space;\rightarrow&space;Y), которая приближала бы целевую функцию 
![alt text](https://latex.codecogs.com/gif.latex?y^*(x)), причём не только на объектах обучающей выборки, но и на всём множестве *X*.

Задачу обучения по прецедентам при Y = R принято называть задачей восстановления регрессии, а решающую функцию a — функцией регрессии.

Алгоритм обучения (learning algorithm), синоним Метод обучения — в задачах обучения по прецедентам — алгоритм \mu, который принимает на входе обучающую выборку данных D, строит и выдаёт на выходе функцию f из заданной модели F, реализующую отображение из множества объектов X во множество ответов Y.

Построенная функция f должна аппроксимировать (восстанавливать) зависимость ответов от объектов, в целом неизвестную, и заданную лишь в конечном числе точек — объектов обучающей выборки D=\bigl((x_1,y_1),\ldots,(x_m,y_m)\bigr)\in X^m.

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


