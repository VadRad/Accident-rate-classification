
#  Определение класса аварийности транспортного средства

По данным телеметрического мониторинга состояния карьерных самосвалов, полученных на предприятиях Coal Inc. в ноябре 2019 года необходимо построить модель определения класса аварийности транспортного средства.

**Описание данных:**

Таблица data:  
- *temperature* — cредняя температура воздуха за бортом;
- *velocity* — cредняя скорость движения самосвала;
- *pressure* — cреднее давление в шинах;
- *incline* — cреднее значение показаний инклинометра;
- *class* — класс аварийности

Предложенный массив данных содержит 135 записей с описанными выше измерениями и классом аварийности самосвала, присвоенным ему ремонтной бригадой.


```python
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats as st
import math
```


```python
data=pd.read_csv('./data/data.csv')
```


```python
data.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>temperature</th>
      <th>velocity</th>
      <th>pressure</th>
      <th>incline</th>
      <th>class</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>5.1</td>
      <td>3.5</td>
      <td>1.4</td>
      <td>0.2</td>
      <td>Class_1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>4.9</td>
      <td>3.0</td>
      <td>1.4</td>
      <td>0.2</td>
      <td>Class_1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>4.7</td>
      <td>3.2</td>
      <td>1.3</td>
      <td>0.2</td>
      <td>Class_1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4.6</td>
      <td>3.1</td>
      <td>1.5</td>
      <td>0.2</td>
      <td>Class_1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5.0</td>
      <td>3.6</td>
      <td>1.4</td>
      <td>0.2</td>
      <td>Class_1</td>
    </tr>
  </tbody>
</table>
</div>



Многие алгоритмы машинного обучения требуют на вход численные признаки, поэтому приведем метку класса к целочисленному типу.


```python
data.loc[:,'class']=data['class'].map({'Class_1': 1, 'Class_2': 2, 'Class_3': 3})
```


```python
data.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 134 entries, 0 to 133
    Data columns (total 5 columns):
    temperature    134 non-null float64
    velocity       134 non-null float64
    pressure       134 non-null float64
    incline        134 non-null float64
    class          134 non-null int64
    dtypes: float64(4), int64(1)
    memory usage: 5.3 KB
    

## Шаг 1. Статистический анализ

Посмотрим на распределение имеющихся данных по классам. 


```python
data['class'].value_counts()
```




    3    45
    1    45
    2    44
    Name: class, dtype: int64



Распределение практически равномерное.

Обратимся к совместному распределению классов.


```python
sns.pairplot(data)
```




    <seaborn.axisgrid.PairGrid at 0x1ec0ca31240>




![png](output_15_1.png)


Построим матрицу корреляции, показывающую линейный коэффициент корреляции между всеми возможными парами переменных.


```python
data.corr()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>temperature</th>
      <th>velocity</th>
      <th>pressure</th>
      <th>incline</th>
      <th>class</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>temperature</th>
      <td>1.000000</td>
      <td>-0.125808</td>
      <td>0.872487</td>
      <td>0.813873</td>
      <td>0.783336</td>
    </tr>
    <tr>
      <th>velocity</th>
      <td>-0.125808</td>
      <td>1.000000</td>
      <td>-0.437006</td>
      <td>-0.381367</td>
      <td>-0.436146</td>
    </tr>
    <tr>
      <th>pressure</th>
      <td>0.872487</td>
      <td>-0.437006</td>
      <td>1.000000</td>
      <td>0.963831</td>
      <td>0.948509</td>
    </tr>
    <tr>
      <th>incline</th>
      <td>0.813873</td>
      <td>-0.381367</td>
      <td>0.963831</td>
      <td>1.000000</td>
      <td>0.961449</td>
    </tr>
    <tr>
      <th>class</th>
      <td>0.783336</td>
      <td>-0.436146</td>
      <td>0.948509</td>
      <td>0.961449</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
</div>



Построим так же сводную таблицу средних значений телеметрических показателей для каждого класса аварийности. 


```python
data.pivot_table(index='class', aggfunc='mean')
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>incline</th>
      <th>pressure</th>
      <th>temperature</th>
      <th>velocity</th>
    </tr>
    <tr>
      <th>class</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>0.251111</td>
      <td>1.462222</td>
      <td>5.022222</td>
      <td>3.442222</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1.325000</td>
      <td>4.256818</td>
      <td>5.938636</td>
      <td>2.772727</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2.042222</td>
      <td>5.548889</td>
      <td>6.591111</td>
      <td>2.973333</td>
    </tr>
  </tbody>
</table>
</div>



## Вывод

Сильная линейная зависимость (с коэффициентром корреляции 0.96) наблюдается классом давлением в шинах и показаниями инклинометра. При превышении наклоном определенного придела давление в шинах значительно повышается. Можно так же заметить, все самосвалы, относящиеся к 1 классу аварийности имели дваление в шинах в пределе 1-2 атм (соответственно работая на в основном в несложных условиях небольших наклонов). Можно сделать вывод, что большая часть аварий, приходится на самосвалы, работающие в более тяжелых условиях. Средний показатель давления в шинах для самосвала отнесенного ко второму классу аварийности - 4.26, для самосвала третьего класса - 5.55. Это хороший показатель для предсказания итогового класса аварийности. С ростом наклона (и соответвующего ему давления в шинах) повышается и класс аварийности. 

Некоторая линейная зависимость (с коэффициентром корреляции 0.78) наблюдается между классом аварийности и температурой окружающего воздуха. Чем выше температура, тем выше класс аварийности, однако этот показатель не такой силен, при одинаковой температуре встречаются самосвалы всех трех классов аварийности. Ко всему прочему температура - параметр который практически невозможно контролировать. Несмотря на это она может помочь предсказать класс аварийности.

Зависимость класса опасности от скорости нелинейна. Однако можно заметить, что самосвалы 1 класса как правило имеют большую среднюю скорость (равную 3.44). Как было выяснено ранее, это может быть связано с тем, что самосвалы 1 класса работают в более легких условиях небольшого наклона и способны достичь более высокой скорости в принципе. Однако при сложных условиях наблюдается зависимость между скоростью и аварийностью. Самосвалы 3 класса двигались со средней скоростью 3, что выше среднего показателя скорости у самосвалов второго класса (составляющей 2.8). В данном случае более высокая скорость коррелирует с повышением класса аварийности. 

**Практические решения:** Наиболее сильно аварийность самосвалов повышается при работе в тяжелых условиях, характеризующихся высокими показаниями инклинометра, ведущими к повышенному давлению в шинах. Для понижения общего уровня аварийности следует своевременно выполненять вертикальную планировку промплощадки, облегчая тем самым условия для ведения работ. В реальных условиях угольного разреза, изобилующего съездами и имеющего высокие темпы ведения горных работ, зачастую невозможно значительно улучшить производственные условия. В этом случае рекомендуется ограничить допустимую скорость движения самосвалов при работе в тяжелых условиях. Это поможет уменьшить число самосвалов, отнесенных к 3 классу аварийности.   
Так же некоторое улучшение может наступить, если проводить ротацию техники, работающую в различных условиях. Это позволит распределить загруженность ремонтных бригад по времени. 


___

## Шаг 2. Построение аналитических моделей. 

Исходные данные содержат небольшое число записей, каждая из которых состоит из 4 численных параметров и 1 метки класса. Классы хорошо различимы даже визуально (например, хорошо видно, что к 1 классу относятся только машины, уровень давления в шинах которых не превышает 2 атм). Для таких данных хорошо подходят простые алгоритмы классификации, вроде k - ближайшах соседей или деревья решений. 

Обучим несколько алгоритмов и сравним полученные результаты. В качестве моделей будут использованы метод k - ближайших соседей, дерево решений и ансамблевый алгоритм random forest (случайный лес).  

В работе использована библиотека scikit-learn.

- Для выполнения контроля за переобучением разобъем имеющийся датасет на тренировочную и тестовую часть. 


```python
from sklearn.model_selection import train_test_split
```

Разделим датасет на две части X, содержащий телеметрические показатели и y, содержащий метки классов. 


```python
X = data.iloc[:, 0:4].values
y = data.iloc[:, 4].values
```

Случайным образом разделим данные на тренировочную и тестовую часть. Размер тестовой части ввиду небольного количества исходных данных примем равным 20% от общего размера датасета.


```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

- Первый исползуемый алгоритм - метод k-ближайших соседей.


```python
from sklearn.neighbors import KNeighborsClassifier
```

В данный момент неизвестно, какие параметры будут наиболее удачными, поэтому примем необходимое число соседей для определения метки класса равным 5. 


```python
knn = KNeighborsClassifier(n_neighbors=5)
```

Обучим модель на тренировочных данных. 


```python
knn.fit(X_train, y_train)
```




    KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='minkowski',
                         metric_params=None, n_jobs=None, n_neighbors=5, p=2,
                         weights='uniform')



Сделаем прогнозы для тестовой выборки. В качестве метрики для определения качества прогноза здесь и далее используем accuracy score - долю правильных ответов. 


```python
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
```


```python
knn_pred = knn.predict(X_test)
```


```python
confusion_matrix(y_test, knn_pred)
```




    array([[12,  0,  0],
           [ 0,  7,  1],
           [ 0,  0,  7]], dtype=int64)




```python
accuracy_score(y_test, knn_pred)
```




    0.9629629629629629



Как видно даже самым простым и ненастроенным алгоритмом удалось добиться хорошей точности в 96,3%

Однако мы можем так-же настроить параметры модели с помощью кросс-валидации. В качестве настраиваемог параметра возьмем необходимое число соседей. Для каждого значения в диапазоне от 1 до 10 будет проведена 5-кратная кросс-валидация, и определено наиболее подходящее значение параметра.


```python
from sklearn.model_selection import GridSearchCV
```


```python
knn_params = {'n_neighbors': range(1, 10)}
```


```python
knn_new = KNeighborsClassifier(n_neighbors=5)
```


```python
knn_grid = GridSearchCV(knn_new, knn_params,
                         cv=5, n_jobs=-1,
                        verbose=True)
```


```python
knn_grid.fit(X_train, y_train)
```

    Fitting 5 folds for each of 9 candidates, totalling 45 fits
    

    [Parallel(n_jobs=-1)]: Using backend LokyBackend with 8 concurrent workers.
    [Parallel(n_jobs=-1)]: Done  45 out of  45 | elapsed:    1.6s finished
    




    GridSearchCV(cv=5, error_score=nan,
                 estimator=KNeighborsClassifier(algorithm='auto', leaf_size=30,
                                                metric='minkowski',
                                                metric_params=None, n_jobs=None,
                                                n_neighbors=5, p=2,
                                                weights='uniform'),
                 iid='deprecated', n_jobs=-1,
                 param_grid={'n_neighbors': range(1, 10)}, pre_dispatch='2*n_jobs',
                 refit=True, return_train_score=False, scoring=None, verbose=True)




```python
knn_grid.best_params_, knn_grid.best_score_
```




    ({'n_neighbors': 7}, 0.980952380952381)



Лучшие результаты модель показывает при параметре n_neighbors равном 7


```python
knn_grid_pred = knn_grid.predict(X_test)
```


```python
confusion_matrix(y_test, knn_grid_pred)
```




    array([[12,  0,  0],
           [ 0,  7,  1],
           [ 0,  0,  7]], dtype=int64)




```python
accuracy_score(y_test, knn_grid_pred)
```




    0.9629629629629629



Видно, что настройка не повлияла на итоговое качество модели. 

- Второй исползуемый алгоритм - дерево решений.


```python
from sklearn.tree import DecisionTreeClassifier
```

Удачные параметры дерева нам неизвестны, поэтому зададим параметры по умолчанию, ограничивая при этом максимальную глубину дерева для избежания переобучения.


```python
tree = DecisionTreeClassifier(random_state=42, max_depth=5)
```


```python
tree.fit(X_train, y_train)
```




    DecisionTreeClassifier(ccp_alpha=0.0, class_weight=None, criterion='gini',
                           max_depth=5, max_features=None, max_leaf_nodes=None,
                           min_impurity_decrease=0.0, min_impurity_split=None,
                           min_samples_leaf=1, min_samples_split=2,
                           min_weight_fraction_leaf=0.0, presort='deprecated',
                           random_state=42, splitter='best')



Сделаем прогнозы по тестовой выборке на основе обученой модели. 


```python
tree_pred = tree.predict(X_test)
```


```python
confusion_matrix(y_test, tree_pred)
```




    array([[12,  0,  0],
           [ 0,  7,  1],
           [ 0,  0,  7]], dtype=int64)




```python
accuracy_score(y_test, tree_pred)
```




    0.9629629629629629



Процент верных предсказаний равен с полученым с помошью метода k-ближайших соседей. 

Произведем настройку параметров дерева. Максимальная глубина и минимальное число элементов в листе настраивается на 5-кратной кросс-валидации.


```python
tree_params = {'max_depth': list(range(1, 5)), 
               'min_samples_leaf': list(range(1, 5)),
                'min_samples_split': list(range(1, 5))}
```


```python
tree_new = DecisionTreeClassifier(max_depth=5, random_state=42)
```


```python
tree_grid = GridSearchCV(tree_new, tree_params,
cv=5, n_jobs=-1,
verbose=True)
```


```python
tree_grid.fit(X_train, y_train)
```

    [Parallel(n_jobs=-1)]: Using backend LokyBackend with 8 concurrent workers.
    

    Fitting 5 folds for each of 64 candidates, totalling 320 fits
    

    [Parallel(n_jobs=-1)]: Done  56 tasks      | elapsed:    0.0s
    [Parallel(n_jobs=-1)]: Done 320 out of 320 | elapsed:    0.2s finished
    




    GridSearchCV(cv=5, error_score=nan,
                 estimator=DecisionTreeClassifier(ccp_alpha=0.0, class_weight=None,
                                                  criterion='gini', max_depth=5,
                                                  max_features=None,
                                                  max_leaf_nodes=None,
                                                  min_impurity_decrease=0.0,
                                                  min_impurity_split=None,
                                                  min_samples_leaf=1,
                                                  min_samples_split=2,
                                                  min_weight_fraction_leaf=0.0,
                                                  presort='deprecated',
                                                  random_state=42,
                                                  splitter='best'),
                 iid='deprecated', n_jobs=-1,
                 param_grid={'max_depth': [1, 2, 3, 4],
                             'min_samples_leaf': [1, 2, 3, 4],
                             'min_samples_split': [1, 2, 3, 4]},
                 pre_dispatch='2*n_jobs', refit=True, return_train_score=False,
                 scoring=None, verbose=True)




```python
tree_grid.best_params_, tree_grid.best_score_
```




    ({'max_depth': 2, 'min_samples_leaf': 1, 'min_samples_split': 2},
     0.9532467532467532)



После определения лучших параметров выполним предсказания. 


```python
tree_grid_pred = tree_grid.predict(X_test)
```


```python
confusion_matrix(y_test, tree_grid_pred)
```




    array([[12,  0,  0],
           [ 0,  7,  1],
           [ 0,  0,  7]], dtype=int64)




```python
accuracy_score(y_test, tree_grid_pred)
```




    0.9629629629629629



Как видно после настройки дерева параметры так же не изменились. 

- Следующий используемый классификатор - случайный лес. 


```python
from sklearn.ensemble import RandomForestClassifier
```

Случайный лес представляет собой совокупность решающих деревьев. Данные полученые по каждому дереву, входящему в состав леса усредняются. Этот способ должен показывать наибольшую точность. Количество деревьев в составе леса зададим равным 100.


```python
forest = RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=42)
```


```python
forest.fit(X_train, y_train)
```




    RandomForestClassifier(bootstrap=True, ccp_alpha=0.0, class_weight=None,
                           criterion='gini', max_depth=None, max_features='auto',
                           max_leaf_nodes=None, max_samples=None,
                           min_impurity_decrease=0.0, min_impurity_split=None,
                           min_samples_leaf=1, min_samples_split=2,
                           min_weight_fraction_leaf=0.0, n_estimators=100,
                           n_jobs=-1, oob_score=False, random_state=42, verbose=0,
                           warm_start=False)



Сделаем прогнозы по тестовой выборке. 


```python
forest_pred = forest.predict(X_test)
```


```python
confusion_matrix(y_test, forest_pred)
```




    array([[12,  0,  0],
           [ 0,  7,  1],
           [ 0,  0,  7]], dtype=int64)




```python
accuracy_score(y_test, forest_pred)
```




    0.9629629629629629




```python
logit_pred = logit_searcher.predict(X_test)
```


```python
accuracy_score(y_test, logit_pred)
```




    0.9629629629629629


