---
title: GridSearchCV na tua cara!
excerpt: "GridSearchCV na prática!"
header:
  image: /assets/images/seek.jpg
categories: [code]
tags: [grid_search]
---
Beleza, a gente viu [como o usar a função cross_validate](https://gusrabbit.com/code/cross_validate/) semana passada então vamos continuar nosso passeio pelas lindezas do scikit-learn e dar uma olhada no GridSearchCV.
Essa função lindia permite que a gente teste um bando de combinação de parâmetros nos nossos modelos, facilitando a gente achar o melhor.

Ela é parecida com a cross_validate mas eu ainda prefiro essa, vocês vão ver o por que.

#Código

Todo o código ta [nesse notebook](https://github.com/gusrabbit/blog-examples/blob/master/GridSearchCV.ipynb) pra quem quiser baixar e ir acompanhando.Vamos lá!
Como sempre, a gente começa importando tudo que vamos usar:

```python
import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.model_selection import GridSearchCV
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression
```
Vamos carregar os dados do dataset de bosto do scikit. Esse dataset tem características e preços de casas em boston, vamos usar para fazer uma regressão! Se você apertar **tab** depois do ponto você vai conseguir ver outros datasets que já vem no scikit. Escolhemos esse dataset pois ele só tem duas categorias e é mais simples.

```python
housing = datasets.load_boston()
```

Que nem da [outra vez](https://gusrabbit.com/code/cross_validate/), a gente separa a variável objetivo das features:
```python
X = housing.data
y = housing.target
```
Vamos pegar um modelo basicão, o DummyRegressor. Ele é um regressor cuja previsão é aleatória. É legal usar esses modelos aleatórios como baseline para comparar a qualidade dos nossos modelos. No mínino tem que ser melhor que isso. Vamos instanciar ele:

```python
baseline = DummyRegressor()
```
De acordo com a [documentação](http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html) ele precisa dos parâmetros a seguir:

GridSearchCV(estimator, **param_grid**, **scoring=None**, fit_params=None, n_jobs=1, iid=True, refit=True, cv=None, verbose=0, pre_dispatch=‘2\*n_jobs’, error_score=’raise’, return_train_score=’warn’)

Mas primeiro! Vamos definir as métricas que queremos medir, lembrando que todas estão no [link do amor](http://scikit-learn.org/stable/modules/model_evaluation.html#scoring-parameter). Vamos definir uma lista com todas as de regressão:



```python
metricas = ['explained_variance', 'neg_mean_absolute_error', 'neg_mean_squared_error', 'neg_mean_squared_log_error', 'neg_median_absolute_error', 'r2']
```
Lembrando que o tchan do Grid Search é exatamente ele fazer os cross validation de vários modelos com hiperparâmetros diferentes de uma vez. Então vamos definir o **param_grid** que vai definir quais modelos com quais parâmetros o nosso Grid Search vai rodar.

Se você olhar na [documentação](http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html) ela diz que pode ser um dicionário ou uma lista de dicionários, qual a ideia?

A vibe é definir um dicionário com os parâmetros do nosso estimador. Vamos usar o método get_params pra ver o nome certinho dos parâmetros do nosso DummyRegressor:

```python
baseline.get_params().keys()
Out: dict_keys(['constant', 'quantile', 'strategy'])
```
Essas são as chaves do nosso dict, os valores são aqueles que queremos que o Grid Search rode. Criamos um dict assim:

```python
hyper = {'strategy':['mean', 'median', 'quantile', 'constant'],
         'quantile':[.75],
         'constant':[300000]}
```

Dessa forma, o nosso Grid Search vai rodar quatro modelos, cada um com uma das estratégias. Isso acontece, pois os parâmetros quantile e constant só são usados naquelas duas estratégias. Vamos definir o Grid Search com tudo que a gente montou até agora e com *verbose=100* pra ele ir falando o que ele ta fazendo passo a passo. Vamos mandar ele fazer o refit no *neg_mean_squared_error*, mas só porque esse é a minha métricas preferida. Isso significa que ele vai escolher o modelo com o menor erro quadrado médio. E por último, a gente pede pra n retornar os scores de treinamento porque eles são meio inúteis.

```python
meu_primeiro_grid = GridSearchCV(baseline, param_grid=hyper, scoring=metricas, verbose=100, refit='neg_mean_squared_error', return_train_score=False)
```

Por padrão, essa função vai separar o dataset em 3 partes e vai devolver a métrica média de cada uma dessas 3. No futuro eu vou postar sobre Kfold e como podemos separar os dados em várias partes para fazer o cross validation como aqui. Tudo isso é controlado pelo parâmetro *cv*. Agora vamos mandar o Grid Search rodar nos dados de boston!

```python
meu_primeiro_grid.fit(X,y)
```
Se você ta rodando junto, vai ver todos os detalhes do que o Grid Search ta fazendo no output do fit!
É legal usar o verbose porque as vezes você ta rodando tantos modelos que demoram tanto que você fica na dúvida se travou : P.

Agora vamos dar uma olhada nos resultados, o melhor estimador foi:

```python
meu_primeiro_grid.best_estimator_
Out: DummyRegressor(constant=300000, quantile=0.75, strategy='median')
```
E o melhor score foi:

```python
meu_primeiro_grid.best_score_
Out: -103.223814229249
```
Que diabéisso? Ok, como a gente colocou *neg_mean_squared_error* no refite ele está devolvendo o melhor score dessa métrica. Essa métrica da a média do quadrado dos erros, só q negativa. Por que negativa? Acho que é pra ficar mais fácil de minimizar, aí eles adotam o padrão de colocar as métricas tudo negativa. O erro é a diferença entre o que o modelo preveu e o valor verdadeiro da casa. Aqui como a melhor estratégia foi a mediana, a previsão do modelo foi a mediana dos valores de treinamento pra toda casa nova. A gente pega a diferença do prebisto pelo verdadeiro, eleva ao quadrado (normal em estatística pra evitar que erros negativos e positivos se anulem, e é mais fácil de derivar do que usar módeulo).

Então pra ter uma ideia se isso ta bom ou ruim basta tirar a raiz disso:

```python
np.sqrt(meu_primeiro_grid.best_score_*-1)
Out: 10.159912117201063
```
Em média nosso modelo errou o preço por 10 mil dólares. O Grid Search retorna um dicionário com todos os resultados, eu gosto de transformar ele num dataframe pra ficar mais fácil de ver (antes vou usar uma manhã do pandas pra que ele mostre todas as colunas):

```python
pd.set_option('max_columns',200)
pd.DataFrame(meu_primeiro_grid.cv_results_)
```
![resultados](/assets/images/df.png)


Vamos ver se uma regressão linear simples fica melhor? Bora instanciar!

```python
ols = LinearRegression()
```


```python
ols.get_params().keys()
Out: dict_keys(['copy_X', 'fit_intercept', 'n_jobs', 'normalize'])
```
Os únicos parâmetros interessantes para a gente são o *fit_intercept* e o *normalize*:

```python
ols_params = {'fit_intercept':[True, False],
              'normalize':[True, False]}
```
Vamos definir o Grid Search!
```python
meu_segundo_grid = GridSearchCV(ols, param_grid=ols_params, scoring=metricas, verbose=100, refit='neg_mean_squared_error', return_train_score=False)
```

```python
meu_segundo_grid.fit(X, y)
```

```python
meu_segundo_grid.best_estimator_
Out: LinearRegression(copy_X=True, fit_intercept=True, n_jobs=1, normalize=False)
```

```python
meu_segundo_grid.best_score_
Out: -168.08917760165718

```


```python
np.sqrt(meu_segundo_grid.best_score_*-1)
Out: 12.964921041088418
```
A regressão ta dando um erro de 12!!! Pior do que sempre chutar a mediana! 

Um beijo e um queijo pra quem colocar no github o código de uma regressão melhor que essas duas! Posto o nome e foto da pessoa aqui!

Qualquer dúvida é só deixar nos comentários ali embaixo! Até a próxima!