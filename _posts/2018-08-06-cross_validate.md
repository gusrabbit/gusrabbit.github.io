---
title: Como fazer cross-validation
excerpt: "Cross-validation na prática!"
header:
  image: /assets/images/kfold.jpg
categories: [code]
tags: [cross_validate]
---

Agora que você rodou seu algortimo você deve estar se perguntando: "E agora? Como eu sei se ele foi uma boa escolha?"
Esse é o momento em que vamos olhar para as métricas da regressão ou classificação que fizemos. 
Essas métricas medem quão bem o modelo prevê a sua variável objetivo. 
Portanto, todas as métricas são calculadas comparando o que o modelo preveu com o que realmente aconteceu.

Mas eu vou ter que esperar até descobrir o que aconteceu de verdade para saber se as previsões foram boas?
Não, pra evitar que a gente coloque um modelo que a gente não sabe a qualidade em produção usamos a técnica de dividir
os dados em [treinamento e teste][blog_train_test], lembra?

Poxa, mas da um super trabalho ficar dividindo o dataset e depois tenho que calcular na mão as métricas?
O scikit-learn já tem as funções, você só tem que colocar o y_predito e o y_real que ele calcula.

Mas ainda ta muito complicado.

Sorte a nossa que a galera do scikit pensou nisso e criou a função [cross_validate][cross_validate]! 
Nesse post vamos dar uma olhada em como usa-la.

## Dataset

Vamos escolher um dataset de fácil acesso para que todos consigam acompanhar.
O scikit mesmo já vem com vários datasets, então vamos pegar um desses mesmo.

Eu deixei um notebook de exemplo [aqui nesse repositório][repo_exemplo], se você quiser baixar para acompanhar.

Primeiro importamos o que vamos usar:
```python
from sklearn import datasets
from sklearn.model_selection import cross_validate
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import RidgeClassifier
```
Vamos carregar os dados do dataset de breast_cancer do scikit, se você apertar **tab** depois do ponto você vai conseguir ver outros datasets que já vem no scikit. Escolhemos esse dataset pois ele só tem duas categorias e é mais simples.

```python
cancer = datasets.load_breast_cancer()
```
O scikit sempre pede pra a gente separar a variável objetivo das features, por isso já vem com essas propriedades:
```python
X = cancer.data
y = cancer.target
```
No futuro vou subir uns exemplos de como importar e tratar arquivos csv ou xlsx de excel.

Vamos escolher um modelo para usar nesse exemplo, que tal o DummyClassifier? Ele é um classificador aleatório, nosso objetivo em qualquer classificação é ficar pelo menos melhor que ele (sim, tem como treinar um modelo pior que o aleatório : P). No scikit a gente tem que instanciar (um dia eu explico) o classificador assim:


```python
classificador_burrao = DummyClassifier()
```

Agora a variável 'classificador_burrao' tem as propriedade do DummyClassifier()! Pronto, agora vamos chamar o cross_validate. Como podemos ver na [documentação](http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.cross_validate.html) ele precisa dos parâmetros a seguir:

cross_validate(estimator, X, y=None, groups=None, scoring=None, cv=None, n_jobs=1, verbose=0, fit_params=None, pre_dispatch=‘2*n_jobs’, return_train_score=’warn’)

Portanto, vamos chamar ele da seguinte forma:


```python
cross_validate(classificador_burrao, X, y, return_train_score=False,
               scoring=['accuracy',
                        'average_precision',
                        'f1',
                        'precision',
                        'recall',
                        'roc_auc'])
```

```
Out:
{'fit_time': array([0.00058603, 0.00030708, 0.00035906]),
 'score_time': array([0.00486493, 0.00419879, 0.00733495]),
 'test_accuracy': array([0.53157895, 0.50526316, 0.56084656]),
 'test_average_precision': array([0.61172048, 0.65517671, 0.62575387]),
 'test_f1': array([0.67226891, 0.65271967, 0.61802575]),
 'test_precision': array([0.64393939, 0.57627119, 0.61206897]),
 'test_recall': array([0.59663866, 0.52941176, 0.58823529]),
 'test_roc_auc': array([0.49674518, 0.48698071, 0.44285714])}
```

Não vou entrar em todos os detalhes de todos o parâmetros dessa função, mas o scoring é a alma do negócio, é nele que a gente fala quais métricas a gente quer que a função retorne. Da pra ver a lista de todas as métricas do scikit [nesse link](http://scikit-learn.org/stable/modules/model_evaluation.html#scoring-parameter), basta copiar e colar os nomes das métricas de classificação em uma lista, que foi o que eu fiz acima.

```python
metricas = ['accuracy', 'average_precision', 'f1', 'precision', 'recall', 'roc_auc']
```

Por padrão, essa função vai separar o dataset em 3 partes, por isso ela devolver 3 valores para cada métrica. No futuro eu vou postar sobre Kfold e como podemos separar os dados em várias partes para fazer o cross validation como aqui.

Dessa forma, vemos que a acurácia do nosso classificador foi 52% no primeiro teste, 50% e 56% no segundo e no terceiro. É interessante pegar a média dos 3 testes, então seria uma precisão de 53% em média.

Seria legal se ele já devolvesse a média direto pra a gente né? Na veradade tem outra função que faz isso, a [GridSearchCV](http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html). Além disso ela ainda deixa a gente rodar várias opções de  parâmetros diferentes e retorna o melhor modelo. Vou postar sobre ela no futuro, acaba que ela é a que a gente mais usa quando ta trabalhando mesmo.

Bom como você viu é muito fácil fazer o cross_validate, tão fácil que agora a gente pode pegar um classificador "de verdade" pra comparar:

```python
ridge = RidgeClassifier()
```

```python
cross_validate(classificador_burrao, X, y, return_train_score=False,
               scoring=metricas)
```

```
Out:
{'fit_time': array([0.00080085, 0.00034523, 0.0003171 ]),
 'score_time': array([0.00347614, 0.00404191, 0.00635886]),
 'test_accuracy': array([0.52631579, 0.63157895, 0.47089947]),
 'test_average_precision': array([0.63973775, 0.60963543, 0.64744377]),
 'test_f1': array([0.59574468, 0.63374486, 0.64347826]),
 'test_precision': array([0.61261261, 0.64516129, 0.60629921]),
 'test_recall': array([0.68067227, 0.67226891, 0.66386555]),
 'test_roc_auc': array([0.53751923, 0.60078116, 0.48487395])}
```

Ele tem uma acurácia média de 54%, um pouquinho melhor do que o nosso classificador aleatório. No futuro vamos dar uma olhada no significado dessas métricas e em como conseguir classificações ainda melhores. Se você rodar essa função várias vezes vai perceber que ela retorna valores diferentes, isso é por que os cortes do dataset são feitos aleatoriamente. Quando formos falar de kFold vamos ver como garantir que sempre cheguemos aos mesmos resultados.

[blog_train_test]:https://gusrabbit.com/intuition/treino-teste/
[cross_validate]:http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.cross_validate.html
[melhor_livro]:https://www.packtpub.com/big-data-and-business-intelligence/python-machine-learning-second-edition
[repo_exemplo]:https://github.com/gusrabbit/blog-examples/blob/master/cross_validate.ipynb