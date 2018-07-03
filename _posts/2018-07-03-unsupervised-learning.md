---
layout: post
title: "Aprendizado não supervisionado"
date: 2018-07-02 10:00:00 -0300
description: Como agrupar dados.
img: unsupervised.jpeg
tags: [aprendizado não supervisionado, unsupervised learning, unsupervised, pca, cluster]
---

Diferente do aprendizado supervisionado em que temos as respostas ou *ground truths* do passado relacionadas ao que estamos tentando prever, o aprendizado não supervisionado vem para responder a pergunta: 

### Como eu faço se eu não tenho as respostas?

![home alone](https://media.giphy.com/media/yYCXPHoEHTn0I/giphy.gif)

Nesse caso, temos duas opções:

1. Classificar cada ponto dos nossos dados com o que queremos prever, como "bom" ou "ruim" etc.
2. Deixar os dados falarem por si só.

Eu considero a opção 1 ruim por alguns motivos:

- Se o for uma quantidade de dados grande dá trabalho.
- Mesmo que que haja um critério objetivo de classificação podem haver problemas na classificação humana.
- O custo benefício não vai valer a pena.

Nesse post vamos discutir sobre a opção 2. Essa opção representa a área de aprendizado de máquina conhecida como [aprendizado não supervisionado](http://scikit-learn.org/stable/unsupervised_learning.html). Nela, basicamente o computador vai olhar para os dados e aprender o que eles dizem sobre si mesmos.

## Agrupamento (Clustering)

Existem várias técnicas de [agrupamento](http://scikit-learn.org/stable/modules/clustering.html). Elas basicamente agrupam os dados de acordo com sua proximidade no espaço vetorial. Se X está mais perto de Z do que de Y, X e Z pertecerão ao grupo 1 enquanto Y ao grupo 2. É fácil pensar nesse exemplo em apenas uma dimensão:

![one dimensional clustering](https://planspace.org/20150520-practical_mergic/img/one_dimensional.png)
[Fonte da figura](https://planspace.org/20150520-practical_mergic)

A ideia é que os pontos sejam semelhantes aqueles dentro do mesmo grupo e diferentes aos pertencentes a outros grupos. Os vários métodos de agrupamento buscam minimizar a distância entre os pontos do mesmo grupo maximizando a distância entre os diferentes grupos de várias formas diferentes. Se um post discutindo alguns desses algoritmos lhe interessa deixe um comentário!

## Redução de dimensionalidade ([PCA](http://scikit-learn.org/stable/modules/decomposition.html#principal-component-analysis-pca), [FA](http://scikit-learn.org/stable/modules/decomposition.html#factor-analysis))

A [redução de dimensionalidade](http://scikit-learn.org/stable/modules/decomposition.html) busca decompor os sinais em componentes. Isso pode ser interessante por vários motivos:

1. Caso você queira evitar a [maldição da dimensionalidade](https://en.wikipedia.org/wiki/Curse_of_dimensionality), relevante principalmente em classificação e agrupamento.
2. Caso você tenha muitas variáveis (features) e um poder de processamento limitado.
3. Caso você precise deixar suas variáveis ortogonais (sem relação uma com a outra) pra satisfazer a hipótese de um dos seus modelos.
4. Pra facilitar a interpretação de uma quantiadade grande de dados, criando uma espécie de indicador, no caso da Análise Fatorial (Factor Analysis).

E aí, o que vocês acham de aprendizado não supervisionado? Tem muitos mais métodos e detalhes a entrar, podemos ir expandindo nisso com o tempo.

Até mais!
