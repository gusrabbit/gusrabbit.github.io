---
layout: post
title: "Treino é treino, teste é teste"
date: 2018-01-26 17:20:00 -0300
description: Pra que e por que dividir meu dataset em teste e treino?
img: treino.jpg
tags: [Train-test]
---
E aí pessoal!

Esse é o primeiro de uma série de posts sobre machine learning que eu pretendo fazer. 
A ideia é ir de conceito a conceito até conseguirmos fazer uma predição. 
Em todos os posts vou usar o famoso dataset do tutorial do [Titanic do Kaggle](https://www.kaggle.com/c/titanic), então:

### Cuidado! Spoilers à frente!

Se você está pensando "perae gustavo, você ta começando meio que pelo meio já, né?". Você está certa! Mas relaxa que a ideia é que os posts 
se completem, mas provavelmente vou soltar eles fora da ordem : P.

## Qual a ideia?

A divisão em datasets de treino e teste é parte da escolha e otimizaçào de modelos, ou sea, escolher o modelo certo (ou um ensemble de modelos)
 ou os melhores parâmetros para o mesmo. 

A ideia é dividir o dataset em duas partes, uma pra treinar e outra pra testar e ter
uma noção da qualidade das nossas previsões (e, portanto, 
do nosso modelo) antes de enviar nosso "chute" pro Kaggle dar a nota 
(ou antes de cologar em produção, se você for gente grande).

## Comofaz?
Simples! A gente pega a base de dados e divide na metade,
 a metade de cima vai pra treino e a de baixo para o teste.
 
## Opa perae!
Pegadinha do malandro! Antes de continuar lendo tente pensar por que
dividir no meio talvez não seja uma ideia tão boa.

-------
Então, existem alguns problema possíveis. Por exemplo, se eu tiver o meu DataFrame
 ordenado do maior para o menor na variável idade. Aí meu algoritmo
vai treinar só nos velinhos, e na hora de prever se os novinhos sobreviveram, ele não 
vai se dar tão bem (lembrando que mulheres e crianças primeiro!).

Então #comofaz? Lembrando lá de estatística, a gente ta pegando uma "amostra"
do dataset. Então seria interssante a gente pegar uma amostra representativa para 
o nosso algoritmo ter uma boa noção do nosso dataset. E o que nossa professora de estatistica
sempre falava?

## Vamos pegar amostras aleatórias!

Nós queremos dividir em duas partes, divididas aleatoriamente
para que elas representem bem. E olha só que beleza! A biblioteca 
scikit-learn já tem uma função pronta pra isso! Basta escrever o código a seguir:

{% highlight python %}

import pandas as pd

train = pd.read_csv('train.csv')

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(vaiq,
                                                    train['Survived'],
                                                    test_size=0.3,
                                                    random_state=42)
	
{% endhighlight %}
