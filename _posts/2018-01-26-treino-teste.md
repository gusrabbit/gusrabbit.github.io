---
layout: posts
title: Treino é treino, teste é teste
excerpt: "Pra que e por que dividir meu dataset em teste e treino?"
header:
  image: /assets/images/treino.jpg
categories: [intuition]
tags: [cross validation]
---


---
layout: post
title: "Treino é treino, teste é teste"
date: 2018-02-28 15:00:00 -0300
description: Pra que e por que dividir meu dataset em teste e treino?
img: treino.jpg
tags: [Train-test]
---
E aí pessoal!

Esse é o primeiro de uma série de posts sobre machine learning que eu pretendo fazer. 
A ideia é ir de conceito a conceito até conseguirmos fazer uma predição. 
Em todos os posts vou usar o famoso dataset do tutorial do [Titanic do Kaggle][desafio do titanic], então:

### Cuidado! Spoilers à frente!

Se você está pensando "perae gustavo, você ta começando meio que pelo meio já, né?". Você está certa! Mas relaxa que a ideia é que os posts 
se completem, mas provavelmente vou soltar eles fora da ordem : P.

## Qual a ideia?

A divisão em datasets de treino e teste é parte da escolha e otimização de modelos, ou seja, escolher o modelo certo (ou um ensemble de modelos)
 ou os melhores parâmetros para o mesmo. 

A ideia é dividir o dataset em duas partes, uma pra treinar e outra pra testar e ter
uma noção da qualidade das nossas previsões (e, portanto, 
do nosso modelo) antes de enviar nosso "chute" pro Kaggle dar a nota. 
Ou antes de colocar em produção, se você for gente grande.

## #Comofaz?
Simples! A gente pega a base de dados e divide na metade,
 a metade de cima vai pra treino e a de baixo para o teste.
 
## Opa perae!
Pegadinha do malandro! Antes de continuar lendo tente pensar por que
dividir no meio talvez não seja uma ideia tão boa.

-------
Então, existem alguns problemas possíveis. Por exemplo, se eu tiver o meu DataFrame
 ordenado do maior para o menor na variável idade. Aí meu algoritmo
vai treinar só nos velinhos, e na hora de prever se os novinhos sobreviveram, ele não 
vai se dar tão bem (lembrando que mulheres e crianças primeiro!).

Então #comofaz? Lembrando lá de estatística, a gente ta pegando uma "amostra"
do dataset. Então seria interessante a gente pegar uma amostra representativa para 
o nosso algoritmo ter uma boa noção do nosso dataset. E o que nossa professora de estatística
sempre falava?

## Vamos pegar amostras aleatórias!

Nós queremos dividir aleatoriamente em duas partes, 
para que elas representem bem o nosso dataset. E olha só que beleza! A biblioteca 
scikit-learn já tem uma função pronta pra isso! Basta escrever o código a seguir:


```python
import pandas as pd

train = pd.read_csv('train.csv')

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(train.drop('Survived',
                                                    axis=1),
                                                    train['Survived'],
                                                    test_size=0.3,
                                                    random_state=42)
	
```

Nesse código importamos a biblioteca pandas na primeira linha. Na segunda, pedimos que ela leia o arquivo train.csv e transforme num Data Frame com nome train.

Na terceira linha importamos a função train_test_split do módulo model_selection da biblioteca scikit-learn. Na quarta linha associamos quatro variáveis ao resultado da função.

Você pode olhar a [documentação da função][documentacao do traintest split] para ver os detalhes, mas resumidaemente ela vai dividir o seu DataFrame em duas partes, a de teste e a de treino (já vou explicar pq isso gera 4 dfs). Como você pode ver os dois primeiros parâmetros da função são o dataset sem a coluna de interesse (por isso usamos o .drop que tira a coluna 'Survived') e a coluna de interesse separada do resto dos dados.

Isso é apenas pro scikit entender qual é a coluna de interesse. O que ele vai fazer é dividir seus dados em duas partes: 70% para um treino, 30% para teste. Você pode ajusta isso no parâmetros test_size. Podemos falar mais sobre essas proporções no futuro.

## Não se confunda!

A ordem que escrevemos o código costumava me confundir um pouco sobre o que eu tinha que treinar e o que testar. Nós vamos treinar o x_train usando o y_train, ou seja, vamos falar pro algoritmo q o y_train é a coluna de sobreviventes das pessoas no x_train. Depois vamos dar o predict no y_train, o algoritmo vai predizer quem do y_train morreu ou viveu, e comparar o resultado (y_predicted) com o y_test, que é a "resposta certa" para calcular os scores.

Nos próximos posts falaremos melhor sobre como importar os dados, sobre o desafio do titanic em si, e sobre como treinar e testar modelos de Machine Learning. Até lá!

[documentacao do traintest split]:http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html
[desafio do titanic]:https://www.kaggle.com/c/titanic