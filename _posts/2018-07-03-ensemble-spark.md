---
layout: post
title: "Ensemble de regressores no Spark"
date: 2018-07-03 10:00:00 -0300
description: Como montar um ensemble de regressores no Spark usando a api do Scala.
img: scala-spark.png
tags: [aprendizado supervisionado, supervised learning, Scala, Spark, Ensemble]
---

Vou tentar mostrar como fazer um ensemble de regressores na API de Scala do Spark. O exemplo é quase igual pra PySpark, se quiser que eu faça um post desse é só comentar aqui embaixo ;). Mas antes, alguns conceitos básicos.

## O que é ensemble?

Ensemble em aprendizado de máquina se refere à quando você usa um conjunto de modelos diferentes para uma previsão ou classificação. Seria como juntar os poderes de uma [regressão linear](http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html) com os de uma [Regressão de Ridge](http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html).

### Perae, onde eu já vi isso?

Ensembles são muito comuns em aprendizado de máquina, você provavelmente já ouviu falar de [Random Florest](http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html) (Floresta Aleatória). Pois é, ela é um ensemble! Nela se une o poder de predição de várias [Árvores de Decisão](http://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeRegressor.html) geradas por amostras aleatórias da base de dados. Se quiser mais detalhes pede o post lá embaixo!

## Por que Spark, agora que eu aprendi o scikit-learn?

A principal vantagem de utilizar Spark é a velocidade. Spark distribui o seu programa em um cluster paralelizando a execução dele. Isso é útil se você tiver um grupo de computadores (cluster) disponível para realizar o processamento. 

### É sempre mais rápido?

Não necessariamente. Organizar essa paralização tem um custo, chamado de *overhead*, o que pode fazer com que demore ainda mais processar em paralelo. Isso acontece usualmente em bases de dados pequenas, em que processar é tão rápido que um computador sozinho já teria acabado, enquanto o Spark ainda está organizando quem vai processoar o que.

## Por que Scala?

Spark tem várias APIs, ou seja, formas de ser utilizada. Atualmente possui suporte à R, Pyhton (PySpark), Java e Scala. Você pode utilizar a linguagem que você está mais confortável, mas minha impressão é que se você sabe o básico da linguagem o resto é mais entender a forma como as funções do Spark estão estruturadas do que qualquer outra coisa. A única diferença entre essas APIs é que as linguagens R e Python tem uma característica que não permite a implementação de **Datasets**, que são mais eficientes do que os Dataframes. Os Dataframes estão disponíveis em todas as APIs. Isso significa que você pode ter uma vantagem em velocidade usando Java ou Scala. Vale lembrar que as vezes é melhor ter o time todo usando PySpark, por exemplo, que gastar mais tempo debugando Java ou Scala do que se ganharia de velocidade.

## Devo largar tudo e aprender Scala e Spark?

Olha, minha sugestão é sempre começar pelo mais fácil e ir aprendendo conforme a necessidade. Eu mesmo estou aprendendo Scala e Spark para usar na [Rocketmat](http://rocketmat.com/), onde eu trabalho. Meu caminho foi aprender Python bem básico, quebrar a cabeça substituindo Excel pelo [pandas](https://pandas.pydata.org/) e depois usar [scikit-learn](http://scikit-learn.org/stable/index.html) pra modelos de aprendizado de máquina. Como Spark possui alguns detalhes diferentes no seu funcionamento, por exemplo o [lazy](https://en.wikipedia.org/wiki/Lazy_evaluation), que podem causar confusão em quem está usando pela primeira vez, decidi aprender primeiro o PySpark para depois aprender o básico de Scala e aí usar a API dela do Spark. Cada um tem um caminho diferente, esse apenas foi o meu. Cada um desses pontos pode ser facilmente expandido em posts ou mesmo séries de posts. Se tiver interessado em algum deixa um comentário ali embaixo!

## Agora vai

Ok! Agora que já vimos de forma rasa os conceitos que vamos utilizar hoje, partiu código. Primeiro vamos importar algumas classes.

Importamos a classe Pipeline que vai permitir que apliquemos algoritmos em ordem, efetivamente juntando em um único objeto todo o processo de previsão, do tratamento e processamento dos dados até o modelo de previsão.
```scala
import org.apache.spark.ml.Pipeline
```
Agora vamos importar dois modelos de aprendizado de máquina para regressão: a Regressão Linear Generalizada e a Floresta Aleatória.
```scala
import org.apache.spark.ml.regression.{GeneralizedLinearRegression, RandomForestRegressor}
```
Também importamos o VectorAssembler para criar o vetor das features a serem utilizadas nos nossos modelos.
```scala
import org.apache.spark.ml.feature.VectorAssembler
```
Importamos os dados em csv para o formato Dataset do Spark.
```scala
val data = reader.csv("data.csv")
```
Precisamos renomear a variável objetivo como **label** para que o pacote ml do Spark a reconheça automaticamente. Como Spark funciona com lazy evaluation, vamos criar um novo Dataset para forçar ele a realizar a mudança.
```scala
val new_data = data.withColumnRenamed("variável_objetivo", "label")
```
Definimos agora a instância do VectorAssembler que vai transformar as colunas no vetor **features**. Como somos muito criativos vamos chamar de assembler :P. Perceba que escolhemos as nossa features em setInputCols e definimos o nome do vetor em setOutputCol.
```scala
val assembler = new VectorAssembler()
  .setInputCols(Array("coluna1", "coluna2", "coluna3", "coluna4"))
  .setOutputCol("features")
```
Nesse nosso exemplo mínimo, vamos rodar uma Regressão Linear Generalizada nas features e colocar sua predição em uma coluna. Vamos também rodar uma Floresta Aleatória nas features e colocar sua predição em outra coluna.

Por fim, vamos rodar uma nova Floresta Aleatória usando como features as duas colunas de resultado dos dois modelos que rodamos anteriormente.

Vamos lá, passo a passo:

Definimos a Regressão Linear Generalizada da família gaussiana. Perceba que definimos que seu resultado deve ser adicionado à coluna **glr** no nosso Dataset. Por fim, imprimimos os parâmetros da glr, só para podermos ter uma ideia de quais são.
```scala
val glr = new GeneralizedLinearRegression()
  .setFamily("gaussian")
  .setPredictionCol("glr")
println("Generalized Logistic Regression parameter:\n" + glr.explainParams())
```
Agora vamos definir nossa primeira floresta. Vamos chamar ela de rf1, que não é um nome muito bom, então recomendo que você escolha um nome melhor quando for tentar reproduzir esse exemplo. Colocamos apenas 100 árvores pra ser rapidinho. Importante para depois: definimos que os resultados devem ficar na coluna também de nome rf1. Por fim, imprimimos os parâmetros desse modelo.
```scala
val rf1 = new RandomForestRegressor()
  .setNumTrees(100)
  .setPredictionCol("rf1")
println("Random Forest Regression parameter:\n" + rf1.explainParams())
```
Atém então tudo ok, definimos nossas features e nossos modelos. Entretanto, precisamo definir outro VectorAssembler para juntar os resultados dos dois modelos em um novo vetor de features que possa ser usado pelo nosso modelo final. É isso que fazemos a seguir, juntamos as colunas rf1 e glr, que possuem os resultados dos nossos dois modelos, e montamos um vetor chamado **ensemble\_features**.
```scala
val assembler2 = new VectorAssembler()
  .setInputCols(Array("rf1", "glr"))
  .setOutputCol("ensemble_features")
```
Agora falta definir nosso último modelo, será um regressor que usará os resultados dos outros regressores como features. Perceba que através do setFeaturesCol definimo que ele deve usar a "coluna" (na verdade vetor) **ensemble\_features** como feature!
```scala
val rf2 = new RandomForestRegressor()
  .setNumTrees(100)
  .setFeaturesCol("ensemble_features")
```
Ok, agora é só rodar um depois do outro né? Sim e não :P. É nesse momento que a classe Pipeline vem para brilhar. Nós definimos um pipeline que começa com a criação dos vetores de features roda os dois algoritmos, cria um vetor com seus resultados e, por fim, roda um último algoritmo usando esse vetor.
```scala
val pipe = new Pipeline()
  .setStages(Array(assembler, rf1, glr, assembler2, rf2))
```
Para treinar o modelo vamos precisar apenas da linha a seguir:
```scala
val model = pipe.fit(new_data)
```
Nem é tão difícil assim né? Existe uma função pipeline no scikit-learn, mas essa implementação seria um pouco diferente lá, pois também precisaríamos utilizar a função FeatureUnion, que é bem interessante. Gostaria de um post sobre isso? Já sabe, deixa um comentário embaixo!