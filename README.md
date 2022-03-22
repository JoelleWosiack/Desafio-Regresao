# Desafio Regressão Linear

## Contexto analítico

<br>
Neste desafio, você deve resolver um case de precificação de imóveis. Esse desafio foi construído em parceria entre a Tera e o QuintoAndar, onde o objetivo é simular um projeto de machine learning com características semelhantes ao que ocorre no dia a dia da empresa.
<br>
Imagine-se na seguinte situação: a área de marketing do QuintoAndar quer montar uma calculadora de preço (como esta aqui: https://mkt.quintoandar.com.br/quanto-cobrar-de-aluguel/), e nesse projeto, os analistas negociais e corretores querem, também, entender as principais variáveis e características chaves que influenciam no valor de venda do imóvel (Ex: quantificar o impacto do aumento da área do imóvel no preço, ou quantificar o impacto de ter piscina, ou não no preço). Você é o cientista de dados que atuará na resolução desse case.
<br>
Para tanto, existem dois objetivos principais:
<br>
Objetivo 1, interpretabilidade: construir uma regressão linear simples, com poucas variáveis importantes, visando gerar insights para corretores e proprietários no quesito precificação dos imóveis. Ou seja, o foco será na interpretação dos coeficientes (ex: se aumentar a área do imóvel em uma unidade isso irá aumentar em Y o preço deste imóvel).
<br>
Objetivo 2, predição: construir um modelo com alto poder preditivo, com mais variáveis, visando um bom desempenho e com o intuito de ser usado em uma página web como a calculadora de preço. Note que, em uma situação real, um alto erro de inferência pode gerar grande insatisfação em um proprietário de imóvel, que pode ficar ofendido com o resultado. Por isso, em casos como esse, queremos ter o menor erro possível, mesmo que o modelo seja complexo e tenha uma interpretação mais difícil.
<br>

## Base de dados

<br>
O conjunto de dados descreve a venda de propriedades residenciais individuais de uma cidade americana, de 2006 a 2010. O conjunto de dados contém 2.930 observações e um grande número de features (23 nominais, 23 ordinais, 14 discretas e 20 contínuas) envolvidas na avaliação do valor dos imóveis, ou seja, são 80 variáveis explicativas.
<br>
Geralmente, as 20 features contínuas estão relacionadas com várias dimensões de área para cada imóvel. Além do típico tamanho do lote e da metragem quadrada total da área habitável, outras variáveis mais específicas são quantificadas no conjunto de dados. Medidas da área do porão, área da sala de estar e até mesmo das varandas estão presentes e divididas em categorias individuais com base na qualidade e no tipo. O grande número de variáveis contínuas neste conjunto de dados deve fornecer muitas oportunidades de discretização e construção de novas features.
<br>
As 14 features discretas normalmente quantificam o número de itens que existem na casa. Como, por exemplo: o número de cozinhas, quartos e banheiro discretizados por sua localização (porão ou acima do térreo). Adicionalmente, a capacidade da garagem e as datas de construção/reforma também estão registadas.
<br>
As features nominais normalmente identificam várias categorias de moradias, garagens, materiais e condições ambientais, enquanto as variáveis ordinais normalmente classificam vários itens na propriedade.
<br>

## Roteiro para o desafio:

<br>
Análise exploratória
<br>
Você pode tentar o seguinte:
<br>
- Verificar a distribuição da variável de interesse (valor de venda)
<br>
- Contar o número de valores faltantes
<br>
- Verificar a matriz de correlação entre as features continuas
<br>
- Scatterplots são úteis para visualizar duas variáveis continuas
<br>
- Plotar a distribuição do valor de venda (histogramas ou boxplots) para as diferentes variáveis - categóricas
<br>
- Ao final, escreva um pouco sobre o que você conseguiu entender, extrapolar e interpretar a partir da análise exploratória
<br>
O número de variáveis é alto, então é importante ser criterioso na montagem dos gráficos exploratórios. Use sua intuição e raciocínio crítico para mostrar as variáveis e encontrar a informação que importa para prosseguir com sua modelagem.
<br>

## Pré-Preprocessamento, limpeza dos dados, construção de features

<br>
Você pode tentar o seguinte:
<br>
- Dropar colunas(features) com muitos valores faltantes
<br>
- Buscar algum erro de preenchimento no dataset
<br>
- Tente criar features (Exemplo: dividir o número de quartos pela área, elevar a área ao quadrado)
<br>
- Não remova linhas com valores faltantes, já que isso pode modificar a distribuição do dataset de validação
<br>

## Objetivo 1: interpretabilidade usando uma regressão linear

<br>
- Utilize apenas exemplos onde a variável SaleCondition é igual à Normal (isso ajuda a diminuir o ruído no dado e levar a uma melhor interpretação). Crie um novo dataset aplicando esse filtro (esse dataset será usado apenas nessa etapa)
<br>
- Aplique uma transformação logarítmica na sua variável de interesse (**SalePrice**)
<br>
- Selecione 6 features para o seu modelo: onde pelo menos uma das features é uma feature construída com os valores de área do imóvel
<br>
- As outras features devem estar relacionadas ao estado de conservação da casa, suas amenidades ou instalações internas
<br>
- Trate os valores faltantes
<br>
- Aplique as transformações nas variáveis categóricas que você julgar necessárias (One hot encoding, ordinal encoding, etc…)
<br>
- Utilize a lib statsmodel para fitar a regressão linear; use a função summary para conseguir interpretar os coeficientes
<br>
- Importante lembrar que a interpretação do coeficiente muda ao aplicar uma transformação lograritmica
<br>
- Verifique a distribuição dos resíduos da regressão linear, e quais as implicações do resultado obtido
<br>
- Reporte o R² dessa regressão, e a sua interpretação desse resultado
<br>
- As features não podem ter alta correlação (utilizar o EDA feito previamente para encontrar as features que você julgue relevantes)
<br>
- Verifique se os pressupostos da regressão linear estão sendo atendidos
<br>
- Em um breve sumário, discorra sobre a interpretação dos coeficientes obtidos pela regressão linear
<br>
Constraints do modelo: essa regressão linear deve ter um R² mínimo de 0.85 e conter exatamente 6 features/variáveis. Todas as features devem ter um P-value maior do que 0.95 e não podem ser colineares. Não é necessário separar esse dataset em treino e teste, já que o foco nessa etapa é na interpretação dos coeficientes de uma regressão linear.
<br>

## Objetivo 2: poder preditivo, regressão via random forest

<br>
A ideia dessa segunda parte é treinar um modelo mais robusto visando o poder preditivo e a obtenção de um modelo para uso em produção (uso real em uma aplicação web)
<br>
- Transforme o sua variável de interesse usando uma transformação logarítmica (ex: y_log=np.log(y))
<br>
- Separe seu dataset original em treino, teste, validação (ver instruções de como fazer a separação abaixo)
<br>
- Impute os valores faltantes das variáveis numéricas com a mediana e os valores faltantes das variáveis categóricas com a moda, os imputers devem ser fitados usando o dataset de treino para depois serem aplicadas nos datasets de validação e teste, isso evitará data leakage (Dica: usar os simpleimputer do sklearn)
<br>
- Aplique as transformações nas variáveis categóricas que você julgar necessárias (One hot encoding, ordinal encoding e etc…), as transformações também devem ser fitadas usando o dataset de treino para depois serem aplicadas nos datasets de validação e teste, isso evitará data leakage (Dica: usar os transformers do sklearn)
<br>
- Treine uma árvore inicial (modelo baseline), usando todas as features e sem mexer nos hiperparâmetros do modelo (usando n_estimators = 150), compute as métricas de avaliação no dataset de validação para você ter um baseline.
<br>
- Tente tunar o seu random forest. Teste diferentes hiperparâmetros, veja as instruções abaixo, use a documentação do sklearn para entender os hiperparâmetros que você for testar
<br>
- Compute a importância das features no dataset de validação (usar permutation_importance do sklearn), usando a importância das features remova do seu treinamento as features menos importantes para que o seu modelo tenha no máximo 40 features, verifique novamente a performance com esse número reduzido de features (isso pode melhorar a performance e a velocidade do seu modelo)
<br>
- Adicione um breve texto com sua interpretação em relação à importância das features
<br>
- Finalmente, compute as métricas de avaliação no dataset de teste para obter o proxy de performance do seu modelo em um ambiente em produção (ambiente real online).
<br>
- Adicione uma conclusão para fechar o seu case
<br>
Dica: sempre que você for avaliar o seu modelo, você deve reverter suas predições da escala log para a escala normal usando uma função exponencial (ex: y_pred = np.exp(y_pred)),
<br>
Separação dos dados em treinamento e validação: os dados devem ser separados em treino, validação e teste, na fase de exploração e modelagem você pode avaliar o modelo usando o dataset de validação para evitar overfitting, e depois, com estudo fechado aplicar as métricas de avaliação no dataset de teste (simulando a performance em exemplos nunca vistos). Para esse caso você deve separar os datasets usando a função train_test_split do sklearn, usando como random state o número 42:
<br>
- Primeiro use a função função train_test_split para separar 70% para treino e 30% para validação e teste
<br>
- Segundo, aplique novamente essa função para quebrar esses 30% em dois datasets, sendo 50% para teste e 50% para validação. Assim obtendo 70% para treino, 15% para validação e 15% para teste
<br>
Testando diferentes hiperparâmetros do seu Random Forest: Primeiro, aumente o número de árvores (n_estimators) até que a precisão pare de melhorar. Em seguida, usando o número de árvores da primeira etapa, tente alguns valores de max_features e escolha aquele que forneça a melhor métrica. Finalmente, usando o melhor max_features, execute min_samples_leaf de 1 até 15, novamente escolhendo o melhor.
<br>
- n_estimators (10, 30, 50, 70, 100, 150, 200)
<br>
- max_features ('sqrt', 0.1 até 0.6)
<br>
- min_samples_leaf(1 até 15)
<br>
Avaliação do modelo de regressão: Para fazer a avaliação do seu modelo você deve aplicar métricas de avaliação no dataset de validação (e no final do estudo no dataset de teste), as seguintes métricas são comuns em modelos de regressão:
<br>
- R²: pense nesse score como uma medida do desempenho do nosso modelo em comparação com um modelo trivial que retorna sempre a média para qualquer previsão solicitada. (o valor de 1.0 representa um modelo perfeito, já um valor de 0.0 representa um modelo equivalente a um modelo aleatório)
<br>
- Valor absoluto médio (MAE): que é apenas a diferença absoluta média entre os valores previstos e verdadeiros. O valor absoluto evita que desvios negativos e positivos se cancelem.
<br>
- Em vez de tomar o valor absoluto, poderíamos elevar ao quadrado as diferenças, dando-nos o erro quadrático médio (MSE). Elevar a diferença também tem o efeito de enfatizar quaisquer previsões que estejam muito longe de seus verdadeiros valores.
<br>
- Para ignorar algumas previsões significativamente desviantes (outliers), é melhor usar o MAE no lugar MSE. Tudo depende do que você está buscando
<br>
- Como as unidades do MSE são o quadrado das unidades da variável de interesse, é útil usar a raiz do erro quadrático médio (RMSE) como métrica de avaliação.
