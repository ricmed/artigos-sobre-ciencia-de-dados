# Introdução à Regressão Linear

A regressão linear é uma das técnicas estatísticas mais simples e amplamente utilizadas em Data Science, especialmente útil para prever um valor quantitativo. Os modelos de regressão linear tentam prever uma variável dependente (também chamada de variável de resposta) com base em uma ou mais variáveis independentes (ou preditores), assumindo que a relação entre as variáveis é linear.

## Forma Matemática da Regressão Linear

### Regressão Linear Simples
No caso mais simples, com uma única variável preditora, a relação entre a variável dependente \( y \) e a variável independente \( x \) pode ser expressa como:

$$ y = \beta_0 + \beta_1 x + \epsilon $$

Onde:
- **$( \beta_0 )$** é o intercepto,
- **$( \beta_1 )$** é o coeficiente da variável independente \( x \),
- **$( \epsilon )$** é o termo de erro, que capta todas as outras influências sobre \( y \) que não são explicadas por \( x \).

### Regressão Linear Múltipla
Quando há mais de uma variável independente, a equação se expande para:

$$ y = \beta_0 + \beta_1 x_1 + \beta_2 x_2 + \cdots + \beta_n x_n + \epsilon $$

Aqui, cada $( x_i )$ representa uma variável independente diferente, e $( \beta_i )$ são os coeficientes correspondentes.

## Aplicações da Regressão Linear

- **Economia:** Previsão de indicadores econômicos como PIB, inflação, etc.
- **Biologia:** Modelagem de crescimento populacional ou relações dose-resposta.
- **Marketing:** Previsão de vendas baseada em gastos com publicidade.
- **Finanças:** Estimativa de riscos ou retornos esperados de ativos.

## Métricas de Avaliação

Para avaliar o desempenho de um modelo de regressão linear, as métricas mais comuns são:

1. **Erro Quadrático Médio (MSE):**

$MSE = \frac{1}{n} \sum_{i=1}^n (y_i - \hat{y}_i)^2$

onde $( \hat{y}_i )$ é o valor predito e $( y_i )$ é o valor real.

2. **Raiz do Erro Quadrático Médio (RMSE):**

$RMSE = \sqrt{MSE}$

fornece uma medida da magnitude do erro em termos das unidades originais da variável de saída.

3. **Coeficiente de Determinação** ( R^2 ):

$$ R^2 = 1 - \frac{\sum_{i=1}^n (y_i - \hat{y}_i)^2}{\sum_{i=1}^n (y_i - \bar{y})^2} $$


Onde:
- $( y_i )$ são os valores reais das observações,
- $( \hat{y}_i )$ são os valores preditos pelo modelo,
- $( \bar{y} )$ é a média dos valores reais.

O $( R^2 )$ mede a proporção da variância na variável dependente que é previsível a partir das variáveis independentes.

## Exemplo Prático em Python

Para ilustrar a regressão linear em Python, vamos usar a biblioteca `scikit-learn` para prever preços de casas com base em características como área, número de quartos, etc.

```python
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd

# Supondo que temos um DataFrame `df` com as colunas 'preco', 'area', 'quartos'
X = df[['area', 'quartos']]  # Variáveis independentes
y = df['preco']  # Variável dependente

# Dividir os dados em conjuntos de treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Criar e treinar o modelo
modelo = LinearRegression()
modelo.fit(X_train, y_train)

# Previsões
y_pred = modelo.predict(X_test)

# Avaliação do modelo
mse = mean_squared_error(y_test, y_pred)
rmse = mse**0.5
r2 = r2_score(y_test, y_pred)

print("MSE:", mse)
print("RMSE:", rmse)
print("R^2:", r2)
```
