# One-Hot Encoding vs. Label Encoding: Entendendo e Aplicando em Python com Scikit-Learn e Pandas

No mundo da ciência de dados, lidar com variáveis categóricas é uma tarefa comum. Para utilizar algoritmos de machine learning, é necessário converter essas variáveis em um formato numérico. Dois métodos populares para essa conversão são o Label Encoding e o One-Hot Encoding. Neste artigo, vamos explorar o que são esses métodos, as diferenças entre eles e como implementá-los em Python usando as bibliotecas Scikit-Learn e Pandas.

## O que são Variáveis Categóricas?
Variáveis categóricas são aquelas que representam categorias ou grupos, como grau de instrução, níveis de satisfação ou categorias de produtos. Elas podem ser:

- **Nominais**: Não possuem uma ordem intrínseca (por exemplo, cores: vermelho, azul, verde).
- **Ordinais**: Possuem uma ordem intrínseca (por exemplo, nível de satisfação: insatisfeito, neutro, satisfeito).

## Por que Codificar Variáveis Categóricas?
A maioria dos algoritmos de machine learning não consegue trabalhar diretamente com dados categóricos. Portanto, é necessário converter essas variáveis em um formato numérico sem perder a essência da informação que elas carregam.

## Label Encoding
O Label Encoding converte cada valor de categoria em um número inteiro único. Esse método é especialmente útil para variáveis ordinais, onde há uma relação de ordem entre as categorias.

### Como Garantir a Ordem das Classes com LabelEncoder?
Por padrão, o LabelEncoder do Scikit-Learn atribui números às classes com base na ordem alfabética das categorias, o que pode não refletir a ordem de importância desejada. Para garantir a ordem das classes, você pode utilizar o LabelEncoder em conjunto com um mapeamento personalizado.

Exemplo em Python:

Imagine a seguinte lista de níveis de satisfação do cliente:
`niveis_satisfacao = ['insatisfeito', 'neutro', 'satisfeito', 'muito satisfeito', 'neutro']
`

E queremos garantir a seguinte ordem de importância:
- insatisfeito (0)
- neutro (1)
- satisfeito (2)
- muito satisfeito (3)

```python
import numpy as np
from sklearn.preprocessing import LabelEncoder

# Dados de exemplo
niveis_satisfacao = np.array(['insatisfeito', 'neutro', 'satisfeito', 'muito satisfeito', 'neutro'])

# Definindo a ordem das categorias
ordem = ['insatisfeito', 'neutro', 'satisfeito', 'muito satisfeito']

# Criando um mapeamento personalizado
mapping = {label: idx for idx, label in enumerate(ordem)}

# Aplicando o mapeamento aos dados
niveis_encoded = np.array([mapping[label] for label in niveis_satisfacao])

print(niveis_encoded)
# [0 1 2 3 1]
```

**Vantagens**: Permite controlar a ordem das categorias conforme a importância.

**Desvantagens**: Requer a criação manual do mapeamento.

### Usando OrdinalEncoder para Garantir a Ordem:
Outra maneira é utilizar o OrdinalEncoder do Scikit-Learn, que permite especificar a ordem das categorias.

```python
from sklearn.preprocessing import OrdinalEncoder
import numpy as np

# Dados de exemplo
niveis_satisfacao = np.array(['insatisfeito', 'neutro', 'satisfeito', 'muito satisfeito', 'neutro']).reshape(-1, 1)

# Definindo a ordem das categorias
ordem = [['insatisfeito', 'neutro', 'satisfeito', 'muito satisfeito']]

# Inicializando o OrdinalEncoder com a ordem especificada
ordinal_encoder = OrdinalEncoder(categories=ordem)

# Ajustando e transformando os dados
niveis_encoded = ordinal_encoder.fit_transform(niveis_satisfacao)

print(niveis_encoded.flatten().astype(int))
# [0 1 2 3 1]
```

### Implementação com Pandas:
Usando Pandas, podemos definir categorias ordenadas usando o tipo CategoricalDtype.
    
```python
import pandas as pd

# Dados de exemplo
df = pd.DataFrame({'satisfacao': niveis_satisfacao.flatten()})

# Definindo a ordem das categorias
ordem_categorias = ['insatisfeito', 'neutro', 'satisfeito', 'muito satisfeito']
tipo_categoria = pd.api.types.CategoricalDtype(categories=ordem_categorias, ordered=True)

# Convertendo para categoria ordenada
df['satisfacao'] = df['satisfacao'].astype(tipo_categoria)

# Codificando as categorias
df['satisfacao_encoded'] = df['satisfacao'].cat.codes

print(df)
#            satisfacao  satisfacao_encoded
#0       insatisfeito                   0
#1             neutro                   1
#2         satisfeito                   2
#3   muito satisfeito                   3
#4             neutro                   1
```
**Vantagens**: Preserva a ordem intrínseca das categorias e integra bem com o fluxo de trabalho do Pandas.

**Desvantagens**: Pode ser menos eficiente para grandes conjuntos de dados se não utilizado corretamente.


## One-Hot Encoding
O One-Hot Encoding cria uma nova coluna binária para cada categoria, onde 1 indica a presença da categoria e 0 a ausência. Esse método é ideal para variáveis nominais.

Exemplo em Python:
Usando a seguinte lista de cores: `cores = ['vermelho', 'azul', 'verde', 'azul', 'vermelho']`

```python
from sklearn.preprocessing import OneHotEncoder

# Dados de exemplo
df_cores = pd.DataFrame({'cor': cores})

# Inicializando o OneHotEncoder
onehot_encoder = OneHotEncoder(sparse_output=False)

# Ajustando e transformando os dados
cores_onehot = onehot_encoder.fit_transform(df_cores[['cor']])

print(cores_onehot)
print(onehot_encoder.get_feature_names_out(['cor']))
# [[0. 0. 1.]
# [1. 0. 0.]
# [0. 1. 0.]
# [1. 0. 0.]
# [0. 0. 1.]]
# ['cor=azul' 'cor=verde' 'cor=vermelho']
```

### Implementação com Pandas:
Uma maneira mais direta é usar o método `get_dummies` do Pandas:

```python
# Aplicando One-Hot Encoding usando pandas
cores_onehot = pd.get_dummies(df_cores['cor'], prefix='cor')

print(cores_onehot)
#    cor_azul  cor_verde  cor_vermelho
# 0         0          0             1
# 1         1          0             0
# 2         0          1             0
# 3         1          0             0
# 4         0          0             1
```

**Vantagens**: Evita relações ordinais inexistentes e é fácil de implementar com Pandas.

**Desvantagens**: Pode aumentar significativamente o número de features, especialmente com variáveis que possuem muitas categorias.

## Diferenças Principais
- Relação Ordinal:
    - Label Encoding preserva a ordem das categorias.
    - One-Hot Encoding não considera nenhuma ordem entre categorias.

- Dimensionalidade:
    - Label Encoding mantém a dimensionalidade original.
    - One-Hot Encoding aumenta o número de categorias.

## Quando Usar Cada Método?
- Label Encoding: Use para variáveis categóricas ordinais, onde a ordem das categorias é significativa para o modelo.
- One-Hot Encoding: Use para variáveis categóricas nominais, onde não há relação ordinal entre as categorias.

## Conclusão
A escolha entre Label Encoding e One-Hot Encoding depende do tipo de variável categórica e do contexto do problema. Para garantir a ordem de importância das classes ao usar o `LabelEncoder`, é necessário criar um mapeamento personalizado ou utilizar o `OrdinalEncoder`. Usar o Pandas para essas transformações pode simplificar o processo e integrar melhor o fluxo de trabalho de análise de dados.
