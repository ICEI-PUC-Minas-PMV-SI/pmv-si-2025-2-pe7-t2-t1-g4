# 1. Análise descritiva e exploratória dos dados

## Introdução

O dataset analisado refere-se a transações de vendas, com informações detalhadas sobre pedidos, clientes, produtos, datas, regiões, entre outros atributos. O principal objetivo desta etapa foi compreender suas principais características estruturais, explorar padrões centrais e de dispersão, detectar outliers e iniciar a identificação de possíveis relações entre variáveis.

## Origem do dataset

O dataset utilizado nesta análise foi obtido em: [Sales Forecasting - Kaggle](https://www.kaggle.com/datasets/rohitsahoo/sales-forecasting)

## Localização do notebook de análise

O notebook da análise descritiva e exploratória encontra-se em:

```
src/sales.ipynb
```

A base de dados (`train.csv`) utilizada está localizada em:

```
src/data/train.csv
```

## Como executar o notebook

1. Navegue até o diretório `src`:

```sh
cd src
```
2. Ative o ambiente virtual (caso ainda não ativado):

```sh
source .venv/bin/activate
```
3. Instale todas as dependências necessárias declaradas em `pyproject.toml`:

```sh
uv pip install
```
4. Inicie o Jupyter Lab:

```sh
uv run jupyter lab --port 8888 --ip 0.0.0.0 --ServerApp.disable_check_xsrf=True --ServerApp.token=''
```
5. Abra o notebook `sales.ipynb` e execute as células normalmente.

---

## Estrutura e limpeza dos dados

A base dispõe de 9.800 registros e 18 colunas, abarcando informações numéricas, categóricas e temporais. Realizou-se padronização dos nomes das colunas (minúsculas, sem espaços) e as datas foram convertidas para o tipo `datetime`. Não há registros duplicados; os únicos valores nulos detectados concentram-se em `postal_code` (apenas 11 casos). Não foi necessário realizar imputação neste estágio.

## Estatísticas descritivas e dispersão

As principais medidas de tendência central e dispersão das variáveis numéricas podem ser sintetizadas conforme a tabela abaixo:

```python
df.describe().T[['mean', '50%', 'std', 'min', '25%', '75%', 'max']]
```

- **Sales**: Média ≈ 230,77; Mediana ≈ 54,49; Desvio padrão elevado (> 626), distribuição bastante assimétrica (mínimo de 0,444 e máximo superior a 22.600).
- **Postal Code**: Não representa valor contínuo; utilizada apenas para localização.

A observação dos boxplots e histogramas revela distribuição fortemente assimétrica para `Sales`, com presença clara de outliers (valores extremos) que correspondem a vendas pouco comuns no conjunto. A maior parte das vendas concentra-se em valores baixos, com cauda longa à direita.

## Variáveis categóricas

As categorias possuem a seguinte frequência:
```python
for col in ['category', 'region', 'segment']:
    print(df[col].value_counts())
```

- **Category**: Office Supplies (60%), Furniture (21%), Technology (19%).
- **Region**: West (32%), East (28%), Central (23%), South (16%).
- **Segment**: Consumer (52%), Corporate (30%), Home Office (18%).

Os gráficos de barras confirmam forte predominância de vendas em Office Supplies e em consumidores finais.

## Visualização e detecção de outliers

Os boxplots e histogramas apresentados evidenciam concentração dos dados de vendas em patamares baixos, com ocorrência de valores atípicos significativos na distribuição de vendas. Há forte assimetria positiva, refletindo grande dispersão e presença de outliers.

## Relações entre variáveis (correlação)

Para uma análise inicial das relações entre variáveis numéricas:
```python
correl = df.corr(numeric_only=True)
import seaborn as sns
import matplotlib.pyplot as plt
plt.figure(figsize=(5, 3))
sns.heatmap(correl, annot=True, cmap="coolwarm")
plt.title("Matriz de correlação das variáveis numéricas")
plt.show()
```
Observa-se que:
- Não há correlações fortes entre as variáveis numéricas disponíveis (com destaque apenas para conexões internas de identificação, não de valor preditivo).
- Como o dataset não possui colunas de lucro ou quantidade, a variável de vendas é isolada para análise descritiva, limitando o aprofundamento das relações quantitativas.

## Achados relevantes

- O dataset é composto majoritariamente por vendas de baixo valor, sendo as vendas altas raras e caracterizando cauda longa na distribuição de `Sales`. Isso sugere recomendação explícita de tratar atipicidades e transformar a variável para métodos analíticos futuros.
- Não há problemas estruturais graves: as colunas estão bem definidas, a ausência de dados é restrita e não compromete a análise inicial.
- As categorias 'Office Supplies' e clientes do segmento 'Consumer' são a maioria expressiva das vendas, sugerindo padrão de concentração de mercado que pode ser explorado em análises futuras.
- Não foi identificada, até este ponto, correlação forte entre as variáveis numéricas disponíveis. O padrão identificável é, sobretudo, de dispersão e assimetria em 'Sales'.

## Ferramentas e tecnologias utilizadas

Todas as análises foram desenvolvidas em **Python**, utilizando as seguintes ferramentas:
- **pandas**: para carregamento, limpeza, manipulação e sumarização dos dados.
- **matplotlib** e **seaborn**: para visualização dos dados, boxplots, histogramas, gráficos de barras e mapas de calor.
- **Jupyter Notebook**: ambiente interativo para execução de código e apresentação integrada dos artefatos.

O código implementado nas células do notebook contempla tanto os cálculos estatísticos quanto a visualização dos dados e pode ser referenciado e adaptado para etapas posteriores do projeto.
