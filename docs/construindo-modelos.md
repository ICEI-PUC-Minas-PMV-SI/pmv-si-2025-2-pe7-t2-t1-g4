# Relatório de Análise de Modelos de Recomendação

## 1. Preparação dos Dados

Na preparação dos dados, enfrentamos alguns desafios reais. O dataset é relativamente limpo, mas encontramos algumas surpresas:

*   **Valores Extremos na Coluna de Quantity:** encontramos alguns valores estranhos na coluna de `Quantity`. Havia um pedido com **14 unidades** de um mesmo produto, o que parecia fora do padrão quando a maioria era de 1 a 5 unidades. Decidimos tratar esses casos limitando os valores extremos usando o método **IQR**, mas sem excluir completamente para não perder informações valiosas.
*   **Clientes com Poucas Compras (Cold Start):** Cerca de **15% dos clientes** tinham menos de 5 compras no histórico. Isso é um problema real para sistemas de recomendação. Para esses casos, a estratégia adotada foi usar as **categorias dos produtos** que eles compraram como *fallback* (solução alternativa).
*   **Extração de Features Temporais:** Nas datas, percebemos que poderíamos extrair mais informações. Criamos *flags* para **fins de semana** e também separei por **trimestres**, já que no varejo isso faz diferença.

## 2. Descrição dos Modelos

### Filtragem Colaborativa Baseada em Itens

Escolhemos esse modelo primeiro porque é mais intuitivo: se alguém compra um notebook, provavelmente vai precisar de uma mochila.

*   **Implementação:** Direta, utilizando **similaridade de cosseno** (*cosine similarity*).
*   **Ajuste:** tivemos que ajustar o *threshold* de similaridade para **0.2**, pois com valores muito baixos as recomendações ficavam sem sentido.

### ALS (Alternating Least Squares)

Este foi o modelo mais complexo de ajustar.

*   **Ajuste de Fatores Latentes:** Testei com diferentes números de fatores latentes — comecei com 32, depois 64, e vi que com 128 já começava o *overfitting*. O número final escolhido foi **64 fatores**, que proporcionou um bom balanço entre performance e qualidade.
*   **Limitação:** Uma coisa chata do ALS é que ele **não explica bem as recomendações** — você recomenda, mas não sabe muito bem por quê.

### Modelo Híbrido

Aqui foi onde gastamos mais tempo, testando várias combinações entre o colaborativo e o baseado em conteúdo.

*   **Combinação Final:** Usar **70% do *score* do ALS** e **30% do baseado em conteúdo** funcionou melhor para o nosso caso.
*   **Benefício:** Essa abordagem híbrida ajuda especialmente com **produtos novos**, onde o ALS sozinho não funciona devido à falta de dados de interação.

## 3. Avaliação dos Modelos

### Métricas Utilizadas

Focamos em métricas que refletem a experiência real do usuário, onde a posição das recomendações é crucial.

*   **Precision e Recall:** Utilizadas porque na prática ninguém olha além das primeiras recomendações.
*   **NDCG (Normalized Discounted Cumulative Gain):** Considera a ordem da recomendação — afinal, se o produto mais relevante está em 10º lugar, não adianta muita coisa.
*   **Diversidade:** Uma métrica que nos surpreendeu. O modelo baseado em conteúdo naturalmente recomenda coisas mais variadas, enquanto o ALS tende a recomendar sempre os produtos mais populares.

### Resultados Observados

| Modelo | Métrica | Valor | Observação |
| :--- | :--- | :--- | :--- |
| **ALS** | Precision | 0.168 | Campeão em precisão, mas com tendência a recomendar produtos populares. |
| **Baseado em Conteúdo** | Cobertura | 0.78 | Cobertura maior, mas acerta menos. |
| **Híbrido** | - | - | Fica no meio do caminho, mas é mais útil na prática por cobrir mais situações. |

**Insight Interessante:**

*   Para **clientes com mais de 10 compras**, o ALS funciona muito melhor.
*   Para **clientes novos (Cold Start)**, o modelo baseado em conteúdo é a salvação.

## 4. Pipeline de Análise

Nosso fluxo de trabalho foi meio bagunçado no começo, mas depois conseguimos organizá-lo:

1.  **Exploração dos Dados:** Primeiramente, exploramos os dados. Fizemos uns gráficos básicos de distribuição e observamos que havia muita coisa em *Office Supplies*. Também notamos que as vendas crescem no final do ano, o que é esperado.
2.  **Preparação:** Gastamos um tempo decidindo como codificar as categorias. No final, **Label Encoding** foi suficiente para a maioria.
3.  **Modelagem:** Começamos com o mais simples (*item-based*) e fomos complicando.
4.  **Validação:** Fizemos uma **validação temporal** — separamos os últimos 20% dos dados por data para teste, o que é mais realista.


