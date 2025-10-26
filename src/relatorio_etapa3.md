# Etapa 3 — Pré-processamento, Modelo e Avaliação

Este relatório sintetiza o que foi implementado e executado na Etapa 3 do projeto: descrição do pré-processamento e tratamento de dados, construção do modelo proposto e avaliação com métricas.

## Visão Geral
- Dataset temporal com divisão: treino (2015–2017) e teste (2018).
- Tarefa de classificação ordinal da demanda: `very low`, `low`, `neutral`, `high`, `very high`.
- Modelo principal: `CatBoostClassifier` com suporte nativo a variáveis categóricas.
- Validações executadas: avaliação simples (com `product_id`) e validação cruzada com `GroupKFold` por produto; avaliação final em 2018 sem `product_id`.

## Pré-processamento e Tratamento
- Alvo ordinal mapeado para inteiros:
  - `{'very low': 0, 'low': 1, 'neutral': 2, 'high': 3, 'very high': 4}` → colunas `demand_ord` para treino e teste.
- Remoção de colunas não utilizadas no treino direto:
  - `['demand', 'demand_ord', 'row_id', 'order_id', 'customer_name', 'product_name', 'order_date', 'ship_date']`.
- Seleção automática de variáveis categóricas para CatBoost:
  - Índices das colunas com `dtype == 'object'` são passados em `cat_features` ao `Pool`.
- Observações:
  - Não foi necessário normalização/escala; CatBoost lida com numéricas e categóricas.
  - Balanceamento não foi aplicado.

## Construção do Modelo
- Algoritmo: `CatBoostClassifier(loss_function='MultiClass')`.
- Hiperparâmetros principais:
  - `iterations = 300`, `learning_rate = 0.1`, `depth = 6`, `random_state = 42`, `verbose = False`.
- Treino e predição:
  - `train_pool = Pool(X_train, label=y_train, cat_features=cat_features)`;
  - Predição por probabilidade (argmax) para o conjunto de teste.
- Validação por produto (generalização):
  - `GroupKFold(n_splits=5)`, com `groups = df_treino['product_id']`.
  - Para evitar vazamento, o CV usa `X_full = df_treino.drop(columns=cols_to_drop + ['product_id'])`.
- Avaliação final em 2018 sem `product_id`:
  - `X_train_final = X_train.drop(columns=['product_id'])` e `X_test_final = X_test.drop(columns=['product_id'])`.

## Avaliação do Modelo

### 1) Teste simples com `product_id` presente
- Acurácia (train): `1.000`
- Acurácia (test): `0.993`
- Gap (train-test): `+0.007` → sem evidência forte de overfitting.
- Classification report – Test set:
```
              precision    recall  f1-score   support

    very low       0.99      1.00      0.99       458
         low       1.00      0.99      0.99       659
     neutral       0.98      0.99      0.99       685
        high       0.99      0.99      0.99       698
   very high       1.00      1.00      1.00       758

    accuracy                           0.99      3258
   macro avg       0.99      0.99      0.99      3258
```

Interpretação rápida:
- Como os mesmos produtos aparecem em treino e teste, o modelo memoriza bem (CatBoost usa estatísticas de categoria/CTR). O alto desempenho é consistente com uso de `product_id` como preditor.

### 2) Validação cruzada com `GroupKFold` (produto exclusivo por fold)
- Folds (acurácia): `0.5783`, `0.5921`, `0.5887`, `0.5795`, `0.6315`.
- Acurácia média: `0.5940 ± 0.0195`.
- Classification report – Out-of-Fold (treino 2015–17):
```
              precision    recall  f1-score   support

    very low       0.59      0.52      0.55       940
         low       0.48      0.56      0.52      1356
     neutral       0.49      0.43      0.46      1295
        high       0.56      0.56      0.56      1350
   very high       0.82      0.82      0.82      1601

    accuracy                           0.59      6542
   macro avg       0.59      0.58      0.58      6542
   weighted avg    0.60      0.59      0.59      6542
```

Interpretação rápida:
- Métricas significativamente menores quando garantimos que produtos não se repetem entre treino e validação → estimativa mais realista de generalização.

### 3) Teste 2018 sem `product_id` (generalização)
- Acurácia (2018 sem `product_id`): `0.586`.
- Classification report – Teste 2018:
```
              precision    recall  f1-score   support

    very low       0.61      0.53      0.57       458
         low       0.49      0.61      0.54       659
     neutral       0.50      0.41      0.45       685
        high       0.53      0.52      0.53       698
   very high       0.79      0.81      0.80       758

    accuracy                           0.59      3258
   macro avg       0.58      0.58      0.58      3258
   weighted avg    0.59      0.59      0.58      3258
```

Interpretação rápida:
- Desempenho cai quando removemos `product_id`, indicando dependência do modelo nessa coluna. Sem `product_id`, o modelo precisa generalizar usando somente atributos transversais (e.g., categoria, segmento, região, datas derivadas), o que é mais difícil.

## Conclusões e Próximos Passos
- Para avaliar generalização real, mantenha `product_id` fora dos features e use CV por grupos (produto por fold).
- Considerar enriquecer o conjunto de features com atributos derivados de tempo (sazonalidade), interações, estatísticas por categoria/região.
- Explorar tuning de hiperparâmetros (grid/bayes) e variações de profundidade/iterações.
- Comparar com outros modelos (e.g., LightGBM, XGBoost) e técnicas orientadas a ordinal (loss específica).

—
Relatório gerado automaticamente a partir da execução de `src/sales.ipynb` (exportado para `src/sales.md`).