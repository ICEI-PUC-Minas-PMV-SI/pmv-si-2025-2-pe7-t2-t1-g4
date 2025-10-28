# Etapa 4 — Construção de Modelos, Avaliação e Comparação

## Pré-processamento e Ajustes
- Remoção de "product_id" dos features para evitar vazamento por produto.
- Codificação categórica via OneHotEncoder em colunas de menor cardinalidade: ['ship_mode','segment','country','state','region','category','sub_category'].
- Imputação: categóricas por moda, numéricas por mediana.
- Validação por produto com GroupKFold (n_splits=5).

## Modelos Implementados
- LogisticRegression (multinomial, solver=saga): baseline linear, robusta em alta dimensionalidade após one-hot.
- HistGradientBoostingClassifier: boosting baseado em histogramas, forte em dados tabulares.

## Métrica Principal
- Macro F1, com acurácia como métrica auxiliar.

### LogisticRegression
- CV mean_acc = 0.4465 ± 0.0154
- CV macro_f1 = 0.3422
- Test 2018 acc = 0.444
- Classification Report (2018):
```
              precision    recall  f1-score   support

           0       0.00      0.00      0.00       458
           1       0.37      0.78      0.50       659
           2       0.32      0.12      0.17       685
           3       0.41      0.37      0.38       698
           4       0.60      0.79      0.68       758

    accuracy                           0.44      3258
   macro avg       0.34      0.41      0.35      3258
weighted avg       0.37      0.44      0.38      3258

```

### HistGradientBoosting
- CV mean_acc = 0.5593 ± 0.0157
- CV macro_f1 = 0.5455
- Test 2018 acc = 0.571
- Classification Report (2018):
```
              precision    recall  f1-score   support

           0       0.58      0.48      0.53       458
           1       0.48      0.59      0.53       659
           2       0.49      0.43      0.46       685
           3       0.51      0.52      0.51       698
           4       0.78      0.78      0.78       758

    accuracy                           0.57      3258
   macro avg       0.57      0.56      0.56      3258
weighted avg       0.57      0.57      0.57      3258

```

### RandomForestClassifier
- CV mean_acc = 0.5310 ± 0.0159
- CV macro_f1 = 0.5152
- Test 2018 acc = 0.538
- Classification Report (2018):
```
              precision    recall  f1-score   support

           0       0.50      0.48      0.49       458
           1       0.45      0.49      0.47       659
           2       0.44      0.40      0.42       685
           3       0.49      0.48      0.48       698
           4       0.76      0.79      0.77       758

    accuracy                           0.54      3258
   macro avg       0.53      0.53      0.53      3258
weighted avg       0.53      0.54      0.54      3258

```

## Comparação por Macro F1 (CV)
1. HistGradientBoosting: macro_f1=0.5455
2. RandomForestClassifier: macro_f1=0.5152
3. LogisticRegression: macro_f1=0.3422

## Refinamento do Pipeline
- Pipeline modular (pré-processamento + modelo), permitindo troca de algoritmos sem alterar etapas anteriores.
- Suporte a validação por grupos e avaliação temporal (2018).
- Próximos passos: ajuste fino de hiperparâmetros, inclusão de encoding mais eficiente para alta cardinalidade (e.g., target/frequency encoding), e comparação com modelos de gradient boosting adicionais.
