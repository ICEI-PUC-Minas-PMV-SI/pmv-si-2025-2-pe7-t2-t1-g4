# Etapa 4 — Construção de Modelos, Avaliação e Comparação

## Pré-processamento e Ajustes
- Remoção de 'product_id' dos features para evitar vazamento por produto.
- Codificação categórica via OneHotEncoder em colunas de menor cardinalidade: ['ship_mode','segment','country','state','region','category','sub_category'].
- Imputação: categóricas por moda, numéricas por mediana.
- Validação por produto com GroupKFold (n_splits=5).

## Modelos Implementados
- LogisticRegression (multinomial, solver=saga): baseline linear, robusta em alta dimensionalidade após one-hot.
- HistGradientBoostingClassifier: boosting baseado em histogramas, forte em dados tabulares.

## Métrica Principal
- Macro F1, com acurácia como métrica auxiliar.

### LogisticRegression
- CV mean_acc = 0.4352 ± 0.0232
- CV macro_f1 = 0.3921
- Test 2018 acc = 0.432
- Classification Report (2018):
```
              precision    recall  f1-score   support

           0       0.34      0.80      0.48       458
           1       0.30      0.05      0.09       659
           2       0.36      0.26      0.30       685
           3       0.40      0.39      0.39       698
           4       0.62      0.74      0.68       758

    accuracy                           0.43      3258
   macro avg       0.40      0.45      0.39      3258
weighted avg       0.41      0.43      0.39      3258

```

### LogisticRegressionCW
- CV mean_acc = 0.4276 ± 0.0186
- CV macro_f1 = 0.3664
- Test 2018 acc = 0.428
- Classification Report (2018):
```
              precision    recall  f1-score   support

           0       0.33      0.82      0.47       458
           1       0.00      0.00      0.00       659
           2       0.36      0.28      0.31       685
           3       0.38      0.42      0.40       698
           4       0.66      0.70      0.68       758

    accuracy                           0.43      3258
   macro avg       0.35      0.44      0.37      3258
weighted avg       0.36      0.43      0.38      3258

```

### HistGradientBoosting
- CV mean_acc = 0.5679 ± 0.0145
- CV macro_f1 = 0.5536
- Test 2018 acc = 0.568
- Classification Report (2018):
```
              precision    recall  f1-score   support

           0       0.57      0.45      0.51       458
           1       0.47      0.59      0.52       659
           2       0.49      0.43      0.46       685
           3       0.52      0.51      0.51       698
           4       0.78      0.79      0.79       758

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

### LogisticRegression_calibrada
- CV mean_acc = 0.4276 ± 0.0186
- CV macro_f1 = 0.3664
- Test 2018 acc = 0.426
- Classification Report (2018):
```
              precision    recall  f1-score   support

           0       0.00      0.00      0.00       458
           1       0.39      0.71      0.50       659
           2       0.29      0.18      0.22       685
           3       0.26      0.29      0.27       698
           4       0.72      0.79      0.75       758

    accuracy                           0.43      3258
   macro avg       0.33      0.39      0.35      3258
weighted avg       0.36      0.43      0.38      3258

```

## Comparação por Macro F1 (CV)
1. HistGradientBoosting: macro_f1=0.5536
2. RandomForestClassifier: macro_f1=0.5152
3. LogisticRegression: macro_f1=0.3921
4. LogisticRegressionCW: macro_f1=0.3664
5. LogisticRegression_calibrada: macro_f1=0.3664

## Refinamento do Pipeline
- Pipeline modular (pré-processamento + modelo), permitindo troca de algoritmos sem alterar etapas anteriores.
- Suporte a validação por grupos e avaliação temporal (2018).
- Próximos passos: ajuste fino de hiperparâmetros, inclusão de encoding mais eficiente para alta cardinalidade (e.g., target/frequency encoding), e comparação com modelos de gradient boosting adicionais.
