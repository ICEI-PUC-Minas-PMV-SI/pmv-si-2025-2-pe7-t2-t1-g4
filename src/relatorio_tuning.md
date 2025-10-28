# Tuning CatBoost — Execução Final

## Configuração Selecionada
- params: {'iterations': 400, 'depth': 6, 'learning_rate': 0.06, 'l2_leaf_reg': 3}
- Validação: treinado em 2015-2017 e avaliado em 2018 (sem product_id).

## Avaliação Final (2018 sem product_id)
- Acurácia: 0.587
- Classification Report:
```
              precision    recall  f1-score   support

           0       0.61      0.52      0.56       458
           1       0.49      0.61      0.54       659
           2       0.49      0.41      0.44       685
           3       0.53      0.54      0.54       698
           4       0.81      0.81      0.81       758

    accuracy                           0.59      3258
   macro avg       0.58      0.58      0.58      3258
weighted avg       0.59      0.59      0.58      3258

```
