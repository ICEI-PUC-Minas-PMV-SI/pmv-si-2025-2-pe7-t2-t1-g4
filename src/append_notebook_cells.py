from pathlib import Path
import nbformat as nbf

MD_VIS = """
## Visualizações — Confusão e Importâncias

As células abaixo treinam o modelo final (sem `product_id`) e geram:
- Matriz de confusão para 2018 (`sales_files/confusion_matrix_2018.png`).
- Importância de features (Top 20) (`sales_files/feature_importance_final.png`).
"""

CODE_CONFUSION = r"""
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from catboost import CatBoostClassifier
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Utilitários
def build_cat_features(X: pd.DataFrame):
    return [i for i, col in enumerate(X.columns) if X[col].dtype == 'object']

cwd = Path().resolve()
csv_path = cwd / 'data' / 'train.csv'
print(f"Carregando dados: {csv_path}")

df = pd.read_csv(csv_path)
df.columns = (
    df.columns
    .str.strip()
    .str.lower()
    .str.replace(' ', '_')
    .str.replace('-', '_')
)
if 'order_date' in df.columns:
    df['order_date'] = pd.to_datetime(df['order_date'], errors='coerce', dayfirst=True)
if 'ship_date' in df.columns:
    df['ship_date'] = pd.to_datetime(df['ship_date'], errors='coerce', dayfirst=True)

# Alvo ordinal
product_demand = df.groupby('product_id')['sales'].sum().reset_index()
labels = ['very low', 'low', 'neutral', 'high', 'very high']
product_demand['demand'] = pd.qcut(product_demand['sales'], q=5, labels=labels)
df = pd.merge(left=df, right=product_demand[['product_id','demand']], on='product_id', how='left')
ordinal_mapping = {
    'very low': 0,
    'low': 1,
    'neutral': 2,
    'high': 3,
    'very high': 4,
}
df['demand_ord'] = df['demand'].map(ordinal_mapping).astype('int8')

# Split temporal
train_mask = df['order_date'].dt.year.isin([2015, 2016, 2017])
test_mask = df['order_date'].dt.year == 2018
df_train = df.loc[train_mask].reset_index(drop=True)
df_test = df.loc[test_mask].reset_index(drop=True)

# Features (sem product_id)
cols_to_drop = ['demand', 'demand_ord', 'row_id', 'order_id', 'customer_name', 'product_name', 'order_date', 'ship_date']
X_train = df_train.drop(columns=cols_to_drop + ['product_id'], errors='ignore')
X_test = df_test.drop(columns=cols_to_drop + ['product_id'], errors='ignore')
y_train = df_train['demand_ord']
y_test = df_test['demand_ord']

cat_features = build_cat_features(X_train)

best_params = {'iterations': 400, 'depth': 6, 'learning_rate': 0.06, 'l2_leaf_reg': 3}
model_final = CatBoostClassifier(
    loss_function='MultiClass',
    random_state=42,
    verbose=False,
    thread_count=-1,
    **best_params
)
print("Treinando modelo final para visualizações...")
model_final.fit(X_train, y_train, cat_features=cat_features)

y_pred = model_final.predict(X_test).astype(int).flatten()
acc = (y_pred == y_test).mean()
print(f"Acurácia 2018 (sem product_id): {acc:.3f}")

# Matriz de confusão
cm = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots(figsize=(7, 6))
ConfusionMatrixDisplay(confusion_matrix=cm).plot(ax=ax, cmap='Blues')
ax.set_title('Matriz de Confusão — 2018 (sem product_id)')
fig.tight_layout()
img_path = cwd / 'sales_files' / 'confusion_matrix_2018.png'
fig.savefig(img_path)
print(f"Figura salva em: {img_path}")
plt.show()
"""

CODE_IMPORTANCE = r"""
import pandas as pd
import matplotlib.pyplot as plt

# Importâncias de features
importances = model_final.get_feature_importance()
fi = pd.DataFrame({'feature': X_train.columns, 'importance': importances})
fi = fi.sort_values('importance', ascending=False)

fig, ax = plt.subplots(figsize=(8, 6))
fi_top = fi.head(20)
ax.barh(fi_top['feature'][::-1], fi_top['importance'][::-1])
ax.set_title('Importância de Features (Top 20)')
ax.set_xlabel('Score')
fig.tight_layout()
img_path = Path().resolve() / 'sales_files' / 'feature_importance_final.png'
fig.savefig(img_path)
print(f"Figura salva em: {img_path}")
plt.show()

fi.head(10)
"""


def main():
    cwd = Path(__file__).resolve().parent
    nb_path = cwd / 'sales.ipynb'
    nb = nbf.read(nb_path.open('r', encoding='utf-8'), as_version=4)

    # Append markdown + code cells
    nb.cells.append(nbf.v4.new_markdown_cell(MD_VIS.strip()))
    nb.cells.append(nbf.v4.new_code_cell(CODE_CONFUSION.strip()))
    nb.cells.append(nbf.v4.new_code_cell(CODE_IMPORTANCE.strip()))

    nbf.write(nb, nb_path.open('w', encoding='utf-8'))
    print(f"Appended visualization cells to {nb_path}")


if __name__ == '__main__':
    main()