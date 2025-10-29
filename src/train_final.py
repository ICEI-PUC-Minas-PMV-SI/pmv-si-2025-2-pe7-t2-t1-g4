from pathlib import Path
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score, classification_report


def build_cat_features(X: pd.DataFrame):
    return [i for i, col in enumerate(X.columns) if X[col].dtype == 'object']


def load_data(csv_path: Path):
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
    return df


def map_target(df: pd.DataFrame):
    ordinal_mapping = {
        'very low': 0,
        'low': 1,
        'neutral': 2,
        'high': 3,
        'very high': 4,
    }
    df['demand_ord'] = df['demand'].map(ordinal_mapping).astype('int8')
    return ordinal_mapping


def split_train_test(df: pd.DataFrame):
    if 'order_date' not in df.columns:
        raise ValueError("Coluna 'order_date' não encontrada para efetuar split temporal")
    train_mask = df['order_date'].dt.year.isin([2015, 2016, 2017])
    test_mask = df['order_date'].dt.year == 2018
    df_train = df.loc[train_mask].reset_index(drop=True)
    df_test = df.loc[test_mask].reset_index(drop=True)
    return df_train, df_test


def prepare_features(df_train: pd.DataFrame, df_test: pd.DataFrame):
    cols_to_drop = ['demand', 'demand_ord', 'row_id', 'order_id', 'customer_name',
                    'product_name', 'order_date', 'ship_date']
    X_train = df_train.drop(columns=cols_to_drop + ['product_id'], errors='ignore')
    X_test = df_test.drop(columns=cols_to_drop + ['product_id'], errors='ignore')
    y_train = df_train['demand_ord']
    y_test = df_test['demand_ord']
    return X_train, X_test, y_train, y_test


def write_md_report(md_path: Path, params: dict, test_acc_final: float, report_text: str):
    lines = []
    lines.append("# Tuning CatBoost — Execução Final\n")
    lines.append("\n## Configuração Selecionada\n")
    lines.append(f"- params: {params}\n")
    lines.append("- Validação: treinado em 2015-2017 e avaliado em 2018 (sem product_id).\n")
    lines.append("\n## Avaliação Final (2018 sem product_id)\n")
    lines.append(f"- Acurácia: {test_acc_final:.3f}\n")
    lines.append("- Classification Report:\n")
    lines.append("```\n")
    lines.append(report_text)
    lines.append("\n```\n")
    md_path.write_text(''.join(lines), encoding='utf-8')
    print(f"\nRelatório gravado em: {md_path}")


def main():
    cwd = Path(__file__).resolve().parent
    csv_path = cwd / 'data' / 'train.csv'
    md_out = cwd / 'relatorio_tuning.md'

    print(f"Carregando dados de: {csv_path}")
    df = load_data(csv_path)
    print("Colunas normalizadas:", list(df.columns))
    required = {"product_id", "sales", "order_date"}
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"Colunas necessárias ausentes: {missing}. Disponíveis: {list(df.columns)}")

    print("Derivando labels 'demand' por quantis de sales e mapeando alvo ordinal...")
    product_demand = df.groupby('product_id')['sales'].sum().reset_index()
    labels = ['very low', 'low', 'neutral', 'high', 'very high']
    product_demand['demand'] = pd.qcut(product_demand['sales'], q=5, labels=labels)
    df = pd.merge(left=df, right=product_demand[['product_id','demand']], on='product_id', how='left')
    _ = map_target(df)

    print("Realizando split temporal (2015-17 / 2018)...")
    df_train, df_test = split_train_test(df)

    print("Preparando features finais (sem product_id)...")
    X_train, X_test, y_train, y_test = prepare_features(df_train, df_test)

    cat_features_final = build_cat_features(X_train)
    best_params = {'iterations': 400, 'depth': 6, 'learning_rate': 0.06, 'l2_leaf_reg': 3}

    print("Treinando modelo final com melhores hiperparâmetros observados...")
    model_final = CatBoostClassifier(
        loss_function='MultiClass',
        random_state=42,
        verbose=False,
        thread_count=-1,
        **best_params
    )
    model_final.fit(X_train, y_train, cat_features=cat_features_final)

    print("Avaliando em 2018 sem product_id...")
    test_preds_final = model_final.predict(X_test).astype(int).flatten()
    test_acc_final = accuracy_score(y_test, test_preds_final)
    report_text = classification_report(y_test, test_preds_final)

    print(f"\nAcurácia final (2018 sem product_id): {test_acc_final:.3f}")
    print("\nClassification Report (2018 sem product_id):\n")
    print(report_text)

    print("Gravando relatório final...")
    write_md_report(md_out, best_params, test_acc_final, report_text)


if __name__ == '__main__':
    main()