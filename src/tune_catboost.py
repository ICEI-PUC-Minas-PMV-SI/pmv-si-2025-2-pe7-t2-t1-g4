import os
from pathlib import Path
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool
from sklearn.model_selection import GroupKFold
from sklearn.metrics import accuracy_score, classification_report, f1_score


def build_cat_features(X: pd.DataFrame):
    return [i for i, col in enumerate(X.columns) if X[col].dtype == 'object']


def load_data(csv_path: Path):
    df = pd.read_csv(csv_path)
    # Padronizar nomes de colunas para snake_case
    df.columns = (
        df.columns
        .str.strip()
        .str.lower()
        .str.replace(' ', '_')
        .str.replace('-', '_')
    )
    # Garantir parsing de datas (formato DD/MM/YYYY no dataset)
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
    # Treino: 2015-2017, Teste: 2018
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
    X_train = df_train.drop(columns=cols_to_drop, errors='ignore')
    X_test = df_test.drop(columns=cols_to_drop, errors='ignore')
    y_train = df_train['demand_ord']
    return X_train, X_test, y_train, cols_to_drop


def run_groupkfold_tuning(df_train: pd.DataFrame, cols_to_drop):
    # Remover product_id dos features para CV
    X_full = df_train.drop(columns=cols_to_drop + ['product_id'], errors='ignore')
    y_full = df_train['demand_ord']
    groups = df_train['product_id']

    gkf = GroupKFold(n_splits=5)
    cat_features_cv = build_cat_features(X_full)

    param_grid = [
        {'iterations': 400, 'depth': 6, 'learning_rate': 0.06, 'l2_leaf_reg': 3}
    ]

    results = []

    print("Iniciando tuning com GroupKFold (produto exclusivo por fold)...")
    for params in param_grid:
        fold_acc = []
        oof_preds = np.empty(len(df_train), dtype=int)

        for fold_idx, (tr_idx, val_idx) in enumerate(gkf.split(X_full, y_full, groups)):
            X_tr, X_val = X_full.iloc[tr_idx], X_full.iloc[val_idx]
            y_tr, y_val = y_full.iloc[tr_idx], y_full.iloc[val_idx]

            model = CatBoostClassifier(
                loss_function='MultiClass',
                random_state=42,
                verbose=False,
                thread_count=-1,
                **params
            )
            print(f"[TUNING] Fold {fold_idx+1}/5 com params {params}...")
            model.fit(X_tr, y_tr, cat_features=cat_features_cv, eval_set=(X_val, y_val))
            val_preds = model.predict(X_val).astype(int).flatten()
            acc = accuracy_score(y_val, val_preds)
            print(f"[TUNING] Fold {fold_idx+1} acurácia = {acc:.4f}")
            fold_acc.append(acc)
            oof_preds[val_idx] = val_preds

        mean_acc = float(np.mean(fold_acc))
        std_acc = float(np.std(fold_acc))
        macro_f1 = float(f1_score(y_full, oof_preds, average='macro'))
        results.append({
            'params': params,
            'mean_acc': mean_acc,
            'std_acc': std_acc,
            'macro_f1': macro_f1
        })
        print(f"Params {params} -> mean_acc={mean_acc:.4f} ± {std_acc:.4f}, macro_f1={macro_f1:.4f}")

    # Escolher melhor por macro_f1; desempate por mean_acc
    results_sorted = sorted(results, key=lambda r: (r['macro_f1'], r['mean_acc']), reverse=True)
    best = results_sorted[0]
    print("\nMelhores hiperparâmetros:")
    print(best)
    return best, results_sorted


def train_final_and_eval(df_train: pd.DataFrame, df_test: pd.DataFrame, cols_to_drop, best_params):
    X_train_final = df_train.drop(columns=cols_to_drop + ['product_id'], errors='ignore')
    X_test_final = df_test.drop(columns=cols_to_drop + ['product_id'], errors='ignore')
    y_train = df_train['demand_ord']
    y_test = df_test['demand_ord']

    cat_features_final = build_cat_features(X_train_final)

    final_params = {k: v for k, v in best_params['params'].items() if k not in ('use_best_model','od_type','od_wait','early_stopping_rounds')}
    model_final = CatBoostClassifier(
        loss_function='MultiClass',
        random_state=42,
        verbose=False,
        thread_count=-1,
        **final_params
    )
    model_final.fit(X_train_final, y_train, cat_features=cat_features_final)

    test_preds_final = model_final.predict(X_test_final).astype(int).flatten()
    test_acc_final = accuracy_score(y_test, test_preds_final)
    report = classification_report(y_test, test_preds_final)

    print(f"\nAcurácia final (2018 sem product_id): {test_acc_final:.3f}")
    print("\nClassification Report (2018 sem product_id):\n")
    print(report)

    return test_acc_final, report


def write_md_report(md_path: Path, best, results_sorted, test_acc_final, report_text):
    lines = []
    lines.append("# Tuning CatBoost — GroupKFold por produto\n")
    lines.append("\n## Resumo do Tuning\n")
    lines.append("- Métrica primária: macro F1 (desempate por acurácia média).\n")
    lines.append("- Validação: GroupKFold (n_splits=5), grupos = product_id, features sem product_id.\n")
    lines.append("\n### Melhores Hiperparâmetros\n")
    lines.append(f"- params: {best['params']}\n")
    lines.append(f"- mean_acc: {best['mean_acc']:.4f} ± {best['std_acc']:.4f}\n")
    lines.append(f"- macro_f1: {best['macro_f1']:.4f}\n")

    lines.append("\n### Top 5 combinações\n")
    for i, r in enumerate(results_sorted[:5], start=1):
        lines.append(f"{i}. params={r['params']}, mean_acc={r['mean_acc']:.4f} ± {r['std_acc']:.4f}, macro_f1={r['macro_f1']:.4f}\n")

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

    print("Preparando features de treino/teste...")
    X_train, X_test, y_train, cols_to_drop = prepare_features(df_train, df_test)

    print("Executando tuning com GroupKFold...")
    best, results_sorted = run_groupkfold_tuning(df_train, cols_to_drop)

    print("Treinando modelo final e avaliando em 2018 sem product_id...")
    test_acc_final, report_text = train_final_and_eval(df_train, df_test, cols_to_drop, best)

    print("Gravando relatório de tuning...")
    write_md_report(md_out, best, results_sorted, test_acc_final, report_text)


if __name__ == '__main__':
    main()