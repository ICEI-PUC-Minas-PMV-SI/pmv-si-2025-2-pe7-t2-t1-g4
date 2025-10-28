from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import GroupKFold
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
try:
    from catboost import CatBoostClassifier
    HAS_CATBOOST = True
except Exception:
    CatBoostClassifier = None
    HAS_CATBOOST = False


def load_data(csv_path: Path) -> pd.DataFrame:
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
    ordinal_mapping = {'very low': 0, 'low': 1, 'neutral': 2, 'high': 3, 'very high': 4}
    df['demand_ord'] = df['demand'].map(ordinal_mapping).astype('int8')
    return ordinal_mapping


def derive_labels(df: pd.DataFrame) -> pd.DataFrame:
    product_demand = df.groupby('product_id')['sales'].sum().reset_index()
    labels = ['very low', 'low', 'neutral', 'high', 'very high']
    product_demand['demand'] = pd.qcut(product_demand['sales'], q=5, labels=labels)
    df = pd.merge(left=df, right=product_demand[['product_id','demand']], on='product_id', how='left')
    _ = map_target(df)
    return df


def split_train_test(df: pd.DataFrame):
    if 'order_date' not in df.columns:
        raise ValueError("Coluna 'order_date' não encontrada para efetuar split temporal")
    train_mask = df['order_date'].dt.year.isin([2015, 2016, 2017])
    test_mask = df['order_date'].dt.year == 2018
    df_train = df.loc[train_mask].reset_index(drop=True)
    df_test = df.loc[test_mask].reset_index(drop=True)
    return df_train, df_test


def select_features(df_train: pd.DataFrame, df_test: pd.DataFrame):
    cols_drop = ['demand', 'demand_ord', 'row_id', 'order_id', 'customer_name', 'product_name', 'order_date', 'ship_date']
    X_train = df_train.drop(columns=cols_drop + ['product_id'], errors='ignore')
    X_test = df_test.drop(columns=cols_drop + ['product_id'], errors='ignore')
    y_train = df_train['demand_ord']
    y_test = df_test['demand_ord']
    # Selecionar categóricas de menor cardinalidade para evitar explosão de one-hot
    allowed_cats = {'ship_mode','segment','country','state','region','category','sub_category'}
    cat_cols = [c for c in X_train.columns if X_train[c].dtype == 'object' and c in allowed_cats]
    num_cols = [c for c in X_train.columns if X_train[c].dtype != 'object']
    return X_train, X_test, y_train, y_test, cat_cols, num_cols


def build_preprocessor(cat_cols, num_cols):
    categorical = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    numeric = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median'))
    ])
    preprocessor = ColumnTransformer(
        transformers=[
            ('categorical', categorical, cat_cols),
            ('numeric', numeric, num_cols)
        ],
        remainder='drop'
    )
    return preprocessor


def build_cat_features(X: pd.DataFrame):
    return [i for i, col in enumerate(X.columns) if X[col].dtype == 'object']


def run_groupkfold_cv(X: pd.DataFrame, y: pd.Series, groups: pd.Series, pipeline: Pipeline, n_splits: int = 5):
    gkf = GroupKFold(n_splits=n_splits)
    fold_acc = []
    oof_preds = np.empty(len(X), dtype=int)
    for fold_idx, (tr_idx, val_idx) in enumerate(gkf.split(X, y, groups)):
        X_tr, X_val = X.iloc[tr_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[tr_idx], y.iloc[val_idx]
        pipeline.fit(X_tr, y_tr)
        val_preds = pipeline.predict(X_val).astype(int)
        acc = accuracy_score(y_val, val_preds)
        fold_acc.append(acc)
        oof_preds[val_idx] = val_preds
        print(f"[CV] Fold {fold_idx+1}/{n_splits} acc={acc:.4f}")
    mean_acc = float(np.mean(fold_acc))
    std_acc = float(np.std(fold_acc))
    macro_f1 = float(f1_score(y, oof_preds, average='macro'))
    return mean_acc, std_acc, macro_f1


def run_catboost_groupkfold_cv(X: pd.DataFrame, y: pd.Series, groups: pd.Series, params: dict, n_splits: int = 5):
    if not HAS_CATBOOST:
        raise RuntimeError("CatBoost não disponível no ambiente")
    gkf = GroupKFold(n_splits=n_splits)
    fold_acc = []
    oof_preds = np.empty(len(X), dtype=int)
    cat_features = build_cat_features(X)
    for fold_idx, (tr_idx, val_idx) in enumerate(gkf.split(X, y, groups)):
        X_tr, X_val = X.iloc[tr_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[tr_idx], y.iloc[val_idx]
        model = CatBoostClassifier(
            loss_function='MultiClass',
            random_state=42,
            verbose=False,
            thread_count=-1,
            **params
        )
        model.fit(X_tr, y_tr, cat_features=cat_features)
        val_preds = model.predict(X_val).astype(int).flatten()
        acc = accuracy_score(y_val, val_preds)
        fold_acc.append(acc)
        oof_preds[val_idx] = val_preds
        print(f"[CV-CB] Fold {fold_idx+1}/{n_splits} acc={acc:.4f}")
    mean_acc = float(np.mean(fold_acc))
    std_acc = float(np.std(fold_acc))
    macro_f1 = float(f1_score(y, oof_preds, average='macro'))
    return mean_acc, std_acc, macro_f1


def eval_on_test(pipeline: Pipeline, X_train, y_train, X_test, y_test, fig_prefix: str):
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test).astype(int)
    acc = float(accuracy_score(y_test, y_pred))
    report = classification_report(y_test, y_pred)
    # Matriz de confusão
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(7, 6))
    from sklearn.metrics import ConfusionMatrixDisplay
    ConfusionMatrixDisplay(confusion_matrix=cm).plot(ax=ax, cmap='Blues')
    ax.set_title(f'Matriz de Confusão — 2018 ({fig_prefix})')
    fig.tight_layout()
    out_dir = Path(__file__).resolve().parent / 'sales_files'
    img_path = out_dir / f'cm_{fig_prefix}.png'
    fig.savefig(img_path)
    plt.close(fig)
    print(f"Figura de confusão salva em: {img_path}")
    return acc, report


def eval_catboost_on_test(X_train, y_train, X_test, y_test, params: dict, fig_prefix: str):
    if not HAS_CATBOOST:
        raise RuntimeError("CatBoost não disponível no ambiente")
    cat_features = build_cat_features(X_train)
    model = CatBoostClassifier(
        loss_function='MultiClass',
        random_state=42,
        verbose=False,
        thread_count=-1,
        **params
    )
    model.fit(X_train, y_train, cat_features=cat_features)
    y_pred = model.predict(X_test).astype(int).flatten()
    acc = float(accuracy_score(y_test, y_pred))
    report = classification_report(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(7, 6))
    from sklearn.metrics import ConfusionMatrixDisplay
    ConfusionMatrixDisplay(confusion_matrix=cm).plot(ax=ax, cmap='Blues')
    ax.set_title(f'Matriz de Confusão — 2018 ({fig_prefix})')
    fig.tight_layout()
    out_dir = Path(__file__).resolve().parent / 'sales_files'
    img_path = out_dir / f'cm_{fig_prefix}.png'
    fig.savefig(img_path)
    plt.close(fig)
    print(f"Figura de confusão salva em: {img_path}")
    return acc, report


def write_report(md_path: Path, details: dict):
    lines = []
    lines.append("# Etapa 4 — Construção de Modelos, Avaliação e Comparação\n\n")
    lines.append("## Pré-processamento e Ajustes\n")
    lines.append("- Remoção de 'product_id' dos features para evitar vazamento por produto.\n")
    lines.append("- Codificação categórica via OneHotEncoder em colunas de menor cardinalidade: ['ship_mode','segment','country','state','region','category','sub_category'].\n")
    lines.append("- Imputação: categóricas por moda, numéricas por mediana.\n")
    lines.append("- Validação por produto com GroupKFold (n_splits=5).\n\n")

    lines.append("## Modelos Implementados\n")
    lines.append("- LogisticRegression (multinomial, solver=saga): baseline linear, robusta em alta dimensionalidade após one-hot.\n")
    lines.append("- HistGradientBoostingClassifier: boosting baseado em histogramas, forte em dados tabulares.\n\n")

    lines.append("## Métrica Principal\n")
    lines.append("- Macro F1, com acurácia como métrica auxiliar.\n\n")

    for name, info in details.items():
        lines.append(f"### {name}\n")
        lines.append(f"- CV mean_acc = {info['cv_mean_acc']:.4f} ± {info['cv_std_acc']:.4f}\n")
        lines.append(f"- CV macro_f1 = {info['cv_macro_f1']:.4f}\n")
        lines.append(f"- Test 2018 acc = {info['test_acc']:.3f}\n")
        lines.append("- Classification Report (2018):\n")
        lines.append("```\n")
        lines.append(info['report_text'])
        lines.append("\n```\n\n")

    # Comparação (macro F1)
    lines.append("## Comparação por Macro F1 (CV)\n")
    items = [(n, v['cv_macro_f1']) for n, v in details.items()]
    items_sorted = sorted(items, key=lambda x: x[1], reverse=True)
    for i, (n, score) in enumerate(items_sorted, start=1):
        lines.append(f"{i}. {n}: macro_f1={score:.4f}\n")
    lines.append("\n")

    lines.append("## Refinamento do Pipeline\n")
    lines.append("- Pipeline modular (pré-processamento + modelo), permitindo troca de algoritmos sem alterar etapas anteriores.\n")
    lines.append("- Suporte a validação por grupos e avaliação temporal (2018).\n")
    lines.append("- Próximos passos: ajuste fino de hiperparâmetros, inclusão de encoding mais eficiente para alta cardinalidade (e.g., target/frequency encoding), e comparação com modelos de gradient boosting adicionais.\n")

    md_path.write_text(''.join(lines), encoding='utf-8')
    print(f"Relatório gravado em: {md_path}")


def main():
    cwd = Path(__file__).resolve().parent
    csv_path = cwd / 'data' / 'train.csv'
    md_out = cwd / 'relatorio_etapa4.md'

    print(f"Carregando dados de: {csv_path}")
    df = load_data(csv_path)
    required = {"product_id", "sales", "order_date"}
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"Colunas necessárias ausentes: {missing}. Disponíveis: {list(df.columns)}")

    print("Derivando labels por quantis de sales e mapeando alvo ordinal...")
    df = derive_labels(df)

    print("Realizando split temporal (2015-17 / 2018)...")
    df_train, df_test = split_train_test(df)

    print("Selecionando features e colunas para pré-processamento...")
    X_train, X_test, y_train, y_test, cat_cols, num_cols = select_features(df_train, df_test)
    groups = df_train['product_id']

    preprocessor = build_preprocessor(cat_cols, num_cols)

    models = [
        ("LogisticRegression", LogisticRegression(max_iter=500, multi_class='multinomial', solver='saga')),
        ("HistGradientBoosting", HistGradientBoostingClassifier(max_depth=6, learning_rate=0.06, max_iter=300, random_state=42))
    ]

    details = {}
    for name, est in models:
        print(f"\n===== Rodando CV para {name} =====")
        pipe = Pipeline(steps=[('preprocess', preprocessor), ('model', est)])
        cv_mean_acc, cv_std_acc, cv_macro_f1 = run_groupkfold_cv(X_train, y_train, groups, pipe, n_splits=5)
        print(f"CV {name}: acc={cv_mean_acc:.4f} ± {cv_std_acc:.4f}, macro_f1={cv_macro_f1:.4f}")
        print(f"Avaliando {name} em 2018...")
        test_acc, report_text = eval_on_test(pipe, X_train, y_train, X_test, y_test, fig_prefix=name)
        details[name] = {
            'cv_mean_acc': cv_mean_acc,
            'cv_std_acc': cv_std_acc,
            'cv_macro_f1': cv_macro_f1,
            'test_acc': test_acc,
            'report_text': report_text
        }

    # Terceiro modelo: CatBoost se disponível; caso contrário, fallback para RandomForest
    if HAS_CATBOOST:
        print("\n===== Rodando CV para CatBoostClassifier =====")
        cb_params = {'iterations': 400, 'depth': 6, 'learning_rate': 0.06, 'l2_leaf_reg': 3}
        cv_mean_acc, cv_std_acc, cv_macro_f1 = run_catboost_groupkfold_cv(X_train, y_train, groups, cb_params, n_splits=5)
        print(f"CV CatBoostClassifier: acc={cv_mean_acc:.4f} ± {cv_std_acc:.4f}, macro_f1={cv_macro_f1:.4f}")
        print("Avaliando CatBoostClassifier em 2018...")
        test_acc, report_text = eval_catboost_on_test(X_train, y_train, X_test, y_test, cb_params, fig_prefix='CatBoostClassifier')
        details['CatBoostClassifier'] = {
            'cv_mean_acc': cv_mean_acc,
            'cv_std_acc': cv_std_acc,
            'cv_macro_f1': cv_macro_f1,
            'test_acc': test_acc,
            'report_text': report_text
        }
    else:
        print("\nCatBoost indisponível; usando RandomForestClassifier como terceiro modelo.")
        rf = RandomForestClassifier(n_estimators=400, max_depth=None, random_state=42, n_jobs=-1)
        name = 'RandomForestClassifier'
        print(f"===== Rodando CV para {name} =====")
        pipe = Pipeline(steps=[('preprocess', preprocessor), ('model', rf)])
        cv_mean_acc, cv_std_acc, cv_macro_f1 = run_groupkfold_cv(X_train, y_train, groups, pipe, n_splits=5)
        print(f"CV {name}: acc={cv_mean_acc:.4f} ± {cv_std_acc:.4f}, macro_f1={cv_macro_f1:.4f}")
        print(f"Avaliando {name} em 2018...")
        test_acc, report_text = eval_on_test(pipe, X_train, y_train, X_test, y_test, fig_prefix=name)
        details[name] = {
            'cv_mean_acc': cv_mean_acc,
            'cv_std_acc': cv_std_acc,
            'cv_macro_f1': cv_macro_f1,
            'test_acc': test_acc,
            'report_text': report_text
        }

    print("\nGerando relatório da Etapa 4...")
    write_report(md_out, details)


if __name__ == '__main__':
    main()