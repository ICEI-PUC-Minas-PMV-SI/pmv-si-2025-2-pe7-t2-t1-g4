from pathlib import Path
import nbformat as nbf


def main():
    cwd = Path(__file__).resolve().parent
    nb_path = cwd / 'sales.ipynb'

    nb = nbf.read(nb_path.open('r', encoding='utf-8'), as_version=4)

    md_lines = [
        "# Resumo do Tuning Final (CatBoost)\n",
        "\n",
        "- Hiperparâmetros: `iterations=400`, `depth=6`, `learning_rate=0.06`, `l2_leaf_reg=3`.\n",
        "- Validação: `GroupKFold` (5 folds) por `product_id`, sem `product_id` nos features.\n",
        "- Resultados CV (depth=6): `mean_acc=0.5986 ± 0.0179`, `macro_f1=0.5853`.\n",
        "- Avaliação 2018 (sem `product_id`): Acurácia `0.587`, Macro F1 `0.58`.\n",
        "- Relatório detalhado: [relatorio_tuning.md](./relatorio_tuning.md).\n",
    ]

    new_cell = nbf.v4.new_markdown_cell(''.join(md_lines))
    nb.cells.append(new_cell)

    nbf.write(nb, nb_path.open('w', encoding='utf-8'))
    print(f"Added tuning summary cell to {nb_path}")


if __name__ == '__main__':
    main()