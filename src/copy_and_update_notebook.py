from pathlib import Path
import shutil
import json


def ensure_backup(src_nb: Path, backup_nb: Path):
    if not src_nb.exists():
        raise FileNotFoundError(f"Notebook não encontrado: {src_nb}")
    if not backup_nb.exists():
        shutil.copy2(src_nb, backup_nb)
        print(f"Backup criado: {backup_nb}")
    else:
        print(f"Backup já existe: {backup_nb}")


def append_etapa4_cells(nb_path: Path):
    with open(nb_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)

    new_cells = []

    def md_cell(text: str):
        return {
            "cell_type": "markdown",
            "metadata": {},
            "source": text,
        }

    def code_cell(code: str):
        return {
            "cell_type": "code",
            "metadata": {},
            "execution_count": None,
            "outputs": [],
            "source": code,
        }

    md_intro = (
        "# Etapa 4 — Modelos, Avaliação e Comparação\n\n"
        "Este notebook integra os resultados da Etapa 4 (construção de modelos, validação por grupos e avaliação em 2018) ao fluxo existente. "
        "A métrica principal é Macro F1, com acurácia como auxiliar.\n\n"
        "- Relatório consolidado: `src/relatorio_etapa4.md`\n"
        "- Figuras geradas em: `src/sales_files/`\n"
    )
    new_cells.append(md_cell(md_intro))

    code_show_report = (
        "from IPython.display import Markdown, display\n"
        "import os\n\n"
        "md_path = 'relatorio_etapa4.md'\n"
        "with open(md_path, 'r', encoding='utf-8') as f:\n"
        "    text = f.read()\n"
        "display(Markdown(text))\n"
    )
    new_cells.append(code_cell(code_show_report))

    code_show_images = (
        "from IPython.display import Image, display\n"
        "import os\n\n"
        "base = 'sales_files'\n"
        "imgs = [\n"
        "    ('LogisticRegression', os.path.join(base, 'cm_LogisticRegression.png')),\n"
        "    ('HistGradientBoosting', os.path.join(base, 'cm_HistGradientBoosting.png')),\n"
        "    ('RandomForestClassifier', os.path.join(base, 'cm_RandomForestClassifier.png')),\n"
        "    ('CatBoostClassifier', os.path.join(base, 'cm_CatBoostClassifier.png')),\n"
        "]\n\n"
        "found = False\n"
        "for name, p in imgs:\n"
        "    if os.path.exists(p):\n"
        "        found = True\n"
        "        print(f'Matriz de Confusão — {name}')\n"
        "        display(Image(filename=p))\n\n"
        "if not found:\n"
        "    print('Nenhuma imagem de matriz de confusão encontrada em sales_files/')\n"
    )
    new_cells.append(code_cell(code_show_images))

    md_reexec = (
        "### Reexecutar Etapa 4 (opcional)\n\n"
        "Execute a célula abaixo para regenerar o relatório e figuras da Etapa 4 usando `src/etapa4_models.py`.\n"
    )
    new_cells.append(md_cell(md_reexec))

    code_reexec = (
        "import subprocess, sys\n"
        "print('Executando: python src/etapa4_models.py')\n"
        "res = subprocess.run([sys.executable, 'src/etapa4_models.py'], capture_output=True, text=True)\n"
        "print(res.stdout)\n"
        "print(res.stderr)\n"
    )
    new_cells.append(code_cell(code_reexec))

    nb.setdefault("cells", [])
    nb["cells"].extend(new_cells)

    with open(nb_path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, ensure_ascii=False, indent=1)
    print(f"Células da Etapa 4 anexadas em: {nb_path}")


def main():
    src_dir = Path(__file__).resolve().parent
    sales_nb = src_dir / 'sales.ipynb'
    backup_nb = src_dir / 'sales_etapa3.ipynb'

    ensure_backup(sales_nb, backup_nb)
    append_etapa4_cells(sales_nb)


if __name__ == '__main__':
    main()