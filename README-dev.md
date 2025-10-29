# Guia de desenvolvimento (README-dev)

Este documento descreve como instalar as dependências, preparar o ambiente e executar os artefatos principais do projeto, tanto via `uv` (recomendado) quanto via `pip + venv`.

## Pré-requisitos
- `python` >= `3.11` (o projeto roda com `3.13` também)
- `git` instalado
- Acesso à internet para baixar pacotes
- Windows 10/11 (testado) — outros OS podem funcionar com ajustes mínimos

## Estrutura relevante do projeto
- `src/pyproject.toml` — especifica dependências do projeto
- `src/data/train.csv` — base de dados utilizada
- `src/etapa4_models.py` — construção e avaliação de modelos (CV, teste 2018)
- `src/sales.ipynb`, `src/sales_etapa4.ipynb` — notebooks principais
- `src/execute_sales_notebook.py` — executa um notebook e salva `*_exec.ipynb`
- `src/generate_sales_report_html.py` — gera `sales_report.html` a partir de Markdown e imagens
- `src/sales_files/*.png` — figuras (matrizes de confusão, etc.)
- `src/relatorio_etapa4.md` — relatório consolidado (Markdown)

## Dependências
Core (segundo `pyproject.toml`):
- `ipykernel`, `jupyterlab`, `jupyter-collaboration`
- `numpy`, `pandas`, `matplotlib`, `seaborn`, `plotly`
- `scikit-learn`, `catboost`, `us`

Opcionais (usados se instalados):
- `xgboost`, `lightgbm`

## Instalação (recomendada): uv
1) Instalar `uv` (uma vez no sistema):
```
powershell -ExecutionPolicy Bypass -Command "iwr -useb https://astral.sh/uv/install.ps1 | iex"
```
2) Clonar e entrar no repositório:
```
git clone <URL-do-GitHub>
cd kfcaio-pmv-si-2025-2-pe7-t2-t1-g4
```
3) Sincronizar dependências do projeto (o `pyproject.toml` está em `src`):
```
cd src
uv sync
```
4) (Opcional) Registrar o kernel do Jupyter com nome amigável:
```
uv run python -m ipykernel install --user --name sales-env --display-name "Python (sales)"
```

## Instalação alternativa: pip + venv
1) Clonar e entrar no repositório:
```
git clone <URL-do-GitHub>
cd kfcaio-pmv-si-2025-2-pe7-t2-t1-g4
```
2) Criar e ativar ambiente virtual:
```
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -U pip
```
3) Instalar dependências principais:
```
pip install ipykernel jupyterlab jupyter-collaboration numpy pandas matplotlib seaborn plotly scikit-learn catboost us
```
4) (Opcional) Instalar pacotes extras:
```
pip install xgboost lightgbm
```
5) (Opcional) Registrar kernel do Jupyter:
```
python -m ipykernel install --user --name sales-env --display-name "Python (sales)"
```

## Execução dos artefatos
Sempre execute comandos que acessam dados com `cwd` em `src` para garantir que `data/train.csv` seja encontrado.

### Pipeline rápido (CLI)
Com `uv`:
```
cd src
uv run python execute_sales_notebook.py sales.ipynb
uv run python generate_sales_report_html.py
```
Com `pip + venv`:
```
cd src
python execute_sales_notebook.py sales.ipynb
python generate_sales_report_html.py
```
Resultado esperado:
- `src/sales_exec.ipynb` gerado/atualizado (execução do notebook)
- `src/sales_report.html` e `sales_report.html` (cópia na raiz)
- `src/relatorio_etapa4.md` e figuras em `src/sales_files/`

### Abrir notebooks no JupyterLab
Com `uv`:
```
uv run jupyter lab
```
Com `pip + venv`:
```
python -m jupyterlab
```
No JupyterLab, abra:
- `src/sales.ipynb`
- `src/sales_etapa4.ipynb`
- `src/sales_exec.ipynb`

### Scripts de modelagem
- Treino final (CatBoost) e relatório:
```
cd src
uv run python train_final.py
```
Saída: `src/relatorio_tuning.md`

- Tuning exploratório (grid leve CatBoost):
```
cd src
uv run python tune_catboost.py
```

- Utilitários para notebooks (se necessário):
```
cd src
uv run python copy_and_update_notebook.py
uv run python append_notebook_cells.py
uv run python update_notebook.py
```

## Dicas e solução de problemas
- Codificação de texto: relatórios são gravados em UTF-8. Se visualizar acentos quebrados fora do Jupyter, garanta que o editor/navegador respeita UTF-8. Um fallback ASCII é salvo em `src/relatorio_etapa4_ascii.txt`.
- Erro "Notebook does not appear to be JSON": apague/regenere `src/sales_exec.ipynb` executando `execute_sales_notebook.py` novamente (com `cwd` em `src`).
- `FileNotFoundError: data/train.csv`: rode os scripts a partir de `src` ou ajuste o caminho do CSV.
- Aviso de event loop no Windows (ZeroMQ/tornado): é inofensivo. Para silenciar em automações:
```
python -c "import asyncio; asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())"
```
- Pacotes opcionais `xgboost` e `lightgbm` podem exigir tooling extra. Se a instalação falhar, são opcionais; o pipeline principal roda sem eles.

## Checklist para a equipe
- Instalar `python` >= 3.11 e `git`
- Clonar o repositório e entrar na pasta
- Escolher instalação com `uv` (recomendado) ou `pip + venv`
- Rodar (em `src`): `execute_sales_notebook.py` e `generate_sales_report_html.py`
- Abrir `sales_report.html` (raiz) para visualizar relatório estático ou usar JupyterLab para notebooks