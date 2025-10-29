# Instruções de utilização

Este diretório contém os notebooks e scripts principais do projeto. Abaixo há um quickstart para instalação e execução. Para detalhes completos (dependências, alternativas e troubleshooting), consulte `../README-dev.md`.

## Instalação rápida (uv)

```
pip install uv
uv venv
uv pip install -e .
```

## Alternativa (pip + venv)

```
python -m venv .venv
.\.venv\Scripts\activate
pip install -e .
```

## Executar pipeline rápido

```
cd src
python execute_sales_notebook.py sales.ipynb
python generate_sales_report_html.py
```

## Abrir JupyterLab

```
cd ..
python -m jupyterlab --no-browser --ip=127.0.0.1 --port 8888
```

## Artefatos gerados

- `src/sales_exec.ipynb` e `src/sales_exec_log.txt`
- `src/sales_report.html` (cópia também na raiz: `../sales_report.html`)

## Guia completo

Para instruções detalhadas (setup, scripts de modelagem, dicas e solução de problemas), veja `../README-dev.md`.

## Instalação do Site

O site em HTML/CSS/JS é um projeto estático, logo pode ser utilizado tanto em servidores...

## Histórico de versões

### [0.1.0] - DD/MM/AAAA
#### Adicionado
- Adicionado ...