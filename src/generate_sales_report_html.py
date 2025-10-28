from pathlib import Path
import unicodedata

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="pt-BR">
<head>
  <meta charset="utf-8" />
  <title>Etapa 4 — Relatório e Visualizações</title>
  <style>
    body {{ font-family: Arial, sans-serif; line-height: 1.5; margin: 24px; }}
    h1, h2 {{ color: #222; }}
    .section {{ margin-bottom: 28px; }}
    .images {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(360px, 1fr)); gap: 16px; }}
    .img-card {{ border: 1px solid #ddd; border-radius: 6px; padding: 8px; }}
    .img-card img {{ max-width: 100%; height: auto; display: block; }}
    pre {{ background: #f7f7f7; padding: 12px; border-radius: 6px; white-space: pre-wrap; }}
    a {{ color: #0645ad; }}
  </style>
  </head>
<body>
  <h1>Etapa 4 — Relatório e Visualizações</h1>
  <div class="section">
    <p>Este documento estático consolida o relatório da Etapa 4 e as matrizes de confusão geradas. Para executar novamente, use o script <code>src/etapa4_models.py</code> ou abra o notebook <code>src/sales.ipynb</code> (quando o ambiente Jupyter estiver disponível).</p>
    <ul>
      <li><a href="relatorio_etapa4.md">Abrir relatorio_etapa4.md (Markdown)</a></li>
      <li><a href="sales.ipynb">Abrir sales.ipynb (Notebook)</a> — requer suporte a Jupyter no navegador/IDE.</li>
    </ul>
  </div>

  <div class="section">
    <h2>Relatório — Markdown bruto</h2>
    <pre>{md_content}</pre>
  </div>

  <div class="section">
    <h2>Matrizes de Confusão</h2>
    <div class="images">
      {images_html}
    </div>
  </div>
</body>
</html>
"""


def main():
    root = Path(__file__).resolve().parent
    md_path = root / 'relatorio_etapa4.md'
    images_dir = root / 'sales_files'
    out_path = root / 'sales_report.html'

    md_content = "(relatorio_etapa4.md não encontrado)"
    if md_path.exists():
        md_content = md_path.read_text(encoding='utf-8')
        # Fallback ASCII (sem acentos) para visualizadores que não respeitam UTF-8
        ascii_fallback = unicodedata.normalize('NFKD', md_content).encode('ascii', 'ignore').decode('ascii')
        (root / 'relatorio_etapa4_ascii.txt').write_text(ascii_fallback, encoding='utf-8')

    candidates = [
        ('LogisticRegression', images_dir / 'cm_LogisticRegression.png'),
        ('HistGradientBoosting', images_dir / 'cm_HistGradientBoosting.png'),
        ('RandomForestClassifier', images_dir / 'cm_RandomForestClassifier.png'),
        ('CatBoostClassifier', images_dir / 'cm_CatBoostClassifier.png'),
    ]

    cards = []
    for name, p in candidates:
        if p.exists():
            rel = p.relative_to(root)
            cards.append(f'<div class="img-card"><strong>{name}</strong><img src="{rel.as_posix()}" alt="{name}" /></div>')

    if not cards:
        cards.append('<p>Nenhuma figura encontrada em <code>src/sales_files/</code>.</p>')

    html = HTML_TEMPLATE.format(md_content=md_content, images_html='\n'.join(cards))
    out_path.write_text(html, encoding='utf-8')
    print(f"HTML gerado em: {out_path}")
    # Também gerar uma cópia na raiz do projeto para facilitar abertura direta
    repo_root = root.parent
    (repo_root / 'sales_report.html').write_text(html, encoding='utf-8')
    print(f"HTML gerado em (cópia): {repo_root / 'sales_report.html'}")


if __name__ == '__main__':
    main()