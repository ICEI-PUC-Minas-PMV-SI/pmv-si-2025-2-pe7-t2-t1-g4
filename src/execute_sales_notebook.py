import sys
from pathlib import Path
import nbformat
from nbclient import NotebookClient


def detect_section_titles(nb):
    sections = {}
    current = "(Sem seção)"
    for idx, cell in enumerate(nb.cells):
        if cell.get("cell_type") == "markdown":
            text = cell.get("source", "")
            # Captura títulos de sessão (H2/H3 ou linhas com "Sessão"/"Etapa")
            lines = text.splitlines()
            for line in lines:
                s = line.strip()
                if s.startswith("##"):
                    # Remove hashes e espaços
                    title = s.lstrip("#").strip()
                    current = title or current
                    break
                if ("Sessão" in s) or ("Etapa" in s):
                    current = s
                    break
        sections[idx] = current
    return sections


def collect_errors(nb):
    errors = []
    for i, cell in enumerate(nb.cells):
        if cell.get("cell_type") != "code":
            continue
        outputs = cell.get("outputs", [])
        for out in outputs:
            if out.get("output_type") == "error":
                errors.append({
                    "cell_index": i,
                    "ename": out.get("ename"),
                    "evalue": out.get("evalue"),
                    "traceback": out.get("traceback", []),
                })
    return errors


def main():
    if len(sys.argv) < 2:
        print("Uso: python src/execute_sales_notebook.py <path_para_notebook>")
        sys.exit(2)
    nb_path = Path(sys.argv[1])
    if not nb_path.exists():
        print(f"ERROR: arquivo não encontrado: {nb_path}")
        sys.exit(3)

    nb = nbformat.read(nb_path, as_version=4)
    sections = detect_section_titles(nb)

    print(f"[INFO] Executando notebook: {nb_path}")
    client = NotebookClient(nb, timeout=180, allow_errors=True)
    client.execute()

    # Salva notebook executado
    exec_out = nb_path.with_name(nb_path.stem + "_exec.ipynb")
    nbformat.write(nb, exec_out)
    print(f"[INFO] Notebook executado salvo em: {exec_out}")

    errors = collect_errors(nb)
    if not errors:
        print("[OK] Nenhum erro durante a execução.")
        return

    # Resumo por seção
    log_path = nb_path.with_name(nb_path.stem + "_exec_log.txt")
    with open(log_path, "w", encoding="utf-8") as log:
        for err in errors:
            idx = err["cell_index"]
            section = sections.get(idx, "(Sem seção)")
            ename = err.get("ename")
            evalue = err.get("evalue")
            tb = "\n".join(err.get("traceback", [])[:6])
            msg = (
                f"\n[ERRO] Seção: {section}\n"
                f"Cell #{idx} — {ename}: {evalue}\n"
                f"Traceback:\n{tb}\n"
            )
            log.write(msg)
            print(msg)
    print(f"[INFO] Log de execução salvo em: {log_path}")


if __name__ == "__main__":
    main()