#!/usr/bin/env python3
import json
import shutil
from pathlib import Path
import argparse


def main(args):
    nb_path = Path(args.file)
    if not nb_path.exists():
        raise FileNotFoundError(f"Notebook não encontrado: {nb_path}")

    backup_path = nb_path.with_suffix('.ipynb.bak')
    print('Lendo notebook:', nb_path)
    with open(nb_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    shutil.copy2(nb_path, backup_path)
    print('Backup criado em:', backup_path)

    changed_count = 0

    moji_tokens = (
        'Ã', 'Â', 'â€”', 'â€“', 'â€œ', 'â€˜', 'â€™', 'â€³', 'â€¢', 'â€¦', '�'
    )
    skip_keys = {'image/png', 'image/jpeg', 'image/svg+xml'}

    def looks_mojibake(s: str) -> bool:
        return any(tok in s for tok in moji_tokens)

    def fix_once(s: str) -> str:
        try:
            return s.encode('latin-1', errors='ignore').decode('utf-8', errors='ignore')
        except Exception:
            return s

    def fix_repeat(s: str) -> str:
        nonlocal changed_count
        prev = s
        for _ in range(3):
            new = fix_once(prev)
            if new == prev:
                break
            prev = new
        if prev != s:
            changed_count += 1
        return prev

    def deep_fix(obj, path=()):
        if isinstance(obj, str):
            if path and path[-1] in skip_keys:
                return obj
            if looks_mojibake(obj):
                return fix_repeat(obj)
            return obj
        if isinstance(obj, list):
            return [deep_fix(v, path) for v in obj]
        if isinstance(obj, dict):
            return {k: deep_fix(v, path + (k,)) for k, v in obj.items()}
        return obj

    fixed = deep_fix(data)

    with open(nb_path, 'w', encoding='utf-8') as f:
        json.dump(fixed, f, ensure_ascii=False, indent=1)

    print('Strings corrigidas:', changed_count)

    with open(nb_path, 'r', encoding='utf-8') as f:
        s = f.read()
    residual = sum(s.count(tok) for tok in moji_tokens)
    print('Resíduos de mojibake no arquivo:', residual)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Sanear caracteres mojibake em um notebook .ipynb')
    parser.add_argument('--file', required=True, help='Caminho do notebook .ipynb')
    args = parser.parse_args()
    main(args)