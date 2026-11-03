"""Index explicitly scoped project artifacts; deduplicate failed-run copies only.

Does not execute scientific code, import data from elsewhere, or stage Git files.
The only optional deletion is a SHA-identical r1 artifact already stored in r2.
"""
from pathlib import Path
import argparse
import csv
import hashlib
import json

BASE = Path(__file__).resolve().parents[1]
OUTPUTS = {'data/manifest.csv', 'file_inventory.json'}
FOLDERS = ['configs', 'src', 'scripts', 'tests', 'validation', 'docs',
           'reports', 'data', 'results', 'logs']
EXTENSIONS = {'.py', '.md', '.json', '.csv', '.npz', '.txt', '.wl', '.log'}

def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

def candidates():
    paths = [BASE/'README.md', BASE/'requirements-lock.txt',
             BASE/'references/source_inventory.json', BASE/'references/README.md']
    for folder in FOLDERS:
        paths += [p for p in (BASE/folder).rglob('*') if p.is_file()
                  and p.suffix in EXTENSIONS and '__pycache__' not in p.parts]
    return sorted(set(p for p in paths if p.is_file()
                      and p.relative_to(BASE).as_posix() not in OUTPUTS))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--deduplicate', action='store_true')
    args = parser.parse_args()
    mapping = {}
    # Retain the complete accepted r2 files, while recording r1 historical names.
    for folder in ['data/raw', 'data/origin']:
        failed = BASE/folder/'stage1_20261103_r1'
        passed = BASE/folder/'stage1_20261103_r2'
        for src in sorted(failed.glob('*')):
            dst = passed/src.name
            if src.is_file() and dst.is_file() and sha(src) == sha(dst):
                mapping[src.relative_to(BASE).as_posix()] = dst.relative_to(BASE).as_posix()
    # Preserve earlier dedup mappings on subsequent refreshes.
    old_path = BASE/'file_inventory.json'
    if old_path.exists():
        old = json.loads(old_path.read_text(encoding='utf-8'))
        for item in old['files']:
            if item['historical_path'] != item['stored_path']:
                mapping.setdefault(item['historical_path'], item['stored_path'])
    records = []
    all_paths = {p.relative_to(BASE).as_posix():p for p in candidates()}
    for src in mapping:
        all_paths.setdefault(src, BASE/src)
    for rel in sorted(all_paths):
        stored = mapping.get(rel, rel)
        p = (BASE/stored).resolve(strict=True)
        p.relative_to(BASE.resolve())
        run = next((part for part in Path(rel).parts if part.startswith(('stage1_2026', 'stage2_2026'))), '')
        record = dict(historical_path=rel, stored_path=stored, sha256=sha(p),
                      bytes=p.stat().st_size, run_id=run,
                      status='failed_serialization_preserved' if run=='stage1_20261103_r1' else 'saved')
        if record['bytes'] > 2*1024*1024:
            raise ValueError(f'Above 2MiB commit limit: {stored}')
        records.append(record)
    (BASE/'data').mkdir(exist_ok=True)
    with (BASE/'data/manifest.csv').open('w', encoding='utf-8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['historical_path','stored_path','sha256','bytes','run_id','status'])
        w.writeheader(); w.writerows(records)
    old_path.write_text(json.dumps(dict(schema=1, date='2026-11-04',
        scope='Explicit QL-TIP-1D project files; excludes environments, copyright full texts and previews',
        files=records), ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    # Write the reconstruction map before removing only verified duplicate bytes.
    if args.deduplicate:
        for historical, stored in mapping.items():
            src, dst = (BASE/historical).resolve(), (BASE/stored).resolve(strict=True)
            src.relative_to(BASE.resolve()); dst.relative_to(BASE.resolve())
            if src.exists():
                if src == dst or sha(src) != sha(dst):
                    raise ValueError(f'Duplicate verification failed: {historical}')
                src.unlink()
    print(json.dumps(dict(historical_files=len(records), unique_files=len({r['stored_path'] for r in records}),
                          dedup_mappings=len(mapping), total_unique_bytes=sum(
                              (BASE/p).stat().st_size for p in {r['stored_path'] for r in records}),
                          deduplicated=args.deduplicate), indent=2))

if __name__ == '__main__':
    main()
