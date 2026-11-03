#!/usr/bin/env python3
"""Q-MATH file inventory, byte-preserving import and local Git checkpointing.

Python standard library only. Never imports or runs research code; never pushes.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import re
import shutil
import subprocess
import sys
from urllib.parse import unquote
import zipfile

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / 'Q-MATH'
MANIFEST = BASE / 'manifest.json'
MAX_BYTES = 2 * 1024 * 1024
ALLOWED = {'.md', '.txt', '.py', '.wl', '.wls', '.json', '.npz', '.npy',
           '.png', '.svg', '.csv', '.tsv', '.in', '.nml', '.sh'}
SECRET = re.compile(rb'-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----|'
                    rb'gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{30,}|'
                    rb'AKIA[A-Z0-9]{16}|sk-[A-Za-z0-9_-]{32,}')
GENERATED = {'Q-MATH/manifest.json', 'Q-MATH/INDEX.md'}
CONTROL = {'AGENTS.md', 'tools/qmath/manage.py', 'Q-MATH/README.md',
           'Q-MATH/SYNC.md', 'Q-MATH/REPRODUCE.md', 'Q-MATH/.gitignore',
           'Q-MATH/.gitattributes', 'Q-MATH/VALIDATION.md',
           'docs/04_BSV量子光/.gitattributes',
           'QL-TIP-1D/.gitignore', 'QL-TIP-1D/.gitattributes'}


def digest(data):
    return hashlib.sha256(data).hexdigest()


def git(*args):
    return subprocess.check_output(['git', '-C', str(ROOT), *args], stderr=subprocess.STDOUT)


def load():
    if MANIFEST.exists():
        return json.loads(MANIFEST.read_text(encoding='utf-8'))
    return {'schema': 1, 'sources': [], 'packages': {}, 'artifacts': {}}


def safe_name(name):
    p = PurePosixPath(name.replace('\\', '/'))
    if p.is_absolute() or any(x in {'..', '.git'} or ':' in x for x in p.parts):
        raise ValueError(f'Unsafe archive path: {name}')
    if not p.parts:
        raise ValueError('Empty path')
    return p.as_posix()


def category(name):
    ext = Path(name).suffix.lower()
    if ext in {'.md', '.txt'}:
        return 'reports'
    if ext in {'.py', '.wl', '.wls', '.sh', '.in', '.nml'}:
        return 'code'
    return 'results'


def acceptable(name, data):
    p = PurePosixPath(name)
    if any(x.startswith('.') or x in {'__pycache__', 'node_modules', '_preview'} for x in p.parts):
        return 'cache-or-hidden'
    if p.suffix.lower() not in ALLOWED:
        return 'format-not-selected (original retained externally)'
    if len(data) > MAX_BYTES:
        return 'larger-than-2MiB (original retained externally)'
    if re.search(r'(?i)(credential|password|id_rsa|id_ed25519|\.env)', name):
        return 'sensitive-name'
    if SECRET.search(data):
        raise ValueError(f'Potential secret detected in {name}; import stopped')
    return None


def existing_hashes(m):
    known = {a['sha256']: p for p, a in m['artifacts'].items()}
    # Reuse existing research files byte-for-byte, without touching their locations.
    candidates = list((ROOT / 'tools/b0_small_model').glob('qmath_r1*'))
    candidates += [ROOT / 'tools/b0_small_model' / n for n in
                   ['q8_m0_model.py', 'q8_tperp_static_audit.py']]
    doc = ROOT / 'docs/04_BSV量子光'
    candidates += list(doc.glob('*QMATH*.md')) + list(doc.glob('qmath*.json'))
    candidates += list(doc.glob('qmath*.txt'))
    candidates += [doc / 'PHYSICS_CONTRACT.md', doc / 'SOURCES_PROVENANCE.md']
    candidates += sorted((doc / 'qmath_pack_16_11_20261023').rglob('*'))
    for p in candidates:
        if not p.is_file() or acceptable(p.relative_to(ROOT).as_posix(), p.read_bytes()):
            continue
        h = digest(p.read_bytes())
        known.setdefault(h, p.relative_to(ROOT).as_posix())
    return known, candidates


def retain(m, known, package, name, data, origin):
    h = digest(data)
    path = known.get(h)
    if path is None:
        path = f'Q-MATH/{category(name)}/{package}/{safe_name(name)}'
        dest = ROOT / path
        if dest.exists() and dest.read_bytes() != data:
            raise ValueError(f'Refusing to overwrite original: {path}')
        dest.parent.mkdir(parents=True, exist_ok=True)
        if not dest.exists():
            dest.write_bytes(data)
        known[h] = path
    a = m['artifacts'].setdefault(path, {'sha256': h, 'bytes': len(data),
                                        'kind': category(name), 'origins': []})
    if a['sha256'] != h:
        raise ValueError(f'Original changed: {path}')
    if origin not in a['origins']:
        a['origins'].append(origin)
    return path


def import_archive(m, known, data, package, label, depth=0):
    if depth > 8:
        raise ValueError('Archive nesting exceeds 8 levels')
    if package in m['packages']:
        if m['packages'][package]['sha256'] != digest(data):
            raise ValueError(f'Package name reused with different bytes: {package}')
        return
    record = {'source': label, 'sha256': digest(data), 'files': {}, 'excluded': []}
    m['packages'][package] = record
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        for entry in archive.infolist():
            if entry.is_dir():
                continue
            name = safe_name(entry.filename)
            if name in record['files']:
                raise ValueError(f'Duplicate archive path: {name}')
            if entry.file_size > 20 * MAX_BYTES:
                record['excluded'].append({'name': name, 'reason': 'oversize archive member'})
                continue
            raw = archive.read(entry)  # Also checks ZIP CRC; no code is executed.
            if name.lower().endswith('.zip'):
                version = re.search(r'Q-MATH-(\d+\.\d+)', name, re.I)
                baseline = re.search(r'baseline_(\d+)_(\d+)', name, re.I)
                if version:
                    child = 'v' + version.group(1)
                elif baseline:
                    child = f'v{baseline.group(1)}.{baseline.group(2)}'
                elif '16_10' in name:
                    child = 'history-to-16.10'
                elif '16_9' in name:
                    child = 'history-to-16.9'
                else:
                    child = package + '__' + re.sub(r'[^A-Za-z0-9_.-]', '_', Path(name).stem)
                import_archive(m, known, raw, child, label + '!' + name, depth + 1)
                record['excluded'].append({'name': name, 'sha256': digest(raw),
                                           'reason': 'nested package catalogued', 'package': child})
                continue
            reason = acceptable(name, raw)
            if reason:
                record['excluded'].append({'name': name, 'sha256': digest(raw), 'reason': reason})
                continue
            origin = label + '!' + name
            record['files'][name] = retain(m, known, package, name, raw, origin)


def import_paths(m, paths, package_override=None):
    known, candidates = existing_hashes(m)
    # Register original in-repository Q-MATH reports and audits as authoritative paths.
    for p in candidates:
        if p.is_file() and not acceptable(p.relative_to(ROOT).as_posix(), p.read_bytes()):
            rel = p.relative_to(ROOT).as_posix()
            retain(m, known, 'line-Q', rel, p.read_bytes(), 'existing-repository:' + rel)
    for p in paths:
        p = p.resolve(strict=True)
        data = p.read_bytes()
        label = p.name  # No private absolute machine path in the shared inventory.
        src = {'name': label, 'sha256': digest(data), 'bytes': len(data)}
        if src in m['sources']:
            continue
        match = re.search(r'(?:Q-?MATH[-_]?)(\d+(?:\.\d+)?)', label, re.I)
        package = ('v' + match.group(1)) if match else (
            'history-to-16.11' if label.startswith('Q-MATH_') else 'foundation-M0')
        if package_override is not None:
            package = safe_name(package_override)
            if '/' in package:
                raise ValueError('Import package must be a single directory name')
        if p.suffix.lower() == '.zip':
            import_archive(m, known, data, package, label)
        else:
            reason = acceptable(label, data)
            if reason:
                raise ValueError(f'{label}: {reason}')
            retain(m, known, package, label, data, label)
        m['sources'].append(src)


def check(m):
    for rel, a in m['artifacts'].items():
        p = ROOT / safe_name(rel)
        if not p.is_file() or digest(p.read_bytes()) != a['sha256']:
            raise ValueError(f'Missing or altered original: {rel}')
        if acceptable(rel, p.read_bytes()):
            raise ValueError(f'Unexpected excluded file: {rel}')
    # No disconnected new research files may be silently left outside the inventory.
    for folder in ['reports', 'code', 'results']:
        for p in (BASE / folder).rglob('*'):
            if p.is_file() and p.relative_to(ROOT).as_posix() not in m['artifacts']:
                raise ValueError(f'Unregistered artifact (use register): {p.relative_to(ROOT)}')
    for pkg in m['packages'].values():
        for rel in pkg['files'].values():
            if rel not in m['artifacts']:
                raise ValueError(f'Unregistered package dependency: {rel}')
    available = set(m['artifacts']) | CONTROL | GENERATED
    available |= set(git('-c', 'core.quotepath=false', 'ls-files', '-z').decode('utf-8').split('\0'))
    for name in ['README.md', 'SYNC.md', 'REPRODUCE.md']:
        p = BASE / name
        if not p.exists():
            continue
        for href in re.findall(r'\]\(([^)]+)\)', p.read_text(encoding='utf-8')):
            if '://' in href or href.startswith('#'):
                continue
            dest = (p.parent / unquote(href.split('#')[0])).resolve()
            rel = dest.relative_to(ROOT).as_posix()
            if not dest.is_file() or rel not in available:
                raise ValueError(f'Link would be missing from Git: {name}: {href}')


def refresh(m):
    check(m)
    BASE.mkdir(exist_ok=True)
    MANIFEST.write_text(json.dumps(m, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    lines = ['# Q-MATH 文件索引（自动生成）', '',
             '由 `python tools/qmath/manage.py refresh` 更新；仅核验文件，不运行科研程序。', '',
             f"归档来源 {len(m['sources'])} 个；独立文件 {len(m['artifacts'])} 个；"
             f"可重建历史包 {len(m['packages'])} 个。", '',
             '所有文件的完整 SHA-256、原包成员名与去重映射见 [manifest.json](manifest.json)。', '',
             '## 报告、程序、结果与测试', '', '| 类型 | 文件 | 字节 | SHA-256 前 12 位 |',
             '|---|---|---:|---|']
    for rel, a in sorted(m['artifacts'].items()):
        href = Path(__import__('os').path.relpath(ROOT / rel, BASE)).as_posix().replace(' ', '%20')
        lines.append(f"| {a['kind']} | [{rel}]({href}) | {a['bytes']} | `{a['sha256'][:12]}` |")
    lines += ['', '## 历史包复现布局', '',
              'prepare 只从已核验文件还原相对目录到忽略目录，不执行程序。', '']
    for name, pkg in m['packages'].items():
        lines.append(f"- `{name}`：{len(pkg['files'])} 个入选成员；{len(pkg['excluded'])} 个排除/嵌套成员。")
    (BASE / 'INDEX.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')


def finish(m, message, resume_owned_index=False):
    staged_before = git('-c', 'core.quotepath=false', 'diff', '--cached', '--name-only').decode('utf-8').splitlines()
    owned = sorted(set(m['artifacts']) | CONTROL | GENERATED)
    if staged_before and not (resume_owned_index and set(staged_before).issubset(owned)):
        raise ValueError('Index already contains staged changes; preserve them and stop')
    refresh(m)
    # Inspect all candidates before staging. Never use git add . / -A.
    for rel in owned:
        if not (ROOT / rel).is_file():
            raise ValueError(f'Missing checkpoint file: {rel}')
        data = (ROOT / rel).read_bytes()
        if len(data) > MAX_BYTES or SECRET.search(data):
            raise ValueError(f'Size/secret check failed: {rel}')
    for i in range(0, len(owned), 40):
        git('add', '--', *owned[i:i + 40])
    # Reapply the byte-preserving attributes even if Git's stat cache says clean.
    for i in range(0, len(owned), 40):
        git('add', '--renormalize', '--', *owned[i:i + 40])
    staged = git('-c', 'core.quotepath=false', 'diff', '--cached', '--name-only').decode('utf-8').splitlines()
    if not set(staged).issubset(owned):
        raise ValueError('Unexpected staged paths; commit stopped')
    # Git attributes must not silently change the imported original bytes.
    for rel in set(staged) & set(m['artifacts']):
        if digest(git('show', ':' + rel)) != m['artifacts'][rel]['sha256']:
            raise ValueError(f'Git newline conversion changed original bytes: {rel}')
    if not staged:
        print('No Q-MATH changes to commit.')
        return
    print(git('commit', '-m', message).decode('utf-8', errors='replace'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('import'); p.add_argument('paths', nargs='+', type=Path)
    p.add_argument('--package', help='Explicit independent-project/version archive label')
    p = sub.add_parser('register'); p.add_argument('paths', nargs='+', type=Path)
    sub.add_parser('refresh'); sub.add_parser('check')
    p = sub.add_parser('finish'); p.add_argument('--message', required=True)
    p.add_argument('--resume-owned-index', action='store_true',
                   help='Resume only staging created by this task after an interrupted finish')
    p = sub.add_parser('prepare'); p.add_argument('package')
    args = parser.parse_args()
    m = load()
    if args.command == 'import':
        import_paths(m, args.paths, args.package); refresh(m)
    elif args.command == 'register':
        known, _ = existing_hashes(m)
        for p in args.paths:
            rel = p.resolve(strict=True).relative_to(ROOT).as_posix()
            raw = p.read_bytes()
            reason = acceptable(rel, raw)
            if reason:
                raise ValueError(reason)
            # Explicit registration preserves its chosen authoritative location.
            known[digest(raw)] = rel
            retain(m, known, 'new-round', rel, raw, 'registered:' + rel)
        refresh(m)
    elif args.command == 'refresh':
        refresh(m)
    elif args.command == 'check':
        check(m); print(f"PASS: {len(m['artifacts'])} original hashes and all package links")
    elif args.command == 'finish':
        finish(m, args.message, args.resume_owned_index)
    elif args.command == 'prepare':
        check(m)
        pkg = m['packages'][args.package]
        target = ROOT / 'local_runs/qmath_reproduce' / safe_name(args.package)
        if target.exists():
            raise ValueError(f'Refusing to overwrite reproduction workspace: {target}')
        for name, rel in pkg['files'].items():
            dest = target / safe_name(name)
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(ROOT / rel, dest)
        print(f'Prepared files only; no research code executed: {target}')


if __name__ == '__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')
    try:
        main()
    except (ValueError, KeyError, OSError, subprocess.CalledProcessError) as exc:
        print(f'ERROR: {exc}', file=sys.stderr)
        if isinstance(exc, subprocess.CalledProcessError) and exc.output:
            print(exc.output.decode('utf-8', errors='replace'), file=sys.stderr)
        raise SystemExit(1)
