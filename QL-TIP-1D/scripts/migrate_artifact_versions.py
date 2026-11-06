"""Migrate only mismatched registered artifacts to immutable SHA history.

This is version management, not scientific execution. It never changes an old
artifact's recorded SHA; it relocates the old bytes and lets the active path be
registered as a new version.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "Q-MATH"
MANIFEST = BASE / "manifest.json"
HISTORY = BASE / "history/by_sha256"
MIGRATION = BASE / "version_migration.json"
CONTROL = {"Q-MATH/README.md"}


def sha(data: bytes) -> str:
    """Return SHA-256 of bytes."""
    return hashlib.sha256(data).hexdigest()


def git(*args: str) -> bytes:
    """Return raw Git bytes without text transcoding."""
    return subprocess.check_output(["git", "-C", str(ROOT), *args])


def find_original(path: str, expected: str) -> bytes:
    """Find a historical Git blob whose bytes match the recorded SHA."""
    commits = git("rev-list", "--all", "--", path).decode().splitlines()
    for commit in commits:
        try:
            data = git("show", f"{commit}:{path}")
        except subprocess.CalledProcessError:
            continue
        if sha(data) == expected:
            return data
    raise RuntimeError(f"No Git blob matches recorded SHA for {path}")


def main() -> int:
    """Run a dry-run or apply the narrowly scoped migration."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    mismatches = []
    for rel, record in list(manifest["artifacts"].items()):
        path = ROOT / rel
        if not path.is_file():
            continue
        current = sha(path.read_bytes())
        if current != record["sha256"]:
            mismatches.append((rel, record["sha256"], current, path.suffix or ".bin"))
    print(json.dumps({"mismatch_count": len(mismatches), "mismatches": mismatches}, indent=2))
    if not args.apply:
        return 0
    HISTORY.mkdir(parents=True, exist_ok=True)
    records = []
    updated_artifacts = dict(manifest["artifacts"])
    replacements = {}
    for rel, old_sha, current_sha, suffix in mismatches:
        old_bytes = find_original(rel, old_sha)
        history_rel = f"Q-MATH/history/by_sha256/{old_sha}{suffix}"
        history_path = ROOT / history_rel
        if history_path.exists() and sha(history_path.read_bytes()) != old_sha:
            raise RuntimeError(f"History path has different bytes: {history_rel}")
        if not history_path.exists():
            history_path.write_bytes(old_bytes)
        record = updated_artifacts.pop(rel)
        record["origins"] = list(record.get("origins", [])) + [f"migrated-from:{rel}"]
        updated_artifacts[history_rel] = record
        replacements[rel] = history_rel
        records.append({"original_path": rel, "old_sha256": old_sha, "current_sha256": current_sha,
                        "history_path": history_rel, "git_commit": "history-resolved-by-blob",
                        "purpose": "preserve original bytes while active path changes"})
        if rel in CONTROL:
            # CONTROL files must not remain immutable artifacts.
            continue
    for package in manifest.get("packages", {}).values():
        for name, rel in list(package.get("files", {}).items()):
            if rel in replacements:
                package["files"][name] = replacements[rel]
        for excluded in package.get("excluded", []):
            value = excluded.get("file")
            if value in replacements:
                excluded["file"] = replacements[value]
    manifest["artifacts"] = updated_artifacts
    MIGRATION.write_text(json.dumps({"schema": 1, "records": records}, indent=2, ensure_ascii=False) + "\n",
                         encoding="utf-8")
    MANIFEST.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"migrated={len(records)} manifest={MANIFEST}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
