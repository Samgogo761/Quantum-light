"""Hash every local solo-v1 artifact and mark Git-size eligibility."""
from __future__ import annotations

import csv
import hashlib
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]


def digest(path: Path) -> str:
    """Stream a SHA-256 digest without loading large arrays into memory."""
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def main() -> int:
    """Write a recovery table for files that intentionally stay local."""
    rows = []
    for root_name in ["results/bias_v1", "data/raw/bias_v1"]:
        root = BASE / root_name
        if not root.exists():
            continue
        for path in sorted(item for item in root.rglob("*") if item.is_file()):
            size = path.stat().st_size
            rows.append([str(path.relative_to(BASE)).replace("\\", "/"), size, digest(path),
                         "candidate_under_2MiB" if size < 2 * 1024 * 1024 else "local_only_over_2MiB"])
    output = BASE / "results/bias_v1/local_artifact_hashes.csv"
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["path", "bytes", "sha256", "archive_eligibility"])
        writer.writerows(rows)
    print(f"hashed={len(rows)} file={output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
