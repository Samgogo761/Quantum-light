"""Hash continuation-v2 local artifacts without modifying scientific files."""
from __future__ import annotations

import csv
import hashlib
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]


def digest(path: Path) -> str:
    """Return a streaming SHA-256 digest."""
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main() -> int:
    """Write a recovery manifest for v2 results and raw states."""
    rows = []
    for root_name in ["results/bias_v2", "data/raw/bias_v2"]:
        root = PROJECT / root_name
        if not root.exists():
            continue
        for path in sorted(item for item in root.rglob("*") if item.is_file()):
            size = path.stat().st_size
            rows.append([str(path.relative_to(PROJECT)).replace("\\", "/"), size, digest(path),
                         "candidate_under_2MiB" if size < 2 * 1024 * 1024 else "local_only_over_2MiB"])
    output = PROJECT / "results/bias_v2/local_artifact_hashes.csv"
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["path", "bytes", "sha256", "archive_eligibility"])
        writer.writerows(rows)
    print(f"hashed={len(rows)} file={output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
