"""Small diagnostic artifacts: provenance and atomic, non-overwriting writes."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import tempfile


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def provenance(paths):
    return {str(Path(p).resolve()): file_sha256(p) for p in paths}


def write_new_json(path, report):
    """Publish a complete file without replacing an existing result (NTFS)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"Preserve existing audit; choose a new output: {path}")
    temp = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                         prefix=".audit-", suffix=".tmp", delete=False) as stream:
            temp = Path(stream.name)
            json.dump(report, stream, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        # Windows rename is atomic and refuses to replace an existing name.
        # On POSIX rename would replace it; a hard link preserves exclusivity.
        if os.name == "nt":
            os.rename(temp, path)
        else:
            os.link(temp, path)
    finally:
        if temp is not None:
            temp.unlink(missing_ok=True)
