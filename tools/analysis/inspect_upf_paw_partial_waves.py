#!/usr/bin/env python3
"""Inventory PAW CHI / BETA / AE-PS partial waves in a UPF v2 file."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _attr(blob: str, key: str) -> str | None:
    m = re.search(rf"""(?:^|[\s]){key}\s*=\s*['"]([^'"]+)['"]""", blob, re.I)
    return m.group(1) if m else None


def _header_val(text: str, key: str) -> str | None:
    m = re.search(rf"{key}\s*=\s*'?([^'\n]+)'?", text, re.I)
    return m.group(1).strip().strip("'\"") if m else None


def _block_floats(body: str) -> list[float]:
    vals: list[float] = []
    for tok in body.replace(",", " ").split():
        try:
            vals.append(float(tok))
        except ValueError:
            continue
    return vals


def parse_upf(path: Path) -> dict:
    text = path.read_text(errors="replace")
    info: dict = {
        "file": path.name,
        "path": str(path),
        "size": path.stat().st_size,
        "sha256": sha256(path),
    }
    for key in (
        "element",
        "pseudo_type",
        "relativistic",
        "is_ultrasoft",
        "is_paw",
        "z_valence",
        "l_max",
        "number_of_wfc",
        "number_of_proj",
        "mesh_size",
    ):
        info[key] = _header_val(text, key)

    tags = sorted(set(re.findall(r"PP_[A-Z][A-Z0-9_]*", text)))
    info["pp_tags"] = tags

    chis = []
    for m in re.finditer(
        r"<PP_CHI\.(\d+)([^>]*)>(.*?)</PP_CHI\.\1>",
        text,
        re.S | re.I,
    ):
        attrs = m.group(2)
        radial = _block_floats(m.group(3))
        chis.append(
            {
                "index": int(m.group(1)),
                "label": _attr(attrs, "label"),
                "n": _attr(attrs, "n"),
                "l": _attr(attrs, "l"),
                "occupation": _attr(attrs, "occupation"),
                "n_mesh": len(radial),
                "abs_max": max((abs(x) for x in radial), default=None),
            }
        )
    info["atomic_wfc"] = chis

    betas = []
    for m in re.finditer(
        r"<PP_BETA\.(\d+)([^>]*)>(.*?)</PP_BETA\.\1>",
        text,
        re.S | re.I,
    ):
        attrs = m.group(2)
        radial = _block_floats(m.group(3))
        betas.append(
            {
                "index": int(m.group(1)),
                "label": _attr(attrs, "label"),
                "l": _attr(attrs, "l"),
                "angular_momentum": _attr(attrs, "angular_momentum"),
                "n_mesh": len(radial),
                "abs_max": max((abs(x) for x in radial), default=None),
            }
        )
    info["beta"] = betas

    partial = []
    for kind in ("AEWFC", "PSWFC", "AEWFC_REL", "PSWFC_REL"):
        for m in re.finditer(
            rf"<PP_{kind}\.(\d+)([^>]*)>(.*?)</PP_{kind}\.\1>",
            text,
            re.S | re.I,
        ):
            attrs = m.group(2)
            radial = _block_floats(m.group(3))
            partial.append(
                {
                    "kind": kind,
                    "index": int(m.group(1)),
                    "label": _attr(attrs, "label"),
                    "l": _attr(attrs, "l") or _attr(attrs, "angular_momentum"),
                    "n_mesh": len(radial),
                    "abs_max": max((abs(x) for x in radial), default=None),
                }
            )
    info["partial_waves"] = partial
    info["n_chi"] = len(chis)
    info["n_beta"] = len(betas)
    info["n_partial"] = len(partial)
    info["has_d_chi"] = any(str(c.get("l")) == "2" for c in chis)
    info["d_beta_indices"] = [
        b["index"] for b in betas if str(b.get("l")) == "2" or str(b.get("label") or "").upper().endswith("D")
    ]
    info["d_partial"] = [
        p
        for p in partial
        if str(p.get("l")) == "2" or str(p.get("label") or "").upper().endswith("D")
    ]
    return info


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("upf", nargs="+", type=Path)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    report = {"schema": "upf_paw_partial_waves_v1", "files": [parse_upf(p) for p in args.upf]}
    text = json.dumps(report, indent=2)
    print(text)
    if args.out is not None:
        args.out.write_text(text, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
