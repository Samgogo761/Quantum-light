#!/usr/bin/env python3
"""Inventory UPF atomic wavefunctions vs nonlocal beta projectors.

projwfc.x uses PP_PSWFC / PP_CHI channels, not PP_BETA. A UPF may have
l=2 beta (PAW reconstruction) without an I-d atomic-wfc channel.
"""
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


def parse_upf(path: Path) -> dict:
    text = path.read_text(errors="replace")
    info: dict = {
        "file": path.name,
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
    ):
        m = re.search(rf"{key}\s*=\s*'?([^'\n]+)'?", text, re.I)
        if m:
            info[key] = m.group(1).strip().strip("'\"")

    chis = []
    for m in re.finditer(r"<PP_CHI\.(\d+)([^>]*)>", text):
        attrs = m.group(2)
        chis.append(
            {
                "index": int(m.group(1)),
                "label": _attr(attrs, "label"),
                "n": _attr(attrs, "n"),
                "l": _attr(attrs, "l"),
                "occupation": _attr(attrs, "occupation"),
                "source": "PP_CHI",
            }
        )
    if not chis:
        m = re.search(r"<PP_PSWFC>(.*?)</PP_PSWFC>", text, re.S | re.I)
        if m:
            for line in m.group(1).splitlines():
                s = line.strip()
                if not s or s.startswith("#") or s.startswith("<"):
                    continue
                if re.search(r"[spdfg]", s, re.I) and re.search(r"\d", s):
                    chis.append({"raw": s[:160], "source": "PP_PSWFC_text"})

    betas = []
    for m in re.finditer(r"<PP_BETA\.(\d+)([^>]*)>", text):
        attrs = m.group(2)
        betas.append(
            {
                "index": int(m.group(1)),
                "label": _attr(attrs, "label"),
                "l": _attr(attrs, "l"),
                "source": "PP_BETA",
            }
        )

    info["atomic_wfc"] = chis
    info["n_atomic_wfc"] = len(chis)
    info["beta"] = betas
    info["n_beta"] = len(betas)
    info["atomic_wfc_l"] = sorted(
        {str(c.get("l")) for c in chis if c.get("l") is not None}
    )
    info["beta_l"] = sorted({str(b.get("l")) for b in betas if b.get("l") is not None})
    labels = [str(c.get("label") or c.get("raw") or "") for c in chis]
    info["has_d_atomic_wfc"] = any(
        (c.get("l") == "2") or re.search(r"(^|[^a-z])d([^a-z]|$)", str(c.get("label") or ""), re.I)
        for c in chis
    )
    info["projwfc_can_see_d"] = info["has_d_atomic_wfc"]
    info["labels"] = labels
    return info


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("upf", nargs="+", type=Path)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    report = {"schema": "upf_pswfc_inventory_v1", "files": []}
    for p in args.upf:
        report["files"].append(parse_upf(p))
    text = json.dumps(report, indent=2)
    print(text)
    if args.out is not None:
        args.out.write_text(text, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
