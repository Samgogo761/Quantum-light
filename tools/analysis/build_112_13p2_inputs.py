#!/usr/bin/env python3
"""Build the 112_[-6,13.2] window-only control from 112_large.

Same num_wann=112, same Cr-d+I-p projectors, same frozen window, same 400-band
NSCF / AMN / MMN / EIG. Only dis_win_max changes 8.0 -> 13.2.

This is the missing control between 112_large[-6,8] and 232_I-d[-6,13.2].
Not production. Not I-d. Not WINDOW144.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

SCHEMA = "ref112_13p2_inputs_v1"
N_WANN = 112
N_BANDS = 400
DIS_WIN = (-6.0, 13.2)
DIS_FROZ = (-1.5, 0.8)


def sha256(path: Path) -> str:
    import hashlib

    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def patch_win(text: str) -> str:
    if "num_wann = 112" not in text:
        raise RuntimeError("expected num_wann = 112")
    if "num_bands = 400" not in text:
        raise RuntimeError("112_13.2 reuses the 400-band NSCF")
    if re.search(r"^I:\s+d\s*$", text, re.M):
        raise RuntimeError("window control must not add I:d")
    if "I:   p" not in text and not re.search(r"^I:\s+p\s*$", text, re.M):
        raise RuntimeError("missing I: p projection")
    if not re.search(r"dis_froz_min\s*=\s*-1\.5", text) or not re.search(
        r"dis_froz_max\s*=\s*0\.8", text
    ):
        raise RuntimeError("frozen window must stay [-1.5, 0.8]")
    out = re.sub(
        r"dis_win_max\s*=\s*8\.0",
        f"dis_win_max  = {DIS_WIN[1]}",
        text,
        count=1,
    )
    if "13.2" not in out:
        raise RuntimeError("failed to set dis_win_max=13.2")
    if "dis_win_max  =  8.0" in out or re.search(r"dis_win_max\s*=\s*8\.0", out):
        raise RuntimeError("old dis_win_max=8.0 still present")
    if "num_wann = 232" in out:
        raise RuntimeError("window control must stay num_wann=112")
    if re.search(r"^I:\s+d\s*$", out, re.M):
        raise RuntimeError("I:d leaked into window control")
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--win-112large", type=Path, required=True)
    ap.add_argument("--out-win", type=Path, required=True)
    ap.add_argument("--out-json", type=Path, required=True)
    args = ap.parse_args()
    win = patch_win(args.win_112large.read_text(encoding="utf-8"))
    args.out_win.parent.mkdir(parents=True, exist_ok=True)
    args.out_win.write_text(win, encoding="utf-8")
    payload = {
        "schema": SCHEMA,
        "status": "WINDOW_ONLY_CONTROL_NOT_PRODUCTION",
        "num_wann": N_WANN,
        "num_bands": N_BANDS,
        "dis_win_eV": list(DIS_WIN),
        "dis_froz_eV": list(DIS_FROZ),
        "projections": ["Cr1: d", "Cr2: d", "I: p"],
        "chemistry": "Cr-d + I-p",
        "reuses": ["112_large AMN", "112_large MMN", "112_large EIG", "112_large nnkp"],
        "changes": ["dis_win_max 8.0 -> 13.2"],
        "not": [
            "production parent",
            "I-d",
            "num_wann=232",
            "WINDOW144",
            "GH7",
            "k40/k50",
            "k10/k20 HHG",
        ],
        "role": "112_[-6,8] -> 112_[-6,13.2] tests outer window only; I-d is 112_13.2 vs 232_13.2",
        "win_sha256": sha256(args.out_win),
    }
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {args.out_win}")
    print(f"wrote {args.out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
