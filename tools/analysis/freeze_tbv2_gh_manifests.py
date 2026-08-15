#!/usr/bin/env python3
"""Generate and freeze ES25.17 GH3/GH5 manifests for TB-v2 (two selected states).

Checks: n_nodes, weight sum, Q0 moments, exact antipodes after disk round-trip, SHA256.
Does not overwrite 27992 manifests.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "deploy" / "a0_layerA_m88full"))
sys.path.insert(0, str(ROOT / "tools" / "analysis"))

from make_gh_nodes import ALPHA0_TOL, TWOPI, make_sv_nodes, write_manifest  # noqa: E402
from validate_qlight_nodes_production import load_manifest  # noqa: E402
from validate_qlight_nodes_strict import validate_sv  # noqa: E402

STATES = ((2.5, 0.0), (2.5, 180.0))
I_BAR = 1.0e11


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def check_antipodes(rows: list[dict]) -> dict:
    by_alpha = {(round(r["re"], 12), round(r["im"], 12)): r for r in rows}
    max_dphi = 0.0
    n_pairs = 0
    n_self = 0
    for r in rows:
        a = complex(float(r["re"]), float(r["im"]))
        if abs(a) <= ALPHA0_TOL:
            n_self += 1
            if abs(float(r["intensity"])) > 1e-30 or abs(float(r["phase"])) > 1e-30:
                raise ValueError(f"alpha=0 node id={r['id']} has nonzero I/phi")
            continue
        key = (round(-float(r["re"]), 12), round(-float(r["im"]), 12))
        if key not in by_alpha:
            raise ValueError(f"missing antipode for id={r['id']}")
        p = by_alpha[key]
        if abs(float(r["weight"]) - float(p["weight"])) > 1e-16:
            raise ValueError(f"weight mismatch id={r['id']}")
        if abs(float(r["intensity"]) - float(p["intensity"])) > 1e-16:
            raise ValueError(f"I mismatch id={r['id']}")
        dphi = abs(((float(p["phase"]) - (float(r["phase"]) + math.pi)) + math.pi) % TWOPI - math.pi)
        max_dphi = max(max_dphi, dphi)
        n_pairs += 1
    if max_dphi > 1e-15:
        raise ValueError(f"max |phi_bar-(phi+pi)| = {max_dphi}")
    return {"n_self_alpha0": n_self, "n_antipode_directed": n_pairs, "max_dphi": max_dphi}


def freeze_one(outdir: Path, r: float, th: float, gh: int, i_bar: float) -> dict:
    rows = make_sv_nodes(r, th, i_bar, gh)
    name = f"nodes_sv_r{str(r).replace('.','p')}_th{int(th)}_gh{gh}.dat"
    path = outdir / name
    write_manifest(path, rows, r, th, gh, i_bar)
    loaded = load_manifest(path)
    expected_n = gh * gh
    if len(loaded) != expected_n:
        raise ValueError(f"{path.name}: n_nodes={len(loaded)} expected {expected_n}")
    q0 = validate_sv(loaded, r, th, i_bar, tol=2.0e-6)
    anti = check_antipodes(loaded)
    rec = {
        "file": str(path.as_posix()),
        "name": name,
        "r": r,
        "theta_deg": th,
        "gh_order": gh,
        "n_nodes": len(loaded),
        "sha256": sha256_file(path),
        "q0": q0,
        "antipodes": anti,
    }
    print(f"OK {name} n={len(loaded)} sha256={rec['sha256']}")
    return rec


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--outdir",
        type=Path,
        default=ROOT / "deploy" / "a0_layerA_m88full" / "nodes_tbv2_es25",
    )
    ap.add_argument("--I-bar", type=float, default=I_BAR)
    args = ap.parse_args()
    gh3_dir = args.outdir / "gh3"
    gh5_dir = args.outdir / "gh5"
    gh3_dir.mkdir(parents=True, exist_ok=True)
    gh5_dir.mkdir(parents=True, exist_ok=True)

    records = []
    for r, th in STATES:
        records.append(freeze_one(gh3_dir, r, th, 3, args.I_bar))
        records.append(freeze_one(gh5_dir, r, th, 5, args.I_bar))

    sums = args.outdir / "SHA256SUMS.txt"
    lines = [f"{rec['sha256']}  {Path(rec['file']).relative_to(args.outdir).as_posix()}\n" for rec in records]
    sums.write_text("".join(lines), encoding="utf-8")
    freeze = {
        "policy": "TB-v2 GH3/GH5 manifests; ES25.17; exact antipodes; two selected states only",
        "I_bar": args.I_bar,
        "states": [{"r": r, "theta_deg": th} for r, th in STATES],
        "files": records,
        "sha256sums": str(sums.as_posix()),
    }
    report = args.outdir / "MANIFEST_FREEZE.json"
    report.write_text(json.dumps(freeze, indent=2) + "\n", encoding="utf-8")
    print("wrote", report)
    print("wrote", sums)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
