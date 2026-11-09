#!/usr/bin/env python3
"""Build 232_I-d Wannier inputs from 112_large.

232 cannot use dis_win=[-6,8]: that window has only 174-185 Bloch states.
The locked technical outer window is [-6, 13.2] eV (N_outer>=299 on the 8-k
count). Frozen window and 400-band NSCF stay as in 112_large.

This is a reference diagnostic parent, not a production model.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from extract_upf_defined_radials import (
    KNOWN_SHA256,
    build_all,
    parse_upf_radials,
    qe_radialpart_mesh,
    sha256,
)

SCHEMA = "ref232_id_inputs_v1"
N_WANN = 232
N_BANDS = 400
DIS_WIN = (-6.0, 13.2)
DIS_FROZ = (-1.5, 0.8)
SHELLS_232 = ("Cr_d", "I_p", "I_d")
N_BY_SHELL = {"Cr_d": 40, "I_p": 72, "I_d": 120}
L_OF = {"Cr_d": 2, "I_p": 1, "I_d": 2}


def parse_eig(path: Path, n_k: int, n_b: int) -> np.ndarray:
    e = np.zeros((n_k, n_b), dtype=float)
    n = 0
    with path.open() as fh:
        for line in fh:
            p = line.split()
            if len(p) < 3:
                continue
            ib, ik = int(p[0]) - 1, int(p[1]) - 1
            e[ik, ib] = float(p[2])
            n += 1
    if n != n_k * n_b:
        raise RuntimeError(f"{path} got {n} rows, expected {n_k * n_b}")
    return e


def window_count(e: np.ndarray, lo: float, hi: float) -> dict:
    nin = np.sum((e >= lo) & (e <= hi), axis=1)
    return {
        "lo_eV": lo,
        "hi_eV": hi,
        "min": int(nin.min()),
        "max": int(nin.max()),
        "n_k_le_232": int(np.sum(nin <= N_WANN)),
        "ok_strict_gt_232": bool(nin.min() > N_WANN),
        "ok_surplus_299": bool(nin.min() >= 299),
    }


def patch_win(text: str) -> str:
    if "num_wann = 112" not in text:
        raise RuntimeError("expected production-style num_wann = 112")
    if "num_bands = 400" not in text:
        raise RuntimeError("232_I-d reuses the 400-band NSCF")
    out = text.replace("num_wann = 112", f"num_wann = {N_WANN}", 1)
    out = re.sub(
        r"dis_win_max\s*=\s*8\.0",
        f"dis_win_max  = {DIS_WIN[1]}",
        out,
        count=1,
    )
    if f"num_wann = {N_WANN}" not in out:
        raise RuntimeError("failed to set num_wann=232")
    if "13.2" not in out:
        raise RuntimeError("failed to set dis_win_max=13.2")
    if not re.search(r"dis_froz_min\s*=\s*-1\.5", out) or not re.search(r"dis_froz_max\s*=\s*0\.8", out):
        raise RuntimeError("frozen window must stay [-1.5, 0.8]")
    if re.search(r"^I:\s+d\s*$", out, re.M):
        raise RuntimeError("I:d already present")
    if "I:   p" not in out and not re.search(r"^I:\s+p\s*$", out, re.M):
        raise RuntimeError("missing I: p projection")
    out = re.sub(r"(I:\s+p\s*\n)end projections", r"\1I:   d\nend projections", out, count=1)
    if not re.search(r"^I:\s+d\s*$", out, re.M):
        raise RuntimeError("failed to add I: d")
    return out


def write_radials(cr_upf: Path, i_upf: Path, radial_id: Path, radial_R: Path) -> dict:
    cr = parse_upf_radials(cr_upf)
    iodine = parse_upf_radials(i_upf)
    for rec in (cr, iodine):
        expect = KNOWN_SHA256[rec["file"]]
        if rec["sha256"] != expect:
            raise RuntimeError(f"UPF hash mismatch {rec['file']}")
    r = qe_radialpart_mesh()
    shells = build_all(cr, iodine, r)
    ids: list[int] = []
    for i, name in enumerate(SHELLS_232, start=1):
        n = N_BY_SHELL[name]
        if n <= 0:
            raise RuntimeError(name)
        ids.extend([i] * n)
    if len(ids) != N_WANN:
        raise RuntimeError(f"radial_id count {len(ids)} != {N_WANN}")
    radial_id.write_text(str(len(ids)) + "\n" + "\n".join(str(i) for i in ids) + "\n", encoding="utf-8")
    lines = [f"{len(SHELLS_232)} {r.size}"]
    lines.extend(f"{x:.16e}" for x in r)
    for name in SHELLS_232:
        lines.append(f"{name} {L_OF[name]}")
        lines.extend(f"{x:.16e}" for x in shells[name]["R"])
    radial_R.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {
        "n_proj": len(ids),
        "n_type": len(SHELLS_232),
        "mesh": int(r.size),
        "mean_r_bohr": {name: shells[name]["mean_r_bohr"] for name in SHELLS_232},
        "upf": {
            "Cr": {"file": cr["file"], "sha256": cr["sha256"]},
            "I": {"file": iodine["file"], "sha256": iodine["sha256"]},
        },
        "I_d_channels": shells["I_d"]["channel_indices"],
        "I_d_rejected_exploding": shells["I_d"]["rejected_exploding_indices"],
    }


def nnkpts_block(nnkp: Path) -> str:
    text = nnkp.read_text(encoding="utf-8", errors="replace")
    m = re.search(r"begin nnkpts\n(.*?)end nnkpts\n", text, re.S)
    if m is None:
        raise RuntimeError(f"no nnkpts in {nnkp}")
    return m.group(1)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--win-112large", type=Path, required=True)
    ap.add_argument("--eig", type=Path, default=None)
    ap.add_argument("--cr-upf", type=Path, required=True)
    ap.add_argument("--i-upf", type=Path, required=True)
    ap.add_argument("--out-win", type=Path, required=True)
    ap.add_argument("--radial-id", type=Path, required=True)
    ap.add_argument("--radial-R", type=Path, required=True)
    ap.add_argument("--out-json", type=Path, required=True)
    args = ap.parse_args()
    win = patch_win(args.win_112large.read_text(encoding="utf-8"))
    args.out_win.write_text(win, encoding="utf-8")
    rad = write_radials(args.cr_upf, args.i_upf, args.radial_id, args.radial_R)
    mesh = None
    if args.eig is not None:
        e = parse_eig(args.eig, 576, 400)
        mesh = {
            "advertised_m6_8": window_count(e, -6.0, 8.0),
            "locked_m6_13p2": window_count(e, DIS_WIN[0], DIS_WIN[1]),
            "frozen": window_count(e, DIS_FROZ[0], DIS_FROZ[1]),
        }
        if not mesh["locked_m6_13p2"]["ok_strict_gt_232"]:
            raise RuntimeError("576-k mesh has k with N[-6,13.2] <= 232")
    payload = {
        "schema": SCHEMA,
        "status": "REFERENCE_DIAGNOSTIC_PARENT_NOT_PRODUCTION",
        "num_wann": N_WANN,
        "num_bands": N_BANDS,
        "dis_win_eV": list(DIS_WIN),
        "dis_froz_eV": list(DIS_FROZ),
        "projections": ["Cr1: d", "Cr2: d", "I: p", "I: d"],
        "chemistry": "Cr-d + I-p + I-d",
        "not": [
            "production parent",
            "chemical I-d 144",
            "WINDOW144",
            "GH7",
            "k40/k50",
            "same outer window [-6,8] (N_outer < 232)",
        ],
        "radial": rad,
        "mesh_eig": mesh,
        "win_sha256": sha256(args.out_win),
    }
    args.out_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {args.out_win}")
    print(f"wrote {args.radial_id}")
    print(f"wrote {args.radial_R}")
    print(f"wrote {args.out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
