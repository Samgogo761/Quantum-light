#!/usr/bin/env python3
"""Compare J_2, J_5, J_7, J_9 of WINDOW144 vs full112 classical k20 runs.

Reads solver HHG.dat (and HHG_complex.dat if present).  Does not interpret
the ratio as a finished Hilbert verdict by itself; IMPACT_ON_HHG stays
unresolved until the numbers are recorded in the protocol.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

ORDERS = (2, 5, 7, 9)
HALF_WIDTH = 0.4


def load_hhg(path: Path) -> tuple[np.ndarray, np.ndarray]:
    rows = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        parts = line.split()
        rows.append([float(parts[0]), float(parts[4])])
    data = np.asarray(rows, dtype=float)
    return data[:, 0], data[:, 1]


def load_complex(path: Path) -> dict[str, np.ndarray]:
    rows = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        p = line.split()
        rows.append([float(x) for x in p[:6]])
    data = np.asarray(rows, dtype=float)
    jx = data[:, 2] + 1j * data[:, 3]
    jy = data[:, 4] + 1j * data[:, 5]
    return {"order": data[:, 0], "jx": jx, "jy": jy}


def peak_total(order_grid: np.ndarray, tot: np.ndarray, n: int) -> dict:
    sel = (order_grid >= n - HALF_WIDTH) & (order_grid <= n + HALF_WIDTH)
    if not np.any(sel):
        return {"order": n, "HHG_total": 0.0, "bin_order": None}
    i = int(np.argmax(tot[sel]))
    idx = np.flatnonzero(sel)[i]
    return {
        "order": n,
        "HHG_total": float(tot[idx]),
        "bin_order": float(order_grid[idx]),
    }


def peak_complex(cplx: dict[str, np.ndarray], n: int) -> dict:
    sel = (cplx["order"] >= n - HALF_WIDTH) & (cplx["order"] <= n + HALF_WIDTH)
    tot = np.abs(cplx["jx"]) ** 2 + np.abs(cplx["jy"]) ** 2
    i = int(np.argmax(tot[sel]))
    idx = np.flatnonzero(sel)[i]
    jx, jy = complex(cplx["jx"][idx]), complex(cplx["jy"][idx])
    return {
        "order": n,
        "Jx": [jx.real, jx.imag],
        "Jy": [jy.real, jy.imag],
        "mod_J": float(np.hypot(abs(jx), abs(jy))),
        "bin_order": float(cplx["order"][idx]),
    }


def compare_dirs(dir_112: Path, dir_144: Path) -> dict:
    o112, t112 = load_hhg(dir_112 / "HHG.dat")
    o144, t144 = load_hhg(dir_144 / "HHG.dat")
    out: dict = {
        "dir_112": str(dir_112),
        "dir_144": str(dir_144),
        "orders": list(ORDERS),
        "half_width": HALF_WIDTH,
        "harmonics": {},
        "IMPACT_ON_HHG": "UNRESOLVED",
    }
    c112 = dir_112 / "HHG_complex.dat"
    c144 = dir_144 / "HHG_complex.dat"
    cx112 = load_complex(c112) if c112.is_file() else None
    cx144 = load_complex(c144) if c144.is_file() else None
    for n in ORDERS:
        a = peak_total(o112, t112, n)
        b = peak_total(o144, t144, n)
        ratio = b["HHG_total"] / a["HHG_total"] if a["HHG_total"] != 0.0 else None
        rec: dict = {
            "full112": a,
            "window144": b,
            "HHG_total_ratio_144_over_112": ratio,
            "HHG_total_rel_diff": (ratio - 1.0) if ratio is not None else None,
        }
        if cx112 is not None and cx144 is not None:
            ja, jb = peak_complex(cx112, n), peak_complex(cx144, n)
            rec["full112_complex"] = ja
            rec["window144_complex"] = jb
            if ja["mod_J"] != 0.0:
                rec["mod_J_ratio_144_over_112"] = jb["mod_J"] / ja["mod_J"]
        out["harmonics"][str(n)] = rec
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dir-112", required=True, type=Path)
    ap.add_argument("--dir-144", required=True, type=Path)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    report = compare_dirs(args.dir_112, args.dir_144)
    text = json.dumps(report, indent=2)
    print(text)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
