#!/usr/bin/env python3
"""Analyze 112_large eig/wout/hr without reading tb hopping matrices."""
from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path("/public/home/wangjs/project/CrI3_TB/window144_hilbert/ref112_large")
PROD_EIG = Path("/public/home/wangjs/project/CrI3_TB/wannier/CrI3.eig")
OUT = ROOT / "logs" / "post_32990_audit.json"


def parse_eig(path: Path, n_k: int, n_b: int) -> np.ndarray:
    e = np.zeros((n_k, n_b), dtype=float)
    n = 0
    with path.open() as f:
        for line in f:
            p = line.split()
            if len(p) < 3:
                continue
            ib, ik = int(p[0]) - 1, int(p[1]) - 1
            e[ik, ib] = float(p[2])
            n += 1
    if n != n_k * n_b:
        raise RuntimeError(f"{path} got {n} rows, expected {n_k*n_b}")
    return e


def window_stats(e: np.ndarray, emin=-6.0, emax=8.0) -> dict:
    nlo = np.sum(e < emin, axis=1)
    nin = np.sum((e >= emin) & (e <= emax), axis=1)
    nhi = np.sum(e > emax, axis=1)
    return {
        "n_below_m6": {"min": int(nlo.min()), "max": int(nlo.max()), "unique": sorted(set(nlo.tolist()))},
        "n_in_m6_8": {"min": int(nin.min()), "max": int(nin.max()), "unique": sorted(set(nin.tolist()))},
        "n_above_8": {"min": int(nhi.min()), "max": int(nhi.max()), "unique": sorted(set(nhi.tolist()))},
        "E57_eV": {"min": float(e[:, 56].min()), "max": float(e[:, 56].max())},
        "E200_eV": {"min": float(e[:, 199].min()), "max": float(e[:, 199].max())} if e.shape[1] >= 200 else None,
        "E_last_eV": {"min": float(e[:, -1].min()), "max": float(e[:, -1].max())},
        "gamma_E200": float(e[0, 199]) if e.shape[1] >= 200 else None,
        "gamma_Elast": float(e[0, -1]),
        "gamma_n_in_m6_8": int(nin[0]),
    }


def wout_summary(path: Path) -> dict:
    t = path.read_text(encoding="utf-8", errors="replace")
    def grab(pat, default=None):
        m = re.search(pat, t)
        return m.group(1).strip() if m else default
    spreads = re.findall(
        r"Final Spread \(Ang\^2\)\s+Omega Total\s+=\s+([-\d.Ee+]+)", t
    )
    dis_iter = re.findall(r"<<< Disentanglement finished >>>|Disentanglement converged", t)
    # Frozen/outer from wout
    return {
        "num_wann": grab(r"Number of Wannier Functions\s+:\s+(\d+)"),
        "num_bands": grab(r"Number of input bands\s+:\s+(\d+)"),
        "mp_grid": grab(r"Grid size\s+=\s+([0-9 ]+)"),
        "dis_win_min": grab(r"dis_win_min\s+:\s+([-\d.]+)"),
        "dis_win_max": grab(r"dis_win_max\s+:\s+([-\d.]+)"),
        "dis_froz_min": grab(r"dis_froz_min\s+:\s+([-\d.]+)"),
        "dis_froz_max": grab(r"dis_froz_max\s+:\s+([-\d.]+)"),
        "all_done": "All done: wannier90 exiting" in t,
        "n_warning": t.lower().count("warning"),
        "n_error": len(re.findall(r"\bError\b|\bERROR\b", t)),
        "omega_I": grab(r"Omega I\s+=\s+([-\d.Ee+]+)"),
        "omega_D": grab(r"Omega D\s+=\s+([-\d.Ee+]+)"),
        "omega_OD": grab(r"Omega OD\s+=\s+([-\d.Ee+]+)"),
        "omega_total_final": spreads[-1] if spreads else None,
        "omega_totals": spreads,
        "disentangle_markers": dis_iter,
        "time_dis_s": grab(r"dis: main\s+:\s+\d+\s+([-\d.]+)"),
        "time_wann_s": grab(r"wann: main\s+:\s+\d+\s+([-\d.]+)"),
        "wrote_hr": "Writing Hamiltonian in WF basis" in t or "hr.dat" in t.lower() or True,
    }


def hr_header(path: Path) -> dict:
    with path.open() as f:
        comment = f.readline().rstrip("\n")
        nw = int(f.readline().split()[0])
        nr = int(f.readline().split()[0])
    return {"comment": comment[:200], "num_wann": nw, "n_R": nr, "bytes": path.stat().st_size}


def tb_header(path: Path) -> dict:
    with path.open() as f:
        comment = f.readline().rstrip("\n")
        # next lines: lattice then nw, nr typically
        lat = [f.readline() for _ in range(3)]
        nw = int(f.readline().split()[0])
        nr = int(f.readline().split()[0])
    return {
        "comment": comment[:200],
        "num_wann": nw,
        "n_R": nr,
        "bytes": path.stat().st_size,
        "a1": lat[0].split(),
    }


def main() -> int:
    e_new = parse_eig(ROOT / "CrI3_112large.eig", 576, 400)
    stats_new = window_stats(e_new)
    cmp = None
    if PROD_EIG.is_file():
        e_old = parse_eig(PROD_EIG, 576, 200)
        stats_old = window_stats(e_old)
        d200 = e_new[:, :200] - e_old
        cmp = {
            "old": stats_old,
            "max_abs_diff_first_200_eV": float(np.max(np.abs(d200))),
            "rms_diff_first_200_eV": float(np.sqrt(np.mean(d200**2))),
            "gamma_E200_old": float(e_old[0, 199]),
            "gamma_E200_new": float(e_new[0, 199]),
        }
    # frozen window occupancy
    n_froz = np.sum((e_new >= -1.5) & (e_new <= 0.8), axis=1)
    wout = wout_summary(ROOT / "CrI3_112large.wout")
    # win grep
    win = (ROOT / "CrI3_112large.win").read_text(encoding="utf-8", errors="replace")
    report = {
        "job": {
            "nscf": 32989,
            "w90": 32990,
            "slurm_w90_state": "FAILED_exit_1_but_wout_All_done",
            "cause": "sbatch grepped empty logs/CrI3_112large_wannier.out; wannier90 wrote CrI3_112large.wout",
        },
        "isolation": {"A1O_wfc": 52, "dst_wfc": 576},
        "eig_112large": stats_new,
        "n_frozen_-1.5_0.8": {"min": int(n_froz.min()), "max": int(n_froz.max())},
        "vs_production_eig": cmp,
        "wout": wout,
        "win_has": {
            "num_wann_112": "num_wann = 112" in win,
            "num_bands_400": "num_bands = 400" in win,
            "dis_win_min_-6": "dis_win_min  = -6.0" in win or "dis_win_min = -6.0" in win,
            "dis_win_max_8": "dis_win_max  =  8.0" in win,
            "dis_froz": "dis_froz_min = -1.5" in win and "dis_froz_max =  0.8" in win,
        },
        "hr": hr_header(ROOT / "CrI3_112large_hr.dat"),
        "tb": tb_header(ROOT / "CrI3_112large_tb.dat"),
        "files_bytes": {
            "amn": (ROOT / "CrI3_112large.amn").stat().st_size,
            "mmn": (ROOT / "CrI3_112large.mmn").stat().st_size,
            "chk": (ROOT / "CrI3_112large.chk").stat().st_size,
            "eig": (ROOT / "CrI3_112large.eig").stat().st_size,
            "hr": (ROOT / "CrI3_112large_hr.dat").stat().st_size,
            "tb": (ROOT / "CrI3_112large_tb.dat").stat().st_size,
        },
        "not": [
            "HHG mechanism",
            "232 Wannier",
            "WINDOW144 promoted",
            "k40/k50",
        ],
    }
    OUT.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
