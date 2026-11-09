#!/usr/bin/env python3
"""Gabor / STFT of total J(t). Candidate time windows, not band-k attribution."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

HBAR_OMEGA_EV = 1239.84193 / 3200.0
FS_PER_AU = 0.024188843265857  # matches solver au_to_fs if dt was written in fs already


def load_jt(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    t, jx, jy = [], [], []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        p = line.split()
        t.append(float(p[1]))
        jx.append(float(p[2]))
        jy.append(float(p[3]))
    return np.asarray(t), np.asarray(jx), np.asarray(jy)


def gabor_power(t: np.ndarray, j: np.ndarray, omega: float, sigma: float, t0: np.ndarray) -> np.ndarray:
    dt = float(t[1] - t[0])
    out = np.empty(t0.size, dtype=float)
    for i, c in enumerate(t0):
        w = np.exp(-0.5 * ((t - c) / sigma) ** 2)
        z = np.sum(j * w * np.exp(-1j * omega * t)) * dt
        out[i] = float(np.abs(z) ** 2)
    return out


def analyze_one(t, jx, jy, sigma_fs: float, t0: np.ndarray, omega: dict) -> dict:
    harmonics = {}
    curves = {}
    for n, w in omega.items():
        px = gabor_power(t, jx, w, sigma_fs, t0)
        py = gabor_power(t, jy, w, sigma_fs, t0)
        tot = px + py
        imax = int(np.argmax(tot))
        harmonics[f"H{n}"] = {
            "omega_1_per_fs": w,
            "t_peak_fs": float(t0[imax]),
            "power_peak": float(tot[imax]),
        }
        curves[f"H{n}"] = [{"t_fs": float(tt), "power": float(pp)} for tt, pp in zip(t0, tot)]
    return {"sigma_fs": sigma_fs, "harmonics": harmonics, "curves": curves}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--jt", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--sigma-fs", type=float, default=None, help="single window; default 1.0 cycle")
    ap.add_argument(
        "--sigma-cycles",
        type=str,
        default="",
        help="comma-separated optical-cycle multiples, e.g. 0.5,0.75,1.0,1.5",
    )
    ap.add_argument("--t-opt-fs", type=float, default=10.67, help="optical period at 3200 nm")
    args = ap.parse_args()
    t, jx, jy = load_jt(args.jt)
    hbar_ev_fs = 0.658211951
    omega = {n: n * HBAR_OMEGA_EV / hbar_ev_fs for n in (2, 5, 7, 9)}
    t0 = t[:: max(1, len(t) // 200)]
    if args.sigma_cycles.strip():
        cycles = [float(x) for x in args.sigma_cycles.split(",")]
    elif args.sigma_fs is not None:
        cycles = [args.sigma_fs / args.t_opt_fs]
    else:
        cycles = [1.0]
    scans = []
    for cyc in cycles:
        sigma = cyc * args.t_opt_fs
        one = analyze_one(t, jx, jy, sigma, t0, omega)
        scans.append({"sigma_cycles": cyc, **one})
        print(json.dumps({"sigma_cycles": cyc, "harmonics": one["harmonics"]}, indent=2))
    h5 = [s["harmonics"]["H5"]["t_peak_fs"] for s in scans]
    h7 = [s["harmonics"]["H7"]["t_peak_fs"] for s in scans]
    order = [h7i > h5i for h5i, h7i in zip(h5, h7)]
    out = {
        "label": "CANDIDATE_TIME_WINDOW_NOT_ATTRIBUTION",
        "jt": str(args.jt),
        "t_opt_fs": args.t_opt_fs,
        "n_samples": int(t.size),
        "t_max_fs": float(t[-1]),
        "note": "Do not convert a ~1-cycle window peak split into a path delay.",
        "h5_before_h7_at_all_sigma": all(order),
        "scans": [
            {
                "sigma_cycles": s["sigma_cycles"],
                "sigma_fs": s["sigma_fs"],
                "harmonics": s["harmonics"],
            }
            for s in scans
        ],
        "curves": {f"cyc_{s['sigma_cycles']}": s["curves"] for s in scans},
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
