"""P4: Post-pulse drift of the linearized vertex u(t) and contact D(t).

Question: after the 8-cycle P1 pulse ends, H0 is strictly static and
rho_cl(t) = exp(-iH0(t-Tp)) rho_cl(Tp) exp(+iH0(t-Tp)). Unless
[rho_cl(Tp), H0] = 0, the linearized vertices u(t) = sum_k sqrt(w_k) X(k,t)
and the contact term D(t) keep evolving. This script measures that drift on
the full Gamma12 grid (no propagation of quantum columns; classical
background only), to test whether a "static-H0 free-electron frequency sum"
could represent the full feedback tail.

Cheap: one 8x8 frame per k, 1024 macro-steps per cycle, cycles 0..20.
Writes JSON only.
"""
from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np

from q8_m0_classical import p1_a2_protocol
from q8_m0_model import build_geometry
from q8_m0_p1_h2_finite_kcv_bz_access_20261005 import grid_sample
from q8_m0_p1_h2_finite_kcv_gamma12_backend_short_20261006 import (
    advance_half_frame, vertices_from_frame)

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results/q_line/g1_kernel_quadrature_20261017/postpulse_vertex_drift.json"


def main():
    if OUT.exists():
        raise FileExistsError(OUT)
    prot = p1_a2_protocol(0.8)
    geo = build_geometry()
    sample, _inv = grid_sample(geo, 12, 0.0, neel=1)
    h = float(prot.period) / 1024.0
    frame = sample["frame"].copy()
    rec = {}
    n_total = 20 * 1024
    for step in range(n_total):
        t = step * h
        for _when_off, when in enumerate((t + h / 4.0, t + 3.0 * h / 4.0)):
            frame = advance_half_frame(geo, sample["kc"], prot, sample["neel"], frame, when, h)
        done = step + 1
        if done % 1024 == 0:
            cycle = done / 1024.0
            t_mid = (step + 0.5) * h
            u, d = vertices_from_frame(geo, sample["kc"], prot, sample["neel"],
                                       t_mid, frame, sample["weights"])
            rec[cycle] = (u.copy(), d.copy())
            if int(cycle) % 4 == 0 or cycle >= 8:
                print("cycle", cycle, float(np.max(np.abs(u))), float(np.max(np.abs(d))), flush=True)
    ref = rec[9.0]
    drift = []
    for c in sorted(rec):
        if c < 9.0:
            continue
        u, d = rec[c]
        drift.append({
            "cycle": c,
            "max_abs_u": float(np.max(np.abs(u))),
            "max_abs_d": float(np.max(np.abs(d))),
            "max_abs_u_minus_u9": float(np.max(np.abs(u - ref[0]))),
            "max_abs_d_minus_d9": float(np.max(np.abs(d - ref[1]))),
            "rel_u_drift_vs_u9": float(np.max(np.abs(u - ref[0])) / np.max(np.abs(ref[0]))),
            "rel_d_drift_vs_d9": float(np.max(np.abs(d - ref[1])) / max(np.max(np.abs(ref[1])), 1e-30)),
        })
    import json
    with open(OUT, "w", encoding="utf-8") as fh:
        json.dump({
            "task": "P4 post-pulse vertex/contact drift, Gamma12 classical background",
            "date": "2026-10-17",
            "reference": "cycle 9.0 (mid-step values)",
            "per_cycle": drift,
            "conclusion_note": "If rel drift is non-negligible, u(t) and D(t) are NOT static after the pulse: a static-H0 free-electron frequency sum cannot represent the full feedback tail without proof of equivalence.",
        }, fh, ensure_ascii=False, indent=1)
    print("saved", OUT, flush=True)


if __name__ == "__main__":
    main()
