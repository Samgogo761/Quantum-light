#!/usr/bin/env python3
"""One incoming-column block: driven and undriven, write ΔU,V. All k stay inside the block.

Usage: py -3 this.py --freq-npy X.npy --out Y.npz [--pump-window]
Does not start the full 78472-column job.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import q8_m0_p1_h2_finite_kcv_gamma12_backend_short_20261006 as be
import q8_m0_p1_h2_finite_kcv_input_close_20261005 as q
from q8_m0_classical import p1_a2_protocol
from q8_m0_continuum_columns_20261004 import stationary_photon_columns
from q8_m0_gsheet import gamma_self
from q8_m0_incoming_spectral_completeness_20261004 import dark_subspace, stationary_dark_basis_columns
from q8_m0_model import build_geometry
from q8_m0_p1_h2_finite_kcv_bz_access_20261005 import grid_sample
from q8_m0_p1_h2_finite_kcv_caller_fix_20261005 import free_A_increment
from q8_m0_s1_kernel import parallel_perp_basis


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--freq-npy", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--include-dark", action="store_true")
    p.add_argument("--pump-window", action="store_true")
    p.add_argument("--n-dark", type=int, default=be.N_DARK_KEEP)
    args = p.parse_args()
    geo = build_geometry()
    gamma = float(gamma_self(geo))
    prot_d = p1_a2_protocol(q.A_STAR)
    prot_u = p1_a2_protocol(0.0)
    pol = np.vstack(parallel_perp_basis(geo))
    h = prot_d.period / q.NTPC
    g12, _ = grid_sample(geo, 12, 0.0)
    oms, wts = q.input_quadrature(g12["gap"], g12["u"], gamma, n_panel=96)
    dnu = {round(float(o), 12): float(w) for o, w in zip(oms, wts)}
    freq = np.load(args.freq_npy)
    photon = stationary_photon_columns(g12["gap"], g12["u"], g12["D"], gamma, freq)

    class _B:
        def __init__(self, ph):
            self.photon = ph
            self.n_photon_columns = int(ph.initial.b.shape[1])

    q.attach_weights(_B(photon), dnu)
    n_pt = 0
    if args.include_dark:
        basis, info, _n = dark_subspace(g12["gap"], g12["u"])
        dark, _l, _g = stationary_dark_basis_columns(g12["gap"], g12["u"], basis, info)
        initial, n_pt = be.pack_photon_dark(photon, dark, args.n_dark)
    else:
        initial = photon.initial.copy()
    step0 = be.PUMP_STEP0 if args.pump_window else 0
    t0 = step0 * h
    t_mid = (step0 + np.arange(be.NSTEP) + 0.5) * h
    k, _ = be.kernel_at_times(gamma, prot_d, t_mid)
    vd = be.vertices_window(geo, g12, prot_d, be.NSTEP, h, step0)
    vu = be.vertices_window(geo, g12, prot_u, be.NSTEP, h, step0)
    _d, _u, merged = be.run_pair(photon, gamma, initial, n_pt, k, k, vd, vu, be.NSTEP, h, pol, t0)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, delta_u=merged["delta_u"], v=merged["v"], freq=freq, n_point=n_pt)
    meta = {
        "out": str(out),
        "n_freq": int(freq.size),
        "ncol": int(initial.b.shape[1]),
        "n_point": int(n_pt),
        "driven_wall_s": merged["driven_wall_s"],
        "undriven_wall_s": merged["undriven_wall_s"],
        "pump_window": bool(args.pump_window),
        "algebraic_if_pump_window": bool(args.pump_window),
    }
    print(json.dumps(meta), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
