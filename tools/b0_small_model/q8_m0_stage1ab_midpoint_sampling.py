#!/usr/bin/env python3
"""Zero-drive comparison of interpolated versus truly sampled midpoint K(t).

One two-k algorithm diagnostic only. The photon quadrature is intentionally
small and is not asserted to approximate the continuum.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from q8_m0_cjj import propagate_cjj
from q8_m0_classical import p1_a2_protocol
from q8_m0_mode_io import eta2_ncell
from q8_m0_model import R0, build_geometry
from q8_m0_phgauss import (cayley_step, linear_K, n_ph_osc, pack_waveforms,
                           pauli_from_phi, sqrt_w, symplectic_J)
from q8_m0_s1_kernel import parallel_perp_basis
from q8_m0_stage1s_continuum import open_grid
from q8_m0_stage1t_continuum_pauli import continuum_waveforms

REPO = Path(__file__).resolve().parents[2]


def one_run(x, md, fr, fi, w, eta, dt, mode):
    sw = sqrt_w(w)
    nph = fr.shape[0]
    nb = n_ph_osc(len(w))
    n = nph + nb
    phi = np.eye(2*n)
    for j in range((len(x)-1)//2):
        a, b, m = 2*j, 2*j+2, 2*j+1
        xm = x[m] if mode in ("x", "both") else 0.5*(x[a]+x[b])
        dm = md[m] if mode in ("d", "both") else 0.5*(md[a]+md[b])
        frm = fr[:,m] if mode in ("f", "both") else 0.5*(fr[:,a]+fr[:,b])
        fim = fi[:,m] if mode in ("f", "both") else 0.5*(fi[:,a]+fi[:,b])
        k = linear_K(nph, sw, xm, frm, fim, dm, eta)
        phi = cayley_step(phi, k, dt)
    p = pauli_from_phi(phi, nph, nb, w)
    jmat = symplectic_J(n)
    sym = float(np.max(abs(phi.T@jmat@phi-jmat)))
    return {"C_P_final": p["Ncell_times_eps_Pauli"],
            "C_C_final": p["Ncell_times_C_minus_I"],
            "Phi_maxabs": float(np.max(abs(phi))),
            "symplectic_abs": sym,
            "symplectic_scaled": sym/max(1.0,float(np.max(abs(phi)))**2)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ntpc", type=int, nargs="+", default=[128,256])
    ap.add_argument("--nquad", type=int, default=8)
    args = ap.parse_args()
    geo = build_geometry()
    prot = p1_a2_protocol(0.0)
    epar, eperp = parallel_perp_basis(geo)
    om, wt = open_grid(8*prot.omega0, args.nquad)
    eta = float(np.sqrt(eta2_ncell(geo)))
    rows = []
    for ntpc in args.ntpc:
        rec = propagate_cjj(geo, R0, prot, nt_per_cycle=2*ntpc,
                            neel=+1, sign=+1, mu=-0.009010620372198276,
                            verbose=False,
                            kf_frac=np.array([[0.0,0.0],[0.17,0.41]]),
                            save_full_x=True, save_mean_D=True)
        fr, fi = pack_waveforms(continuum_waveforms(om, wt, rec["times"], epar, eperp))
        x = rec["X_full"]
        md = rec["mean_D_t"]
        w = rec["w_k"]
        dt = 2*float(rec["dt"])
        row = {"ntpc": ntpc, "nquad": args.nquad,
               "endpoint_average": one_run(x,md,fr,fi,w,eta,dt,"none"),
               "sampled_midpoint": one_run(x,md,fr,fi,w,eta,dt,"both"),
               "only_electronic_midpoint": one_run(x,md,fr,fi,w,eta,dt,"x"),
               "only_photon_midpoint": one_run(x,md,fr,fi,w,eta,dt,"f")}
        rows.append(row)
        print(ntpc, row["endpoint_average"]["C_P_final"],
              row["sampled_midpoint"]["C_P_final"], flush=True)
    steps = "-".join(str(n) for n in args.ntpc)
    path = REPO / f"results/q_line/stage1ab_midpoint_sampling_n{args.nquad}_ntpc{steps}_20260923.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"status":"numerical_sampling_diagnostic_only",
                                "a_star":0.0,"rows":rows},indent=2), encoding="utf-8")
    print(path)


if __name__ == "__main__":
    main()
