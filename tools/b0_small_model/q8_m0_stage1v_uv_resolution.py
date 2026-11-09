#!/usr/bin/env python3
"""Separate photon cutoff from quadrature resolution for one stored-source type.

The electronic trajectory is recomputed with the frozen M0/P1 protocol.  The
test treats one X(t) component as a prescribed current; it does not propagate
the joint photon/electron state or qualify a physical UV cutoff.
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from q8_m0_cjj import propagate_cjj
from q8_m0_classical import p1_a2_protocol
from q8_m0_model import R0, build_geometry
from q8_m0_s1_kernel import parallel_perp_basis
from q8_m0_stage1s_continuum import compare, field_from_kernel, photon_retarded_kernel
from q8_m0_stage1v_real_source_kernel import gauss_two_segment, open_grid

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "results/q_line/stage1v_uv_resolution_20260923.json"


def main():
    geo = build_geometry()
    prot = p1_a2_protocol(0.8)
    _, e_perp = parallel_perp_basis(geo)
    rec = propagate_cjj(
        geo, R0, prot, nt_per_cycle=256, neel=+1, sign=+1,
        mu=-0.009010620372198276, verbose=False,
        kf_frac=np.array([[0.0, 0.0], [0.17, 0.41]]),
        save_full_x=True, save_mean_D=False,
    )
    x = np.asarray(rec["X_full"])
    ch = tuple(int(i) for i in np.unravel_index(np.argmax(np.sum(abs(x)**2, axis=0)), x.shape[1:]))
    z = x[(slice(None),) + ch]
    current = np.real(z)[:, None] * e_perp[None, :]
    times = np.asarray(rec["times"])
    dt = float(rec["dt"])
    lag = times - times[0]
    gamma = 1.0
    target = field_from_kernel(current, np.r_[0.0, np.ones(len(times)-1)], dt)
    configs = [
        (8, 96, 0), (8, 384, 0),
        (64, 96, 96), (64, 96, 768), (64, 96, 1536),
    ]
    rows = []
    for cutoff, nlow, nuv in configs:
        if nuv == 0:
            omega, weight = open_grid(cutoff * prot.omega0, nlow)
        else:
            omega, weight = gauss_two_segment(
                0.0, 8.0 * prot.omega0, cutoff * prot.omega0, nlow, nuv)
        kernel = photon_retarded_kernel(lag, omega, weight, gamma)
        field = field_from_kernel(current, kernel, dt)
        metric = compare(field, target, current, dt, gamma)
        row = {"cutoff_over_omega0": cutoff, "n_low": nlow,
               "n_uv": nuv, "n_total": len(omega), **metric}
        rows.append(row)
        print(cutoff, nlow, nuv, metric["A_rel_max"], metric["dA_rel_max"], flush=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({"status": "prescribed_source_cutoff_vs_resolution_only",
                               "channel": ch, "ntpc": 256, "rows": rows}, indent=2),
                   encoding="utf-8")


if __name__ == "__main__":
    main()
