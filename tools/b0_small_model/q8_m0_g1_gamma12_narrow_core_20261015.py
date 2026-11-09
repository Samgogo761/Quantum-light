"""Continue the narrow core that still moved at 320 nodes."""
from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np

import q8_m0_gamma12_transpose_input_refine_20261008 as base
from q8_m0_audit_artifacts_20261014 import provenance, write_new_json
from q8_m0_classical import p1_a2_protocol
from q8_m0_g1_gamma12_tail_columns_20261014 import (
    BASELINE, contribution, load_rows, match_saved, output_columns,
)
from q8_m0_gamma12_input_wing_20261008 import wing_quadrature
from q8_m0_p1_h2_finite_kcv_input_close_20261005 import t_gauss_panel

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results/q_line/g1_gamma12_panel_budget_20261015/narrow_core_ladder.json"


def pack(m):
    m = np.asarray(m, complex)
    return {"real": np.real(m).tolist(), "imag": np.imag(m).tolist(),
            "maxabs": float(np.max(np.abs(m)))}


def main():
    if OUT.exists():
        raise FileExistsError(OUT)
    rows, cache, t0, _meta = load_rows()
    gamma = float(np.asarray(cache["gamma"]))
    pol = np.array(cache["pol"], copy=True)
    prot = p1_a2_protocol(base.q.A_STAR)
    grid = base.q.omega_grid(prot.omega0, prot.t_pulse)
    grid = grid[grid > 0]
    _phi, norm = base.q.normalize_phi(grid, base.q.phi_unnormalized(grid, 2 * prot.omega0, prot.t_pulse))
    phi_of = lambda nu: base.q.phi_at(nu, prot, norm)
    eta = base.q.eta_of(gamma)
    panels = []
    wing_quadrature(cache["gaps"], cache["u0"], gamma, n_panel=160, wing_factor=2,
                    omega_lo=1.0e-4, omega_hi=12.0, panels=panels)
    panel = next(item for item in panels if item["kind"] == "core" and abs(item["pole"] - 0.5089945843) < 1e-10)
    with np.load(BASELINE, allow_pickle=False) as saved_file:
        saved_oms, saved_wts = np.array(saved_file["oms"]), np.array(saved_file["wts"])
        saved_u, saved_v = np.array(saved_file["U"]), np.array(saved_file["V"])
    _ids, cols = match_saved(saved_oms, saved_wts, panel["omega"], panel["weights"])
    levels = {"160": contribution(saved_u[:, cols], saved_v[:, cols])}
    steps = {}
    prev_n = 160
    prev = levels["160"]
    for n in (320, 640, 1280, 2560):
        om, wt = t_gauss_panel(panel["pole"], panel["gwid"], panel["t0"], panel["t1"], n)
        if abs(float(np.sum(wt) - panel["weight_sum"])) > 1.0e-6:
            raise AssertionError(f"Weight changed at n={n}: {float(np.sum(wt) - panel['weight_sum'])}")
        u, v = output_columns(rows, cache, gamma, pol, phi_of, eta, t0, om, wt)
        got = contribution(u, v)
        key = f"{prev_n}_to_{n}"
        steps[key] = {"normal": pack(got[0] - prev[0]), "anomalous": pack(got[1] - prev[1])}
        change = steps[key]["normal"]["maxabs"]
        print(prev_n, n, change, flush=True)
        prev, prev_n = got, n
        if change < 1.0e-6:
            break
    write_new_json(OUT, {
        "pole": panel["pole"], "omega_lo": panel["omega_lo"], "omega_hi": panel["omega_hi"],
        "gwid": panel["gwid"], "t0": panel["t0"], "t1": panel["t1"], "weight_sum": panel["weight_sum"],
        "steps": steps, "production_G1_pass": False,
        "provenance_sha256": provenance([__file__]),
    })


if __name__ == "__main__":
    main()
