"""Double every H2-window wing on the saved 4096-step rows."""
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
from q8_m0_p1_h2_finite_kcv_input_close_20261005 import gauss_segment

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results/q_line/g1_gamma12_tail_column_20261014/h2_wings_doubled.json"


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
    with np.load(BASELINE, allow_pickle=False) as saved_file:
        saved = {key: np.array(saved_file[key]) for key in ("oms", "wts", "U", "V")}
    total_n = np.zeros((2, 2), complex)
    total_a = np.zeros((2, 2), complex)
    rows_out = []
    for panel in panels:
        if panel["kind"] != "h2_wing":
            continue
        _ids, cols = match_saved(saved["oms"], saved["wts"], panel["omega"], panel["weights"])
        new_o, new_w = gauss_segment(panel["omega_lo"], panel["omega_hi"], 2 * int(panel["requested"]))
        new_u, new_v = output_columns(rows, cache, gamma, pol, phi_of, eta, t0, new_o, new_w)
        old_n, old_a = contribution(saved["U"][:, cols], saved["V"][:, cols])
        new_n, new_a = contribution(new_u, new_v)
        total_n += new_n - old_n
        total_a += new_a - old_a
        rows_out.append({
            "pole": panel["pole"], "omega_lo": panel["omega_lo"], "omega_hi": panel["omega_hi"],
            "old_nodes": panel["nodes"], "new_nodes": int(new_o.size),
            "normal_change": pack(new_n - old_n), "anomalous_change": pack(new_a - old_a),
        })
        print(panel["omega_lo"], panel["omega_hi"], float(np.max(np.abs(new_n - old_n))), flush=True)
    write_new_json(OUT, {
        "wings": rows_out, "combined_normal": pack(total_n), "combined_anomalous": pack(total_a),
        "production_G1_pass": False, "provenance_sha256": provenance([__file__]),
    })
    print("combined", float(np.max(np.abs(total_n))), float(np.max(np.abs(total_a))))


if __name__ == "__main__":
    main()
