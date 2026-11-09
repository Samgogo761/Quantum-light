"""Which saved panels carry the refined off-diagonal and anomalous plateau.

Panels already doubled are left as they are. The largest remaining
contributors are doubled once at the 4096-step rows.
"""
from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np

import q8_m0_gamma12_transpose_input_refine_20261008 as base
from q8_m0_audit_artifacts_20261014 import provenance, write_new_json
from q8_m0_classical import p1_a2_protocol
from q8_m0_g1_gamma12_pulse_screen_20261014 import panel_rules, phase_delta, resolvent, times_of_interest
from q8_m0_g1_gamma12_refined_time_20261015 import under_resolved
from q8_m0_g1_gamma12_tail_columns_20261014 import (
    BASELINE, contribution, load_rows, match_saved, output_columns,
)
from q8_m0_gamma12_input_wing_20261008 import wing_quadrature
from q8_m0_p1_h2_finite_kcv_input_close_20261005 import gauss_segment, t_gauss_panel

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results/q_line/g1_gamma12_tail_column_20261014/plateau_columns.json"


def pack(m):
    m = np.asarray(m, complex)
    return {"real": float(np.real(m)), "imag": float(np.imag(m)), "abs": float(abs(m))}


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
    oms, wts, _counts = wing_quadrature(
        cache["gaps"], cache["u0"], gamma, n_panel=160, wing_factor=2,
        omega_lo=1.0e-4, omega_hi=12.0, panels=panels)
    with np.load(BASELINE, allow_pickle=False) as saved_file:
        saved = {key: np.array(saved_file[key]) for key in ("oms", "wts", "U", "V")}
    if not (np.allclose(saved["oms"], oms) and np.allclose(saved["wts"], wts)):
        raise AssertionError("Baseline nodes changed")
    clock = times_of_interest()
    ranked = []
    for panel in panels:
        _ids, cols = match_saved(saved["oms"], saved["wts"], panel["omega"], panel["weights"])
        normal, anomalous = contribution(saved["U"][:, cols], saved["V"][:, cols])
        ranked.append({
            "kind": panel["kind"], "pole": panel["pole"],
            "omega_lo": panel["omega_lo"], "omega_hi": panel["omega_hi"],
            "nodes": panel["nodes"], "already_doubled": bool(under_resolved(panel, clock)),
            "normal_01": pack(normal[0, 1]), "normal_11": pack(normal[1, 1]),
            "anomalous_01": pack(anomalous[0, 1]),
            "score": max(abs(normal[0, 1]), abs(normal[1, 1]), abs(anomalous[0, 1])),
        })
    ranked.sort(key=lambda row: row["score"], reverse=True)
    fresh = [row for row in ranked if not row["already_doubled"]][:8]
    trials = []
    by_key = {(panel["kind"], round(panel["pole"], 10), round(panel["omega_lo"], 8)): panel for panel in panels}
    for row in fresh:
        panel = by_key[(row["kind"], round(row["pole"], 10), round(row["omega_lo"], 8))]
        _ids, cols = match_saved(saved["oms"], saved["wts"], panel["omega"], panel["weights"])
        if panel["kind"] == "h2_wing":
            new_o, new_w = gauss_segment(panel["omega_lo"], panel["omega_hi"], 2 * int(panel["requested"]))
        else:
            new_o, new_w = t_gauss_panel(
                panel["pole"], panel["gwid"], panel["t0"], panel["t1"], 2 * int(panel["requested"]))
        new_u, new_v = output_columns(rows, cache, gamma, pol, phi_of, eta, t0, new_o, new_w)
        old_n, old_a = contribution(saved["U"][:, cols], saved["V"][:, cols])
        new_n, new_a = contribution(new_u, new_v)
        trials.append({
            "pole": panel["pole"], "kind": panel["kind"],
            "omega_lo": panel["omega_lo"], "omega_hi": panel["omega_hi"],
            "old_nodes": panel["nodes"], "new_nodes": int(new_o.size),
            "normal_01_change": pack((new_n - old_n)[0, 1]),
            "normal_11_change": pack((new_n - old_n)[1, 1]),
            "anomalous_01_change": pack((new_a - old_a)[0, 1]),
            "normal_change_maxabs": float(np.max(np.abs(new_n - old_n))),
        })
        print(panel["pole"], trials[-1]["normal_change_maxabs"], flush=True)
    dark = slice(4 * len(oms), None)
    dark_n, dark_a = contribution(saved["U"][:, dark], saved["V"][:, dark])
    write_new_json(OUT, {
        "largest_saved_contributions": ranked[:12],
        "doubling_trials": trials,
        "dark": {"normal_01": pack(dark_n[0, 1]), "normal_11": pack(dark_n[1, 1]),
                 "anomalous_01": pack(dark_a[0, 1])},
        "production_G1_pass": False,
        "provenance_sha256": provenance([__file__]),
    })


if __name__ == "__main__":
    main()
