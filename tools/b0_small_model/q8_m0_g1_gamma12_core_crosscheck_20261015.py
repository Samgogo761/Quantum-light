"""Independent quadratures for the narrow core at 0.508995.

The production rule is one Gauss panel in t=arcsinh((omega-Delta)/Gamma).
The checks use the same total node count: two t-halves, and Gauss directly in omega.
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
from q8_m0_g1_gamma12_tail_columns_20261014 import contribution, load_rows, output_columns
from q8_m0_gamma12_input_wing_20261008 import wing_quadrature
from q8_m0_p1_h2_finite_kcv_input_close_20261005 import gauss_segment, t_gauss_panel

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results/q_line/g1_gamma12_panel_budget_20261015/narrow_core_crosscheck.json"


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
    pole, gwid, ta, tb = panel["pole"], panel["gwid"], panel["t0"], panel["t1"]
    single = t_gauss_panel(pole, gwid, ta, tb, 1280)
    mid = 0.5 * (ta + tb)
    left, right = t_gauss_panel(pole, gwid, ta, mid, 640), t_gauss_panel(pole, gwid, mid, tb, 640)
    split = (np.concatenate((left[0], right[0])), np.concatenate((left[1], right[1])))
    flat = gauss_segment(panel["omega_lo"], panel["omega_hi"], 1280)
    rules = {"t_gauss_1280": single, "t_split_640_640": split, "omega_gauss_1280": flat}
    got = {}
    for name, (om, wt) in rules.items():
        u, v = output_columns(rows, cache, gamma, pol, phi_of, eta, t0, om, wt)
        normal, anomalous = contribution(u, v)
        got[name] = {"weight": float(np.sum(wt)), "normal": pack(normal), "anomalous": pack(anomalous)}
        print(name, got[name]["normal"]["maxabs"], flush=True)
    ref_n = np.array(got["t_gauss_1280"]["normal"]["real"]) + 1j * np.array(got["t_gauss_1280"]["normal"]["imag"])
    ref_a = np.array(got["t_gauss_1280"]["anomalous"]["real"]) + 1j * np.array(got["t_gauss_1280"]["anomalous"]["imag"])
    diffs = {}
    for name in ("t_split_640_640", "omega_gauss_1280"):
        normal = np.array(got[name]["normal"]["real"]) + 1j * np.array(got[name]["normal"]["imag"])
        anomalous = np.array(got[name]["anomalous"]["real"]) + 1j * np.array(got[name]["anomalous"]["imag"])
        diffs[name] = {"normal": pack(normal - ref_n), "anomalous": pack(anomalous - ref_a)}
        print("diff", name, diffs[name]["normal"]["maxabs"], diffs[name]["anomalous"]["maxabs"], flush=True)
    write_new_json(OUT, {
        "pole": pole, "omega_lo": panel["omega_lo"], "omega_hi": panel["omega_hi"],
        "gwid": gwid, "t0": ta, "t1": tb,
        "rules": got, "difference_from_t_gauss_1280": diffs,
        "production_G1_pass": False, "provenance_sha256": provenance([__file__]),
    })


if __name__ == "__main__":
    main()
