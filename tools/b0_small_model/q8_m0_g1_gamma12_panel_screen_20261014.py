"""Screen Gamma12 quadrature panels. No column propagation.

The current 32904-node measure is wing factor 2, core 160, omega from 1e-4 to 12.
Each complete tail is compared with itself at twice the node count. A large
probe change marks a candidate. It does not by itself move the commutator.
"""
from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np

from q8_m0_audit_artifacts_20261014 import provenance, write_new_json
from q8_m0_gamma12_input_wing_20261008 import wing_quadrature
from q8_m0_gsheet import gamma_self
from q8_m0_model import build_geometry
from q8_m0_p1_h2_finite_kcv_bz_access_20261005 import grid_sample
from q8_m0_p1_h2_finite_kcv_input_close_20261005 import gauss_segment, t_gauss_panel

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results/q_line/g1_gamma12_panel_screen_20261014/tail_screen.json"
OMEGA0 = 0.05


def probe(om, wt):
    om = np.asarray(om, float)
    wt = np.asarray(wt, float)
    if om.size == 0:
        return 0j
    return complex(np.sum(wt * np.exp(1j * om / OMEGA0)))


def refine(panel):
    n = int(panel["requested"])
    if panel["kind"] == "sinh_tail":
        coarse = t_gauss_panel(panel["pole"], panel["gwid"], panel["t0"], panel["t1"], n)
        fine = t_gauss_panel(panel["pole"], panel["gwid"], panel["t0"], panel["t1"], 2 * n)
    elif panel["kind"] == "outer_direct":
        coarse = gauss_segment(panel["omega_lo"], panel["omega_hi"], n)
        fine = gauss_segment(panel["omega_lo"], panel["omega_hi"], 2 * n)
    else:
        return None
    return coarse, fine


def summarize(panel, coarse, fine):
    om, wt = coarse
    gap = float(np.max(np.diff(np.sort(om)))) if om.size > 1 else float(om[0] if om.size else 0.0)
    delta = probe(*fine) - probe(*coarse)
    return {
        "kind": panel["kind"],
        "pole": panel["pole"],
        "gwid": panel["gwid"],
        "omega_lo": panel["omega_lo"],
        "omega_hi": panel["omega_hi"],
        "span": panel["omega_hi"] - panel["omega_lo"],
        "requested": panel["requested"],
        "nodes": panel["nodes"],
        "weight_sum": panel["weight_sum"],
        "max_omega_gap": gap,
        "probe_delta_abs": float(abs(delta)),
        "probe_delta_over_weight": float(abs(delta) / max(panel["weight_sum"], 1e-30)),
    }


def main():
    if OUT.exists():
        raise FileExistsError(OUT)
    geo = build_geometry()
    sample, _inv = grid_sample(geo, 12, 0.0, neel=+1)
    gamma = float(gamma_self(geo))
    panels = []
    oms, wts, counts = wing_quadrature(
        sample["gap"], sample["u"], gamma, n_panel=160, wing_factor=2,
        omega_lo=1.0e-4, omega_hi=12.0, panels=panels)
    print("freq", len(oms), "panels", len(panels), flush=True)
    rows = []
    for panel in panels:
        got = refine(panel)
        if got is None:
            continue
        rows.append(summarize(panel, *got))
    from q8_m0_g1_two_k_tail_refine_20261014 import lowest_tail
    center, width, ta, tb, n_coarse = lowest_tail()
    cal_coarse = t_gauss_panel(center, width, ta, tb, n_coarse)
    cal_fine = t_gauss_panel(center, width, ta, tb, 72)
    calibration = {
        "model": "two-k lowest bright left tail",
        "n_coarse": n_coarse, "n_fine": 72,
        "probe_delta_abs": float(abs(probe(*cal_fine) - probe(*cal_coarse))),
        "weight": float(np.sum(cal_coarse[1])),
        "note": "This panel moved two-k Delta C by about 2.3e-5 at fixed timestep.",
    }
    rows.sort(key=lambda row: row["probe_delta_abs"], reverse=True)
    kind_counts = {}
    for panel in panels:
        kind_counts[panel["kind"]] = kind_counts.get(panel["kind"], 0) + 1
    report = {
        "quadrature": {"nk": 12, "n_panel": 160, "wing_factor": 2,
                       "omega_lo": 1.0e-4, "omega_hi": 12.0,
                       "frequencies": int(len(oms)), "weight_sum": float(np.sum(wts))},
        "panel_counts": kind_counts,
        "request_counts": counts,
        "probe": "sum w exp(i omega / 0.05), compared at n and 2n on the same interval",
        "two_k_calibration": calibration,
        "top_probe_changes": rows[:40],
        "panels_with_probe_delta_above_1e-4": int(sum(row["probe_delta_abs"] > 1e-4 for row in rows)),
        "panels_with_probe_delta_above_1e-3": int(sum(row["probe_delta_abs"] > 1e-3 for row in rows)),
        "largest_omega_gaps": sorted(rows, key=lambda row: row["max_omega_gap"], reverse=True)[:15],
        "production_G1_pass": False,
        "commutator_not_recomputed": True,
        "provenance_sha256": provenance([__file__]),
    }
    write_new_json(OUT, report)
    print("top", rows[0]["probe_delta_abs"], "pole", rows[0]["pole"], "span", rows[0]["span"])
    print("above 1e-3", report["panels_with_probe_delta_above_1e-3"],
          "above 1e-4", report["panels_with_probe_delta_above_1e-4"])


if __name__ == "__main__":
    main()
