"""Longer-time screen of every Gamma12 tail. No column propagation.

exp(i omega / 0.05) only tests t=20. The eight-cycle pulse lasts about 1005.
Each tail is refined on its own interval and scored at several times, and
again with the pole resolvent kept in the integrand.
"""
from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np

from q8_m0_audit_artifacts_20261014 import provenance, write_new_json
from q8_m0_classical import p1_a2_protocol
from q8_m0_gamma12_input_wing_20261008 import wing_quadrature
from q8_m0_gsheet import gamma_self
from q8_m0_model import build_geometry
from q8_m0_p1_h2_finite_kcv_bz_access_20261005 import grid_sample
from q8_m0_p1_h2_finite_kcv_input_close_20261005 import gauss_segment, t_gauss_panel

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results/q_line/g1_gamma12_panel_screen_20261014/tail_screen_pulse_times.json"


def times_of_interest():
    period = float(p1_a2_protocol(0.8).period)
    return {
        "t20": 20.0,
        "early_quarter_cycle": 0.25 * period,
        "one_cycle": period,
        "mid_pulse": 4.0 * period,
        "pulse_end": 8.0 * period,
        "one_cycle_difference": period,
    }


def panel_rules(panel):
    n = int(panel["requested"])
    if panel["kind"] == "sinh_tail":
        return (t_gauss_panel(panel["pole"], panel["gwid"], panel["t0"], panel["t1"], n),
                t_gauss_panel(panel["pole"], panel["gwid"], panel["t0"], panel["t1"], 2 * n))
    if panel["kind"] == "outer_direct":
        return (gauss_segment(panel["omega_lo"], panel["omega_hi"], n),
                gauss_segment(panel["omega_lo"], panel["omega_hi"], 2 * n))
    if panel["kind"] == "h2_wing":
        return (gauss_segment(panel["omega_lo"], panel["omega_hi"], n),
                gauss_segment(panel["omega_lo"], panel["omega_hi"], 2 * n))
    return None


def phase_delta(coarse, fine, t, weight):
    def acc(om, wt):
        return np.sum(wt * weight(om) * np.exp(1j * om * t))
    return complex(acc(*fine) - acc(*coarse))


def flat(_om):
    return 1.0


def resolvent(panel):
    pole = float(panel["pole"])
    width = panel["gwid"]
    scale = 1e-12 if width is None else max(float(width), 1e-12)

    def weight(om):
        return 1.0 / (pole - om + 1j * scale)
    return weight


def main():
    if OUT.exists():
        raise FileExistsError(OUT)
    geo = build_geometry()
    sample, _inv = grid_sample(geo, 12, 0.0, neel=+1)
    gamma = float(gamma_self(geo))
    panels = []
    oms, _wts, counts = wing_quadrature(
        sample["gap"], sample["u"], gamma, n_panel=160, wing_factor=2,
        omega_lo=1.0e-4, omega_hi=12.0, panels=panels)
    clock = times_of_interest()
    rows = []
    for panel in panels:
        rules = panel_rules(panel)
        if rules is None:
            continue
        coarse, fine = rules
        scores = {}
        worst = 0.0
        worst_name = ""
        for name, t in clock.items():
            if name == "one_cycle_difference":
                continue
            delta = abs(phase_delta(coarse, fine, t, flat))
            scores[name] = float(delta)
            if delta > worst:
                worst, worst_name = float(delta), name
        response = {}
        response_worst = 0.0
        weight = resolvent(panel)
        for name, t in (("mid_pulse", clock["mid_pulse"]), ("pulse_end", clock["pulse_end"])):
            delta = abs(phase_delta(coarse, fine, t, weight))
            response[name] = float(delta)
            response_worst = max(response_worst, float(delta))
        rows.append({
            "kind": panel["kind"], "pole": panel["pole"], "gwid": panel["gwid"],
            "omega_lo": panel["omega_lo"], "omega_hi": panel["omega_hi"],
            "span": panel["omega_hi"] - panel["omega_lo"],
            "requested": panel["requested"], "nodes": panel["nodes"],
            "weight_sum": panel["weight_sum"],
            "flat_phase": scores, "flat_worst": worst, "flat_worst_time": worst_name,
            "resolvent_phase": response, "resolvent_worst": response_worst,
        })
    rows.sort(key=lambda row: max(row["flat_worst"], row["resolvent_worst"]), reverse=True)
    above = lambda key, tol: int(sum(row[key] > tol for row in rows))
    report = {
        "frequencies": int(len(oms)),
        "request_counts": counts,
        "times": clock,
        "pulse_period": float(p1_a2_protocol(0.8).period),
        "note": "A small flat-phase difference does not certify the production commutator.",
        "flat_worst_above_1e-3": above("flat_worst", 1e-3),
        "flat_worst_above_1e-6": above("flat_worst", 1e-6),
        "resolvent_worst_above_1e-3": above("resolvent_worst", 1e-3),
        "top": rows[:25],
        "production_G1_pass": False,
        "provenance_sha256": provenance([__file__]),
    }
    write_new_json(OUT, report)
    print("freq", len(oms), "tails", len(rows))
    print("flat>1e-3", report["flat_worst_above_1e-3"], "flat>1e-6", report["flat_worst_above_1e-6"])
    print("top", rows[0]["pole"], rows[0]["flat_worst"], rows[0]["flat_worst_time"],
          "resolvent", rows[0]["resolvent_worst"])


if __name__ == "__main__":
    main()
