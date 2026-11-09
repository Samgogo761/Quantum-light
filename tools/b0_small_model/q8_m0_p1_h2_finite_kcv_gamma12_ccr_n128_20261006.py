#!/usr/bin/env python3
"""Γ12 n_panel=128 refine against archived n=96. Does not overwrite gamma12_ccr_fix.json."""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import q8_m0_p1_h2_finite_kcv_gamma12_ccr_fix_20261006 as fx
import q8_m0_p1_h2_finite_kcv_input_close_20261005 as q
from q8_m0_gsheet import gamma_self
from q8_m0_model import build_geometry

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "results/q_line/p1_h2_finite_kcv_gamma12_ccr_fix_20261006"
OLD = OUT / "gamma12_ccr_fix.json"
q.OUT = OUT
fx.OUT = OUT


def main() -> int:
    t0 = time.perf_counter()
    if not OLD.exists():
        raise FileNotFoundError(OLD)
    prev = json.loads(OLD.read_text(encoding="utf-8"))
    fx.progress("n128 refine start")
    geo = build_geometry()
    gamma = float(gamma_self(geo))
    g12, inv12 = fx.grid_sample(geo, 12, 0.0)
    oms, wts = q.input_quadrature(g12["gap"], g12["u"], gamma, n_panel=128)
    mass = fx.mass_report(oms, wts)
    fx.progress(f"n128 nodes={oms.size} mass_err={mass['mass_rel_err']:.3g}")
    meas = fx.bright_ccr(g12, gamma, oms, wts)
    prev96 = prev["gamma12_access"]["bright_ccr_fine"]
    refine_tr = abs(meas["trace_photon_bb"] - prev96["trace_photon_bb"]) / max(abs(prev96["trace_photon_bb"]), 1e-30)
    refine_n = abs(meas["electronic_ccr_normal_maxabs"] - prev96["electronic_ccr_normal_maxabs"]) / max(
        abs(prev96["electronic_ccr_normal_maxabs"]), 1e-30
    )
    fx.progress(
        f"n128 CCR_n={meas['electronic_ccr_normal_maxabs']:.4g} CCR_a={meas['electronic_ccr_anomalous_maxabs']:.4g} "
        f"refine_tr={refine_tr:.4g} refine_n={refine_n:.4g} t={meas['elapsed_s']:.1f}s"
    )
    gate = bool(
        meas["electronic_ccr_normal_maxabs"] < 0.05
        and meas["electronic_ccr_anomalous_maxabs"] < 0.05
        and prev96["electronic_ccr_normal_maxabs"] < 0.05
        and prev96["electronic_ccr_anomalous_maxabs"] < 0.05
        and refine_tr < 0.05
        and int(meas.get("skipped_poles", 0)) == 0
    )
    payload = {
        "status": "p1_h2_gamma12_ccr_n96_vs_n128_not_domain",
        "did_not_overwrite_gamma12_ccr_fix": True,
        "n96_from": str(OLD.relative_to(REPO)),
        "n96": {
            "ccr_normal": prev96["electronic_ccr_normal_maxabs"],
            "ccr_anomalous": prev96["electronic_ccr_anomalous_maxabs"],
            "trace_photon_bb": prev96["trace_photon_bb"],
            "min_eig": prev96["electronic_ccr_min_eig"],
        },
        "n128": meas,
        "mass": mass,
        "refine_trace_bb": refine_tr,
        "refine_ccr_normal": refine_n,
        "working_grid_is_n_panel_96": True,
        "n_panel_64_not_enough_for_gamma12_offdiag": True,
        "gate_n96_with_n128_refine": gate,
        "did_not_start_pulse": True,
        "elapsed_s": time.perf_counter() - t0,
        "production": False,
    }
    path = OUT / "gamma12_ccr_n128.json"
    if path.exists():
        raise FileExistsError(path)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({"gate": gate, "n128_CCR_n": meas["electronic_ccr_normal_maxabs"], "refine_tr": refine_tr, "out": str(path)}, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
