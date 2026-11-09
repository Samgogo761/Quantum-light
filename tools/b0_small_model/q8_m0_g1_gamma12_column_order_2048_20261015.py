"""Finer midpoint and one finer Richardson level for the same 16 columns."""
from __future__ import annotations

import json
import os
from pathlib import Path

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np

from q8_m0_audit_artifacts_20261014 import provenance, write_new_json
from q8_m0_classical import p1_a2_protocol
from q8_m0_continuum_columns_20261004 import ColumnState
from q8_m0_g1_gamma12_column_order_20261015 import (
    CACHE, CYCLES, TARGETS, column_gram, load_saved, vertices,
)
from q8_m0_g1_gamma12_tail_columns_20261014 import load_rows
from q8_m0_g1_local_richardson_20261014 import propagate_local_richardson
from q8_m0_g1_two_k_timestep_20261014 import pack, propagate
from q8_m0_model import build_geometry
from q8_m0_p1_h2_finite_kcv_bz_access_20261005 import grid_sample
import q8_m0_gamma12_transpose_input_refine_20261008 as base

ROOT = Path(__file__).resolve().parents[2]
PREV = ROOT / "results/q_line/g1_gamma12_time_order_20261015/column_richardson.json"
OUT = ROOT / "results/q_line/g1_gamma12_time_order_20261015/column_order_2048.json"


def matrix_from(block):
    return np.array(block["real"], float) + 1j * np.array(block["imag"], float)


def main():
    if OUT.exists():
        raise FileExistsError(OUT)
    previous = json.loads(PREV.read_text(encoding="utf-8"))
    _rows, cache, _t0, _meta = load_rows()
    geo = build_geometry()
    sample, _inv = grid_sample(geo, 12, 0.0, neel=1)
    prot = p1_a2_protocol(0.8)
    quiet = p1_a2_protocol(0.0)
    gamma = float(np.asarray(cache["gamma"]))
    pol = np.array(cache["pol"], copy=True)
    saved_oms = np.load(CACHE / "oms.npy")
    saved_wts = np.load(CACHE / "wts.npy")
    chosen = []
    for target in TARGETS:
        index = int(np.argmin(np.abs(saved_oms - target)))
        chosen.append((float(saved_oms[index]), float(saved_wts[index])))
    packet = base.q.omega_grid(prot.omega0, prot.t_pulse)
    packet = packet[packet > 0.0]
    _phi, norm = base.q.normalize_phi(packet, base.q.phi_unnormalized(packet, 2.0 * prot.omega0, prot.t_pulse))
    phi_of = lambda nu: base.q.phi_at(nu, prot, norm)
    ph, u0 = base.generated_photon_block(
        cache, gamma, np.array([o for o, _w in chosen]), np.array([w for _o, w in chosen]),
        pol, phi_of, base.q.eta_of(gamma))
    prep = {
        "initial": ColumnState(ph.initial.b, ph.initial.bdag, ph.initial.A),
        "free": ph.free_A0, "omega": ph.omega, "gamma": gamma, "pol": pol,
        "u0": u0, "drive": prot,
    }
    print("vertices 2048", flush=True)
    h = float(prot.period) / 2048
    fine = {
        "driven": vertices(sample, geo, prot, CYCLES * 2048, h),
        "undriven": vertices(sample, geo, quiet, CYCLES * 2048, h),
    }
    print("midpoint 2048", flush=True)
    driven = propagate(prep, fine["driven"], h)
    undriven = propagate(prep, fine["undriven"], h)
    normal, anomalous = column_gram(prep, driven, undriven)
    print("richardson 1024", flush=True)
    coarse = {
        "driven": load_saved("driven", CYCLES * 1024),
        "undriven": load_saved("undriven", CYCLES * 1024),
    }
    h_macro = float(prot.period) / 1024
    driven_r = propagate_local_richardson(prep, coarse["driven"], fine["driven"], h_macro)
    undriven_r = propagate_local_richardson(prep, coarse["undriven"], fine["undriven"], h_macro)
    normal_r, anomalous_r = column_gram(prep, driven_r, undriven_r)
    levels = {
        "midpoint_2048": {"normal": pack(normal), "anomalous": pack(anomalous)},
        "richardson_1024": {"normal": pack(normal_r), "anomalous": pack(anomalous_r)},
    }
    series = {}
    for kind in ("normal", "anomalous"):
        mids = [matrix_from(previous["levels"][name][kind]) for name in
                ("midpoint_256", "midpoint_512", "midpoint_1024")]
        mids.append(normal if kind == "normal" else anomalous)
        rich = [matrix_from(previous["levels"][name][kind]) for name in
                ("richardson_256", "richardson_512")]
        rich.append(normal_r if kind == "normal" else anomalous_r)
        def ratios(mats):
            diffs = [mats[i] - mats[i + 1] for i in range(len(mats) - 1)]
            out = []
            for left, right in zip(diffs, diffs[1:]):
                denom = np.max(np.abs(right))
                out.append(None if denom < 1e-14 else float(np.max(np.abs(left)) / denom))
            return out
        series[kind] = {
            "midpoint_successive_maxabs_ratio": ratios(mids),
            "richardson_successive_maxabs_ratio": ratios(rich),
            "richardson_1024_minus_midpoint_2048": pack(
                (normal_r if kind == "normal" else anomalous_r) - mids[-1]),
        }
    write_new_json(OUT, {
        "levels": levels, "order_ratios": series,
        "production_G1_pass": False, "provenance_sha256": provenance([__file__]),
    })
    print(series["normal"]["midpoint_successive_maxabs_ratio"],
          series["normal"]["richardson_successive_maxabs_ratio"], flush=True)


if __name__ == "__main__":
    main()
