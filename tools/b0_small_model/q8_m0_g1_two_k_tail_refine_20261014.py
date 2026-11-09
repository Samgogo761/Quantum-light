"""Nested refinement of one tail, with the integrated weight held fixed.

The closed-packet quadrature has 18 Gauss nodes on the left tail of the
lowest bright pole, from about 0.32 to 2.07. Four of them carry the whole
[0.3, 1) weight. This run replaces those 18 nodes by 72 nodes on the same
t interval. Every other input node stays put.
"""
from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np

from q8_m0_audit_artifacts_20261014 import provenance, write_new_json
from q8_m0_g1_production_time_integral_20261014 import prepare
from q8_m0_g1_two_k_timestep_20261014 import matrices, pack, propagate, vertices
from q8_m0_gsheet import gamma_self
from q8_m0_model import build_geometry
from q8_m0_p1_h2_finite_kcv_caller_fix_20261005 import TWO_K, WEIGHTS, m0_sample
from q8_m0_p1_h2_finite_kcv_input_close_20261005 import (
    N_TAIL_CAP, N_TAIL_MIN, T_MAX_CAP, T_MAX_MIN, T_TAIL_CAP, cluster_rank_width,
    even_count, grouped_bright_gaps, input_quadrature, t_gauss_panel)

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results/q_line/g1_two_k_timestep_20261014/tail18_to_72.json"


def lowest_tail():
    geo = build_geometry()
    sample = m0_sample(geo, TWO_K, WEIGHTS, +1)
    gamma = float(gamma_self(geo))
    bright, widths, groups = grouped_bright_gaps(sample["gap"], sample["u"], gamma)
    cuts = np.empty(len(groups) + 1)
    cuts[0], cuts[-1] = 1.0e-3, 12.0
    g_left = np.array([float(bright[grp][0]) for grp in groups])
    g_right = np.array([float(bright[grp][-1]) for grp in groups])
    for i in range(len(groups) - 1):
        cuts[i + 1] = 0.5 * (g_right[i] + g_left[i + 1])
    for i, grp in enumerate(groups):
        dgs, gs = bright[grp], widths[grp]
        pole_cuts = (np.concatenate(([cuts[i]], 0.5 * (dgs[:-1] + dgs[1:]), [cuts[i + 1]]))
                     if dgs.size > 1 else np.array([cuts[i], cuts[i + 1]]))
        for j, dg in enumerate(dgs):
            a, b = float(pole_cuts[j]), float(pole_cuts[j + 1])
            gwid = max(float(gs[j]), 1e-16)
            _rank, all_w = cluster_rank_width(sample["gap"], sample["u"], gamma, float(dg))
            g_max = float(np.max(all_w)) if all_w.size else gwid
            t_left = float(np.clip(np.arcsinh((a - float(dg)) / gwid), -T_TAIL_CAP, T_TAIL_CAP))
            half = max(min(float(dg) - a, b - float(dg)), 8.0 * gwid)
            t_cell = float(np.arcsinh(half / gwid))
            t_wide = float(np.arcsinh(8.0 * g_max / gwid))
            t_core = min(T_MAX_CAP, max(T_MAX_MIN, t_wide, min(t_cell, T_MAX_CAP)))
            t_core = min(t_core, max(-t_left, 0.0))
            o0 = float(dg) + gwid * np.sinh(t_left)
            o1 = float(dg) + gwid * np.sinh(-t_core)
            if not (min(o0, o1) < 0.30 < max(o0, o1)):
                continue
            x0, x1 = 0.30, max(o0, o1)
            ta = float(np.arcsinh((x0 - float(dg)) / gwid))
            tb = float(np.arcsinh((x1 - float(dg)) / gwid))
            if tb < ta:
                ta, tb = tb, ta
            n = min(even_count(int(N_TAIL_MIN + 2.0 * (tb - ta)), N_TAIL_MIN), N_TAIL_CAP)
            return float(dg), gwid, ta, tb, n
    raise RuntimeError("Lowest-pole tail not found.")


def refined_nodes():
    center, width, ta, tb, n_coarse = lowest_tail()
    if n_coarse != 18:
        raise AssertionError(f"Expected the 18-node tail, found {n_coarse}")
    coarse_o, coarse_w = t_gauss_panel(center, width, ta, tb, n_coarse)
    fine_o, fine_w = t_gauss_panel(center, width, ta, tb, 72)
    geo = build_geometry()
    sample = m0_sample(geo, TWO_K, WEIGHTS, +1)
    gamma = float(gamma_self(geo))
    oms, wts = input_quadrature(sample["gap"], sample["u"], gamma, n_panel=32, n_h2=96,
                                omega_lo=1.0e-3, omega_hi=12.0)
    keep = np.ones(oms.size, dtype=bool)
    for om, wt in zip(coarse_o, coarse_w):
        hit = np.where(np.abs(oms - om) < 1e-10)[0]
        if hit.size != 1 or abs(float(wts[hit[0]] - wt)) > 1e-9:
            raise AssertionError(f"Tail node {om} is not a clean quadrature node.")
        keep[hit[0]] = False
    if abs(float(wts[~keep].sum() - coarse_w.sum())) > 1e-9:
        raise AssertionError("Removed weight does not match the tail panel.")
    if abs(float(fine_w.sum() - coarse_w.sum())) > 1e-8:
        raise AssertionError("Fine panel changed the integrated weight.")
    oms = np.concatenate((oms[keep], fine_o))
    wts = np.concatenate((wts[keep], fine_w))
    order = np.argsort(oms)
    return oms[order], wts[order], {
        "center": center, "width": width, "t_a": ta, "t_b": tb,
        "n_coarse": n_coarse, "n_fine": 72,
        "panel_weight": float(coarse_w.sum()),
    }


def main():
    if OUT.exists():
        raise FileExistsError(OUT)
    oms, wts, info = refined_nodes()
    prep = prepare(oms=oms, wts=wts)
    print("freq", prep["n_freq"], "mass", prep["mass"], "panel", info["panel_weight"], flush=True)
    if not (prep["mass"] > 1.0 - 1.0e-8):
        raise AssertionError(f"Packet mass moved: {prep['mass']}")
    levels, raw = {}, {}
    for ntpc in (1024, 2048):
        h = float(prep["drive"].period) / ntpc
        print("ntpc", ntpc, flush=True)
        quiet = vertices(prep, 0.0, 0.0, 8 * ntpc, h)
        driven = vertices(prep, 0.8, 0.0, 8 * ntpc, h)
        uu, uv = propagate(prep, quiet, h)
        du, dv = propagate(prep, driven, h)
        got = matrices(prep, du - uu, dv - uv)
        raw[ntpc] = got
        levels[str(ntpc)] = {name: pack(mat) for name, mat in got.items()}
        print("  dC", got["against_U0_gram"].real[0, 0], flush=True)
    d = raw[2048]["against_U0_gram"] - raw[1024]["against_U0_gram"]
    floor = raw[2048]["against_U0_gram"] + d / 3.0
    da = raw[2048]["anomalous"] - raw[1024]["anomalous"]
    floor_a = raw[2048]["anomalous"] + da / 3.0
    write_new_json(OUT, {
        "replacement": info, "frequencies": prep["n_freq"], "packet_mass": prep["mass"],
        "levels": levels,
        "richardson_1024_2048_against_U0_gram": pack(floor),
        "richardson_1024_2048_anomalous": pack(floor_a),
        "production_G1_pass": False,
        "provenance_sha256": provenance([__file__]),
    })
    print("floor", floor.real[0, 0], "max", float(np.max(np.abs(floor))))
    print("anomalous floor max", float(np.max(np.abs(floor_a))))


if __name__ == "__main__":
    main()
