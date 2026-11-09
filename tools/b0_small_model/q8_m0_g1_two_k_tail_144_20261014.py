"""72, 108 and 144 nodes on the same lowest-pole left tail.

The 18-node panel and its weight stay the reference. Each finer rule uses
that same t interval. Packet nodes and every other input node stay put.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np

from q8_m0_audit_artifacts_20261014 import provenance, write_new_json
from q8_m0_g1_production_time_integral_20261014 import prepare
from q8_m0_g1_two_k_tail_refine_20261014 import lowest_tail
from q8_m0_g1_two_k_timestep_20261014 import matrices, pack, propagate, vertices
from q8_m0_gsheet import gamma_self
from q8_m0_model import build_geometry
from q8_m0_p1_h2_finite_kcv_caller_fix_20261005 import TWO_K, WEIGHTS, m0_sample
from q8_m0_p1_h2_finite_kcv_input_close_20261005 import input_quadrature, t_gauss_panel

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "results/q_line/g1_two_k_timestep_20261014"
OUT = BASE / "tail_72_108_144.json"


def nodes_for(n_fine):
    center, width, ta, tb, n_coarse = lowest_tail()
    if n_coarse != 18:
        raise AssertionError(f"Expected the 18-node tail, found {n_coarse}")
    coarse_o, coarse_w = t_gauss_panel(center, width, ta, tb, n_coarse)
    fine_o, fine_w = t_gauss_panel(center, width, ta, tb, n_fine)
    weight_gap = abs(float(fine_w.sum() - coarse_w.sum()))
    if weight_gap > 1e-8:
        raise AssertionError(f"n={n_fine} changed the panel weight by {weight_gap}")
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
    oms = np.concatenate((oms[keep], fine_o))
    wts = np.concatenate((wts[keep], fine_w))
    order = np.argsort(oms)
    info = {"n_fine": int(n_fine), "panel_weight": float(coarse_w.sum()),
            "weight_gap_vs_18": weight_gap, "t_a": ta, "t_b": tb}
    return oms[order], wts[order], info


def load_saved(ntpc, name):
    if ntpc == 512:
        block = json.loads((BASE / "tail18_to_72_ntpc512.json").read_text(encoding="utf-8"))
        if name != "against_U0_gram":
            raise KeyError(name)
        return np.array(block[name]["real"]) + 1j * np.array(block[name]["imag"])
    data = json.loads((BASE / "tail18_to_72.json").read_text(encoding="utf-8"))
    block = data["levels"][str(ntpc)][name]
    return np.array(block["real"]) + 1j * np.array(block["imag"])


def run_level(prep, ntpc):
    h = float(prep["drive"].period) / ntpc
    quiet = vertices(prep, 0.0, 0.0, 8 * ntpc, h)
    driven = vertices(prep, 0.8, 0.0, 8 * ntpc, h)
    uu, uv = propagate(prep, quiet, h)
    du, dv = propagate(prep, driven, h)
    return matrices(prep, du - uu, dv - uv)


def main():
    if OUT.exists():
        raise FileExistsError(OUT)
    plan = {108: (2048,), 144: (1024, 2048)}
    levels = {}
    raw = {}
    for n_fine, steps in plan.items():
        oms, wts, info = nodes_for(n_fine)
        prep = prepare(oms=oms, wts=wts)
        if not (prep["mass"] > 1.0 - 1.0e-8):
            raise AssertionError(f"Packet mass moved at n={n_fine}: {prep['mass']}")
        print("n", n_fine, "freq", prep["n_freq"], "mass", prep["mass"], flush=True)
        levels[str(n_fine)] = {"info": info, "frequencies": prep["n_freq"],
                               "packet_mass": prep["mass"], "steps": {}}
        for ntpc in steps:
            print("  ntpc", ntpc, flush=True)
            got = run_level(prep, ntpc)
            raw[(n_fine, ntpc)] = got
            levels[str(n_fine)]["steps"][str(ntpc)] = {name: pack(mat) for name, mat in got.items()}
            print("   dC", got["against_U0_gram"].real[0, 0], flush=True)
    names = ("against_I", "against_U0_gram", "against_mass_identity", "anomalous")
    versus_72 = {}
    for n_fine, ntpc in ((108, 2048), (144, 1024), (144, 2048)):
        entry = {}
        for name in names:
            if n_fine == 108 and ntpc == 2048 and name != "against_U0_gram" and ntpc == 512:
                continue
            try:
                old = load_saved(ntpc, name)
            except KeyError:
                continue
            delta = raw[(n_fine, ntpc)][name] - old
            entry[name] = pack(delta)
        versus_72[f"n{n_fine}_ntpc{ntpc}"] = entry
        print("vs72", n_fine, ntpc, "dC", entry["against_U0_gram"]["maxabs"],
              "RN", entry.get("against_I", {}).get("maxabs"),
              "RA", entry.get("anomalous", {}).get("maxabs"), flush=True)
    d = raw[(144, 2048)]["against_U0_gram"] - raw[(144, 1024)]["against_U0_gram"]
    floor = raw[(144, 2048)]["against_U0_gram"] + d / 3.0
    write_new_json(OUT, {
        "levels": levels,
        "difference_from_72_same_timestep": versus_72,
        "richardson_144_1024_2048_against_U0_gram": pack(floor),
        "production_G1_pass": False,
        "provenance_sha256": provenance([__file__]),
    })
    print("144 floor", floor.real[0, 0], "max", float(np.max(np.abs(floor))))


if __name__ == "__main__":
    main()
