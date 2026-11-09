"""Denser resonance-core nodes at fixed H2-window density.

n_h2 stays 96, so the static packet mass is already closed. n_panel goes
from 32 to 64. Time steps 1024 and 2048 match the dense H2-window run.
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

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results/q_line/g1_two_k_timestep_20261014/panel64_h2_96.json"


def main():
    if OUT.exists():
        raise FileExistsError(OUT)
    prep = prepare(n_panel=64, n_h2=96)
    print("freq", prep["n_freq"], "mass", prep["mass"], flush=True)
    levels = {}
    raw = {}
    for ntpc in (1024, 2048):
        h = float(prep["drive"].period) / ntpc
        nstep = 8 * ntpc
        print("ntpc", ntpc, flush=True)
        quiet = vertices(prep, 0.0, 0.0, nstep, h)
        driven = vertices(prep, 0.8, 0.0, nstep, h)
        uu, uv = propagate(prep, quiet, h)
        du, dv = propagate(prep, driven, h)
        got = matrices(prep, du - uu, dv - uv)
        raw[ntpc] = got
        levels[str(ntpc)] = {name: pack(mat) for name, mat in got.items()}
        print("  dC", got["against_U0_gram"].real[0, 0], "RN", got["against_I"].real[0, 0], flush=True)
    d = raw[2048]["against_U0_gram"] - raw[1024]["against_U0_gram"]
    floor = raw[2048]["against_U0_gram"] + d / 3.0
    da = raw[2048]["anomalous"] - raw[1024]["anomalous"]
    floor_a = raw[2048]["anomalous"] + da / 3.0
    report = {
        "n_panel": 64, "n_h2": 96,
        "frequencies": prep["n_freq"], "packet_mass": prep["mass"],
        "levels": levels,
        "richardson_1024_2048_against_U0_gram": pack(floor),
        "richardson_1024_2048_anomalous": pack(floor_a),
        "production_G1_pass": False,
        "provenance_sha256": provenance([__file__]),
    }
    write_new_json(OUT, report)
    print("floor", floor.real[0, 0], "max", float(np.max(np.abs(floor))))
    print("anomalous floor max", float(np.max(np.abs(floor_a))))


if __name__ == "__main__":
    main()
