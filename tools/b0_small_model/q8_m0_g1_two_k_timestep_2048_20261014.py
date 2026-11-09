"""Add the 2048-step level to the fixed 526-frequency two-k ladder."""
from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import json
import numpy as np

from q8_m0_audit_artifacts_20261014 import provenance, write_new_json
from q8_m0_g1_two_k_timestep_20261014 import matrices, pack, prepare, propagate, vertices

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "results/q_line/g1_two_k_timestep_20261014"
OUT = BASE / "timestep_2048.json"
NPZ = BASE / "columns_ntpc_2048.npz"


def load_matrix(level, name):
    block = level[name]
    return np.array(block["real"]) + 1j * np.array(block["imag"])


def main():
    if OUT.exists():
        raise FileExistsError(OUT)
    prep = prepare()
    if prep["n_freq"] != 526:
        raise AssertionError(f"Frequency count changed: {prep['n_freq']}")
    ntpc = 2048
    h = float(prep["drive"].period) / ntpc
    nstep = 8 * ntpc
    print("ntpc", ntpc, "steps", nstep, flush=True)
    quiet = vertices(prep, 0.0, 0.0, nstep, h)
    driven = vertices(prep, 0.8, 0.0, nstep, h)
    uu, uv = propagate(prep, quiet, h)
    du, dv = propagate(prep, driven, h)
    delta_u, delta_v = du - uu, dv - uv
    got = matrices(prep, delta_u, delta_v)
    level = {name: pack(mat) for name, mat in got.items()}
    np.savez(NPZ, U=prep["u0"] + delta_u, V=delta_v, delta_u=delta_u, delta_v=delta_v)
    previous = json.loads((BASE / "timestep_ladder.json").read_text(encoding="utf-8"))
    names = ("against_I", "against_U0_gram", "against_mass_identity", "anomalous")
    mats = {}
    for name in names:
        mats[name] = [load_matrix(previous["levels"][k], name) for k in ("512", "1024")]
        mats[name].append(got[name])
    comparison = {}
    for name in names:
        m512, m1024, m2048 = mats[name]
        d_prev = m1024 - m512
        d_new = m2048 - m1024
        # R(h/2) = R(h) + (R(h) - R(2h)) / 4. The minus sign extrapolates backward.
        predicted = m1024 + d_prev / 4.0
        old_floor = m1024 + d_prev / 3.0
        new_floor = m2048 + d_new / 3.0
        comparison[name] = {
            "predicted_2048_from_512_1024": pack(predicted),
            "actual_2048": pack(m2048),
            "prediction_minus_actual": pack(predicted - m2048),
            "richardson_512_1024": pack(old_floor),
            "richardson_1024_2048": pack(new_floor),
            "floor_shift": pack(new_floor - old_floor),
        }
    report = {
        "frequencies": 526,
        "packet_mass": prep["mass"],
        "ntpc": 2048,
        "steps": nstep,
        "h": h,
        "columns_npz": NPZ.name,
        "level": level,
        "comparison_with_512_1024": comparison,
        "production_G1_pass": False,
        "provenance_sha256": provenance([__file__, Path(__file__).with_name(
            "q8_m0_g1_two_k_timestep_20261014.py")]),
    }
    write_new_json(OUT, report)
    dc = comparison["against_U0_gram"]
    print("actual dC00", level["against_U0_gram"]["real"][0][0])
    print("predicted", dc["predicted_2048_from_512_1024"]["real"][0][0])
    print("old floor", dc["richardson_512_1024"]["real"][0][0])
    print("new floor", dc["richardson_1024_2048"]["real"][0][0])
    print("floor shift", dc["floor_shift"]["maxabs"])


if __name__ == "__main__":
    main()
