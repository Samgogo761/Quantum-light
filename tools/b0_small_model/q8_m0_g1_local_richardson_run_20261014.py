"""Cost check of the local Richardson step on the 72-node tail.

Macro steps 256 and 512. Each macro step costs three production midpoints.
The 72-node quadrature is the one whose extrapolated floor is already near 1e-6.
"""
from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np

from q8_m0_audit_artifacts_20261014 import provenance, write_new_json
from q8_m0_g1_local_richardson_20261014 import propagate_local_richardson
from q8_m0_g1_production_time_integral_20261014 import prepare, vertices
from q8_m0_g1_two_k_tail_144_20261014 import nodes_for
from q8_m0_g1_two_k_timestep_20261014 import matrices, pack

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results/q_line/g1_two_k_timestep_20261014/local_richardson_72.json"


def run(prep, ntpc):
    h = float(prep["drive"].period) / ntpc
    nstep = 8 * ntpc
    coarse_q = vertices(prep, 0.0, 0.0, nstep, h)
    coarse_d = vertices(prep, 0.8, 0.0, nstep, h)
    fine_q = vertices(prep, 0.0, 0.0, 2 * nstep, 0.5 * h)
    fine_d = vertices(prep, 0.8, 0.0, 2 * nstep, 0.5 * h)
    uu, uv = propagate_local_richardson(prep, coarse_q, fine_q, h)
    du, dv = propagate_local_richardson(prep, coarse_d, fine_d, h)
    return matrices(prep, du - uu, dv - uv)


def main():
    if OUT.exists():
        raise FileExistsError(OUT)
    oms, wts, info = nodes_for(72)
    prep = prepare(oms=oms, wts=wts)
    print("freq", prep["n_freq"], "mass", prep["mass"], flush=True)
    levels = {}
    raw = {}
    for ntpc in (256, 512):
        print("macro", ntpc, flush=True)
        got = run(prep, ntpc)
        raw[ntpc] = got
        levels[str(ntpc)] = {name: pack(mat) for name, mat in got.items()}
        print("  dC", got["against_U0_gram"].real[0, 0],
              "max", float(np.max(np.abs(got["against_U0_gram"]))),
              "RA", float(np.max(np.abs(got["anomalous"]))), flush=True)
    d = raw[512]["against_U0_gram"] - raw[256]["against_U0_gram"]
    # A fourth-order step would shrink by about 16, not 4. Record the ratio.
    ratio = np.zeros_like(d)
    mask = np.abs(raw[256]["against_U0_gram"]) > 1e-12
    ratio[mask] = raw[512]["against_U0_gram"][mask] / raw[256]["against_U0_gram"][mask]
    write_new_json(OUT, {
        "n_tail": 72, "frequencies": prep["n_freq"], "packet_mass": prep["mass"],
        "panel": info, "levels": levels,
        "level_ratio_512_over_256_against_U0_gram": pack(ratio),
        "midpoint_equivalent_solves": {"256": 3 * 8 * 256, "512": 3 * 8 * 512},
        "production_G1_pass": False,
        "provenance_sha256": provenance([__file__, Path(__file__).with_name(
            "q8_m0_g1_local_richardson_20261014.py")]),
    })


if __name__ == "__main__":
    main()
