"""One finer local-Richardson level on the same 72-node tail."""
from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np

from q8_m0_audit_artifacts_20261014 import provenance, write_new_json
from q8_m0_g1_local_richardson_run_20261014 import run
from q8_m0_g1_production_time_integral_20261014 import prepare
from q8_m0_g1_two_k_tail_144_20261014 import nodes_for
from q8_m0_g1_two_k_timestep_20261014 import pack

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results/q_line/g1_two_k_timestep_20261014/local_richardson_72_macro1024.json"


def main():
    if OUT.exists():
        raise FileExistsError(OUT)
    oms, wts, _info = nodes_for(72)
    prep = prepare(oms=oms, wts=wts)
    got = run(prep, 1024)
    print("dC", got["against_U0_gram"].real[0, 0], "max", float(np.max(np.abs(got["against_U0_gram"]))))
    print("RN", float(np.max(np.abs(got["against_I"]))))
    print("RA", float(np.max(np.abs(got["anomalous"]))))
    write_new_json(OUT, {
        "n_tail": 72, "macro_ntpc": 1024, "frequencies": prep["n_freq"],
        "packet_mass": prep["mass"],
        "midpoint_solves": 3 * 8 * 1024,
        "level": {name: pack(mat) for name, mat in got.items()},
        "production_G1_pass": False,
        "provenance_sha256": provenance([__file__]),
    })


if __name__ == "__main__":
    main()
