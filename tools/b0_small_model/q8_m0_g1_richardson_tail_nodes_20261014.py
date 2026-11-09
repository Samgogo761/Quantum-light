"""Same fourth-order diagnostic at macro 1024, for 108 and 144 tail nodes.

The 72-node result is already stored. This file only adds the two refinements.
"""
from __future__ import annotations

import json
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
BASE = ROOT / "results/q_line/g1_two_k_timestep_20261014"
OUT = BASE / "local_richardson_1024_tail_nodes.json"


def unpack(block):
    return np.array(block["real"], float) + 1j * np.array(block["imag"], float)


def main():
    if OUT.exists():
        raise FileExistsError(OUT)
    saved = json.loads((BASE / "local_richardson_72_macro1024.json").read_text(encoding="utf-8"))
    reference = {name: unpack(saved["level"][name])
                 for name in ("against_I", "against_U0_gram", "anomalous")}
    report = {"macro_ntpc": 1024, "reference_tail_nodes": 72, "levels": {},
              "difference_from_72": {}}
    for n_fine in (108, 144):
        oms, wts, info = nodes_for(n_fine)
        prep = prepare(oms=oms, wts=wts)
        if abs(prep["mass"] - 1.0) > 1e-8:
            raise AssertionError(f"Packet mass moved: {prep['mass']}")
        print("n", n_fine, "freq", prep["n_freq"], flush=True)
        got = run(prep, 1024)
        report["levels"][str(n_fine)] = {
            "frequencies": prep["n_freq"], "packet_mass": prep["mass"], "panel": info,
            "matrices": {name: pack(mat) for name, mat in got.items()},
        }
        diff = {}
        for name in reference:
            delta = got[name] - reference[name]
            diff[name] = pack(delta)
            print(" ", name, float(np.max(np.abs(delta))), flush=True)
        report["difference_from_72"][str(n_fine)] = diff
    report["production_G1_pass"] = False
    report["provenance_sha256"] = provenance([__file__])
    write_new_json(OUT, report)


if __name__ == "__main__":
    main()
