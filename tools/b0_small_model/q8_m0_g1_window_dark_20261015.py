"""Dark-column window check. Uses the adjoint rows already built for each window.

Dark inputs have zero free field, so the output is the pulled-back row
contracted with the fixed dark initial state. No new propagation.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np

from q8_m0_audit_artifacts_20261014 import provenance, write_new_json
from q8_m0_g1_dynamic_map_audit_20261015 import require_cache
from q8_m0_g1_gamma12_column_order_20261015 import CACHE
from q8_m0_g1_gamma12_ir_tail_20261015 import load_adjoint
from q8_m0_g1_gamma12_richardson_adjoint_20261015 import contract_initial

ROOT = Path(__file__).resolve().parents[2]
OUTDIR = ROOT / "results/q_line/g1_gamma12_time_order_20261015"
PHOTON = OUTDIR / "window_convergence_macro1024.json"
OUT = OUTDIR / "window_dark_columns_macro1024.json"
WINDOWS = {
    "9.0": None,
    "9.5": OUTDIR / "adjoint_rows_macro1024_cycles9p5.npz",
    "10.0": OUTDIR / "adjoint_rows_macro1024_cycles10p0.npz",
    "11.0": OUTDIR / "adjoint_rows_macro1024_cycles11p0.npz",
}


def pack(m):
    m = np.asarray(m, complex)
    return {"real": np.real(m).tolist(), "imag": np.imag(m).tolist(),
            "maxabs": float(np.max(np.abs(m)))}


def unpack(block):
    return np.array(block["real"], float) + 1j * np.array(block["imag"], float)


def dark_map(rows, b, bd, a):
    du, dv = contract_initial(rows[0], b, bd, a)
    uu, uv = contract_initial(rows[2], b, bd, a)
    return du - uu, dv - uv, du, uu, dv, uv


def gram(u, v):
    return u @ u.conj().T - v @ v.conj().T, u @ v.T - v @ u.T


def main():
    if OUT.exists():
        raise FileExistsError(OUT)
    photon = json.loads(PHOTON.read_text(encoding="utf-8"))
    b = np.array(np.load(CACHE / "b_cols.npy", mmap_mode="r")[-1976:].T)
    bd = np.array(np.load(CACHE / "bdag_cols.npy", mmap_mode="r")[-1976:].T)
    a = np.array(np.load(CACHE / "A_cols.npy", mmap_mode="r")[-1976:].T)
    levels = {}
    reference = None
    for label, path in WINDOWS.items():
        if path is None:
            path, _digest = require_cache(1024)
        rows = load_adjoint(path)
        u, v, du, uu, dv, uv = dark_map(rows, b, bd, a)
        normal, anomalous = gram(u, v)
        photon_n = unpack(photon["levels"][label]["grams"]["all"]["normal"])
        photon_a = unpack(photon["levels"][label]["grams"]["all"]["anomalous"])
        level = {
            "dark_columns": 1976,
            "free_field_is_zero_so_only_the_pulled_back_row_enters": True,
            "reused_nine_cycle_dark_gram": False,
            "dark_normal_without_I": pack(normal),
            "dark_anomalous": pack(anomalous),
            "full_normal_residual": pack(photon_n + normal),
            "full_anomalous": pack(photon_a + anomalous),
            "max_abs_driven_raw_u": float(np.max(np.abs(du))),
            "max_abs_undriven_raw_u": float(np.max(np.abs(uu))),
            "max_abs_driven_raw_v": float(np.max(np.abs(dv))),
            "max_abs_undriven_raw_v": float(np.max(np.abs(uv))),
        }
        if reference is None:
            reference = (u, v)
        else:
            level["difference_from_9_cycles"] = {
                "max_abs_delta_U": float(np.max(np.abs(u - reference[0]))),
                "max_abs_delta_V": float(np.max(np.abs(v - reference[1]))),
                "normal": pack(gram(u, v)[0] - gram(*reference)[0]),
                "anomalous": pack(gram(u, v)[1] - gram(*reference)[1]),
            }
        levels[label] = level
        print(label, "dark", float(normal[0, 0].real),
              "full", float((photon_n + normal)[0, 0].real), flush=True)
    write_new_json(OUT, {
        "photon_block_taken_from": str(PHOTON),
        "dark_initial_state_is_the_same_static_basis_at_every_window": True,
        "levels": levels,
        "production_G1_pass": False,
        "provenance_sha256": provenance([__file__]),
    })


if __name__ == "__main__":
    main()
