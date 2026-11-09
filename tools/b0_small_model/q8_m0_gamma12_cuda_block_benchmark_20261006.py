#!/usr/bin/env python3
"""Short CUDA column-block throughput audit for the existing Gamma12 fixture.

This is an engineering benchmark only.  It reuses the already generated
256-column fixture, keeps the full electronic feedback in every propagation,
and does not start a production path or alter the physical model.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE))

from q8_m0_column_step_torch_20261006 import accumulate_torch, gpu_probe
from q8_m0_continuum_columns_20261004 import ColumnState
from q8_m0_p1_h2_finite_kcv_caller_fix_20261005 import free_A_increment


FIXTURE = Path(
    os.environ.get(
        "Q8_BACKEND_FIXTURE",
        str(
            REPO
            / "results/q_line/p1_h2_finite_kcv_gamma12_backend_cuda_resume_20261006/backend_fixture.npz"
        ),
    )
)
NSTEP = 16
REPEATS = 2


class PhotonSlice:
    def __init__(self, omega: np.ndarray, free_A0: np.ndarray):
        self.omega = np.asarray(omega, float)
        self.free_A0 = np.asarray(free_A0, complex)

    def free_A(self, t: float) -> np.ndarray:
        return self.free_A0 * np.exp(-1j * self.omega * float(t))[None, :]


def slice_block(z, lo: int, hi: int, n_photon: int):
    block = ColumnState(z["b"][:, lo:hi], z["bdag"][:, lo:hi], z["A"][:, lo:hi])
    ph_hi = min(hi, n_photon)
    ph_lo = min(lo, n_photon)
    if ph_hi > ph_lo:
        photon = PhotonSlice(z["omega"][ph_lo:ph_hi], z["free_A0"][:, ph_lo:ph_hi])
    else:
        photon = PhotonSlice(np.zeros(0), np.zeros((2, 0)))
    n_point = max(0, hi - max(lo, n_photon))
    return photon, block, n_point


def run_size(z, block_size: int, torch_mod, device) -> dict:
    ncol = int(z["b"].shape[1])
    n_photon = int(z["omega"].size)
    vertices = [
        (z["vdp_u"][j], z["vdp_d"][j])
        for j in range(NSTEP)
    ]
    blocks = []
    for lo in range(0, ncol, block_size):
        hi = min(ncol, lo + block_size)
        blocks.append((lo, hi))

    records = []
    total_start = time.perf_counter()
    for lo, hi in blocks:
        photon, initial, n_point = slice_block(z, lo, hi, n_photon)
        rec = accumulate_torch(
            photon,
            float(z["gamma"]),
            initial,
            n_point,
            z["kp"],
            vertices,
            NSTEP,
            float(z["h"]),
            z["pol"],
            free_A_increment,
            torch_mod,
            device,
            t0=float(z["t_pump"]),
        )
        records.append(
            {
                "lo": lo,
                "hi": hi,
                "ncol": hi - lo,
                "n_photon": int(photon.omega.size),
                "n_point": int(n_point),
                "wall_s": float(rec["wall_s"]),
                "h2d_setup_s": float(rec["h2d_setup_s"]),
                "loop_s": float(rec["loop_s"]),
                "d2h_s": float(rec["d2h_s"]),
                "peak_allocated_bytes": int(rec["peak_allocated_bytes"]),
                "peak_reserved_bytes": int(rec["peak_reserved_bytes"]),
            }
        )
    total_wall = time.perf_counter() - total_start
    return {
        "block_size": int(block_size),
        "n_blocks": len(blocks),
        "n_columns_covered": ncol,
        "total_wall_s": float(total_wall),
        "sum_kernel_wall_s": float(sum(r["wall_s"] for r in records)),
        "peak_allocated_bytes": max(r["peak_allocated_bytes"] for r in records),
        "peak_reserved_bytes": max(r["peak_reserved_bytes"] for r in records),
        "blocks": records,
    }


def main() -> int:
    probe = gpu_probe()
    if not probe.get("cuda_is_available"):
        print(json.dumps({"status": "not_run", "probe": probe}, indent=2))
        return 2

    import torch

    z = np.load(FIXTURE, allow_pickle=False)
    device = torch.device("cuda:0")
    # Warm up the same kernel family before collecting block-size timings.
    photon, initial, n_point = slice_block(z, 0, int(z["b"].shape[1]), int(z["omega"].size))
    _ = accumulate_torch(
        photon,
        float(z["gamma"]),
        initial,
        n_point,
        z["kp"],
        [(z["vdp_u"][j], z["vdp_d"][j]) for j in range(NSTEP)],
        2,
        float(z["h"]),
        z["pol"],
        free_A_increment,
        torch,
        device,
        t0=float(z["t_pump"]),
    )
    torch.cuda.synchronize()

    rows = []
    for block_size in (64, 128, 256):
        reps = [run_size(z, block_size, torch, device) for _ in range(REPEATS)]
        best = min(reps, key=lambda row: row["total_wall_s"])
        rows.append(
            {
                "block_size": block_size,
                "repeats": REPEATS,
                "best": best,
                "all_total_wall_s": [row["total_wall_s"] for row in reps],
            }
        )

    print(
        json.dumps(
            {
                "status": "cuda_block_benchmark_complete",
                "fixture": str(FIXTURE),
                "nstep": NSTEP,
                "pump_window_is_algebraic_not_physical_from_t0": True,
                "probe": probe,
                "rows": rows,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

