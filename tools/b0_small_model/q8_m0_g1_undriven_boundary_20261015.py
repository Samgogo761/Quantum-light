"""Compare the undriven time integral with the stationary U0 formula.

Production sets U = U0 + raw_driven - raw_undriven, so an undriven run equals
U0 by subtraction. This script does not subtract. It contracts the saved
adjoint rows and asks whether that integral already equals U0. No new
propagation and no overwrite of the adjoint caches.
"""
from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np

import q8_m0_gamma12_transpose_input_refine_20261008 as base
from q8_m0_audit_artifacts_20261014 import provenance, write_new_json
from q8_m0_classical import p1_a2_protocol
from q8_m0_g1_dynamic_map_audit_20261015 import EXPECTED, require_cache
from q8_m0_g1_gamma12_ir_tail_20261015 import load_adjoint
from q8_m0_g1_gamma12_richardson_adjoint_20261015 import contract_initial
from q8_m0_g1_gamma12_richardson_full_20261015 import accumulate_arrays
from q8_m0_g1_gamma12_tail_columns_20261014 import load_rows

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results/q_line/g1_gamma12_time_order_20261015/undriven_boundary_match.json"
PROBES = np.array([0.11, 0.25, 1.0, 1.5, 4.0])


def column_report(raw_u, raw_v, u0):
    gap = raw_u - u0
    scale = np.maximum(np.max(np.abs(u0), axis=0), 1e-30)
    return {
        "max_abs_raw_u": float(np.max(np.abs(raw_u))),
        "max_abs_raw_v": float(np.max(np.abs(raw_v))),
        "max_abs_U0": float(np.max(np.abs(u0))),
        "max_abs_raw_u_minus_U0": float(np.max(np.abs(gap))),
        "max_relative_per_column": float(np.max(np.max(np.abs(gap), axis=0) / scale)),
        "port_order": ["L0", "L1", "R0", "R1"],
        "relative_per_column": (np.max(np.abs(gap), axis=0) / scale).tolist(),
    }


def main():
    if OUT.exists():
        raise FileExistsError(OUT)
    _saved, cache, t0, _meta = load_rows()
    if abs(t0) > 0:
        raise AssertionError(t0)
    gaps = np.asarray(cache["gaps"], float)
    if np.min(np.abs(PROBES[:, None] - gaps[None, :])) < 1e-4:
        raise AssertionError("A probe sits on an electron gap")
    gamma = float(np.asarray(cache["gamma"]))
    pol = np.array(cache["pol"], copy=True)
    prot = p1_a2_protocol(0.8)
    grid = base.q.omega_grid(prot.omega0, prot.t_pulse)
    grid = grid[grid > 0.0]
    _phi, norm = base.q.normalize_phi(grid, base.q.phi_unnormalized(grid, 2.0 * prot.omega0, prot.t_pulse))
    phi_of = lambda nu: base.q.phi_at(nu, prot, norm)
    eta = base.q.eta_of(gamma)
    reports = {}
    for macro in (1024, 2048):
        path, digest = require_cache(macro)
        rows = load_adjoint(path)
        level = {}
        for omega in PROBES:
            ph, u0 = base.generated_photon_block(
                cache, gamma, np.array([float(omega)]), np.array([1.0]), pol, phi_of, eta)
            undriven = accumulate_arrays(
                rows[2], rows[3], ph.initial.b, ph.initial.bdag, ph.initial.A, ph.free_A0, ph.omega)
            driven = accumulate_arrays(
                rows[0], rows[1], ph.initial.b, ph.initial.bdag, ph.initial.A, ph.free_A0, ph.omega)
            init_u, init_v = contract_initial(rows[2], ph.initial.b, ph.initial.bdag, ph.initial.A)
            level[f"{omega:.2f}"] = {
                "undriven_versus_U0": column_report(undriven[0], undriven[1], u0),
                "driven_minus_undriven_maxabs": float(np.max(np.abs(driven[0] - undriven[0]))),
                "undriven_initial_history_maxabs": float(np.max(np.abs(init_u))),
                "undriven_force_maxabs": float(np.max(np.abs(undriven[0] - init_u))),
                "cache_sha256": digest,
            }
            print(macro, omega, level[f"{omega:.2f}"]["undriven_versus_U0"]["max_relative_per_column"], flush=True)
        dark_b = np.array(np.load(ROOT / "results/q_line/p1_h2_finite_kcv_gamma12_local_plusN_20261007/arrays/b_cols.npy", mmap_mode="r")[-8:].T)
        dark_bd = np.array(np.load(ROOT / "results/q_line/p1_h2_finite_kcv_gamma12_local_plusN_20261007/arrays/bdag_cols.npy", mmap_mode="r")[-8:].T)
        dark_a = np.array(np.load(ROOT / "results/q_line/p1_h2_finite_kcv_gamma12_local_plusN_20261007/arrays/A_cols.npy", mmap_mode="r")[-8:].T)
        zeros = np.zeros((2, 8), complex)
        dark_u, dark_v = accumulate_arrays(rows[2], rows[3], dark_b, dark_bd, dark_a, zeros, np.zeros(8))
        level["undriven_dark_8columns"] = {
            "max_abs_raw_u": float(np.max(np.abs(dark_u))),
            "max_abs_raw_v": float(np.max(np.abs(dark_v))),
            "U0_is_zero_for_dark": True,
        }
        reports[str(macro)] = level
    write_new_json(OUT, {
        "question": "undriven finite-time adjoint integral versus stationary U0, before driven-minus-undriven subtraction",
        "weight": 1.0,
        "not_a_production_gram": True,
        "probes": reports,
        "production_G1_pass": False,
        "provenance_sha256": provenance([__file__]),
    })


if __name__ == "__main__":
    main()
