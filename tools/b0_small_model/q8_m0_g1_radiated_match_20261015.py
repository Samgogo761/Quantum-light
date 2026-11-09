"""Undriven adjoint versus the radiated sheet field, not the full U0.

The full one-sided U0 also contains direct transmission on the left port.
The current integral is compared with phi*(sqrt(omega)/eta)*(A-A_f) on every port.
Existing macro-2048 rows only.
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
from q8_m0_g1_dynamic_map_audit_20261015 import require_cache
from q8_m0_g1_gamma12_ir_tail_20261015 import load_adjoint
from q8_m0_g1_gamma12_richardson_full_20261015 import accumulate_arrays
from q8_m0_g1_gamma12_tail_columns_20261014 import load_rows

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results/q_line/g1_gamma12_time_order_20261015/undriven_radiated_match.json"
PROBES = np.array([0.11, 0.25, 1.0, 1.5, 4.0])


def radiated_target(ph, omega, phi_of, eta, pol):
    out = np.zeros((2, 4), complex)
    scale = np.sqrt(omega) / eta
    for j in range(4):
        field = ph.initial.A[:, j] - ph.free_A0[:, j]
        out[:, j] = phi_of(omega) * (pol @ (scale * field))
    return out


def main():
    if OUT.exists():
        raise FileExistsError(OUT)
    _saved, cache, t0, _meta = load_rows()
    if abs(float(t0)) > 0:
        raise AssertionError(t0)
    gamma = float(np.asarray(cache["gamma"]))
    pol = np.array(cache["pol"], copy=True)
    prot = p1_a2_protocol(0.8)
    grid = base.q.omega_grid(prot.omega0, prot.t_pulse)
    grid = grid[grid > 0.0]
    _phi, norm = base.q.normalize_phi(grid, base.q.phi_unnormalized(grid, 2.0 * prot.omega0, prot.t_pulse))
    phi_of = lambda nu: base.q.phi_at(nu, prot, norm)
    eta = base.q.eta_of(gamma)
    path, digest = require_cache(2048)
    rows = load_adjoint(path)
    probes = {}
    for omega in PROBES:
        ph, u0 = base.generated_photon_block(
            cache, gamma, np.array([float(omega)]), np.array([1.0]), pol, phi_of, eta)
        raw_u, raw_v = accumulate_arrays(
            rows[2], rows[3], ph.initial.b, ph.initial.bdag, ph.initial.A, ph.free_A0, ph.omega)
        target = radiated_target(ph, float(omega), phi_of, eta, pol)
        probes[f"{omega:.2f}"] = {
            "max_abs_raw_minus_radiated": float(np.max(np.abs(raw_u - target))),
            "max_abs_raw_v": float(np.max(np.abs(raw_v))),
            "max_abs_radiated": float(np.max(np.abs(target))),
            "max_abs_full_U0": float(np.max(np.abs(u0))),
            "left_minus_right_raw": float(np.max(np.abs(raw_u[:, :2] - raw_u[:, 2:]))),
        }
        print(omega, probes[f"{omega:.2f}"]["max_abs_raw_minus_radiated"], flush=True)
    write_new_json(OUT, {
        "identity": "undriven current integral equals phi*(sqrt(omega)/eta)*(A-A_f) on both ports",
        "weight": 1.0,
        "full_U0_also_contains_direct_transmission_on_the_left": True,
        "cache_sha256": digest,
        "probes": probes,
        "production_G1_pass": False,
        "provenance_sha256": provenance([__file__]),
    })


if __name__ == "__main__":
    main()
