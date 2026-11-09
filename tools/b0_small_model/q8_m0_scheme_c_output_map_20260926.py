#!/usr/bin/env python3
"""Small algebraic checks for the scheme-C output map. No dynamics."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from q8_m0_classical import kappa_shift, p1_protocol
from q8_m0_gsheet import two_port_from_current
from q8_m0_model import (
    R0, build_geometry, current_ops, h8, op_P, op_theta_unitary,
)

OUT = Path(__file__).resolve().parents[2] / "results/q_line/scheme_c_output_map_20260926"


def frob(a) -> float:
    return float(np.linalg.norm(np.asarray(a)))


def main() -> int:
    geo = build_geometry()
    p, prot = R0, p1_protocol(0.1)
    t = np.linspace(0.0, prot.t_pulse, 17)
    a = kappa_shift(t, geo, prot, +1)
    minus = kappa_shift(t, geo, prot, -1)
    field_sign = float(np.max(np.abs(minus + a)))
    k = geo.k_cart([0.23, -0.17])
    op_p, ut = op_P(), op_theta_unitary()
    h = h8(geo, p, k + a[3], +1)
    j = current_ops(geo, p, k + a[3], +1)
    h_target = h8(geo, p, -(k + a[3]), -1)
    j_target = current_ops(geo, p, -(k + a[3]), -1)
    p_map = {
        "H": frob(op_p @ h @ op_p.conj().T - h_target),
        "j_minus": frob(-(op_p @ j @ op_p.conj().T) - j_target),
    }
    i = np.array([[0.2, -0.1], [0.0, 0.3], [-0.4, 0.2]])
    rec = two_port_from_current(i, np.zeros_like(i), np.zeros_like(i), 0.01, 0.2)
    port_sum = float(np.max(np.abs(rec["A_R_out"] - rec["A_L_out"])))
    rng = np.random.default_rng(11)
    beta = rng.normal(size=2) + 1j * rng.normal(size=2)
    m = rng.normal(size=(2, 2)) + 1j * rng.normal(size=(2, 2))
    m = 0.5 * (m + m.T)
    phase = np.exp(1j * np.array([0.3, -0.7]))
    beta2 = phase * beta
    m2 = np.outer(phase, phase) * m
    payload = {
        "status": "scheme_C_algebraic_map_only",
        "P1_not_run": True,
        "same_drive_sign_residual": field_sign,
        "driven_P_map_residual": p_map,
        "equal_current_equal_two_ports_residual": port_sum,
        "mode_phase_changes_beta": bool(np.max(np.abs(beta2 - beta)) > 0.1),
        "mode_phase_changes_m": bool(np.max(np.abs(m2 - m)) > 0.1),
        "intensity_invariant_under_common_phase": bool(
            np.max(np.abs(np.abs(beta2) - np.abs(beta))) < 1e-12
        ),
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "output_map_algebra.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
