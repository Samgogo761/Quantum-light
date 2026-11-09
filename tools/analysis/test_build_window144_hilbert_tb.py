#!/usr/bin/env python3
"""Unit tests for WINDOW144 Hilbert-completion helpers (no GB Wannier files)."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_ANALYSIS = Path(__file__).resolve().parent
if str(_ANALYSIS) not in sys.path:
    sys.path.insert(0, str(_ANALYSIS))

from build_window144_hilbert_tb import (
    decide_gates,
    fourier_to_R,
    interpolate_h,
    polar_right_unitary,
    reorth_complement,
)


def require(cond: bool, msg: str) -> None:
    if not cond:
        raise RuntimeError(msg)


def test_polar_identity() -> None:
    ov = 2.0 * np.eye(4, dtype=complex)
    u = polar_right_unitary(ov)
    require(np.max(np.abs(u - np.eye(4))) < 1e-12, f"polar(2I)={u}")
    rng = np.random.default_rng(0)
    a = rng.normal(size=(5, 5)) + 1j * rng.normal(size=(5, 5))
    u = polar_right_unitary(a)
    gram = u.conj().T @ u
    require(np.max(np.abs(gram - np.eye(5))) < 1e-12, "polar not unitary")
    # Re Tr(A U) should beat Re Tr(A I) unless I is already optimal.
    score_u = float(np.real(np.trace(a @ u)))
    score_i = float(np.real(np.trace(a)))
    require(score_u + 1e-12 >= score_i, f"polar score {score_u} < I {score_i}")


def test_reorth() -> None:
    rng = np.random.default_rng(1)
    u = np.linalg.qr(rng.normal(size=(10, 4)) + 1j * rng.normal(size=(10, 4)))[0]
    v_raw = rng.normal(size=(10, 3)) + 1j * rng.normal(size=(10, 3))
    v = reorth_complement(u, v_raw)
    require(np.max(np.abs(u.conj().T @ v)) < 1e-12, "U†V")
    require(np.max(np.abs(v.conj().T @ v - np.eye(3))) < 1e-12, "V†V")


def test_fourier_roundtrip() -> None:
    kfrac = np.array(
        [[0.0, 0.0, 0.0], [0.5, 0.0, 0.0], [0.0, 0.5, 0.0], [0.5, 0.5, 0.0]]
    )
    irvec = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]], dtype=int)
    ndegen = np.ones(4)
    rng = np.random.default_rng(2)
    h_k = np.empty((4, 3, 3), dtype=np.complex128)
    for ik in range(4):
        a = rng.normal(size=(3, 3)) + 1j * rng.normal(size=(3, 3))
        h_k[ik] = 0.5 * (a + a.conj().T)
    h_r = fourier_to_R(kfrac, irvec, h_k)
    for ik in range(4):
        back = interpolate_h(irvec, ndegen, h_r, kfrac[ik])
        err = float(np.max(np.abs(back - h_k[ik])))
        require(err < 1e-12, f"roundtrip k={ik} err={err}")


def test_decide_gates() -> None:
    ok = {
        "hermiticity_maxabs_probe": 1e-12,
        "mesh_roundtrip_maxabs": 1e-8,
        "window144_vs_QE_maxabs_eV": 1e-6,
        "pi_block_vs_Uprod_maxabs_eV": 1e-9,
        "U_V_overlap_probe": 1e-12,
        "hop_QQ": {"frac_Rge12": 0.01},
    }
    status = decide_gates(ok)
    require(status["OVERALL"] == "PASS", status)
    warn = dict(ok)
    warn["hop_QQ"] = {"frac_Rge12": 0.5}
    status = decide_gates(warn)
    require(status["Q_LOCALIZATION"] == "FAIL", status)
    require(status["OVERALL"] == "WARN", status)
    bad = dict(ok)
    bad["window144_vs_QE_maxabs_eV"] = 0.2
    status = decide_gates(bad)
    require(status["OVERALL"] == "FAIL", status)


def main() -> int:
    test_polar_identity()
    print("PASS test_polar_identity")
    test_reorth()
    print("PASS test_reorth")
    test_fourier_roundtrip()
    print("PASS test_fourier_roundtrip")
    test_decide_gates()
    print("PASS test_decide_gates")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
