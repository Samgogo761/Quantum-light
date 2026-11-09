"""Independent CCR identity for a closed quadratic model.

The generator is built from a Hermitian h and a symmetric pairing, using the
frozen gamma and one vertex column only as coefficients. It does not call the
production U,V assembly. A finite oscillator bath is not the production
instantaneous-gamma sheet; both facts are recorded.
"""
from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
from scipy.linalg import expm

from q8_m0_audit_artifacts_20261014 import provenance, write_new_json
from q8_m0_g1_gamma12_tail_columns_20261014 import load_rows
from q8_m0_p1_h2_finite_kcv_input_close_20261005 import eta_of

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results/q_line/g1_gamma12_time_order_20261015/ccr_identity_quadratic.json"


def pack(m):
    m = np.asarray(m, complex)
    return {"real": np.real(m).tolist(), "imag": np.imag(m).tolist(),
            "maxabs": float(np.max(np.abs(m)))}


def nambu(h, pairing):
    """i d_dot = h d + D d†, with the adjoint row fixed by h† and D^T."""
    return np.block([
        [-1j * h, -1j * pairing],
        [1j * np.conjugate(pairing), 1j * np.conjugate(h)],
    ])


def propagate(h, pairing, time):
    sigma = np.block([
        [np.eye(h.shape[0]), np.zeros_like(h)],
        [np.zeros_like(h), -np.eye(h.shape[0])],
    ])
    series = expm(nambu(h, pairing) * time)
    defect = series @ sigma @ series.conj().T - sigma
    n = h.shape[0]
    u, v = series[:n, :n], series[:n, n:]
    return defect, u @ u.conj().T - v @ v.conj().T - np.eye(n), u @ v.T - v @ u.T


def main():
    if OUT.exists():
        raise FileExistsError(OUT)
    _rows, cache, _t0, _meta = load_rows()
    gamma = float(np.asarray(cache["gamma"]))
    eta = float(eta_of(gamma))
    gaps = np.asarray(cache["gaps"], float)
    vertex = np.asarray(cache["u0"])
    bright = int(np.argmax(np.sum(np.abs(vertex) ** 2, axis=0)))
    gap = float(gaps[bright])
    couple = vertex[:, bright]
    freqs = np.array([0.05, 0.10, 0.20, 0.40])
    weights = np.full(4, 0.25)
    # One electron quasi-boson plus one oscillator per frequency.
    # g = eta*sqrt(w/omega) is the frozen oscillator coupling, used once.
    n = 1 + len(freqs)
    h = np.zeros((n, n), complex)
    pairing = np.zeros((n, n), complex)
    h[0, 0] = gap
    for j, (omega, weight) in enumerate(zip(freqs, weights), start=1):
        g = eta * np.sqrt(weight / omega) * couple[0]
        h[j, j] = omega
        h[0, j] = g
        h[j, 0] = np.conjugate(g)
        pairing[0, j] = g
        pairing[j, 0] = g
    hermitian = float(np.max(np.abs(h - h.conj().T)))
    symmetric = float(np.max(np.abs(pairing - pairing.T)))
    defect, full_normal, full_anomalous = propagate(h, pairing, 1.0)
    # Drop the highest oscillator from the input sum. The generator is unchanged.
    series = expm(nambu(h, pairing) * 1.0)
    u, v = series[:n, :n], series[:n, n:]
    kept = np.arange(n - 1)
    dropped = u[:, -1:] @ u[:, -1:].conj().T - v[:, -1:] @ v[:, -1:].conj().T
    partial = u[:, kept] @ u[:, kept].conj().T - v[:, kept] @ v[:, kept].conj().T - np.eye(n)
    # Eliminated sheet chart: (b, b†, A). A is not an extra canonical partner.
    uvec = couple.reshape(2, 1)
    dcontact = np.asarray(cache["D0"], float)
    sheet = np.zeros((4, 4), complex)
    # b_dot = i u^T A, b†_dot = -i u^H A, A_dot = gamma (u* b + u b† - D A)
    sheet[0, 2:4] = 1j * uvec[:, 0]
    sheet[1, 2:4] = -1j * np.conjugate(uvec[:, 0])
    sheet[2:4, 0] = gamma * np.conjugate(uvec[:, 0])
    sheet[2:4, 1] = gamma * uvec[:, 0]
    sheet[2:4, 2:4] = -gamma * dcontact
    sigma_b = np.diag([1.0, -1.0, 0.0, 0.0])
    step = expm(sheet * 0.1)
    sheet_defect = step @ sigma_b @ step.conj().T - sigma_b
    write_new_json(OUT, {
        "identity": "i d_dot = h d + D d† with h Hermitian and D symmetric preserves Sigma=diag(I,-I)",
        "not_the_production_UV_assembly": True,
        "finite_oscillators_are_not_the_instantaneous_gamma_sheet": True,
        "gamma": gamma,
        "eta": eta,
        "electron_gap": gap,
        "hermitian_defect": hermitian,
        "pairing_symmetry_defect": symmetric,
        "full_sigma_defect_after_time_1": pack(defect),
        "full_output_normal_defect": pack(full_normal),
        "full_output_anomalous": pack(full_anomalous),
        "dropped_one_input_normal_defect": pack(partial),
        "dropped_column_normal_gram": pack(dropped),
        "eliminated_sheet_chart_defect": pack(sheet_defect),
        "production_G1_pass": False,
        "provenance_sha256": provenance([__file__]),
    })
    print("sigma", float(np.max(np.abs(defect))),
          "partial", float(np.max(np.abs(partial))),
          "sheet", float(np.max(np.abs(sheet_defect))), flush=True)


if __name__ == "__main__":
    main()
