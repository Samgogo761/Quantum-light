#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Fock / squeezed-vacuum moment selftest. Method check only. Not physics.

Independent of q8_m0 production. Does not assemble U,V, does not propagate
electrons, does not claim AFM HHG. Allowed wall < 2 min.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

OUT_DIR = Path(__file__).resolve().parents[2] / "results" / "q_line" / "qdrive_inventory_20261025"
TASK = "qdrive_fock_moments_selftest"
DATE = "2026-10-25"
NOT_PHYSICS = True
PRODUCTION_G1_PASS = False


def boson_ops(n_max: int):
    dim = n_max + 1
    a = np.zeros((dim, dim), dtype=complex)
    for n in range(n_max):
        a[n, n + 1] = np.sqrt(n + 1.0)
    return a, a.conj().T


def coherent_ket(alpha: complex, n_max: int) -> np.ndarray:
    psi = np.zeros(n_max + 1, dtype=complex)
    log_amp = -0.5 * abs(alpha) ** 2
    psi[0] = np.exp(log_amp)
    for n in range(1, n_max + 1):
        log_amp += np.log(alpha) - 0.5 * np.log(n)
        psi[n] = np.exp(log_amp)
    nrm = np.linalg.norm(psi)
    if nrm == 0.0:
        raise RuntimeError("coherent ket vanished")
    return psi / nrm


def squeezed_vacuum_ket(r: float, theta: float, n_max: int) -> np.ndarray:
    """S(ξ)|0>, ξ = r e^{iθ}. Truncated to n = 0..n_max (even only)."""
    psi = np.zeros(n_max + 1, dtype=complex)
    amp = np.sqrt(1.0 / np.cosh(r))
    psi[0] = amp
    phase_tanh = -np.exp(1j * theta) * np.tanh(r)
    k = 1
    while 2 * k <= n_max:
        amp = amp * phase_tanh * np.sqrt((2 * k - 1) / (2 * k))
        psi[2 * k] = amp
        k += 1
    nrm = np.linalg.norm(psi)
    if nrm == 0.0:
        raise RuntimeError("squeezed ket vanished")
    return psi / nrm


def moments(psi: np.ndarray, a, ad):
    nrm2 = float(np.vdot(psi, psi).real)
    n_exp = float(np.vdot(psi, (ad @ a) @ psi).real)
    aa = complex(np.vdot(psi, (a @ a) @ psi))
    e_like = float(np.vdot(psi, (a + ad) @ psi).real) / np.sqrt(2.0)
    return dict(norm2=nrm2, n=n_exp, aa_re=aa.real, aa_im=aa.imag, e_quad=e_like)


def check(name, cond, detail=""):
    return dict(name=name, ok=bool(cond), detail=detail)


def main() -> int:
    t0 = time.perf_counter()
    rng = np.random.default_rng(20261025)
    a, ad = boson_ops(48)
    tests = []

    vac = np.zeros(49, dtype=complex)
    vac[0] = 1.0
    mv = moments(vac, a, ad)
    tests.append(check("vacuum_norm", abs(mv["norm2"] - 1.0) < 1e-14, f"{mv['norm2']}"))
    tests.append(check("vacuum_n", abs(mv["n"]) < 1e-14, f"{mv['n']}"))
    tests.append(check("vacuum_aa", abs(mv["aa_re"]) + abs(mv["aa_im"]) < 1e-14))

    alpha = 1.7 * np.exp(1j * 0.4)
    psi_c = coherent_ket(alpha, 48)
    mc = moments(psi_c, a, ad)
    tests.append(check("coherent_norm", abs(mc["norm2"] - 1.0) < 1e-12, f"{mc['norm2']}"))
    tests.append(check("coherent_n", abs(mc["n"] - abs(alpha) ** 2) < 1e-10, f"{mc['n']} vs {abs(alpha)**2}"))
    want_aa = alpha * alpha
    tests.append(
        check(
            "coherent_aa",
            abs(complex(mc["aa_re"], mc["aa_im"]) - want_aa) < 1e-10,
            f"{mc['aa_re']:+.6g}{mc['aa_im']:+.6g}j",
        )
    )

    r, theta = 0.8, 0.6
    psi_s = squeezed_vacuum_ket(r, theta, 48)
    ms = moments(psi_s, a, ad)
    n_th = np.sinh(r) ** 2
    aa_th = -np.exp(1j * theta) * np.sinh(r) * np.cosh(r)
    tests.append(check("sv_norm", abs(ms["norm2"] - 1.0) < 1e-12, f"{ms['norm2']}"))
    tests.append(check("sv_n", abs(ms["n"] - n_th) < 1e-8, f"{ms['n']} vs {n_th}"))
    tests.append(
        check(
            "sv_aa",
            abs(complex(ms["aa_re"], ms["aa_im"]) - aa_th) < 1e-7,
            f"got {ms['aa_re']:+.6g}{ms['aa_im']:+.6g}j want {aa_th}",
        )
    )
    tests.append(check("sv_mean_E_near_zero", abs(ms["e_quad"]) < 1e-10, f"{ms['e_quad']}"))

    n_max_list = [8, 16, 24, 32]
    n_err = []
    for n_max in n_max_list:
        aa, aad = boson_ops(n_max)
        psi = squeezed_vacuum_ket(r, theta, n_max)
        n_cut = float(np.vdot(psi, (aad @ aa) @ psi).real)
        n_err.append(abs(n_cut - n_th) / max(n_th, 1e-15))
    tests.append(check("sv_truncation_monotone", n_err[-1] < n_err[0], str(n_err)))
    tests.append(check("sv_n_max32_rel_err", n_err[-1] < 1e-5, f"{n_err[-1]:.3e}"))

    r_hot = 1.6
    n_th_hot = np.sinh(r_hot) ** 2
    psi_hot8 = squeezed_vacuum_ket(r_hot, 0.0, 8)
    aa8, aad8 = boson_ops(8)
    n8 = float(np.vdot(psi_hot8, (aad8 @ aa8) @ psi_hot8).real)
    tests.append(
        check(
            "strong_r_needs_larger_cutoff",
            abs(n8 - n_th_hot) / n_th_hot > 1e-3,
            f"n8={n8:.4g} n_th={n_th_hot:.4g} (truncation visible; not a physics result)",
        )
    )

    _ = rng  # keep seed documented; no sampling physics
    elapsed = time.perf_counter() - t0
    ok = all(t["ok"] for t in tests)
    out = dict(
        task=TASK,
        date=DATE,
        not_physics=NOT_PHYSICS,
        production_G1_pass=PRODUCTION_G1_PASS,
        elapsed_s=elapsed,
        all_ok=ok,
        tests=tests,
        notes=(
            "Fock algebra only. Does not evolve electrons. "
            "Does not equate Husimi-Q sampling with this truncated ket."
        ),
    )
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / "fock_moments_selftest.json"
    path.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"all_ok={ok} elapsed_s={elapsed:.4f} wrote {path}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
