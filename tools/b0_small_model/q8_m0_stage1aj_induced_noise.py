#!/usr/bin/env python3
"""P1=0 prescribed-input induced-current IR ledger.

Dual bookkeeping only:
  S_raw  = K_N(ω) S_κ(ω) K_N(ω)†
  S_delta = ΔK(ω) S_κ(ω) ΔK(ω)†
with frozen isotropic shape S_κ(ω) = 1_2 / ω.

Maps the same tensors to the local sheet source field
  E_src = -γ I_cell,  S_E = γ² S_I
using the existing two-port γ. No self-consistent Maxwell, poles,
intrinsic noise, cross term, collection packet, n/m/V, or initial state.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
import numpy as np

from q8_m0_classical import k_grid
from q8_m0_gsheet import gamma_self
from q8_m0_stage1ag_acceptance_audit import (
    REPO, MU, R0, build_geometry, h8, current_ops, occup_empty_bases,
    x_from_bases, d_ops, mean_d_from_bases, parallel_perp_basis,
)
from q8_m0_stage1ai_lowfreq_kernel import OMEGA0_OVER_TH, pack_matrix, power_fit
from q8_m0_stage1ai_lowfreq_audit import expansion, increment, kernel, spectral_data

OUT = REPO / "results/q_line/stage1aj_induced_noise_20260924"
# Dense log grid well below the ~0.093 direct gap. ω0 is a unit, not a drive.
OMEGA_OVER_OMEGA0 = np.geomspace(1e-6, 3e-2, 37)
OMEGA_MAX_OVER_OMEGA0 = 3e-2
OMEGA_MIN_SCAN_OVER_OMEGA0 = np.geomspace(1e-6, 3e-3, 13)
GRIDS = (
    ("Gamma_96x96", 96, 0.0),
    ("Gamma_128x128", 128, 0.0),
    ("Gamma_192x192", 192, 0.0),
    ("halfshift_192x192", 192, 0.5),
)


def frobenius2(m):
    m = np.asarray(m)
    return float(np.real(np.vdot(m, m)))


def s_from_k(k, omega):
    """S = K (I/ω) K†. ω>0 required."""
    k = np.asarray(k, dtype=complex)
    return (k @ k.conj().T) / float(omega)


def logtrapz(x, y):
    """∫ y dx on a positive, strictly increasing grid via trapezoid."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(x) < 2:
        return 0.0
    return float(np.trapezoid(y, x))


def assemble_grid(geo, name, nk, shift, omegas, omega_max, omega_mins, gamma):
    _, kc, weights = k_grid(geo, nk, shift=shift)
    vv, vc, en = occup_empty_bases(h8(geo, R0, kc, 1), MU)
    x = x_from_bases(current_ops(geo, R0, kc, 1), vv, vc)
    gaps = en[:, 4:, None] - en[:, None, :4]
    gap, c = spectral_data(gaps, x, weights)
    d = np.einsum("k,kab->ab", weights, mean_d_from_bases(d_ops(geo, R0, kc, 1), vv))
    k0 = kernel(d, gap, c, 0)
    linear, quadratic = expansion(gap, c)
    samples = []
    for w in omegas:
        kw = kernel(d, gap, c, w)
        delta, even, odd = increment(gap, c, w)
        s_raw = s_from_k(kw, w)
        s_delta = s_from_k(delta, w)
        rec = {
            "omega": float(w),
            "omega_over_omega0": float(w / OMEGA0_OVER_TH),
            "K_raw": pack_matrix(kw),
            "K_delta": pack_matrix(delta),
            "even_increment_maxabs": float(np.max(np.abs(even))),
            "odd_increment_maxabs": float(np.max(np.abs(odd))),
            "S_I_raw_trace": frobenius2(kw) / float(w),
            "S_I_delta_trace": frobenius2(delta) / float(w),
            "S_I_raw": pack_matrix(s_raw),
            "S_I_delta": pack_matrix(s_delta),
            "S_E_src_raw_trace": gamma**2 * frobenius2(kw) / float(w),
            "S_E_src_delta_trace": gamma**2 * frobenius2(delta) / float(w),
            "raw_times_omega": frobenius2(kw),
            "delta_over_omega3": frobenius2(delta) / (float(w)**4),
            "predicted_raw_M_over_omega": frobenius2(k0) / float(w),
            "predicted_delta_Q2_omega3": frobenius2(quadratic) * float(w)**3,
        }
        samples.append(rec)

    traces_raw = np.array([s["S_I_raw_trace"] for s in samples])
    traces_delta = np.array([s["S_I_delta_trace"] for s in samples])
    om = np.array([s["omega"] for s in samples])
    ir = om / OMEGA0_OVER_TH <= 0.003000001
    fit_raw = power_fit(om[ir], traces_raw[ir])
    fit_delta = power_fit(om[ir], traces_delta[ir])
    fit_delta_all = power_fit(om, traces_delta)

    integrals = []
    for wmin in omega_mins:
        mask = (om >= wmin) & (om <= omega_max)
        integrals.append({
            "omega_min": float(wmin),
            "omega_min_over_omega0": float(wmin / OMEGA0_OVER_TH),
            "omega_max": float(omega_max),
            "omega_max_over_omega0": float(omega_max / OMEGA0_OVER_TH),
            "W_I_raw": logtrapz(om[mask], traces_raw[mask]),
            "W_I_delta": logtrapz(om[mask], traces_delta[mask]),
            "W_E_src_raw": gamma**2 * logtrapz(om[mask], traces_raw[mask]),
            "W_E_src_delta": gamma**2 * logtrapz(om[mask], traces_delta[mask]),
            "n_points": int(np.count_nonzero(mask)),
        })

    return {
        "grid": name,
        "n_k": int(len(kc)),
        "shift": shift,
        "min_direct_gap": float(np.min(gap)),
        "static_K0": pack_matrix(k0),
        "linear_coefficient": pack_matrix(linear),
        "quadratic_coefficient": pack_matrix(quadratic),
        "frobenius2_M": frobenius2(k0),
        "frobenius2_Q": frobenius2(quadratic),
        "power_fit_S_I_raw_omega_le_0p003_omega0": fit_raw,
        "power_fit_S_I_delta_omega_le_0p003_omega0": fit_delta,
        "power_fit_S_I_delta_all": fit_delta_all,
        "samples": samples,
        "W_I_omega_min_scan": integrals,
    }


def compare_grids(rows):
    out = []
    pairs = (("Gamma_128x128", "Gamma_192x192"),
             ("Gamma_192x192", "halfshift_192x192"),
             ("Gamma_96x96", "Gamma_192x192"))
    by = {r["grid"]: r for r in rows}
    for a, b in pairs:
        ra, rb = by[a], by[b]
        ma, mb = ra["frobenius2_M"], rb["frobenius2_M"]
        qa, qb = ra["frobenius2_Q"], rb["frobenius2_Q"]
        checks = []
        for sa, sb in zip(ra["samples"], rb["samples"]):
            checks.append({
                "omega_over_omega0": sa["omega_over_omega0"],
                "raw_trace_relative": float(
                    abs(sa["S_I_raw_trace"] - sb["S_I_raw_trace"]) / max(sb["S_I_raw_trace"], 1e-30)
                ),
                "delta_trace_relative": float(
                    abs(sa["S_I_delta_trace"] - sb["S_I_delta_trace"]) / max(sb["S_I_delta_trace"], 1e-30)
                ),
            })
        out.append({
            "grids": [a, b],
            "M_frobenius2_ratio": float(ma / mb) if mb else None,
            "Q_frobenius2_relative": float(abs(qa - qb) / qb) if qb else None,
            "frequency_checks": checks,
        })
    return out


def main():
    geo = build_geometry()
    gamma = float(gamma_self(geo))
    omegas = OMEGA_OVER_OMEGA0 * OMEGA0_OVER_TH
    omega_max = OMEGA_MAX_OVER_OMEGA0 * OMEGA0_OVER_TH
    omega_mins = OMEGA_MIN_SCAN_OVER_OMEGA0 * OMEGA0_OVER_TH
    rows = []
    for name, nk, shift in GRIDS:
        rec = assemble_grid(geo, name, nk, shift, omegas, omega_max, omega_mins, gamma)
        rows.append(rec)
        print(name,
              "M2", rec["frobenius2_M"],
              "Q2", rec["frobenius2_Q"],
              "fit_raw", rec["power_fit_S_I_raw_omega_le_0p003_omega0"],
              "fit_delta", rec["power_fit_S_I_delta_omega_le_0p003_omega0"],
              flush=True)

    payload = {
        "status": "p1_zero_prescribed_input_induced_current_ledger_only",
        "definition": (
            "S_I = K S_kappa K^dagger with isotropic S_kappa = I_2 / omega; "
            "delta I = -K delta kappa; E_src = -gamma I_cell"
        ),
        "S_kappa": {
            "form": "isotropic_I2_over_omega",
            "not_a_photon_number": True,
            "not_self_consistent_Maxwell": True,
            "note": (
                "Shape of the frozen q_parallel=0 two-port vacuum A-spectrum. "
                "Absolute SI prefactor is not needed for IR scaling. "
                "Do not interpret Tr S_I as a photon occupation."
            ),
        },
        "E_src_map": {
            "E_src": "-gamma I_cell",
            "gamma": gamma,
            "S_E_src": "gamma^2 S_I",
            "not_full_E_out": True,
            "not_collection_packet": True,
        },
        "units": {
            "frequency": "hbar omega / t_h",
            "kappa": "Angstrom^-1",
            "K": "t_h Angstrom^2 per cell",
            "S_I_trace": "code_units of Tr(K K^dagger / omega)",
        },
        "omega0": OMEGA0_OVER_TH,
        "omega_over_omega0": OMEGA_OVER_OMEGA0.tolist(),
        "omega_max_over_omega0": OMEGA_MAX_OVER_OMEGA0,
        "omega_min_scan_over_omega0": OMEGA_MIN_SCAN_OVER_OMEGA0.tolist(),
        "static_remainder_retained": True,
        "increment_is_diagnostic_not_counterterm": True,
        "grids": rows,
        "comparisons": compare_grids(rows),
        "blocked": [
            "full E_out",
            "intrinsic electronic noise",
            "input-response cross term",
            "self-consistent Maxwell",
            "coupled-mode poles",
            "collection packet",
            "n,m,V,S_min",
            "Pauli redefinition",
            "bare vs dressed vacuum",
        ],
        "source_sha256": {
            str(p.relative_to(REPO)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in (
                Path(__file__).resolve(),
                REPO / "tools/b0_small_model/q8_m0_stage1ai_lowfreq_audit.py",
                REPO / "tools/b0_small_model/q8_m0_stage1ai_lowfreq_kernel.py",
                REPO / "tools/b0_small_model/q8_m0_gsheet.py",
            )
        },
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "induced_noise.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()

