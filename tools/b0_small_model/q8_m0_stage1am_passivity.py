#!/usr/bin/env python3
"""P1=0 closed-loop passivity / positive-real audit.

Laplace: s = -i z, Re s > 0 ⇔ Im z > 0. Then
    F(z) = I + iγ K(z)/z  becomes  F_s = I + γ Y,  Y = K(is)/s.
Increment ledger: Y_Δ(s) = ΔK(is)/s = i (ΔK/z)|_{z=is}.
Raw ledger keeps M_N/s and is not expected to be positive-real.

No photons as oscillators, no noise, no initial state, no P1.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from q8_m0_classical import k_grid
from q8_m0_gsheet import gamma_self
from q8_m0_stage1ag_acceptance_audit import (
    REPO, MU, R0, build_geometry, h8, current_ops, occup_empty_bases,
    x_from_bases, d_ops, mean_d_from_bases,
)
from q8_m0_stage1ai_lowfreq_kernel import pack_matrix
from q8_m0_stage1ai_lowfreq_audit import spectral_data
from q8_m0_stage1al_pole_audit import (
    closed_loop_F, det2, increment_over_z, kernel_off_cuts,
    predicted_raw_soft_poles,
)

OUT = REPO / "results/q_line/stage1am_passivity_20260924"
GRIDS = (
    ("Gamma_96x96", 96, 0.0),
    ("Gamma_128x128", 128, 0.0),
    ("Gamma_192x192", 192, 0.0),
    ("halfshift_192x192", 192, 0.5),
)


def y_delta(gap, c, s):
    """Y_Δ(s)=ΔK(is)/s = i (ΔK/z)|_{z=is}."""
    s = complex(s)
    return 1j * increment_over_z(gap, c, 1j * s)[0]


def y_raw(d, gap, c, s):
    s = complex(s)
    z = 1j * s
    return kernel_off_cuts(d, gap, c, z)[0] / s


def herm_part(y):
    y = np.asarray(y, dtype=complex)
    return 0.5 * (y + y.conj().T)


def analytic_re_form(gap, c, s, v):
    """Re v† Y_Δ v from the rank-1 spectral identity, including C*."""
    s = complex(s)
    v = np.asarray(v, dtype=complex)
    x, yim = s.real, s.imag
    alpha = np.einsum("a,abp,b->p", v.conj(), c, v)
    beta = np.einsum("a,abp,b->p", v.conj(), c.conj(), v)
    nminus = (gap + yim) ** 2 + x * x
    nplus = (gap - yim) ** 2 + x * x
    return float(np.real(np.sum((x / gap) * (alpha / nminus + beta / nplus))))


def newton_raw_root(d, gap, c, gamma, z0, niter=24):
    z = complex(z0)
    hist = []
    for _ in range(niter):
        scale = max(abs(z), 1e-18)
        h = 1e-6 * scale + 1e-6j * scale
        zs = np.array([z, z + h], dtype=complex)
        det, _ = eval_raw_det(d, gap, c, gamma, zs)
        val, der = det[0], (det[1] - det[0]) / h
        hist.append({"z_real": z.real, "z_imag": z.imag, "abs_det": float(abs(val))})
        if abs(val) < 1e-16 or abs(der) < 1e-40:
            break
        z = z - val / der
    det, _ = eval_raw_det(d, gap, c, gamma, np.array([z], dtype=complex))
    return {
        "z_real": float(z.real),
        "z_imag": float(z.imag),
        "abs_z": float(abs(z)),
        "abs_det": float(abs(det[0])),
        "converged": bool(abs(det[0]) < 1e-12),
        "last_steps": hist[-3:],
    }


def eval_raw_det(d, gap, c, gamma, zs):
    k = kernel_off_cuts(d, gap, c, zs)
    f = closed_loop_F(k, zs, gamma, from_increment=False)
    return det2(f), f


def probe_s_list(min_gap, rng):
    out = [
        0.02 + 0.0j, 0.01 + 0.03j, 0.01 - 0.03j, 0.05 + 0.5j, 0.05 - 0.5j,
        1e-3 + 1j * 0.5 * min_gap, 1e-3 - 1j * 0.5 * min_gap,
        1e-4 + 1j * min_gap, 1e-4 - 1j * min_gap,
        1e-4 + 1.1j * min_gap, 1e-4 - 1.1j * min_gap,
        1e-3 + 1.0j, 1e-3 - 1.0j, 0.2 + 2.0j, 0.2 - 2.0j, 1.0 + 0.1j,
    ]
    extra = rng.uniform(1e-4, 1.5, size=12) + 1j * rng.uniform(-8.0, 8.0, size=12)
    return [complex(s) for s in list(out) + extra.tolist()]


def assemble_grid(geo, name, nk, shift, gamma, rng):
    _, kc, weights = k_grid(geo, nk, shift=shift)
    vv, vc, en = occup_empty_bases(h8(geo, R0, kc, 1), MU)
    x = x_from_bases(current_ops(geo, R0, kc, 1), vv, vc)
    gaps = en[:, 4:, None] - en[:, None, :4]
    gap, c = spectral_data(gaps, x, weights)
    d = np.einsum("k,kab->ab", weights, mean_d_from_bases(d_ops(geo, R0, kc, 1), vv))
    k0 = kernel_off_cuts(d, gap, c, 0.0)[0]
    min_gap = float(np.min(gap))
    probes = []
    min_eig_delta = np.inf
    min_eig_raw = np.inf
    max_form_err = 0.0
    v = np.array([0.6 + 0.2j, -0.3 + 0.4j], dtype=complex)
    v = v / np.linalg.norm(v)
    for s in probe_s_list(min_gap, rng):
        yd = y_delta(gap, c, s)
        yr = y_raw(d, gap, c, s)
        hd = herm_part(yd)
        hr = herm_part(yr)
        eigs_d = np.linalg.eigvalsh(hd)
        eigs_r = np.linalg.eigvalsh(hr)
        form = complex(v.conj() @ yd @ v)
        analytic = analytic_re_form(gap, c, s, v)
        err = abs(form.real - analytic)
        max_form_err = max(max_form_err, err)
        min_eig_delta = min(min_eig_delta, float(np.min(eigs_d)))
        min_eig_raw = min(min_eig_raw, float(np.min(eigs_r)))
        fs = np.eye(2) + gamma * yd
        probes.append({
            "s_real": s.real, "s_imag": s.imag,
            "H_delta_eigs": eigs_d.tolist(),
            "H_raw_eigs": eigs_r.tolist(),
            "Re_vYv_delta": form.real,
            "analytic_Re_vYv_delta": analytic,
            "form_abs_err": err,
            "min_Re_eig_Fs_delta": float(np.min(np.linalg.eigvalsh(herm_part(fs)))),
        })
    poles = []
    for pred in predicted_raw_soft_poles(k0, gamma):
        z0 = pred["z_real"] + 1j * pred["z_imag"]
        found = newton_raw_root(d, gap, c, gamma, z0)
        poles.append({
            **pred,
            "refined": found,
            "abs_z_minus_pred": float(abs(
                (found["z_real"] + 1j * found["z_imag"]) - z0)),
            "gamma_abs_m": float(abs(gamma * (pred["static_eigenvalue"]["real"]
                                              + 1j * pred["static_eigenvalue"]["imag"]))),
        })
        print(name, "raw_root", found["z_imag"], "abs_det", found["abs_det"],
              "shift", poles[-1]["abs_z_minus_pred"], flush=True)
    return {
        "grid": name, "n_k": int(len(kc)), "shift": shift,
        "min_direct_gap": min_gap,
        "static_K0": pack_matrix(k0),
        "passivity_delta_min_eig": min_eig_delta,
        "passivity_raw_min_eig": min_eig_raw,
        "max_quadratic_form_abs_err": max_form_err,
        "n_probes": len(probes),
        "probes": probes,
        "raw_soft_roots": poles,
    }


def main():
    geo = build_geometry()
    gamma = float(gamma_self(geo))
    rng = np.random.default_rng(0)
    rows = []
    for name, nk, shift in GRIDS:
        rec = assemble_grid(geo, name, nk, shift, gamma, rng)
        rows.append(rec)
        print(name, "HY_delta_min", rec["passivity_delta_min_eig"],
              "HY_raw_min", rec["passivity_raw_min_eig"],
              "form_err", rec["max_quadratic_form_abs_err"], flush=True)
    payload = {
        "status": "p1_zero_closed_loop_passivity_audit_only",
        "definition": {
            "s": "-i z, Re s>0 is retarded UHP",
            "Y_delta": "DeltaK(i s)/s = i (DeltaK/z)|_{z=is}",
            "Y_raw": "K(i s)/s includes M_N/s",
            "claim": "H_Y_delta = (Y_delta+Y_delta^dagger)/2 ≽ 0 for Re s>0",
            "then": "Re v^dagger F_s v = ||v||^2 + gamma Re v^dagger Y_delta v > 0",
        },
        "gamma_self": gamma,
        "grids": rows,
        "blocked": [
            "feedback quantum noise / I-O",
            "bare vs dressed vacuum",
            "Pauli redefinition",
            "n,m,V,S_min",
            "P1 drive",
        ],
        "source_sha256": {
            str(p.relative_to(REPO)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in (
                Path(__file__).resolve(),
                REPO / "tools/b0_small_model/q8_m0_stage1al_pole_audit.py",
                REPO / "tools/b0_small_model/q8_m0_stage1ai_lowfreq_audit.py",
            )
        },
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "passivity.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
