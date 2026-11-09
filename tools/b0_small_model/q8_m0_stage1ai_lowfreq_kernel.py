#!/usr/bin/env python3
"""P1=0 full-BZ low-frequency electronic kernel K^A(ω).

Static half-shift check plus retarded D+Π(ω) on frozen M0. No photons,
no joint propagation, no Pauli redefinition, no initial-state change.
"""
from __future__ import annotations
import json
import os
import sys
import time
from pathlib import Path
os.environ.setdefault('OPENBLAS_NUM_THREADS', '4')
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
from q8_m0_classical import k_grid, p1_a2_protocol
from q8_m0_stage1ag_acceptance_audit import (
    REPO, MU, R0, build_geometry, h8, current_ops, occup_empty_bases,
    x_from_bases, d_ops, mean_d_from_bases, parallel_perp_basis)

OUT = REPO / 'results/q_line/stage1ai_ir_kernel_20260924'
OMEGA0_OVER_TH = float(p1_a2_protocol(0.).omega0)
# Physical retarded convention e^{-iωt}: Π(ω)=Σ[C/(ω-Δ)+C*/(-ω-Δ)].
# This reproduces the already-audited static M at ω=0.
OMEGA_OVER_OMEGA0 = (0.0, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 0.1, 0.3, 1.0)


def pack_matrix(m):
    m = np.asarray(m)
    herm = 0.5 * (m + m.conj().T)
    return {
        'real': np.real(m).tolist(),
        'imag': np.imag(m).tolist(),
        'maxabs': float(np.max(np.abs(m))),
        'hermitian_eigenvalues': np.linalg.eigvalsh(herm).tolist(),
        'hermitian_maxabs': float(np.max(np.abs(herm))),
        'anti_hermitian_maxabs': float(np.max(np.abs(m - herm))),
    }


def finite_difference_curvature(geo, kc, eps=1e-4):
    def energy(shift):
        return float(np.mean(np.sum(np.linalg.eigvalsh(h8(geo, R0, kc + shift, 1))[:, :4], axis=1)))
    z = np.zeros(2)
    hess = np.zeros((2, 2))
    e0 = energy(z)
    for a in range(2):
        da = np.eye(2)[a] * eps
        hess[a, a] = (energy(da) - 2 * e0 + energy(-da)) / eps**2
    dx, dy = np.eye(2) * eps
    hess[0, 1] = hess[1, 0] = (
        energy(dx + dy) - energy(dx - dy) - energy(-dx + dy) + energy(-dx - dy)
    ) / (4 * eps**2)
    return hess


def assemble_grid(geo, kf, *, do_fd=False, omegas=None):
    t0 = time.perf_counter()
    kc = geo.k_cart(kf)
    vv, vc, en = occup_empty_bases(h8(geo, R0, kc, 1), MU)
    x = x_from_bases(current_ops(geo, R0, kc, 1), vv, vc)
    gap = en[:, 4:, None] - en[:, None, :4]
    nk = len(kc)
    d = np.mean(mean_d_from_bases(d_ops(geo, R0, kc, 1), vv), axis=0)
    chi0 = 2 * np.einsum('kacv,kbcv,kcv->ab', x.conj(), x, 1 / gap).real / nk
    m0 = d - chi0
    ep, et = parallel_perp_basis(geo)
    static = {
        'n_k': int(nk),
        'min_gap': float(np.min(gap)),
        'D': d.tolist(),
        'static_paramagnetic': chi0.tolist(),
        'occupied_energy_curvature': m0.tolist(),
        'curvature_eigenvalues': np.linalg.eigvalsh(m0).tolist(),
        'parallel_curvature': float(ep @ m0 @ ep),
        'perp_curvature': float(et @ m0 @ et),
        'maxabs': float(np.max(np.abs(m0))),
    }
    if do_fd:
        hess = finite_difference_curvature(geo, kc)
        static['finite_difference_curvature'] = hess.tolist()
        static['finite_difference_maxabs_error'] = float(np.max(np.abs(m0 - hess)))
    kernel_rows = []
    if omegas is not None:
        omegas = np.asarray(omegas, dtype=float)
        den1 = omegas[:, None, None, None] - gap[None]
        den2 = -omegas[:, None, None, None] - gap[None]
        pi = (
            np.einsum('kacv,kbcv,wkcv->wab', x.conj(), x, 1 / den1)
            + np.einsum('kacv,kbcv,wkcv->wab', x, x.conj(), 1 / den2)
        ) / nk
        k_w = d[None] + pi
        for i, w in enumerate(omegas):
            kmat = k_w[i]
            delta = kmat - m0
            rec = {
                'omega': float(w),
                'omega_over_omega0': float(w / OMEGA0_OVER_TH),
                'K': pack_matrix(kmat),
                'K_minus_static': pack_matrix(delta),
                'parallel': complex(ep @ kmat @ ep),
                'perp': complex(et @ kmat @ et),
            }
            rec['parallel'] = {'real': rec['parallel'].real, 'imag': rec['parallel'].imag}
            rec['perp'] = {'real': rec['perp'].real, 'imag': rec['perp'].imag}
            if abs(w) > 0:
                rec['K_minus_static_over_omega2_maxabs'] = float(np.max(np.abs(delta)) / w**2)
                rec['K_minus_static_hermitian_eigs_over_omega2'] = (
                    np.linalg.eigvalsh(0.5 * (delta + delta.conj().T)) / w**2
                ).tolist()
            kernel_rows.append(rec)
    return static, kernel_rows, time.perf_counter() - t0


def power_fit(omegas, values):
    omegas = np.asarray(omegas, dtype=float)
    values = np.abs(np.asarray(values, dtype=float))
    mask = (omegas > 0) & (values > 0)
    if np.count_nonzero(mask) < 2:
        return None
    x = np.log(omegas[mask])
    y = np.log(values[mask])
    slope, intercept = np.polyfit(x, y, 1)
    return {'n_points': int(np.count_nonzero(mask)), 'exponent': float(slope),
            'prefactor': float(np.exp(intercept))}


def main():
    geo = build_geometry()
    omegas = np.array(OMEGA_OVER_OMEGA0, dtype=float) * OMEGA0_OVER_TH
    jobs = [
        ('Gamma_96x96', 96, 0.0, False),
        ('Gamma_128x128', 128, 0.0, False),
        ('Gamma_192x192', 192, 0.0, False),
        ('halfshift_192x192', 192, 0.5, True),
    ]
    grids = []
    for name, nk, shift, do_fd in jobs:
        kf, _kc, _w = k_grid(geo, nk, shift=shift)
        static, kernel_rows, wall = assemble_grid(geo, kf, do_fd=do_fd, omegas=omegas)
        dyn = [row for row in kernel_rows if row['omega'] != 0]
        fit_max = power_fit([r['omega'] for r in dyn], [r['K_minus_static']['maxabs'] for r in dyn])
        # Fit only the IR window ω ≤ 0.03 ω0, where a single power is meaningful.
        ir = [r for r in dyn if r['omega_over_omega0'] <= 0.03]
        fit_ir = power_fit([r['omega'] for r in ir], [r['K_minus_static']['maxabs'] for r in ir])
        rec = {
            'grid': name,
            'shift': shift,
            'wall_seconds': wall,
            'static': static,
            'kernel': kernel_rows,
            'power_fit_K_minus_static_all_positive': fit_max,
            'power_fit_K_minus_static_omega_le_0p03_omega0': fit_ir,
        }
        grids.append(rec)
        print(name, 'static_eigs', static['curvature_eigenvalues'],
              'min_gap', static['min_gap'], 'fit_ir', fit_ir, flush=True)
    half = next(g for g in grids if g['grid'] == 'halfshift_192x192')
    payload = {
        'status': 'p1_zero_bz_lowfreq_kernel_no_model_changes',
        'qualification': 'electronic K^A(ω)=D+Π^R(ω) on frozen insulating M0; no photons',
        'fourier_convention': 'physics_retarded_e^{-i omega t}; iη omitted because all ω < min gap',
        'omega0': OMEGA0_OVER_TH,
        'omega_over_omega0': list(OMEGA_OVER_OMEGA0),
        'grids': grids,
        'halfshift_192_note': (
            'Independent quadrature check of the static residual, not a continuum error bar.'
        ),
        'blocked': [
            'joint electron-photon propagation',
            'coupled-mode poles',
            'bare vs dressed vacuum',
            'covariant Pauli redefinition',
            'n,m,V,S_min',
        ],
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / 'lowfreq_kernel.json').write_text(json.dumps(payload, indent=2), encoding='utf-8')
    (OUT / 'static_halfshift_192.json').write_text(json.dumps({
        'status': 'static_halfshift_diagnostic_no_model_changes',
        'grid': half['grid'],
        'shift': 0.5,
        'static': half['static'],
    }, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
