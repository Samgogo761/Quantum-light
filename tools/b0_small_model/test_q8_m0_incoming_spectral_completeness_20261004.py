#!/usr/bin/env python3
"""Independent completeness checks for joint-ground incoming channels.

No P1 production, no BZ absorption claim, no photon n/m. The electronic
continuum band is a mathematical second interval, not an M0 bath.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from q8_m0_cjj import occup_empty_bases, x_from_bases
from q8_m0_continuum_columns_20261004 import real_frequency_kernel
from q8_m0_gsheet import gamma_self
from q8_m0_incoming_spectral_completeness_20261004 import (
    aa_ii_jumps_from_kernel,
    assemble_incoming_bundle,
    bright_bare_electron_current_jump,
    dark_mask,
    dress_two_band,
    maxabs,
    photon_column_density,
    propagate_bundle,
    two_band_incoming_fixture,
)
from q8_m0_model import R0, build_geometry, current_ops, h8
from q8_m0_response import d_ops, mean_d_from_bases

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results/q_line/incoming_spectral_completeness_20261004"
GAPS = np.array([0.85, 1.55, 2.35])
U0 = np.array(
    [[0.21 + 0.07j, -0.14 + 0.09j, 0.0 + 0.0j],
     [0.06 - 0.16j, 0.24 + 0.04j, 0.0 + 0.0j]],
    complex,
)
D0 = ((U0.conj() / GAPS) @ U0.T + (U0 / GAPS) @ U0.conj().T).real + np.array(
    [[0.05, 0.02], [0.02, 0.04]]
)
GAMMA = 0.22
FREQ = np.array([0.37, 1.05])
MU = -0.009010620372198276
LIMIT = 2e-9


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def finite_electron_photon_checks():
    bundle = assemble_incoming_bundle(GAPS, U0, D0, GAMMA, FREQ)
    dark = dark_mask(U0)
    checks = {
        "dark_index_is_last": bool(np.array_equal(bundle.dark_indices, np.array([2]))),
        "bright_not_treated_as_point": bool(bundle.n_point_columns == 1),
        "photon_columns_are_eight": bool(bundle.n_photon_columns == 8),
    }
    dens_err = []
    overcount = []
    for w in FREQ:
        k = real_frequency_kernel(GAPS, U0, D0, float(w))
        jaa, jii, jai, jia = aa_ii_jumps_from_kernel(k, float(w), GAMMA)
        paa, pii, a_cols, i_cols = photon_column_density(bundle.photon, float(w))
        dens_err.append(max(maxabs(paa - jaa), maxabs(pii - jii),
                            maxabs(a_cols @ i_cols.conj().T - jai)))
        extra = bright_bare_electron_current_jump(GAPS, U0, float(w))
        overcount.append(maxabs(extra))
    checks["photon_columns_match_AA_II_jump"] = max(dens_err)
    checks["bare_bright_electron_current_is_nonzero"] = min(overcount)
    # Photon-only reconstruction of the dark q coordinate must vanish.
    dark_b = bundle.photon.initial.b[2]
    checks["photon_columns_have_zero_dark_b"] = maxabs(dark_b)
    checks["point_column_is_unit_dark_annihilator"] = maxabs(
        bundle.point.b[2] - 1.0
    ) + maxabs(bundle.point.b[:2]) + maxabs(bundle.point.A)
    # Omitting the point column leaves the dark oscillator without an incoming annihilator.
    checks["omitting_point_channel_detected"] = float(1.0 - maxabs(dark_b))
    return bundle, checks, dens_err, overcount


def two_band_checks():
    spec = two_band_incoming_fixture()
    w_ph, w_el = 0.7, 2.3
    q_ph, f_ph = dress_two_band(spec, w_ph, photon=True, electron=False)
    jump_ph = (q_ph - q_ph.conj().T) / (2j * np.pi)
    q_el, f_el = dress_two_band(spec, w_el, photon=False, electron=True)
    jump_el = (q_el - q_el.conj().T) / (2j * np.pi)
    q_wrong, f_wrong = dress_two_band(spec, w_el, photon=True, electron=False)
    jump_wrong_cols = f_wrong @ f_wrong.conj().T
    return {
        "photon_band_columns_span_jump": maxabs(jump_ph - f_ph @ f_ph.conj().T),
        "electron_band_columns_span_jump": maxabs(jump_el - f_el @ f_el.conj().T),
        "photon_only_in_electron_band_column_rank": float(f_wrong.shape[1]),
        "omitting_electron_incoming_detected": maxabs(jump_el - jump_wrong_cols),
    }


def bound_omission_checks():
    """Cheap reuse of the 20261003 bound-mode algebra; no infinite quadrature."""
    from q8_m0_relative_output_20261003 import dress_continuous_channels
    a, b, lam = 0.5, 1.5, 0.8
    width = b - a
    p = 0.5 * np.array([[1, 1j], [-1j, 1]])
    j = np.array([[0.0, 1.0], [-1.0, 0.0]])
    interaction = np.diag([lam, 0.0])
    source = np.array([[1], [-1j]]) / np.sqrt(2 * width)

    def free_q(omega, inside=False):
        plus = (np.log(abs((b - omega) / (a - omega))) + (1j * np.pi if inside else 0.0)) / width
        minus = np.log((b + omega) / (a + omega)) / width
        return p * plus + p.conj() * minus

    omega = 0.9
    qf = free_q(omega, True)
    q, f = dress_continuous_channels(qf, interaction, source)
    jump = (q - q.conj().T) / (2j * np.pi)
    r = np.exp(-2 * width / lam)
    bound = np.sqrt((b * b - r * a * a) / (1 - r))
    qb = free_q(bound)
    derivative = bound / width * (1 / (a * a - bound * bound) - 1 / (b * b - bound * bound))
    residue = np.outer(qb[:, 0], qb[:, 0].conj()) / derivative
    return {
        "in_band_columns_span_jump": maxabs(jump - f @ f.conj().T),
        "bound_equation": abs(1 + lam * qb[0, 0]),
        "bound_residue_norm": maxabs(residue),
        "bound_frequency": float(bound),
    }


def short_propagation_checks(bundle):
    times = np.linspace(0.0, 1.8, 37)

    def u_static(_t):
        return U0 * np.exp(1j * GAPS * _t)[None, :]

    def d_static(_t):
        return D0

    packed = propagate_bundle(bundle, times, u_static, d_static)
    n_el = bundle.n_el
    n_ph = bundle.n_photon_columns
    photon_err = []
    dark_b_err = []
    dark_A_err = []
    for i, t in enumerate(times):
        ref = bundle.photon.state(t).packed()
        got = packed[i, :, :n_ph]
        photon_err.append(maxabs(got - ref))
        dark = packed[i, :, n_ph:]
        dark_b_err.append(maxabs(dark[:n_el] - bundle.point.b))
        dark_A_err.append(maxabs(dark[2 * n_el:]))
    # Drive that turns the last (dark) vertex on. The point column must source A.
    def u_activated(t):
        u = U0 * np.exp(1j * GAPS * t)[None, :]
        u = u.copy()
        u[:, 2] = np.array([0.18, -0.11]) * np.exp(1j * GAPS[2] * t)
        return u

    packed_on = propagate_bundle(bundle, times, u_activated, d_static)
    sourced = maxabs(packed_on[-1, 2 * n_el:, n_ph:])
    return {
        "undriven_photon_vs_analytic": max(photon_err),
        "undriven_dark_b_constant": max(dark_b_err),
        "undriven_dark_A_zero": max(dark_A_err),
        "activating_dark_vertex_sources_A": sourced,
        "dark_must_not_be_dropped": bool(sourced > 1e-4),
    }


def m0_inventory():
    geo = build_geometry()
    gamma = gamma_self(geo)
    kc = geo.k_cart(np.array([[0.0, 0.0], [0.23, 0.12]]))
    weights = np.array([0.5, 0.5])
    missing = [
        "full-BZ retarded electronic continuum boundary values",
        "complete positive-frequency photon packet integral",
        "post-pulse radiation tail of H2",
        "production n/m/V",
    ]
    rows = []
    for neel in (1, -1):
        vv, vc, en = occup_empty_bases(h8(geo, R0, kc, neel), MU)
        x = x_from_bases(current_ops(geo, R0, kc, neel), vv, vc)
        gap = (en[:, 4:, None] - en[:, None, :4]).reshape(-1)
        u = np.einsum("k,kacv->akcv", np.sqrt(weights), x).reshape(2, -1)
        d = np.einsum("k,kab->ab", weights, mean_d_from_bases(d_ops(geo, R0, kc, neel), vv))
        mask = dark_mask(u, relative=1e-10)
        kernels = {}
        hermitian = {}
        for omega in (0.025, 0.10):
            k = real_frequency_kernel(gap, u, d, omega)
            kernels[str(omega)] = {
                "anti_hermitian": maxabs(k - k.conj().T),
                "min_gap_minus_omega": float(np.min(np.abs(gap - omega))),
            }
            hermitian[str(omega)] = bool(maxabs(k - k.conj().T) < 1e-12)
        rows.append({
            "neel": int(neel),
            "n_el": int(u.shape[1]),
            "n_exactly_dark": int(np.count_nonzero(mask)),
            "min_vertex_strength": float(np.min(np.sqrt(np.sum(np.abs(u) ** 2, axis=0)))),
            "note": (
                "n_exactly_dark uses a 1e-10 relative threshold on this 2-k sample; "
                "it is not a BZ dark-space theorem."
            ),
            "kernels_hermitian_off_pole": hermitian,
            "kernel_diagnostics": kernels,
            "legal_electronic_continuum_boundary": False,
        })
    return {
        "gamma": float(gamma),
        "domains": rows,
        "connected_now": [
            "selected photon incoming columns at finite off-pole frequencies",
            "original (k,c,v) labels inside those columns",
            "point-spectrum interface for exactly dark vertices",
        ],
        "missing": missing,
        "why_not_absorption_spectrum": (
            "Finite-k K is Hermitian off poles. Densifying that grid is not a "
            "retarded full-BZ electronic spectral boundary."
        ),
    }


def flags_from(checks, extra_limits=None):
    extra_limits = extra_limits or {}
    out = {}
    for key, val in checks.items():
        if isinstance(val, bool):
            out[key] = val
            continue
        if key in ("bound_frequency",):
            continue
        limit = extra_limits.get(key, LIMIT)
        if key in (
            "bare_bright_electron_current_is_nonzero",
            "omitting_point_channel_detected",
            "omitting_electron_incoming_detected",
            "activating_dark_vertex_sources_A",
            "photon_only_in_electron_band_column_rank",
            "bound_residue_norm",
        ):
            out[key] = bool(val > (0.5 if "rank" in key else 1e-6))
        else:
            out[key] = bool(val < limit)
    return out


def main() -> int:
    t0 = time.perf_counter()
    bundle, finite_checks, dens_err, overcount = finite_electron_photon_checks()
    band = two_band_checks()
    prop = short_propagation_checks(bundle)
    bound = bound_omission_checks()
    inventory = m0_inventory()
    finite_flags = flags_from(finite_checks, {
        "photon_columns_have_zero_dark_b": LIMIT,
        "point_column_is_unit_dark_annihilator": LIMIT,
    })
    band_flags = flags_from(band, {
        "omitting_electron_incoming_detected": 1e-6,
    })
    prop_flags = flags_from(prop, {
        "undriven_photon_vs_analytic": 1e-4,
        "undriven_dark_b_constant": LIMIT,
        "undriven_dark_A_zero": LIMIT,
    })
    bound_flags = flags_from(bound)
    all_flags = {**finite_flags, **band_flags, **prop_flags, **bound_flags}
    payload = {
        "status": "incoming_spectral_completeness_fixture_only",
        "all_checks_pass": bool(all(all_flags.values())),
        "pass_flags": all_flags,
        "finite_electron_plus_photon": finite_checks,
        "two_continuous_bands": band,
        "short_propagation_on_verified_stepper": prop,
        "reused_bound_mode_fixture": bound,
        "M0_interface": inventory,
        "P1_not_run": True,
        "continuum_BZ_claim": False,
        "elapsed_s": time.perf_counter() - t0,
        "source_sha256": {
            "q8_m0_incoming_spectral_completeness_20261004.py": sha(
                Path(__file__).with_name("q8_m0_incoming_spectral_completeness_20261004.py")
            ),
            "test_q8_m0_incoming_spectral_completeness_20261004.py": sha(Path(__file__)),
            "q8_m0_continuum_columns_20261004.py": sha(
                Path(__file__).with_name("q8_m0_continuum_columns_20261004.py")
            ),
        },
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "completeness_checks.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({
        "all_checks_pass": payload["all_checks_pass"],
        "failed": [k for k, v in all_flags.items() if not v],
        "photon_jump": finite_checks["photon_columns_match_AA_II_jump"],
        "omit_electron_band": band["omitting_electron_incoming_detected"],
        "prop_photon": prop["undriven_photon_vs_analytic"],
        "dark_sources_A": prop["activating_dark_vertex_sources_A"],
        "M0_dark": [r["n_exactly_dark"] for r in inventory["domains"]],
        "elapsed_s": payload["elapsed_s"],
    }, indent=2))
    return 0 if payload["all_checks_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
