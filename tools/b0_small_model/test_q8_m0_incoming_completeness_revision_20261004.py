#!/usr/bin/env python3
"""Revision of incoming-channel completeness checks.

Keeps the 20261004 JSON untouched. No P1 production, no full-BZ absorption.
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
    axis_near_zero_mask,
    dark_projector,
    dark_subspace,
    dress_two_band,
    duplicate_column_density,
    electron_column_density,
    embed_block_unitary,
    maxabs,
    photon_column_density,
    propagate_bundle,
    two_band_incoming_fixture,
)
from q8_m0_model import R0, build_geometry, current_ops, h8
from q8_m0_response import d_ops, mean_d_from_bases

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results/q_line/incoming_completeness_revision_20261004"
LIMIT = 2e-9
MU = -0.009010620372198276
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


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def dense_midpoint(packed, u, D, dt, gamma, fa=None):
    packed = np.asarray(packed, complex)
    u = np.asarray(u, complex)
    n, ncol = u.shape[1], packed.shape[1]
    fa = np.zeros((2, ncol), complex) if fa is None else np.asarray(fa, complex)
    k = np.zeros((2 * n + 2, 2 * n + 2), complex)
    k[:n, 2 * n:] = 1j * u.T
    k[n:2 * n, 2 * n:] = -1j * u.conj().T
    k[2 * n:, :n] = gamma * u.conj()
    k[2 * n:, n:2 * n] = gamma * u
    k[2 * n:, 2 * n:] = -gamma * D
    rhs = packed.copy()
    rhs[2 * n:] += fa / 2
    mid = np.linalg.solve(np.eye(2 * n + 2) - 0.5 * dt * k, rhs)
    return 2 * mid - packed


def dark_subspace_checks():
    gaps = np.array([1.1, 1.1, 1.9])
    u = np.array([[0.20, 0.20, 0.15], [0.10, 0.10, -0.08]], complex)
    axis = int(np.count_nonzero(axis_near_zero_mask(u)))
    basis, info, near = dark_subspace(gaps, u)
    rng = np.random.default_rng(4)
    q, _ = np.linalg.qr(rng.normal(size=(2, 2)) + 1j * rng.normal(size=(2, 2)))
    u_rot = u.copy()
    u_rot[:, :2] = u[:, :2] @ q
    basis_rot, _, _ = dark_subspace(gaps, u_rot)
    u_full = embed_block_unitary(q, 3, [0, 1])
    p_orig = dark_projector(basis)
    p_rot = dark_projector(basis_rot)
    p_mapped = u_full.conj().T @ p_orig @ u_full
    p_physical = u_full @ p_rot @ u_full.conj().T
    split = np.array([1.1, 1.8, 1.9])
    u_split = u.copy()
    basis_split, info_split, _ = dark_subspace(split, u_split)
    near_gaps = np.array([1.1, 1.1 + 1e-8, 1.9])
    basis_near, _, near_pairs = dark_subspace(near_gaps, u)
    return {
        "axis_near_zero_count": axis,
        "subspace_nullity": int(basis.shape[1]),
        "axis_count_misses_combination": bool(axis == 0 and basis.shape[1] == 1),
        "rotation_projector_residual": maxabs(p_rot - p_mapped),
        "physical_projector_residual": maxabs(p_physical - p_orig),
        "rotation_dimension_unchanged": bool(basis.shape[1] == basis_rot.shape[1] == 1),
        "cross_gap_nullity": int(basis_split.shape[1]),
        "cross_gap_not_merged": bool(all(item["size"] == 1 for item in info_split)),
        "near_degenerate_not_promoted": int(basis_near.shape[1]),
        "near_degeneracy_kept_as_diagnostic": bool(basis_near.shape[1] == 0 and len(near_pairs) == 1),
        "near_degenerate_pairs": len(near_pairs),
        "dark_coupling_residual": maxabs(u @ basis),
        "rotated_dark_coupling_residual": maxabs(u_rot @ basis_rot),
    }


def retained_local_identities():
    bundle = assemble_incoming_bundle(GAPS, U0, D0, GAMMA, FREQ)
    dens = []
    extra = []
    for w in FREQ:
        k = real_frequency_kernel(GAPS, U0, D0, float(w))
        jaa, jii, jai, _ = aa_ii_jumps_from_kernel(k, float(w), GAMMA)
        paa, pii, a_cols, i_cols = photon_column_density(bundle.photon, float(w))
        dens.append(max(maxabs(paa - jaa), maxabs(pii - jii),
                        maxabs(a_cols @ i_cols.conj().T - jai)))
        err, _ = duplicate_column_density(jaa, a_cols)
        extra.append(err)
    dark_b = bundle.photon.initial.b[2]
    return bundle, {
        "photon_columns_match_AA_II_jump": max(dens),
        "repeating_photon_columns_spectral_error": min(extra),
        "photon_columns_have_zero_dark_b": maxabs(dark_b),
        "dark_subspace_dim": int(bundle.dark_basis.shape[1]),
    }


def electronic_projection_checks():
    """Original-label bright/dark projections; A/I alone is not enough."""
    gaps = np.array([1.1, 1.1, 1.9])
    u0 = np.array([[0.20, 0.20, 0.15], [0.10, 0.10, -0.08]], complex)
    d0 = ((u0.conj() / gaps) @ u0.T + (u0 / gaps) @ u0.conj().T).real + 0.03 * np.eye(2)
    bundle = assemble_incoming_bundle(gaps, u0, d0, GAMMA, FREQ)
    p_dark = dark_projector(bundle.dark_basis)
    p_point = dark_projector(bundle.point.b)
    photon_dark = []
    bright_in_cluster = []
    for w in FREQ:
        rho_b, b_cols = electron_column_density(bundle.photon, float(w))
        photon_dark.append(maxabs(p_dark @ rho_b))
        cluster = rho_b[:2, :2]
        p_bright = np.eye(2) - p_dark[:2, :2]
        bright_in_cluster.append(maxabs(p_bright @ cluster @ p_bright - cluster))
    return {
        "photon_electronic_density_has_zero_dark_projection": max(photon_dark),
        "dark_point_reconstructs_subspace_projector": maxabs(p_point - p_dark),
        "photon_density_lives_in_bright_cluster": max(bright_in_cluster),
        "omitting_dark_combination_detected": float(np.trace(p_dark).real),
        "axis_mask_still_misses_this_combination": bool(
            np.count_nonzero(axis_near_zero_mask(u0)) == 0 and bundle.dark_basis.shape[1] == 1
        ),
    }


def band_checks():
    disjoint = two_band_incoming_fixture()
    overlap = two_band_incoming_fixture(photon_band=(0.5, 2.0), electron_band=(1.2, 2.5))
    w_ph, w_el, w_ov = 0.7, 2.3, 1.5
    q_el, f_el = dress_two_band(disjoint, w_el, photon="support", electron="support")
    jump_el = (q_el - q_el.conj().T) / (2j * np.pi)
    q_omit, f_omit = dress_two_band(disjoint, w_el, photon=False, electron=False)
    jump_omit = f_omit @ f_omit.conj().T if f_omit.size else np.zeros_like(jump_el)
    q_wrong, f_wrong = dress_two_band(disjoint, w_el, photon="force", electron=False)
    q_ov, f_ov = dress_two_band(overlap, w_ov)
    jump_ov = (q_ov - q_ov.conj().T) / (2j * np.pi)
    q_no_ph, f_no_ph = dress_two_band(overlap, w_ov, photon=False, electron="support")
    q_no_el, f_no_el = dress_two_band(overlap, w_ov, photon="support", electron=False)
    dup_err, _ = duplicate_column_density(jump_ov, f_ov)
    q_ph, f_ph = dress_two_band(disjoint, w_ph)
    jump_ph = (q_ph - q_ph.conj().T) / (2j * np.pi)
    rng = np.random.default_rng(5)
    qmix, _ = np.linalg.qr(rng.normal(size=(2, 2)) + 1j * rng.normal(size=(2, 2)))
    f_rot = f_ov @ qmix
    return {
        "photon_only_band_span": maxabs(jump_ph - f_ph @ f_ph.conj().T),
        "electron_only_band_span": maxabs(jump_el - f_el @ f_el.conj().T),
        "omit_electron_source_error": maxabs(jump_el - jump_omit),
        "omit_has_zero_columns": int(f_omit.shape[1]),
        "omit_is_empty": bool(f_omit.shape[1] == 0),
        "wrong_photon_source_error": maxabs(jump_el - f_wrong @ f_wrong.conj().T),
        "omit_and_wrong_are_different": abs(
            maxabs(jump_el - jump_omit) - maxabs(jump_el - f_wrong @ f_wrong.conj().T)
        ),
        "overlap_full_span": maxabs(jump_ov - f_ov @ f_ov.conj().T),
        "overlap_omit_photon": maxabs(jump_ov - f_no_ph @ f_no_ph.conj().T),
        "overlap_omit_electron": maxabs(jump_ov - f_no_el @ f_no_el.conj().T),
        "overlap_duplicate_spectral_error": dup_err,
        "overlap_source_basis_invariance": maxabs(jump_ov - f_rot @ f_rot.conj().T),
        "overlap_interval": list(overlap["overlap"]),
    }


def short_propagation_checks():
    gaps = np.array([1.1, 1.1, 1.9])
    u0 = np.array([[0.20, 0.20, 0.15], [0.10, 0.10, -0.08]], complex)
    d0 = ((u0.conj() / gaps) @ u0.T + (u0 / gaps) @ u0.conj().T).real + 0.03 * np.eye(2)
    bundle = assemble_incoming_bundle(gaps, u0, d0, GAMMA, FREQ)
    times = np.linspace(0.0, 1.8, 37)

    def u_static(t):
        return u0 * np.exp(1j * gaps * t)[None, :]

    def d_static(_t):
        return d0

    packed = propagate_bundle(bundle, times, u_static, d_static)
    n_el, n_ph = bundle.n_el, bundle.n_photon_columns
    abs_err = []
    rel_err = []
    dark_b_err = []
    dark_a_err = []
    for i, t in enumerate(times):
        ref = bundle.photon.state(t).packed()
        got = packed[i, :, :n_ph]
        diff = maxabs(got - ref)
        abs_err.append(diff)
        rel_err.append(diff / max(maxabs(ref), 1e-30))
        dark = packed[i, :, n_ph:]
        dark_b_err.append(maxabs(dark[:n_el] - bundle.point.b))
        dark_a_err.append(maxabs(dark[2 * n_el:]))
    v = bundle.dark_basis[:, 0]
    pol = np.array([0.18, -0.11])

    def u_on(t):
        u = u_static(t)
        return u + np.outer(pol, np.exp(-1j * gaps * t) * v.conj())

    packed_on = propagate_bundle(bundle, times, u_on, d_static)
    independent = [bundle.point.packed()]
    y = independent[0].copy()
    for t0, t1 in zip(times[:-1], times[1:]):
        dt = float(t1 - t0)
        tmid = 0.5 * (t0 + t1)
        y = dense_midpoint(y, u_on(tmid), d0, dt, GAMMA)
        independent.append(y.copy())
    independent = np.stack(independent)
    driven_abs = maxabs(packed_on[:, :, n_ph:] - independent)
    return {
        "undriven_photon_max_absolute": max(abs_err),
        "undriven_photon_max_relative": max(rel_err),
        "relative_denominator": "maxabs of analytic photon reference at each sample",
        "undriven_dark_combination_b": max(dark_b_err),
        "undriven_dark_combination_A": max(dark_a_err),
        "driven_dark_vs_independent_midpoint": driven_abs,
        "not_a_new_convergence_proof": True,
    }


def m0_inventory():
    geo = build_geometry()
    kc = geo.k_cart(np.array([[0.0, 0.0], [0.23, 0.12]]))
    weights = np.array([0.5, 0.5])
    rows = []
    for neel in (1, -1):
        vv, vc, en = occup_empty_bases(h8(geo, R0, kc, neel), MU)
        x = x_from_bases(current_ops(geo, R0, kc, neel), vv, vc)
        gap = (en[:, 4:, None] - en[:, None, :4]).reshape(-1)
        u = np.einsum("k,kacv->akcv", np.sqrt(weights), x).reshape(2, -1)
        axis = int(np.count_nonzero(axis_near_zero_mask(u, relative=1e-10)))
        basis, info, near = dark_subspace(gap, u, gap_atol=1e-12)
        rows.append({
            "neel": int(neel),
            "n_el": int(u.shape[1]),
            "axis_near_zero_count": axis,
            "gap_subspace_nullity": int(basis.shape[1]),
            "n_exact_gap_clusters": len(info),
            "near_degenerate_pairs": len(near),
            "not_a_BZ_dark_dimension": True,
        })
    return {
        "gamma": float(gamma_self(geo)),
        "domains": rows,
        "legal_electronic_continuum_boundary": False,
    }


def flags_from(checks):
    out = {}
    large = {
        "axis_count_misses_combination",
        "repeating_photon_columns_spectral_error",
        "omit_electron_source_error",
        "wrong_photon_source_error",
        "omit_and_wrong_are_different",
        "overlap_omit_photon",
        "overlap_omit_electron",
        "overlap_duplicate_spectral_error",
        "near_degenerate_pairs",
        "cross_gap_not_merged",
        "omitting_dark_combination_detected",
        "axis_mask_still_misses_this_combination",
    }
    skip = {"relative_denominator", "overlap_interval", "not_a_new_convergence_proof",
            "axis_near_zero_count", "near_degenerate_not_promoted", "cross_gap_nullity",
            "omit_has_zero_columns", "subspace_nullity", "dark_subspace_dim"}
    for key, val in checks.items():
        if key in skip:
            continue
        if isinstance(val, bool):
            out[key] = val
            continue
        if key in large:
            out[key] = bool(val > 1e-6)
            continue
        limit = 1e-4 if "absolute" in key or "relative" in key or "independent" in key else LIMIT
        if key == "undriven_photon_max_relative":
            limit = 1e-3
        out[key] = bool(val < limit)
    return out


def main() -> int:
    t0 = time.perf_counter()
    dark = dark_subspace_checks()
    bundle, local = retained_local_identities()
    electronic = electronic_projection_checks()
    band = band_checks()
    prop = short_propagation_checks()
    inventory = m0_inventory()
    flags = {
        **flags_from(dark),
        **flags_from(local),
        **flags_from(electronic),
        **flags_from(band),
        **flags_from(prop),
    }
    payload = {
        "status": "incoming_completeness_revision_not_M0_ground",
        "all_checks_pass": bool(all(flags.values())),
        "pass_flags": flags,
        "dark_subspace": dark,
        "retained_local_identities": local,
        "electronic_projections": electronic,
        "bands": band,
        "short_propagation": prop,
        "M0_two_k_inventory": inventory,
        "old_json_not_overwritten": "results/q_line/incoming_spectral_completeness_20261004/completeness_checks.json",
        "P1_not_run": True,
        "elapsed_s": time.perf_counter() - t0,
        "source_sha256": {
            "q8_m0_incoming_spectral_completeness_20261004.py": sha(
                Path(__file__).with_name("q8_m0_incoming_spectral_completeness_20261004.py")
            ),
            "test_q8_m0_incoming_completeness_revision_20261004.py": sha(Path(__file__)),
        },
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "revision_checks.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({
        "all_checks_pass": payload["all_checks_pass"],
        "failed": [k for k, v in flags.items() if not v],
        "axis_vs_subspace": [dark["axis_near_zero_count"], dark["subspace_nullity"]],
        "omit": band["omit_electron_source_error"],
        "wrong": band["wrong_photon_source_error"],
        "duplicate": band["overlap_duplicate_spectral_error"],
        "photon_abs": prop["undriven_photon_max_absolute"],
        "photon_rel": prop["undriven_photon_max_relative"],
        "driven_vs_independent": prop["driven_dark_vs_independent_midpoint"],
        "M0_axis": [r["axis_near_zero_count"] for r in inventory["domains"]],
        "M0_subspace": [r["gap_subspace_nullity"] for r in inventory["domains"]],
        "elapsed_s": payload["elapsed_s"],
    }, indent=2))
    return 0 if payload["all_checks_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
