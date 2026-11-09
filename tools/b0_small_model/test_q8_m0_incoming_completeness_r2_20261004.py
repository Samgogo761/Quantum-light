#!/usr/bin/env python3
"""Second revision of incoming-channel dark-space and finite-M0 labels.

Keeps completeness_checks.json and revision_checks.json untouched.
No P1, no full-BZ absorption, no photon n/m/V.
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
from q8_m0_continuum_columns_20261004 import (
    ColumnState,
    current,
    real_frequency_kernel,
    rhs,
)
from q8_m0_gsheet import gamma_self
from q8_m0_incoming_spectral_completeness_20261004 import (
    annihilator_current,
    assemble_incoming_bundle,
    axis_near_zero_mask,
    classify_dark_columns,
    dark_column_gaps,
    dark_projector,
    dark_subspace,
    dress_two_band,
    duplicate_column_density,
    electron_column_density,
    electronic_spectral_measure_spec,
    embed_block_unitary,
    maxabs,
    photon_column_density,
    propagate_bundle,
    two_band_incoming_fixture,
)
from q8_m0_model import R0, build_geometry, current_ops, h8
from q8_m0_response import d_ops, mean_d_from_bases
from q8_m0_incoming_spectral_completeness_20261004 import aa_ii_jumps_from_kernel

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results/q_line/incoming_completeness_r2_20261004"
LIMIT = 2e-9
MU = -0.009010620372198276
GAMMA = 0.22
FREQ = np.array([0.37, 1.05])
GAPS = np.array([0.85, 1.55, 2.35])
U0 = np.array(
    [[0.21 + 0.07j, -0.14 + 0.09j, 0.0 + 0.0j],
     [0.06 - 0.16j, 0.24 + 0.04j, 0.0 + 0.0j]],
    complex,
)
D0 = ((U0.conj() / GAPS) @ U0.T + (U0 / GAPS) @ U0.conj().T).real + np.array(
    [[0.05, 0.02], [0.02, 0.04]]
)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def contact_from(u, gaps):
    u = np.asarray(u, complex)
    gaps = np.asarray(gaps, float)
    return ((u.conj() / gaps) @ u.T + (u / gaps) @ u.conj().T).real + 0.03 * np.eye(2)


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
    rhs_vec = packed.copy()
    rhs_vec[2 * n:] += fa / 2
    mid = np.linalg.solve(np.eye(2 * n + 2) - 0.5 * dt * k, rhs_vec)
    return 2 * mid - packed


def complex_kernel_checks():
    """The documented counterexample: ker(u) is not the physical dark space."""
    gaps = np.array([1.1, 1.1])
    u = np.array([[0.2, 0.2j], [0.1, 0.1j]], complex)
    d0 = contact_from(u, gaps)
    basis, info, _ = dark_subspace(gaps, u)
    wrong = u.conj().T @ u
    evals, vecs = np.linalg.eigh(0.5 * (wrong + wrong.conj().T))
    v_wrong = vecs[:, int(np.argmin(evals))]
    state_wrong = ColumnState.zeros(2, 1)
    state_wrong.b[:, 0] = v_wrong
    i_wrong = current(state_wrong, u, d0)
    bundle = assemble_incoming_bundle(gaps, u, d0, GAMMA, FREQ)
    i_right = current(bundle.point, u, d0)
    r_right = rhs(bundle.point, u, d0, GAMMA, np.zeros_like(bundle.point.A))
    rng = np.random.default_rng(4)
    q, _ = np.linalg.qr(rng.normal(size=(2, 2)) + 1j * rng.normal(size=(2, 2)))
    u_rot = u @ q.conj()
    basis_rot, _, _ = dark_subspace(gaps, u_rot)
    u_full = embed_block_unitary(q, 2, [0, 1])
    p_orig = dark_projector(basis)
    p_rot = dark_projector(basis_rot)
    p_mapped = u_full.conj().T @ p_orig @ u_full
    p_physical = u_full @ p_rot @ u_full.conj().T
    i_rot = annihilator_current(u_rot, basis_rot)
    split_gaps = np.array([1.1, 1.8])
    basis_split, info_split, _ = dark_subspace(split_gaps, u)
    v_inst = np.array([[1j], [1.0]], complex) / np.sqrt(2)
    inst = annihilator_current(u, v_inst)
    u_t = u * np.exp(1j * split_gaps * (np.pi / (split_gaps[1] - split_gaps[0])))[None, :]
    later = annihilator_current(u_t, v_inst)
    return {
        "complex_nullity": int(basis.shape[1]),
        "wrong_gram_current": maxabs(i_wrong),
        "physical_current_of_dark_b": maxabs(i_right),
        "physical_rhs_b": maxabs(r_right.b),
        "physical_rhs_bdag": maxabs(r_right.bdag),
        "physical_rhs_A": maxabs(r_right.A),
        "wrong_kernel_is_not_physical": bool(maxabs(i_wrong) > 0.1),
        "rotation_projector_residual": maxabs(p_rot - p_mapped),
        "physical_projector_residual": maxabs(p_physical - p_orig),
        "rotated_physical_current": maxabs(i_rot),
        "rotation_dimension_unchanged": bool(basis.shape[1] == basis_rot.shape[1] == 1),
        "split_gap_nullity": int(basis_split.shape[1]),
        "split_gap_not_merged": bool(all(item["size"] == 1 for item in info_split)),
        "instantaneous_cancel_at_t0": maxabs(inst),
        "same_combination_current_at_later_t": maxabs(later),
        "cross_gap_not_always_dark": bool(maxabs(later) > 1e-3 and maxabs(inst) < 1e-12),
        "metadata_n_point": int(bundle.n_point_columns),
        "metadata_n_gaps": int(len(bundle.point_gaps)),
        "metadata_n_labels": int(len(bundle.point_labels)),
        "metadata_lengths_agree": bool(
            bundle.n_point_columns == len(bundle.point_gaps) == len(bundle.point_labels) == 1
        ),
        "point_gap_value": float(bundle.point_gaps[0]) if bundle.point_gaps.size else None,
    }


def weak_coupling_checks():
    gaps = np.array([1.0, 2.0])
    u = np.array([[1.0, 0.0], [0.0, 1e-7]], complex)
    basis, info, _ = dark_subspace(gaps, u)
    assembled = True
    error = ""
    try:
        assemble_incoming_bundle(gaps, u, contact_from(u, gaps), GAMMA, FREQ)
    except ValueError as exc:
        assembled = False
        error = str(exc)
    return {
        "weak_nullity": int(basis.shape[1]),
        "weak_singular_value": float(info[1]["coupling_singular_values"][0]),
        "weak_kept_bright": bool(basis.shape[1] == 0),
        "weak_does_not_raise_on_assemble": bool(assembled),
        "assemble_error": error,
        "unresolved_count": int(sum(item["unresolved_count"] for item in info)),
    }


def retained_band_and_photon_checks():
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
    q_ph, f_ph = dress_two_band(disjoint, w_ph)
    jump_ph = (q_ph - q_ph.conj().T) / (2j * np.pi)
    return {
        "photon_columns_match_AA_II_jump": max(dens),
        "repeating_photon_columns_spectral_error": min(extra),
        "photon_only_band_span": maxabs(jump_ph - f_ph @ f_ph.conj().T),
        "electron_only_band_span": maxabs(jump_el - f_el @ f_el.conj().T),
        "omit_electron_source_error": maxabs(jump_el - jump_omit),
        "wrong_photon_source_error": maxabs(jump_el - f_wrong @ f_wrong.conj().T),
        "overlap_full_span": maxabs(jump_ov - f_ov @ f_ov.conj().T),
        "overlap_duplicate_spectral_error": duplicate_column_density(jump_ov, f_ov)[0],
    }


def complex_propagation_and_projection():
    gaps = np.array([1.1, 1.1, 1.9])
    u0 = np.array([[0.20, 0.20j, 0.15], [0.10, 0.10j, -0.08]], complex)
    d0 = contact_from(u0, gaps)
    bundle = assemble_incoming_bundle(gaps, u0, d0, GAMMA, FREQ)
    times = np.linspace(0.0, 1.8, 37)

    def u_static(t):
        return u0 * np.exp(1j * gaps * t)[None, :]

    def d_static(_t):
        return d0

    packed = propagate_bundle(bundle, times, u_static, d_static)
    n_el, n_ph = bundle.n_el, bundle.n_photon_columns
    abs_err = []
    dark_b_err = []
    dark_a_err = []
    for i, t in enumerate(times):
        ref = bundle.photon.state(t).packed()
        got = packed[i, :, :n_ph]
        abs_err.append(maxabs(got - ref))
        dark = packed[i, :, n_ph:]
        dark_b_err.append(maxabs(dark[:n_el] - bundle.point.b))
        dark_a_err.append(maxabs(dark[2 * n_el:]))
    p_dark = dark_projector(bundle.dark_basis)
    photon_dark = []
    for w in FREQ:
        rho_b, _ = electron_column_density(bundle.photon, float(w))
        photon_dark.append(maxabs(p_dark @ rho_b))
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
    return {
        "undriven_photon_max_absolute": max(abs_err),
        "undriven_dark_combination_b": max(dark_b_err),
        "undriven_dark_combination_A": max(dark_a_err),
        "photon_electronic_density_has_zero_dark_projection": max(photon_dark),
        "driven_dark_vs_independent_midpoint": maxabs(packed_on[:, :, n_ph:] - independent),
        "metadata_lengths_agree": bool(
            bundle.n_point_columns == len(bundle.point_gaps) == len(bundle.point_labels)
        ),
        "not_a_new_convergence_proof": True,
    }


def m0_finite_label_diagnostics():
    geo = build_geometry()
    kfrac = np.array([[0.0, 0.0], [0.23, 0.12]])
    kc = geo.k_cart(kfrac)
    weights = np.array([0.5, 0.5])
    n_c, n_v = 4, 4
    rows = []
    for neel in (1, -1):
        vv, vc, en = occup_empty_bases(h8(geo, R0, kc, neel), MU)
        x = x_from_bases(current_ops(geo, R0, kc, neel), vv, vc)
        gap = (en[:, 4:, None] - en[:, None, :4]).reshape(-1)
        u = np.einsum("k,kacv->akcv", np.sqrt(weights), x).reshape(2, -1)
        k_of_index = np.repeat(np.arange(len(kc)), n_c * n_v)
        c_of_index = np.tile(np.repeat(np.arange(n_c), n_v), len(kc))
        v_of_index = np.tile(np.arange(n_v), len(kc) * n_c)
        basis, info, near = dark_subspace(gap, u)
        bundle = assemble_incoming_bundle(gap, u, 0.03 * np.eye(2)
                                          + ((u.conj() / np.maximum(gap, 1e-18)) @ u.T
                                             + (u / np.maximum(gap, 1e-18)) @ u.conj().T).real,
                                          GAMMA, FREQ)
        i_dark = current(bundle.point, u, bundle.photon.D0) if bundle.n_point_columns else np.zeros((2, 0))
        classified = classify_dark_columns(info, k_of_index=k_of_index)
        types = {}
        for item in classified:
            types[item["spectral_type"]] = types.get(item["spectral_type"], 0) + int(item["nullity"])
        p_dark = dark_projector(basis)
        photon_dark = []
        for w in FREQ:
            rho_b, _ = electron_column_density(bundle.photon, float(w))
            photon_dark.append(maxabs(p_dark @ rho_b) if basis.size else 0.0)
        rows.append({
            "neel": int(neel),
            "n_el": int(u.shape[1]),
            "n_k": int(len(kc)),
            "axis_near_zero_count": int(np.count_nonzero(axis_near_zero_mask(u, relative=1e-10))),
            "gap_subspace_nullity": int(basis.shape[1]),
            "n_exact_gap_clusters": len(info),
            "near_degenerate_pairs": len(near),
            "physical_current_of_dark_b": maxabs(i_dark),
            "metadata_lengths_agree": bool(
                bundle.n_point_columns == len(bundle.point_gaps) == len(bundle.point_labels)
            ),
            "photon_dark_projection": max(photon_dark) if photon_dark else 0.0,
            "spectral_type_counts": types,
            "original_labels_kept": True,
            "kcv_shape": [int(len(kc)), n_c, n_v],
            "not_a_BZ_dark_dimension": True,
            "finite_sample_not_continuum": True,
        })
        _ = (c_of_index, v_of_index)
    return {
        "gamma": float(gamma_self(geo)),
        "domains": rows,
        "legal_electronic_continuum_boundary": False,
        "spectral_measure": electronic_spectral_measure_spec(),
        "k_sample": kfrac.tolist(),
    }


def flags_from(checks):
    out = {}
    large = {
        "wrong_kernel_is_not_physical",
        "cross_gap_not_always_dark",
        "repeating_photon_columns_spectral_error",
        "omit_electron_source_error",
        "wrong_photon_source_error",
        "overlap_duplicate_spectral_error",
        "same_combination_current_at_later_t",
        "wrong_gram_current",
    }
    skip = {
        "assemble_error", "point_gap_value", "not_a_new_convergence_proof",
        "complex_nullity", "split_gap_nullity", "weak_nullity",
        "unresolved_count", "metadata_n_point", "metadata_n_gaps",
        "metadata_n_labels", "weak_singular_value",
    }
    for key, val in checks.items():
        if key in skip or val is None:
            continue
        if isinstance(val, bool):
            out[key] = val
            continue
        if key in large:
            out[key] = bool(val > 1e-6)
            continue
        limit = 1e-4 if "absolute" in key or "independent" in key else LIMIT
        if key == "photon_dark_projection":
            limit = 1e-8
        out[key] = bool(val < limit)
    return out


def main() -> int:
    t0 = time.perf_counter()
    complex_k = complex_kernel_checks()
    weak = weak_coupling_checks()
    retained = retained_band_and_photon_checks()
    prop = complex_propagation_and_projection()
    inventory = m0_finite_label_diagnostics()
    flags = {
        **flags_from(complex_k),
        **flags_from(weak),
        **flags_from(retained),
        **flags_from(prop),
    }
    for row in inventory["domains"]:
        flags[f"N{row['neel']}_metadata"] = bool(row["metadata_lengths_agree"])
        flags[f"N{row['neel']}_dark_current"] = bool(row["physical_current_of_dark_b"] < 1e-12)
        flags[f"N{row['neel']}_photon_dark"] = bool(row["photon_dark_projection"] < 1e-8)
        flags[f"N{row['neel']}_not_bz"] = bool(row["not_a_BZ_dark_dimension"])
    payload = {
        "status": "incoming_completeness_r2_finite_labels_not_M0_ground",
        "all_checks_pass": bool(all(flags.values())),
        "pass_flags": flags,
        "complex_kernel": complex_k,
        "weak_coupling": weak,
        "retained_band_photon": retained,
        "complex_propagation": prop,
        "M0_finite_labels": inventory,
        "old_json_not_overwritten": [
            "results/q_line/incoming_spectral_completeness_20261004/completeness_checks.json",
            "results/q_line/incoming_completeness_revision_20261004/revision_checks.json",
        ],
        "P1_not_run": True,
        "full_BZ_not_run": True,
        "elapsed_s": time.perf_counter() - t0,
        "source_sha256": {
            "q8_m0_incoming_spectral_completeness_20261004.py": sha(
                Path(__file__).with_name("q8_m0_incoming_spectral_completeness_20261004.py")
            ),
            "test_q8_m0_incoming_completeness_r2_20261004.py": sha(Path(__file__)),
        },
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "r2_checks.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({
        "all_checks_pass": payload["all_checks_pass"],
        "failed": [k for k, v in flags.items() if not v],
        "wrong_gram_I": complex_k["wrong_gram_current"],
        "physical_I": complex_k["physical_current_of_dark_b"],
        "weak_nullity": weak["weak_nullity"],
        "metadata": [complex_k["metadata_n_point"], complex_k["metadata_n_gaps"]],
        "M0_nullity": [r["gap_subspace_nullity"] for r in inventory["domains"]],
        "M0_types": [r["spectral_type_counts"] for r in inventory["domains"]],
        "M0_I": [r["physical_current_of_dark_b"] for r in inventory["domains"]],
        "elapsed_s": payload["elapsed_s"],
    }, indent=2))
    return 0 if payload["all_checks_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())


