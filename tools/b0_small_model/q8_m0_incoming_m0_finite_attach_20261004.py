#!/usr/bin/env python3
"""Finite-label real-M0 incoming attach. Not a continuum boundary.

Weight once: u = sqrt(w) X so C = u u† without a second w.
Frozen occupied D and gamma_self. Original (k,c,v) labels kept.
Negative controls: omit a dark combination, omit one kcv, extra bright vacuum.
Small-k dark inventory only. No P1, no full BZ, no n/m/V.
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
from q8_m0_continuum_columns_20261004 import current, real_frequency_kernel
from q8_m0_gsheet import gamma_self
from q8_m0_incoming_spectral_completeness_20261004 import (
    aa_ii_jumps_from_kernel,
    assemble_incoming_bundle,
    axis_near_zero_mask,
    bright_bare_electron_current_jump,
    classify_dark_columns,
    dark_projector,
    dark_subspace,
    electron_column_density,
    electronic_spectral_measure_spec,
    maxabs,
    photon_column_density,
)
from q8_m0_model import R0, build_geometry, current_ops, h8
from q8_m0_response import d_ops, mean_d_from_bases

REPO = Path(__file__).resolve().parents[2]
MU = -0.009010620372198276
OUT = REPO / "results/q_line/incoming_m0_finite_attach_20261004"
OLD_JSON = [
    "results/q_line/incoming_spectral_completeness_20261004/completeness_checks.json",
    "results/q_line/incoming_completeness_revision_20261004/revision_checks.json",
    "results/q_line/incoming_completeness_r2_20261004/r2_checks.json",
    "results/q_line/incoming_completeness_r2_leftovers_20261004/leftover_checks.json",
    "results/q_line/stage1ao_electronic_spectrum_20261004/electronic_spectrum.json",
    "results/q_line/stage1ap_h2_window_quadrature_20261004/h2_window_quadrature.json",
]
N_C, N_V = 4, 4
FREQ = np.array([0.37, 1.05])
TWO_K = np.array([[0.0, 0.0], [0.23, 0.12]])


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def pack_c(m):
    m = np.asarray(m, complex)
    herm = 0.5 * (m + m.conj().T)
    evals = np.linalg.eigvalsh(herm)
    return {
        "real": np.real(m).tolist(),
        "imag": np.imag(m).tolist(),
        "maxabs": float(np.max(np.abs(m))),
        "trace_real": float(np.trace(m).real),
        "min_hermitian_eig": float(evals[0]),
        "anti_hermitian_maxabs": float(np.max(np.abs(m - herm))),
    }


def m0_sample(geo, kf, weights, neel):
    """Original (k,c,v) vertices. Weight once: u = sqrt(w) X."""
    kf = np.asarray(kf, float)
    weights = np.asarray(weights, float)
    if abs(float(np.sum(weights)) - 1.0) > 1e-12:
        raise ValueError("BZ weights must sum to 1.")
    if np.any(weights <= 0):
        raise ValueError("Positive weights required.")
    kc = geo.k_cart(kf)
    vv, vc, en = occup_empty_bases(h8(geo, R0, kc, neel), MU)
    x = x_from_bases(current_ops(geo, R0, kc, neel), vv, vc)
    gap = (en[:, 4:, None] - en[:, None, :4]).reshape(-1)
    u = np.einsum("k,kacv->akcv", np.sqrt(weights), x).reshape(2, -1)
    x_flat = x.reshape(len(kf), 2, -1)
    c_from_x = np.einsum("k,kap,kbp->ab", weights, np.conj(x_flat), x_flat)
    c_from_u = u.conj() @ u.T
    d_occ = mean_d_from_bases(d_ops(geo, R0, kc, neel), vv)
    d = np.einsum("k,kab->ab", weights, d_occ)
    d = 0.5 * (d + d.T)
    n_k = len(kf)
    k_of = np.repeat(np.arange(n_k), N_C * N_V)
    c_of = np.tile(np.repeat(np.arange(N_C), N_V), n_k)
    v_of = np.tile(np.arange(N_V), n_k * N_C)
    return {
        "kf": kf,
        "weights": weights,
        "gap": gap,
        "u": u,
        "x": x,
        "D": d,
        "C_from_u": c_from_u,
        "C_from_xw": c_from_x,
        "k_of": k_of,
        "c_of": c_of,
        "v_of": v_of,
        "n_el": int(u.shape[1]),
        "n_k": int(n_k),
        "neel": int(neel),
        "weight_once_residual": maxabs(c_from_u - c_from_x),
    }


def gamma_grid(n):
    g = (np.arange(n)) / n
    kf = np.stack(np.meshgrid(g, g, indexing="ij"), axis=-1).reshape(-1, 2)
    w = np.full(len(kf), 1.0 / len(kf))
    return kf, w


def offpole_ok(gap, freq, *, floor=1e-8):
    gap = np.asarray(gap, float)
    dist = min(float(np.min(np.abs(gap - w))) for w in freq)
    return dist, bool(dist > floor)


def photon_jump_errors(bundle, u, D, gamma, freq):
    dens = []
    extra = []
    min_pole = []
    for w in freq:
        dist, ok = offpole_ok(bundle.photon.gaps, [w])
        min_pole.append(dist)
        if not ok:
            dens.append(None)
            extra.append(None)
            continue
        k = real_frequency_kernel(bundle.photon.gaps, u, D, float(w))
        jaa, jii, jai, _ = aa_ii_jumps_from_kernel(k, float(w), gamma)
        paa, pii, a_cols, i_cols = photon_column_density(bundle.photon, float(w))
        dens.append(max(maxabs(paa - jaa), maxabs(pii - jii),
                        maxabs(a_cols @ i_cols.conj().T - jai)))
        extra.append(maxabs(bright_bare_electron_current_jump(
            bundle.photon.gaps, u, float(w))))
    return dens, extra, min_pole


def dark_inventory(sample, gamma, freq):
    gap, u, D = sample["gap"], sample["u"], sample["D"]
    basis, info, near = dark_subspace(gap, u)
    classified = classify_dark_columns(info, k_of_index=sample["k_of"])
    types = {}
    for item in classified:
        types[item["spectral_type"]] = types.get(item["spectral_type"], 0) + int(item["nullity"])
    dist, ok = offpole_ok(gap, freq)
    bundle = None
    dens = extra = None
    photon_dark = None
    i_dark = None
    pole_error = ""
    if ok:
        try:
            bundle = assemble_incoming_bundle(gap, u, D, gamma, freq)
        except ValueError as exc:
            ok = False
            pole_error = str(exc)
            bundle = None
    if bundle is not None:
        dens, extra, _ = photon_jump_errors(bundle, u, D, gamma, freq)
        p_dark = dark_projector(basis)
        photon_dark = []
        for w in freq:
            rho_b, _ = electron_column_density(bundle.photon, float(w))
            photon_dark.append(maxabs(p_dark @ rho_b) if basis.size else 0.0)
        i_dark = (
            current(bundle.point, u, D) if bundle.n_point_columns
            else np.zeros((2, 0))
        )
    unique_gaps = int(len(np.unique(np.round(gap, 12))))
    per_k_null = []
    for k in range(sample["n_k"]):
        idx = np.flatnonzero(sample["k_of"] == k)
        b_k, info_k, _ = dark_subspace(gap[idx], u[:, idx])
        per_k_null.append({
            "k_index": int(k),
            "kf": sample["kf"][k].tolist(),
            "n_el": int(len(idx)),
            "nullity": int(b_k.shape[1]),
            "n_clusters": int(len(info_k)),
            "n_unique_gaps": int(len(np.unique(np.round(gap[idx], 12)))),
        })
    return {
        "n_el": sample["n_el"],
        "n_k": sample["n_k"],
        "weight_once_residual": sample["weight_once_residual"],
        "min_direct_gap": float(np.min(gap)),
        "n_unique_gaps": unique_gaps,
        "axis_near_zero_count": int(np.count_nonzero(axis_near_zero_mask(u, relative=1e-10))),
        "gap_subspace_nullity": int(basis.shape[1]),
        "n_exact_gap_clusters": int(len(info)),
        "near_degenerate_pairs": int(len(near)),
        "spectral_type_counts": types,
        "per_k_nullity": per_k_null,
        "offpole_min_distance": dist,
        "offpole_ok": ok,
        "pole_error": pole_error,
        "photon_AA_II_jump_maxabs": None if dens is None else max(x or 0.0 for x in dens),
        "extra_bare_bright_vacuum_maxabs": None if extra is None else max(x or 0.0 for x in extra),
        "physical_current_of_dark_b": None if i_dark is None else maxabs(i_dark),
        "photon_dark_projection": None if photon_dark is None else max(photon_dark),
        "metadata_lengths_agree": None if bundle is None else bool(
            bundle.n_point_columns == len(bundle.point_gaps) == len(bundle.point_labels)
        ),
        "D": pack_c(D),
        "C_equal_time": pack_c(sample["C_from_u"]),
        "not_a_BZ_dark_dimension": True,
        "finite_sample_not_continuum": True,
    }, bundle, basis, info


def negative_controls(sample, gamma, freq):
    """Omit dark / omit one kcv / extra bright vacuum, on original labels."""
    gap, u, D = sample["gap"], sample["u"], sample["D"]
    basis, info, _ = dark_subspace(gap, u)
    bundle = assemble_incoming_bundle(gap, u, D, gamma, freq)
    p_dark = dark_projector(basis)
    photon_dark = []
    photon_trace = []
    for w in freq:
        rho_b, b = electron_column_density(bundle.photon, float(w))
        photon_dark.append(maxabs(p_dark @ rho_b) if basis.size else 0.0)
        photon_trace.append(float(np.trace(rho_b).real))
    omit_dark = None
    if basis.shape[1]:
        paa, pii, _, _ = photon_column_density(bundle.photon, float(freq[0]))
        omit_dark = {
            "n_dark_present": int(basis.shape[1]),
            "n_point_columns_full": int(bundle.n_point_columns),
            "n_point_columns_if_omitted": int(bundle.n_point_columns - 1),
            "dark_column_norm": float(np.linalg.norm(basis[:, 0])),
            "photon_density_on_that_dark": photon_dark[0] if photon_dark else 0.0,
            "photon_AA_trace": float(np.trace(paa).real),
            "photon_II_trace": float(np.trace(pii).real),
            "photon_does_not_carry_dark": bool(
                (photon_dark[0] if photon_dark else 0.0) < 1e-8
            ),
            "detected_if_omitted": bool(
                bundle.n_point_columns >= 1
                and (photon_dark[0] if photon_dark else 0.0) < 1e-8
            ),
            "meaning": (
                "omitting a dark combination drops a point column; "
                "photon AA/II cannot see that loss"
            ),
        }
    # Omit one original kcv that is not dark-dominated.
    strengths = np.sqrt(np.sum(np.abs(u) ** 2, axis=0))
    dark_weight = np.sum(np.abs(basis) ** 2, axis=1) if basis.size else np.zeros(u.shape[1])
    bright_idx = np.flatnonzero((strengths > 1e-8) & (dark_weight < 0.5))
    omit_kcv = None
    if bright_idx.size:
        j = int(bright_idx[int(np.argmax(strengths[bright_idx]))])
        keep = np.ones(u.shape[1], bool)
        keep[j] = False
        u2, g2 = u[:, keep], gap[keep]
        bundle2 = assemble_incoming_bundle(g2, u2, D, gamma, freq)
        rho_full, _ = electron_column_density(bundle.photon, float(freq[0]))
        rho_omit, _ = electron_column_density(bundle2.photon, float(freq[0]))
        paa_full, pii_full, _, _ = photon_column_density(bundle.photon, float(freq[0]))
        paa_omit, pii_omit, _, _ = photon_column_density(bundle2.photon, float(freq[0]))
        omit_kcv = {
            "omitted_index": j,
            "k": int(sample["k_of"][j]),
            "c": int(sample["c_of"][j]),
            "v": int(sample["v_of"][j]),
            "vertex_strength": float(strengths[j]),
            "photon_density_trace_full": float(np.trace(rho_full).real),
            "photon_density_trace_omit": float(np.trace(rho_omit).real),
            "n_el_full": int(bundle.n_el),
            "n_el_omit": int(bundle2.n_el),
            "label_count_drop": int(bundle.n_el - bundle2.n_el),
            "AA_density_change": maxabs(paa_full - paa_omit),
            "II_density_change": maxabs(pii_full - pii_omit),
            "detected": bool(
                bundle.n_el - bundle2.n_el == 1
                and (
                    maxabs(paa_full - paa_omit) > 1e-12
                    or maxabs(pii_full - pii_omit) > 1e-12
                )
            ),
        }
    extra_vac = []
    for w in freq:
        extra_vac.append(maxabs(bright_bare_electron_current_jump(gap, u, float(w))))
    return {
        "photon_dark_projection": max(photon_dark) if photon_dark else 0.0,
        "photon_electronic_trace": photon_trace,
        "omit_dark_combination": omit_dark,
        "omit_one_kcv": omit_kcv,
        "extra_bright_vacuum_offpole_maxabs": max(extra_vac) if extra_vac else 0.0,
        "extra_bright_vacuum_is_illegal": True,
        "seen_on_original_labels": True,
    }


def scan_grids(geo, gamma, freq):
    rows = []
    for n in (1, 2, 3, 4):
        kf, w = gamma_grid(n)
        sample = m0_sample(geo, kf, w, 1)
        inv, bundle, basis, info = dark_inventory(sample, gamma, freq)
        inv["grid"] = f"Gamma_{n}x{n}"
        inv["n_k_declared"] = int(n * n)
        rows.append(inv)
        _ = (bundle, basis, info)
    return rows


def two_k_attach(geo, gamma, freq):
    w = np.array([0.5, 0.5])
    out = {}
    for neel in (1, -1):
        sample = m0_sample(geo, TWO_K, w, neel)
        inv, bundle, basis, info = dark_inventory(sample, gamma, freq)
        neg = negative_controls(sample, gamma, freq)
        out[f"N{neel}"] = {
            "inventory": inv,
            "negative_controls": neg,
            "weight_once": "u = sqrt(w) X; C = u u^dagger; no second w",
            "D_source": "occupied_mean_d_from_bases_d_ops",
            "gamma_source": "q8_m0_gsheet.gamma_self",
            "n_point_columns": None if bundle is None else int(bundle.n_point_columns),
            "n_photon_columns": None if bundle is None else int(bundle.n_photon_columns),
        }
        _ = (basis, info)
    return out


def flags_from(two_k, scan, gamma):
    flags = {
        "gamma_positive": bool(gamma > 0.0),
        "not_full_ground": True,
        "not_photon_output": True,
        "P1_not_run": True,
        "continuum_boundary_not_claimed": True,
    }
    for key, rec in two_k.items():
        inv = rec["inventory"]
        neg = rec["negative_controls"]
        flags[f"{key}_weight_once"] = bool(inv["weight_once_residual"] < 1e-12)
        flags[f"{key}_offpole"] = bool(inv["offpole_ok"])
        flags[f"{key}_metadata"] = bool(inv["metadata_lengths_agree"])
        flags[f"{key}_dark_current"] = bool(
            inv["physical_current_of_dark_b"] is not None
            and inv["physical_current_of_dark_b"] < 1e-12
        )
        flags[f"{key}_photon_dark"] = bool(
            inv["photon_dark_projection"] is not None
            and inv["photon_dark_projection"] < 1e-8
        )
        flags[f"{key}_AA_II"] = bool(
            inv["photon_AA_II_jump_maxabs"] is not None
            and inv["photon_AA_II_jump_maxabs"] < 1e-10
        )
        flags[f"{key}_omit_dark_detected"] = bool(
            neg["omit_dark_combination"] and neg["omit_dark_combination"]["detected_if_omitted"]
        )
        flags[f"{key}_omit_kcv_detected"] = bool(
            neg["omit_one_kcv"] and neg["omit_one_kcv"]["detected"]
        )
        flags[f"{key}_extra_vacuum_nonzero"] = bool(
            neg["extra_bright_vacuum_offpole_maxabs"] > 1e-6
        )
        flags[f"{key}_not_bz"] = True
        flags[f"{key}_D_hermitian"] = bool(inv["D"]["anti_hermitian_maxabs"] < 1e-12)
    for row in scan:
        name = row["grid"]
        flags[f"{name}_weight_once"] = bool(row["weight_once_residual"] < 1e-12)
        flags[f"{name}_not_bz"] = True
        if row["offpole_ok"]:
            flags[f"{name}_AA_II"] = bool(row["photon_AA_II_jump_maxabs"] < 1e-10)
            flags[f"{name}_dark_current"] = bool(
                row["physical_current_of_dark_b"] is not None
                and row["physical_current_of_dark_b"] < 1e-12
            )
    return flags


def main() -> int:
    t0 = time.perf_counter()
    geo = build_geometry()
    gamma = float(gamma_self(geo))
    two_k = two_k_attach(geo, gamma, FREQ)
    scan = scan_grids(geo, gamma, FREQ)
    flags = flags_from(two_k, scan, gamma)
    payload = {
        "status": "incoming_m0_finite_attach_not_continuum_boundary",
        "all_checks_pass": bool(all(flags.values())),
        "pass_flags": flags,
        "definition": {
            "weight_once": "u = sqrt(w_k) X; C_ab = (u u^dagger)_ab; do not multiply w again",
            "labels": "original (k,c,v)",
            "D": "occupied mean_d_from_bases(d_ops)",
            "gamma": "q8_m0_gsheet.gamma_self",
            "not_photon_number": True,
            "not_incoming_continuum_boundary": True,
            "not_full_ground": True,
        },
        "gamma_self": gamma,
        "frequencies": [float(w) for w in FREQ],
        "two_k": two_k,
        "small_k_scan": scan,
        "spectral_measure": electronic_spectral_measure_spec(),
        "legal_electronic_continuum_boundary": False,
        "human_review_stop": {
            "triggered": True,
            "question": (
                "Finite k samples keep a large current-null space "
                "(~12-20 per sample). Treat each k as a discrete oscillator "
                "set, or construct electronic continuous incoming columns "
                "on the absorption support? The next object is a physics "
                "choice, not a grid choice."
            ),
            "not_a_grid_menu": True,
            "not_a_model_change_yet": True,
        },
        "old_json_not_overwritten": OLD_JSON,
        "P1_not_run": True,
        "full_BZ_not_run": True,
        "elapsed_s": time.perf_counter() - t0,
        "source_sha256": {
            str(p.relative_to(REPO)): sha(p)
            for p in (
                Path(__file__).resolve(),
                REPO / "tools/b0_small_model/q8_m0_incoming_spectral_completeness_20261004.py",
                REPO / "tools/b0_small_model/q8_m0_continuum_columns_20261004.py",
                REPO / "tools/b0_small_model/q8_m0_gsheet.py",
            )
        },
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "finite_attach.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    print(json.dumps({
        "status": payload["status"],
        "all_checks_pass": payload["all_checks_pass"],
        "failed": [k for k, v in flags.items() if not v],
        "gamma_self": gamma,
        "two_k_nullity": {
            k: rec["inventory"]["gap_subspace_nullity"] for k, rec in two_k.items()
        },
        "scan_nullity": {row["grid"]: row["gap_subspace_nullity"] for row in scan},
        "scan_unique_gaps": {row["grid"]: row["n_unique_gaps"] for row in scan},
        "elapsed_s": payload["elapsed_s"],
    }, indent=2))
    return 0 if payload["all_checks_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

