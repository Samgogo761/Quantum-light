#!/usr/bin/env python3
"""Finite-reference radiation ledger + input completeness. Not production n,m.

Same two-k model as the 2026-10-05 caller fix. Keep dark/point columns.
Do not add a second bright electronic vacuum. Three H2-neighbour frequencies
remain wiring-only: a driven mix needs the rest of the input line.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from q8_m0_classical import p1_a2_protocol
from q8_m0_continuum_columns_20261004 import current, real_frequency_kernel
from q8_m0_continuum_ground_bridge_20261003 import ContinuumGroundBridge
from q8_m0_gsheet import gamma_self, gsheet_spec
from q8_m0_incoming_spectral_completeness_20261004 import (
    aa_ii_jumps_from_kernel,
    assemble_incoming_bundle,
    bright_bare_electron_current_jump,
    classify_dark_columns,
    dark_projector,
    dark_subspace,
    electron_column_density,
    electronic_spectral_measure_spec,
    maxabs,
    photon_column_density,
)
from q8_m0_k1_chi_interface import CHI_STATUS
from q8_m0_model import build_geometry
from q8_m0_p1_h2_finite_kcv_caller_fix_20261005 import (
    FREQ,
    OLD_JSON,
    TWO_K,
    WEIGHTS,
    even_odd_from_lr,
    m0_sample,
    photon_port_indices,
    radiation_ledger,
    sha,
)

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "results/q_line/p1_h2_finite_kcv_input_ledger_20261005"
OLD_CONTRACT = REPO / "results/q_line/p1_h2_finite_kcv_contract_20261004/contract.json"
CALLER_FIX = REPO / "results/q_line/p1_h2_finite_kcv_caller_fix_20261005/caller_fix.json"
COMPLEX_Z = (0.3 + 0.2j, 1.0 + 0.4j, 2.5 + 0.15j)
DENSE_FREQ = np.array([0.02, 0.05, 0.10, 0.18, 0.35, 0.60, 0.90, 1.20], float)
COARSE_FREQ = np.array([0.05, 0.10, 0.18], float)
POLE_FLOOR = 1e-8
FROZEN_GAMMA = 0.001080789276
OLD_JSON_KEEP = OLD_JSON + [
    "results/q_line/p1_h2_finite_kcv_caller_fix_20261005/caller_fix.json",
]
BOUNDARY_KEYS = (
    "not_production_n",
    "not_production_m",
    "not_complete_input_vacuum",
    "two_k_not_a_domain_signal",
    "not_eight_cycle",
)


def _int_exp(alpha, tp):
    a = np.asarray(alpha, dtype=complex)
    out = np.empty(a.shape, dtype=complex)
    small = np.abs(a) < 1e-12
    out[small] = tp
    aa = a[~small]
    out[~small] = (np.exp(1j * aa * tp) - 1.0) / (1j * aa)
    return out


def _window_fourier(omega, tp):
    om = np.asarray(omega, dtype=float)
    two = 2.0 * np.pi / tp
    return _int_exp(om, tp) / (2.0 * tp) - _int_exp(om + two, tp) / (4.0 * tp) - _int_exp(om - two, tp) / (4.0 * tp)


def pack_z(z):
    z = complex(z)
    return {"real": float(z.real), "imag": float(z.imag)}


def errors(a, b):
    diff = maxabs(np.asarray(a) - np.asarray(b))
    scale = maxabs(b)
    return {
        "max_absolute": diff,
        "reference_max": scale,
        "max_relative": diff / max(scale, 1e-30),
    }


def keep_offpole(freqs, gaps):
    kept, dropped = [], []
    gaps = np.asarray(gaps, float)
    for w in np.asarray(freqs, float):
        dist = float(np.min(np.abs(gaps - w)))
        rec = {"omega": float(w), "min_pole_distance": dist}
        if dist <= POLE_FLOOR or w <= 0.0:
            rec["kept"] = False
            dropped.append(rec)
        else:
            rec["kept"] = True
            kept.append(rec)
    if not kept:
        raise RuntimeError("All candidate input frequencies sit on poles; no silent drop.")
    return kept, dropped


def a_conversion_check(gamma, omega=0.1):
    eta = np.sqrt(gamma / (2.0 * np.pi))
    a_from_A = np.sqrt(omega / gamma) * (eta / np.sqrt(omega))
    return {
        "omega": float(omega),
        "eta": float(eta),
        "A_free_one_pol": float(eta / np.sqrt(omega)),
        "a_from_A": float(a_from_A),
        "expect_1_over_sqrt_2pi": float(1.0 / np.sqrt(2.0 * np.pi)),
        "residual": float(abs(a_from_A - 1.0 / np.sqrt(2.0 * np.pi))),
        "keep_1_over_sqrt_omega": True,
        "nk_is_not_Ncell": True,
    }


def even_odd_port_algebra():
    """Frozen gsheet ports, not a new Maxwell object."""
    a_L_in = np.array([0.3, -0.1], float)
    a_R_in = np.array([0.2, 0.4], float)
    a_self = np.array([0.05, -0.02], float)
    a_R_out = a_L_in + a_self
    a_L_out = a_R_in + a_self
    even_in = (a_L_in + a_R_in) / np.sqrt(2.0)
    odd_in = (a_L_in - a_R_in) / np.sqrt(2.0)
    even_out = (a_L_out + a_R_out) / np.sqrt(2.0)
    odd_out = (a_L_out - a_R_out) / np.sqrt(2.0)
    return {
        "odd_out_plus_odd_in_maxabs": float(np.max(np.abs(odd_out + odd_in))),
        "even_out_minus_even_in_minus_sqrt2_Aself_maxabs": float(
            np.max(np.abs(even_out - even_in - np.sqrt(2.0) * a_self))
        ),
        "L_R_are_not_even_odd": True,
    }


def kernel_complex(gaps, u, D, z):
    gaps = np.asarray(gaps, float)
    u = np.asarray(u, complex)
    D = np.asarray(D, float)
    z = complex(z)
    pi = (u.conj() / (gaps - z)) @ u.T + (u / (gaps + z)) @ u.conj().T
    return D - pi


def resolvent_checks(sample, gamma):
    bridge = ContinuumGroundBridge(sample["gap"], sample["u"], sample["D"], gamma)
    rows = []
    for z in COMPLEX_Z:
        k_schur = kernel_complex(sample["gap"], sample["u"], sample["D"], z)
        k_bridge = bridge.electronic_kernel(z)
        a_free = np.eye(2, dtype=complex)
        i_free = np.zeros((2, 2), complex)
        loc = bridge.solve_local_fields(z, a_free, i_free)
        qa = (1j if z.imag > 0 else -1j) * gamma / z
        a_schur = np.linalg.solve(np.eye(2) + qa * k_schur, a_free)
        rows.append({
            "z": pack_z(z),
            "K_schur_vs_bridge": errors(k_schur, k_bridge),
            "A_schur_vs_bridge": errors(a_schur, loc["A"]),
            "bdag_not_set_to_b_conjugate": True,
        })
    return rows


def jump_at(sample, gamma, omega):
    k = real_frequency_kernel(sample["gap"], sample["u"], sample["D"], omega)
    jump_aa, jump_ii, _, _ = aa_ii_jumps_from_kernel(k, omega, gamma)
    bundle = assemble_incoming_bundle(sample["gap"], sample["u"], sample["D"], gamma, [omega])
    paa, pii, _, _ = photon_column_density(bundle.photon, omega)
    return {
        "omega": float(omega),
        "AA_columns_vs_jump": errors(paa, jump_aa),
        "II_columns_vs_jump": errors(pii, jump_ii),
        "n_photon_columns": int(bundle.n_photon_columns),
        "n_point_columns": int(bundle.n_point_columns),
    }


def weighted_photon_gram(sample, gamma, freqs):
    """Selected-node diagnostic Gram. Not a complete dν vacuum integral."""
    freqs = np.asarray(freqs, float)
    if len(freqs) < 2:
        raise ValueError("Need at least two nodes to assign a dν width.")
    edges = np.empty(len(freqs) + 1)
    edges[0] = freqs[0] - 0.5 * (freqs[1] - freqs[0])
    edges[-1] = freqs[-1] + 0.5 * (freqs[-1] - freqs[-2])
    edges[1:-1] = 0.5 * (freqs[:-1] + freqs[1:])
    if edges[0] <= 0.0:
        edges[0] = 0.5 * freqs[0]
    dnu = np.diff(edges)
    rho = np.zeros((sample["n_el"], sample["n_el"]), complex)
    for w, dw in zip(freqs, dnu):
        bundle = assemble_incoming_bundle(sample["gap"], sample["u"], sample["D"], gamma, [w])
        b, _ = electron_column_density(bundle.photon, float(w))
        rho += dw * b
    return rho, dnu, freqs


def h2_packet_mass(prot):
    w0 = float(prot.omega0)
    tp = float(prot.t_pulse)
    dw = (2.0 * np.pi / tp) / 64.0
    grid = np.arange(0.0, 12.0 * w0 + 0.5 * dw, dw)
    pos = grid > 0.0
    om = grid[pos]
    raw = tp * _window_fourier(2.0 * w0 - om, tp)
    mass_raw = float(np.trapezoid(np.abs(raw) ** 2, om))
    nrm = float(np.sqrt(mass_raw))
    phi = raw / nrm
    mass = np.abs(phi) ** 2
    def band(lo, hi):
        m = (om >= lo) & (om < hi)
        return float(np.trapezoid(mass[m], om[m])) if np.any(m) else 0.0
    return {
        "packet_center": 2.0 * w0,
        "normalizer": float(nrm),
        "mass_total_positive_grid": float(np.trapezoid(mass, om)),
        "mass_IR_below_0.05": band(0.0, 0.05),
        "mass_near_H2_0.05_to_0.15": band(0.05, 0.15),
        "mass_above_0.15": band(0.15, float(om[-1]) + 1e-15),
        "three_H2_nodes_do_not_truncate_input": True,
        "stationary_packet_tail_is_not_driven_input_tail": True,
    }


def run_domain(geo, gamma, neel, prot):
    started = time.perf_counter()
    sample = m0_sample(geo, TWO_K, WEIGHTS, neel)
    kept_dense, dropped_dense = keep_offpole(DENSE_FREQ, sample["gap"])
    kept_coarse, dropped_coarse = keep_offpole(COARSE_FREQ, sample["gap"])
    freqs_dense = np.array([r["omega"] for r in kept_dense], float)
    freqs_coarse = np.array([r["omega"] for r in kept_coarse], float)
    bundle = assemble_incoming_bundle(sample["gap"], sample["u"], sample["D"], gamma, FREQ)
    classified = classify_dark_columns(bundle.dark_info, k_of_index=sample["k_of"])
    basis, info, near = dark_subspace(sample["gap"], sample["u"])
    p_dark = dark_projector(basis)
    dark_I = current(bundle.point, sample["u"], sample["D"]) if bundle.n_point_columns else np.zeros((2, 0), complex)
    photon_dark = []
    extra_vac = []
    jump_rows = []
    for w in FREQ:
        rho_b, _ = electron_column_density(bundle.photon, float(w))
        photon_dark.append(maxabs(p_dark @ rho_b) if basis.size else 0.0)
        extra_vac.append(maxabs(bright_bare_electron_current_jump(sample["gap"], sample["u"], float(w))))
        jump_rows.append(jump_at(sample, gamma, float(w)))
    left, right = photon_port_indices(bundle.photon.labels)
    even_a, odd_a = even_odd_from_lr(bundle.photon.initial.A, left, right)
    gram_dense, dnu_dense, _ = weighted_photon_gram(sample, gamma, freqs_dense)
    gram_coarse, dnu_coarse, _ = weighted_photon_gram(sample, gamma, freqs_coarse)
    herm = 0.5 * (gram_dense + gram_dense.conj().T)
    evals = np.linalg.eigvalsh(herm) if herm.size else np.zeros(0)
    return {
        "neel": int(neel),
        "n_el": sample["n_el"],
        "weight_once_residual": sample["weight_once_residual"],
        "min_direct_gap": sample["min_direct_gap"],
        "max_direct_gap": sample["max_direct_gap"],
        "n_photon_columns_wiring": int(bundle.n_photon_columns),
        "n_point_columns": int(bundle.n_point_columns),
        "n_dark_classified": int(sum(int(r["nullity"]) for r in classified)),
        "near_degenerate_pairs": int(len(near)),
        "dark_current_t0": maxabs(dark_I),
        "photon_dark_projection": max(photon_dark) if photon_dark else 0.0,
        "extra_bare_bright_vacuum_maxabs": max(extra_vac) if extra_vac else 0.0,
        "A_odd_maxabs": maxabs(odd_a),
        "A_even_maxabs": maxabs(even_a),
        "A_L_minus_A_R_maxabs": maxabs(bundle.photon.initial.A[:, left] - bundle.photon.initial.A[:, right]),
        "kept_dark": True,
        "no_second_bright_vacuum": True,
        "metadata_lengths_agree": bool(
            bundle.n_point_columns
            == len(bundle.point_gaps)
            == len(bundle.point_labels)
            == int(sum(int(r["nullity"]) for r in classified))
        ),
        "bdag_minus_conj_b_maxabs": maxabs(
            bundle.photon.initial.bdag - np.conj(bundle.photon.initial.b)
        ),
        "resolvent_complex": resolvent_checks(sample, gamma),
        "jump_selected": jump_rows,
        "input_nodes": {
            "wiring_FREQ": FREQ.tolist(),
            "dense": kept_dense,
            "dense_dropped_on_pole": dropped_dense,
            "coarse": kept_coarse,
            "coarse_dropped_on_pole": dropped_coarse,
            "silent_node_deletion": False,
            "no_physical_broadening": True,
        },
        "electron_gram_dense": {
            "dnu": dnu_dense.tolist(),
            "trace_real": float(np.trace(gram_dense).real),
            "min_hermitian_eig": float(evals[0]) if evals.size else 0.0,
            "anti_hermitian_maxabs": maxabs(gram_dense - herm),
            "vs_coarse_maxabs": maxabs(gram_dense - gram_coarse),
            "not_complete_input_vacuum": True,
            "selected_node_diagnostic_only": True,
        },
        "h2_packet_mass": h2_packet_mass(prot),
        "wall_seconds": time.perf_counter() - started,
    }


def flags_from(plus, minus, conv, ports, gamma):
    max_k = 0.0
    max_a = 0.0
    for rec in plus["resolvent_complex"] + minus["resolvent_complex"]:
        max_k = max(max_k, rec["K_schur_vs_bridge"]["max_absolute"])
        max_a = max(max_a, rec["A_schur_vs_bridge"]["max_absolute"])
    max_jump = 0.0
    for rec in plus["jump_selected"] + minus["jump_selected"]:
        max_jump = max(
            max_jump,
            rec["AA_columns_vs_jump"]["max_absolute"],
            rec["II_columns_vs_jump"]["max_absolute"],
        )
    return {
        "gamma_positive": bool(gamma > 0.0),
        "gamma_matches_frozen": bool(abs(gamma - FROZEN_GAMMA) < 1e-12),
        "plus_weight_once": bool(plus["weight_once_residual"] < 1e-12),
        "minus_weight_once": bool(minus["weight_once_residual"] < 1e-12),
        "plus_keeps_dark": bool(plus["n_point_columns"] > 0),
        "minus_keeps_dark": bool(minus["n_point_columns"] > 0),
        "plus_dark_matches_classified": bool(
            plus["n_point_columns"] == plus["n_dark_classified"]
        ),
        "minus_dark_matches_classified": bool(
            minus["n_point_columns"] == minus["n_dark_classified"]
        ),
        "plus_metadata_agree": bool(plus["metadata_lengths_agree"]),
        "minus_metadata_agree": bool(minus["metadata_lengths_agree"]),
        "plus_dark_current_small": bool(plus["dark_current_t0"] < 1e-10),
        "minus_dark_current_small": bool(minus["dark_current_t0"] < 1e-10),
        "plus_photon_dark_projection_small": bool(plus["photon_dark_projection"] < 1e-8),
        "minus_photon_dark_projection_small": bool(minus["photon_dark_projection"] < 1e-8),
        "plus_L_equals_R_at_t0": bool(plus["A_L_minus_A_R_maxabs"] < 1e-12),
        "minus_L_equals_R_at_t0": bool(minus["A_L_minus_A_R_maxabs"] < 1e-12),
        "plus_odd_A_small_at_t0": bool(plus["A_odd_maxabs"] < 1e-12),
        "minus_odd_A_small_at_t0": bool(minus["A_odd_maxabs"] < 1e-12),
        "extra_vacuum_is_illegal_and_large": bool(
            plus["extra_bare_bright_vacuum_maxabs"] > 1.0
            and minus["extra_bare_bright_vacuum_maxabs"] > 1.0
        ),
        "a_conversion_residual_lt_1e12": bool(conv["residual"] < 1e-12),
        "odd_port_identity": bool(ports["odd_out_plus_odd_in_maxabs"] < 1e-14),
        "even_port_identity": bool(
            ports["even_out_minus_even_in_minus_sqrt2_Aself_maxabs"] < 1e-14
        ),
        "complex_K_vs_bridge_lt_1e10": bool(max_k < 1e-10),
        "complex_A_vs_bridge_lt_1e10": bool(max_a < 1e-10),
        "selected_jump_lt_1e10": bool(max_jump < 1e-10),
        "old_contract_not_overwritten": bool(OLD_CONTRACT.exists()),
        "caller_fix_json_present": bool(CALLER_FIX.exists()),
        "chi_rad_placeholder_not_flipped": True,
        "not_production_n": True,
        "not_production_m": True,
        "not_complete_input_vacuum": True,
        "two_k_not_a_domain_signal": True,
        "not_eight_cycle": True,
    }


def main() -> int:
    t0 = time.perf_counter()
    geo = build_geometry()
    prot = p1_a2_protocol(0.8)
    gamma = float(gamma_self(geo))
    print(f"gamma_self={gamma:.12g}", flush=True)
    plus = run_domain(geo, gamma, +1, prot)
    minus = run_domain(geo, gamma, -1, prot)
    conv = a_conversion_check(gamma)
    ports = even_odd_port_algebra()
    sheet = gsheet_spec(geo)
    flags = flags_from(plus, minus, conv, ports, gamma)
    physics_flags = {k: v for k, v in flags.items() if k not in BOUNDARY_KEYS}
    ledger = radiation_ledger(gamma)
    ledger["chi_rad_placeholder_status"] = CHI_STATUS
    ledger["gsheet_id"] = sheet.get("ident")
    ledger["gsheet_status"] = sheet.get("status")
    ledger["do_not_flip_chi_rad_flag"] = True
    payload = {
        "status": "p1_h2_finite_kcv_input_ledger_not_production_nm",
        "all_physics_checks_pass": bool(all(physics_flags.values())),
        "pass_flags": flags,
        "scope": {
            "same_two_k_as_caller_fix": True,
            "keep_dark_point_columns": True,
            "no_second_bright_electronic_vacuum": True,
            "selected_frequencies": FREQ.tolist(),
            "selected_frequencies_are_wiring_only": True,
            "dense_input_nodes": DENSE_FREQ.tolist(),
            "coarse_input_nodes": COARSE_FREQ.tolist(),
            "not_complete_input_vacuum": True,
            "H2_output_window_does_not_truncate_input": True,
            "not_eight_cycle": True,
            "not_production_n_m": True,
            "two_k_not_a_domain_signal": True,
            "reuse_frozen_gsheet_and_incoming": True,
            "old_chi_rad_slot_not_a_theory_block": True,
        },
        "radiation_ledger": ledger,
        "a_conversion": conv,
        "even_odd_port_algebra": ports,
        "electronic_spectral_measure": electronic_spectral_measure_spec(),
        "gamma_self": gamma,
        "domains": {"N+1": plus, "N-1": minus},
        "old_json_not_overwritten": OLD_JSON_KEEP,
        "elapsed_s": time.perf_counter() - t0,
        "source_sha256": {
            str(p.relative_to(REPO)): sha(p)
            for p in (
                Path(__file__).resolve(),
                REPO / "tools/b0_small_model/q8_m0_p1_h2_finite_kcv_caller_fix_20261005.py",
                REPO / "tools/b0_small_model/q8_m0_incoming_spectral_completeness_20261004.py",
                REPO / "tools/b0_small_model/q8_m0_gsheet.py",
                REPO / "tools/b0_small_model/q8_m0_continuum_ground_bridge_20261003.py",
                REPO / "tools/b0_small_model/q8_m0_k1_chi_interface.py",
            )
        },
        "old_contract_sha256": sha(OLD_CONTRACT) if OLD_CONTRACT.exists() else None,
        "caller_fix_sha256": sha(CALLER_FIX) if CALLER_FIX.exists() else None,
        "production": False,
        "optical_moments_computed": False,
        "P1_full_pulse": False,
        "legal_electronic_continuum_boundary": False,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    out_json = OUT / "input_ledger.json"
    if out_json.exists():
        raise FileExistsError("Refuse overwrite of input_ledger.json")
    if not OLD_CONTRACT.exists():
        raise FileNotFoundError("Old contract.json missing; refuse to proceed")
    if not CALLER_FIX.exists():
        raise FileNotFoundError("caller_fix.json missing; refuse to proceed")
    for rel in OLD_JSON_KEEP:
        if not (REPO / rel).exists():
            raise FileNotFoundError(f"Protected JSON missing: {rel}")
    out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    failed = [k for k, v in physics_flags.items() if not v]
    print(
        json.dumps(
            {
                "status": payload["status"],
                "all_physics_checks_pass": payload["all_physics_checks_pass"],
                "failed": failed,
                "gamma_self": gamma,
                "plus": {
                    "n_point": plus["n_point_columns"],
                    "dark_I": plus["dark_current_t0"],
                    "photon_dark": plus["photon_dark_projection"],
                    "extra_vac": plus["extra_bare_bright_vacuum_maxabs"],
                    "gram_trace": plus["electron_gram_dense"]["trace_real"],
                    "A_odd": plus["A_odd_maxabs"],
                },
                "minus": {
                    "n_point": minus["n_point_columns"],
                    "dark_I": minus["dark_current_t0"],
                    "photon_dark": minus["photon_dark_projection"],
                    "extra_vac": minus["extra_bare_bright_vacuum_maxabs"],
                    "gram_trace": minus["electron_gram_dense"]["trace_real"],
                },
                "a_residual": conv["residual"],
                "elapsed_s": payload["elapsed_s"],
                "out": str(out_json),
            },
            indent=2,
        ),
        flush=True,
    )
    return 0 if payload["all_physics_checks_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())



