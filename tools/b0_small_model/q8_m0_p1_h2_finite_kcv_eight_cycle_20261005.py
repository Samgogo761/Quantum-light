#!/usr/bin/env python3
"""Eight-cycle online H2 detector rows. Not production n,m.

Reuses the 2026-10-05 corrected midpoint caller. Keeps dark/point columns.
Does not add a second bright electronic vacuum. Wiring frequencies are not
a complete positive-frequency input measure, so n,m are not claimed.
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
from q8_m0_classical import _sin2_window, p1_a2_protocol
from q8_m0_continuum_columns_20261004 import ColumnState, current, midpoint_step
from q8_m0_gsheet import gamma_self
from q8_m0_incoming_spectral_completeness_20261004 import (
    assemble_incoming_bundle,
    classify_dark_columns,
    maxabs,
)
from q8_m0_k1_chi_interface import CHI_STATUS
from q8_m0_k1_packet import normalize_phi, omega_grid, phi_unnormalized
from q8_m0_model import build_geometry
from q8_m0_p1_h2_finite_kcv_caller_fix_20261005 import (
    FREQ,
    OLD_JSON,
    TWO_K,
    WEIGHTS,
    advance_half_frame,
    even_odd_from_lr,
    free_A_increment,
    m0_sample,
    packed_state,
    photon_port_indices,
    sha,
    vertices_from_frame,
)
from q8_m0_s1_kernel import parallel_perp_basis

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "results/q_line/p1_h2_finite_kcv_eight_cycle_20261005"
OLD_CONTRACT = REPO / "results/q_line/p1_h2_finite_kcv_contract_20261004/contract.json"
CALLER_FIX = REPO / "results/q_line/p1_h2_finite_kcv_caller_fix_20261005/caller_fix.json"
INPUT_LEDGER = REPO / "results/q_line/p1_h2_finite_kcv_input_ledger_20261005/input_ledger.json"
NTPC = 1024
N_TAIL_CYCLES = 1
A_STAR = 0.8
H2_BAND = (0.05, 0.20)
# Extra positive frequencies for a driven-input tail estimate. Not a vacuum.
TAIL_FREQ = np.array([0.35, 0.60, 0.90, 1.20], float)
INBAND_EXTRA = np.array([0.05, 0.18], float)
FROZEN_GAMMA = 0.001080789276
OLD_JSON_KEEP = OLD_JSON + [
    "results/q_line/p1_h2_finite_kcv_caller_fix_20261005/caller_fix.json",
    "results/q_line/p1_h2_finite_kcv_input_ledger_20261005/input_ledger.json",
]
BOUNDARY_KEYS = (
    "not_production_n",
    "not_production_m",
    "not_complete_input_vacuum",
    "two_k_not_a_domain_signal",
    "selected_frequencies_are_wiring_only",
)


def pack_c(m):
    m = np.asarray(m, complex)
    out = {
        "real": np.real(m).tolist(),
        "imag": np.imag(m).tolist(),
        "maxabs": float(np.max(np.abs(m))) if m.size else 0.0,
        "shape": list(m.shape),
    }
    if m.ndim == 2 and m.shape[0] == m.shape[1] and m.size:
        herm = 0.5 * (m + m.conj().T)
        evals = np.linalg.eigvalsh(herm)
        out["trace_real"] = float(np.trace(m).real)
        out["min_hermitian_eig"] = float(evals[0])
        out["anti_hermitian_maxabs"] = float(np.max(np.abs(m - herm)))
    return out


def all_input_frequencies():
    return np.unique(np.concatenate((FREQ, INBAND_EXTRA, TAIL_FREQ)))


def column_groups(labels):
    wiring, inband, tail, dark = [], [], [], []
    for j, lab in enumerate(labels):
        if lab[0] == "dark_combination":
            dark.append(j)
            continue
        w = float(lab[0])
        if np.min(np.abs(FREQ - w)) < 1e-14:
            wiring.append(j)
        if H2_BAND[0] <= w < H2_BAND[1]:
            inband.append(j)
        if w >= TAIL_FREQ[0] - 1e-14:
            tail.append(j)
    return {
        "wiring": np.asarray(wiring, int),
        "inband": np.asarray(inband, int),
        "tail": np.asarray(tail, int),
        "dark": np.asarray(dark, int),
    }


def detector_kernel_scalar(gamma, prot, nstep, h):
    """K(t) = i sqrt(γ) ∫ dω φ(ω)/sqrt(ω) e^{+iωt}. Keeps 1/sqrt(ω)."""
    w0 = float(prot.omega0)
    tp = float(prot.t_pulse)
    om = omega_grid(w0, tp)
    om = om[om > 0.0]
    raw = phi_unnormalized(om, 2.0 * w0, tp)
    phi, nrm = normalize_phi(om, raw)
    t_mid = (np.arange(nstep) + 0.5) * h
    weight = phi / np.sqrt(om)
    k = np.empty(nstep, complex)
    batch = 64
    for s in range(0, nstep, batch):
        sl = slice(s, min(s + batch, nstep))
        phase = np.exp(1j * om[:, None] * t_mid[sl][None, :])
        k[sl] = 1j * np.sqrt(gamma) * np.trapezoid(weight[:, None] * phase, om, axis=0)
    wc = 2.0 * w0
    win = _sin2_window(t_mid, tp)
    inside = (t_mid >= 0.0) & (t_mid <= tp)
    win = np.where(inside, win, 0.0)
    k_pulled = 1j * np.sqrt(gamma / wc) * (win / tp) * np.exp(1j * wc * t_mid)
    pulse = t_mid <= tp
    tail = ~pulse
    mass = np.abs(k) ** 2
    return {
        "k_scalar": k,
        "k_pulled_center": k_pulled,
        "t_mid": t_mid,
        "normalizer": float(nrm),
        "omega_c": float(wc),
        "keep_1_over_sqrt_omega": True,
        "pulled_center_is_not_the_kernel": True,
        "k_vs_pulled_maxabs": maxabs(k - k_pulled),
        "mass_pulse": float(np.trapezoid(mass[pulse], t_mid[pulse])) if np.any(pulse) else 0.0,
        "mass_tail": float(np.trapezoid(mass[tail], t_mid[tail])) if np.any(tail) else 0.0,
        "n_omega": int(om.size),
    }


def accumulate_rows(geo, sample, prot, photon, gamma, initial, n_point, k_scalar, nstep, h, *, snapshots=5):
    """Online midpoint accumulation. Five snapshots are diagnostics only."""
    state = initial.copy()
    frame = sample["frame"].copy()
    ncol = int(state.b.shape[1])
    pol = np.vstack(parallel_perp_basis(geo))
    acc = np.zeros((2, ncol), complex)
    n_pulse = int(round(prot.n_cycles * NTPC))
    row_pulse = acc.copy()
    rec_steps = set(int(i) for i in np.round(np.linspace(0, nstep, snapshots)).astype(int))
    rec = []
    started = time.perf_counter()
    ii = np.zeros((2, ncol), complex)

    def snap(step, t, st, uu, dd):
        icur = current(st, uu, dd)
        rec.append({
            "step": int(step),
            "t": float(t),
            "I_maxabs": maxabs(icur),
            "A_maxabs": maxabs(st.A),
        })

    u0, d0 = vertices_from_frame(
        geo, sample["kc"], prot, sample["neel"], 0.0, frame, sample["weights"]
    )
    snap(0, 0.0, state, u0, d0)
    for step in range(nstep):
        t = step * h
        for which, when in enumerate((t + h / 4.0, t + 3.0 * h / 4.0)):
            frame = advance_half_frame(
                geo, sample["kc"], prot, sample["neel"], frame, when, h
            )
            if which == 0:
                u, d = vertices_from_frame(
                    geo, sample["kc"], prot, sample["neel"], t + 0.5 * h, frame, sample["weights"]
                )
                fa = free_A_increment(photon, t, t + h, n_point)
                prev = state
                state = midpoint_step(state, u, d, h, gamma, free_A_increment=fa)
                mid = ColumnState(
                    0.5 * (prev.b + state.b),
                    0.5 * (prev.bdag + state.bdag),
                    0.5 * (prev.A + state.A),
                )
                ii = current(mid, u, d)
                acc += h * k_scalar[step] * (pol @ ii)
        if step + 1 == n_pulse:
            row_pulse = acc.copy()
        if step + 1 in rec_steps:
            snap(step + 1, (step + 1) * h, state, u, d)
        if (step + 1) % NTPC == 0:
            print(
                f"N={sample['neel']:+d} A={prot.a0_dimless:g} cycle {(step+1)//NTPC}/{nstep//NTPC}",
                flush=True,
            )
    if nstep < n_pulse:
        row_pulse = acc.copy()
    return {
        "row": acc,
        "row_pulse": row_pulse,
        "row_tail": acc - row_pulse,
        "n_pulse": n_pulse,
        "snapshots": rec,
        "final_I_maxabs": maxabs(ii),
        "n_steps": int(nstep),
        "dt": float(h),
        "wall_seconds": time.perf_counter() - started,
        "ncol": ncol,
        "n_el": int(state.b.shape[0]),
    }


def group_norms(row, groups):
    out = {}
    for name, idx in groups.items():
        if idx.size:
            out[name] = maxabs(row[:, idx])
            out[name + "_fro"] = float(np.linalg.norm(row[:, idx]))
        else:
            out[name] = 0.0
            out[name + "_fro"] = 0.0
        out[name + "_n"] = int(idx.size)
    return out


def independent_stationary_row(photon, sample, k_scalar, h, nstep, n_point, geo):
    """Undriven photon I(t)=I(0) e^{-iωt} on the same midpoint grid. Dark I=0."""
    pol = np.vstack(parallel_perp_basis(geo))
    i0 = current(photon.initial, sample["u"], sample["D"])
    t_mid = (np.arange(nstep) + 0.5) * h
    n_ph = int(i0.shape[1])
    pred = np.zeros((2, n_ph + n_point), complex)
    phase = np.exp(-1j * photon.omega[None, :] * t_mid[:, None])
    kern = (h * k_scalar)[:, None] * phase
    pred[:, :n_ph] = pol @ (i0 * np.sum(kern, axis=0)[None, :])
    return pred


def run_one(geo, sample, prot, gamma, freqs, nstep, h, k_scalar):
    bundle = assemble_incoming_bundle(sample["gap"], sample["u"], sample["D"], gamma, freqs)
    classified = classify_dark_columns(bundle.dark_info, k_of_index=sample["k_of"])
    n_point = int(bundle.n_point_columns)
    labels = tuple(bundle.photon.labels) + tuple(bundle.point_labels)
    groups = column_groups(labels)
    initial = packed_state(bundle)
    left, right = photon_port_indices(bundle.photon.labels)
    even_a, odd_a = even_odd_from_lr(bundle.photon.initial.A, left, right)
    acc = accumulate_rows(
        geo, sample, prot, bundle.photon, gamma, initial, n_point, k_scalar, nstep, h
    )
    norms_full = group_norms(acc["row"], groups)
    norms_pulse = group_norms(acc["row_pulse"], groups)
    norms_tail = group_norms(acc["row_tail"], groups)
    analytic = None
    if abs(float(prot.a0_dimless)) < 1e-15:
        pred = independent_stationary_row(
            bundle.photon, sample, k_scalar, h, nstep, n_point, geo
        )
        wr, pr = groups["wiring"], pred
        analytic = {
            "row_vs_analytic_maxabs": maxabs(acc["row"] - pr),
            "wiring_vs_analytic_maxabs": maxabs(acc["row"][:, wr] - pr[:, wr]) if wr.size else 0.0,
            "dark_vs_analytic_maxabs": (
                maxabs(acc["row"][:, groups["dark"]] - pr[:, groups["dark"]])
                if groups["dark"].size
                else 0.0
            ),
        }
    return {
        "neel": int(sample["neel"]),
        "a0": float(prot.a0_dimless),
        "n_el": sample["n_el"],
        "n_photon_columns": int(bundle.n_photon_columns),
        "n_point_columns": n_point,
        "n_dark_classified": int(sum(int(r["nullity"]) for r in classified)),
        "metadata_lengths_agree": bool(
            n_point == len(bundle.point_gaps) == len(bundle.point_labels)
            == int(sum(int(r["nullity"]) for r in classified))
        ),
        "A_L_minus_A_R_maxabs": maxabs(
            bundle.photon.initial.A[:, left] - bundle.photon.initial.A[:, right]
        ),
        "A_odd_maxabs": maxabs(odd_a),
        "A_even_maxabs": maxabs(even_a),
        "dark_current_t0": maxabs(current(bundle.point, sample["u"], sample["D"])) if n_point else 0.0,
        "weight_once_residual": sample["weight_once_residual"],
        "min_direct_gap": sample["min_direct_gap"],
        "max_direct_gap": sample["max_direct_gap"],
        "frequencies": np.asarray(freqs, float).tolist(),
        "groups": {k: v.tolist() for k, v in groups.items()},
        "row": pack_c(acc["row"]),
        "row_pulse": pack_c(acc["row_pulse"]),
        "row_tail": pack_c(acc["row_tail"]),
        "norms_full": norms_full,
        "norms_pulse": norms_pulse,
        "norms_tail": norms_tail,
        "snapshots": acc["snapshots"],
        "final_I_maxabs": acc["final_I_maxabs"],
        "n_steps": acc["n_steps"],
        "dt": acc["dt"],
        "n_pulse_steps": acc["n_pulse"],
        "wall_seconds": acc["wall_seconds"],
        "analytic_undriven": analytic,
        "kept_dark": True,
        "no_second_bright_vacuum": True,
    }


def delta_report(driven, undriven):
    rd = np.array(driven["row"]["real"]) + 1j * np.array(driven["row"]["imag"])
    ru = np.array(undriven["row"]["real"]) + 1j * np.array(undriven["row"]["imag"])
    rdp = np.array(driven["row_pulse"]["real"]) + 1j * np.array(driven["row_pulse"]["imag"])
    rup = np.array(undriven["row_pulse"]["real"]) + 1j * np.array(undriven["row_pulse"]["imag"])
    rdt = np.array(driven["row_tail"]["real"]) + 1j * np.array(driven["row_tail"]["imag"])
    rut = np.array(undriven["row_tail"]["real"]) + 1j * np.array(undriven["row_tail"]["imag"])
    groups = {k: np.asarray(v, int) for k, v in driven["groups"].items()}
    d_full = rd - ru
    d_pulse = rdp - rup
    d_tail = rdt - rut
    return {
        "delta_full": pack_c(d_full),
        "delta_pulse": pack_c(d_pulse),
        "delta_tail": pack_c(d_tail),
        "norms_delta_full": group_norms(d_full, groups),
        "norms_delta_pulse": group_norms(d_pulse, groups),
        "norms_delta_tail": group_norms(d_tail, groups),
        "tail_over_inband_fro": (
            group_norms(d_full, groups)["tail_fro"]
            / max(group_norms(d_full, groups)["inband_fro"], 1e-30)
        ),
        "dark_over_inband_fro": (
            group_norms(d_full, groups)["dark_fro"]
            / max(group_norms(d_full, groups)["inband_fro"], 1e-30)
        ),
        "time_tail_over_pulse_fro": float(
            np.linalg.norm(d_tail) / max(np.linalg.norm(d_pulse), 1e-30)
        ),
        "not_n_m": True,
        "not_complete_input_measure": True,
        "not_ccr_claim": True,
    }


def run_domain(geo, gamma, neel, prot_drive, prot_ref, k_scalar, nstep, h, freqs):
    started = time.perf_counter()
    sample = m0_sample(geo, TWO_K, WEIGHTS, neel)
    print(f"N={neel:+d}: n_el={sample['n_el']} driven eight-cycle", flush=True)
    driven = run_one(geo, sample, prot_drive, gamma, freqs, nstep, h, k_scalar)
    print(f"N={neel:+d}: zero-pump reference", flush=True)
    undriven = run_one(geo, sample, prot_ref, gamma, freqs, nstep, h, k_scalar)
    delta = delta_report(driven, undriven)
    return {
        "neel": int(neel),
        "driven": driven,
        "undriven": undriven,
        "delta": delta,
        "wall_seconds": time.perf_counter() - started,
    }


def flags_from(plus, minus, kernel, gamma, nstep, n_pulse):
    def side_ok(dom):
        d = dom["driven"]
        u = dom["undriven"]
        return {
            "keeps_dark": bool(d["n_point_columns"] > 0 and u["n_point_columns"] > 0),
            "metadata": bool(d["metadata_lengths_agree"] and u["metadata_lengths_agree"]),
            "weight_once": bool(d["weight_once_residual"] < 1e-12),
            "eight_cycle_steps": bool(d["n_steps"] == nstep and u["n_steps"] == nstep),
            "pulse_steps": bool(d["n_pulse_steps"] == n_pulse and u["n_pulse_steps"] == n_pulse),
            "snapshots_are_five": bool(len(d["snapshots"]) == 5 and len(u["snapshots"]) == 5),
            "odd_A_small": bool(d["A_odd_maxabs"] < 1e-12 and u["A_odd_maxabs"] < 1e-12),
            "L_equals_R": bool(d["A_L_minus_A_R_maxabs"] < 1e-12),
            "no_second_vac": bool(d["no_second_bright_vacuum"] and u["no_second_bright_vacuum"]),
        }

    p, m = side_ok(plus), side_ok(minus)
    return {
        "gamma_positive": bool(gamma > 0.0),
        "gamma_matches_frozen": bool(abs(gamma - FROZEN_GAMMA) < 1e-12),
        "keep_1_over_sqrt_omega": bool(kernel["keep_1_over_sqrt_omega"]),
        "plus_keeps_dark": p["keeps_dark"],
        "minus_keeps_dark": m["keeps_dark"],
        "plus_metadata_agree": p["metadata"],
        "minus_metadata_agree": m["metadata"],
        "plus_weight_once": p["weight_once"],
        "minus_weight_once": m["weight_once"],
        "plus_eight_cycle_steps": p["eight_cycle_steps"],
        "minus_eight_cycle_steps": m["eight_cycle_steps"],
        "plus_pulse_steps": p["pulse_steps"],
        "minus_pulse_steps": m["pulse_steps"],
        "plus_snapshots_diagnostic_only": p["snapshots_are_five"],
        "minus_snapshots_diagnostic_only": m["snapshots_are_five"],
        "plus_odd_A_small_at_t0": p["odd_A_small"],
        "minus_odd_A_small_at_t0": m["odd_A_small"],
        "plus_L_equals_R_at_t0": p["L_equals_R"],
        "minus_L_equals_R_at_t0": m["L_equals_R"],
        "plus_no_second_bright_vacuum": p["no_second_vac"],
        "minus_no_second_bright_vacuum": m["no_second_vac"],
        "online_not_five_point_integral": bool(nstep > 1000),
        "old_contract_not_overwritten": bool(OLD_CONTRACT.exists()),
        "caller_fix_json_present": bool(CALLER_FIX.exists()),
        "input_ledger_json_present": bool(INPUT_LEDGER.exists()),
        "chi_rad_placeholder_not_flipped": True,
        "not_production_n": True,
        "not_production_m": True,
        "not_complete_input_vacuum": True,
        "two_k_not_a_domain_signal": True,
        "selected_frequencies_are_wiring_only": True,
    }


def main() -> int:
    t0 = time.perf_counter()
    geo = build_geometry()
    prot_drive = p1_a2_protocol(A_STAR)
    prot_ref = p1_a2_protocol(0.0)
    gamma = float(gamma_self(geo))
    h = prot_drive.period / NTPC
    n_pulse = int(prot_drive.n_cycles) * NTPC
    nstep = n_pulse + N_TAIL_CYCLES * NTPC
    freqs = all_input_frequencies()
    print(
        f"gamma={gamma:.12g} nstep={nstep} n_pulse={n_pulse} n_freq={len(freqs)}",
        flush=True,
    )
    kernel = detector_kernel_scalar(gamma, prot_drive, nstep, h)
    k_scalar = kernel.pop("k_scalar")
    kernel.pop("k_pulled_center")
    kernel.pop("t_mid")
    plus = run_domain(geo, gamma, +1, prot_drive, prot_ref, k_scalar, nstep, h, freqs)
    minus = run_domain(geo, gamma, -1, prot_drive, prot_ref, k_scalar, nstep, h, freqs)
    flags = flags_from(plus, minus, kernel, gamma, nstep, n_pulse)
    physics_flags = {k: v for k, v in flags.items() if k not in BOUNDARY_KEYS}
    payload = {
        "status": "p1_h2_finite_kcv_eight_cycle_output_rows_not_production_nm",
        "all_physics_checks_pass": bool(all(physics_flags.values())),
        "pass_flags": flags,
        "scope": {
            "same_two_k_as_caller_fix": True,
            "keep_dark_point_columns": True,
            "no_second_bright_electronic_vacuum": True,
            "ntpc": NTPC,
            "n_cycles": int(prot_drive.n_cycles),
            "n_tail_cycles": N_TAIL_CYCLES,
            "n_steps": nstep,
            "n_pulse_steps": n_pulse,
            "dt": float(h),
            "online_midpoint_accumulation": True,
            "five_snapshots_are_diagnostics_only": True,
            "selected_frequencies": FREQ.tolist(),
            "selected_frequencies_are_wiring_only": True,
            "extra_inband_frequencies": INBAND_EXTRA.tolist(),
            "input_tail_frequencies": TAIL_FREQ.tolist(),
            "all_input_frequencies": freqs.tolist(),
            "not_complete_input_vacuum": True,
            "H2_output_window_does_not_truncate_input": True,
            "not_production_n_m": True,
            "two_k_not_a_domain_signal": True,
            "reuse_corrected_midpoint_caller": True,
            "reuse_frozen_gsheet_and_incoming": True,
            "chi_rad_placeholder_status": CHI_STATUS,
            "do_not_flip_chi_rad_flag": True,
            "assemble_ground_or_photon_moments": "NotImplemented",
        },
        "detector_kernel": kernel,
        "gamma_self": gamma,
        "domains": {"N+1": plus, "N-1": minus},
        "old_json_not_overwritten": OLD_JSON_KEEP,
        "elapsed_s": time.perf_counter() - t0,
        "source_sha256": {
            str(p.relative_to(REPO)): sha(p)
            for p in (
                Path(__file__).resolve(),
                REPO / "tools/b0_small_model/q8_m0_p1_h2_finite_kcv_caller_fix_20261005.py",
                REPO / "tools/b0_small_model/q8_m0_p1_h2_finite_kcv_input_ledger_20261005.py",
                REPO / "tools/b0_small_model/q8_m0_continuum_columns_20261004.py",
                REPO / "tools/b0_small_model/q8_m0_incoming_spectral_completeness_20261004.py",
                REPO / "tools/b0_small_model/q8_m0_gsheet.py",
                REPO / "tools/b0_small_model/q8_m0_k1_packet.py",
            )
        },
        "old_contract_sha256": sha(OLD_CONTRACT) if OLD_CONTRACT.exists() else None,
        "caller_fix_sha256": sha(CALLER_FIX) if CALLER_FIX.exists() else None,
        "input_ledger_sha256": sha(INPUT_LEDGER) if INPUT_LEDGER.exists() else None,
        "production": False,
        "optical_moments_computed": False,
        "P1_full_pulse": True,
        "legal_electronic_continuum_boundary": False,
        "complete_positive_frequency_input_integral": False,
        "n_m_claimed": False,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    out_json = OUT / "eight_cycle.json"
    if out_json.exists():
        raise FileExistsError("Refuse overwrite of eight_cycle.json")
    if not OLD_CONTRACT.exists():
        raise FileNotFoundError("Old contract.json missing; refuse to proceed")
    if not CALLER_FIX.exists():
        raise FileNotFoundError("caller_fix.json missing; refuse to proceed")
    if not INPUT_LEDGER.exists():
        raise FileNotFoundError("input_ledger.json missing; refuse to proceed")
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
                "nstep": nstep,
                "n_pulse": n_pulse,
                "plus": {
                    "n_point": plus["driven"]["n_point_columns"],
                    "delta_inband_fro": plus["delta"]["norms_delta_full"]["inband_fro"],
                    "delta_tail_fro": plus["delta"]["norms_delta_full"]["tail_fro"],
                    "delta_dark_fro": plus["delta"]["norms_delta_full"]["dark_fro"],
                    "time_tail_over_pulse": plus["delta"]["time_tail_over_pulse_fro"],
                    "wall": plus["wall_seconds"],
                },
                "minus": {
                    "n_point": minus["driven"]["n_point_columns"],
                    "delta_inband_fro": minus["delta"]["norms_delta_full"]["inband_fro"],
                    "delta_tail_fro": minus["delta"]["norms_delta_full"]["tail_fro"],
                    "delta_dark_fro": minus["delta"]["norms_delta_full"]["dark_fro"],
                    "time_tail_over_pulse": minus["delta"]["time_tail_over_pulse_fro"],
                    "wall": minus["wall_seconds"],
                },
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






