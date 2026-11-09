#!/usr/bin/env python3
"""a2 linear-drive selection rules (static). No new propagation.

Records how P, M_a2, C2(n̂) act on (N,k,A,J) and the implied
mean-current / CJJ tables. Also a cheap Δ–gap sidecar (not the main line).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from q8_m0_cjj import dump  # noqa: E402
from q8_m0_classical import k_grid  # noqa: E402
from q8_m0_model import (  # noqa: E402
    M0Params,
    R0,
    build_geometry,
    current_ops,
    h8,
    mirror_matrix_inplane,
    op_c2_nhat,
    op_mirror_a2,
    op_P,
)

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "results/q_line/a2_selection_20260917"
NK = 24


def unit(v) -> np.ndarray:
    v = np.asarray(v, dtype=float)[:2]
    return v / np.linalg.norm(v)


def angle_deg(v, w) -> float:
    v, w = unit(v), unit(w)
    return float(np.degrees(np.arccos(np.clip(np.dot(v, w), -1.0, 1.0))))


def residual_h(u, h_src, h_tgt) -> float:
    mapped = np.einsum("ij,kjl,lm->kim", u, h_src, u.conj().T)
    return float(np.max(np.abs(h_tgt - mapped)))


def residual_j_polar(u, j_src, j_tgt, r2) -> float:
    """U j_src_a U† vs Σ_b R_{ab} j_tgt_b, R 2×2 on Cartesian components."""
    mapped = np.einsum("ij,kajl,lm->kaim", u, j_src, u.conj().T)
    rot = np.einsum("ab,kbij->kaij", r2, j_tgt)
    return float(np.max(np.abs(mapped - rot)))


def static_maps(geo, nk: int = NK) -> dict:
    m = mirror_matrix_inplane(geo)
    r_c2 = -m
    a2 = unit(geo.a2_cart)
    nh = unit(geo.mirror_normal_inplane)
    xh = np.array([1.0, 0.0])
    yh = np.array([0.0, 1.0])
    kf, kc, _w = k_grid(geo, nk, shift=0.0)
    k_m = kc @ m.T
    k_c2 = kc @ r_c2.T
    um = op_mirror_a2(geo)
    uc2 = op_c2_nhat(geo)
    up = op_P()
    h_p = h8(geo, R0, kc, +1)
    h_m = h8(geo, R0, kc, -1)
    j_p = current_ops(geo, R0, kc, +1)
    j_m = current_ops(geo, R0, kc, -1)
    j_p_mk = current_ops(geo, R0, k_m, +1)
    j_m_mk = current_ops(geo, R0, k_m, -1)
    h_m_mk = h8(geo, R0, k_m, -1)
    h_p_c2 = h8(geo, R0, k_c2, +1)
    j_p_c2 = current_ops(geo, R0, k_c2, +1)
    return {
        "nk": nk,
        "a2_cart_unit": a2.tolist(),
        "n_hat_unit": nh.tolist(),
        "angle_a2_from_plus_x_deg": float(np.degrees(np.arctan2(a2[1], a2[0]))),
        "angle_n_from_plus_x_deg": float(np.degrees(np.arctan2(nh[1], nh[0]))),
        "angle_x_to_a2_deg": angle_deg(xh, a2),
        "angle_x_to_n_deg": angle_deg(xh, nh),
        "M_a2_minus_a2_abs": float(np.linalg.norm(m @ a2 - a2)),
        "M_n_plus_n_abs": float(np.linalg.norm(m @ nh + nh)),
        "C2_a2_plus_a2_abs": float(np.linalg.norm(r_c2 @ a2 + a2)),
        "C2_n_minus_n_abs": float(np.linalg.norm(r_c2 @ nh - nh)),
        "M_x_equals_pm_x": bool(np.allclose(m @ xh, xh, atol=1e-8) or np.allclose(m @ xh, -xh, atol=1e-8)),
        "C2_x_equals_pm_x": bool(
            np.allclose(r_c2 @ xh, xh, atol=1e-8) or np.allclose(r_c2 @ xh, -xh, atol=1e-8)
        ),
        "same_pulse_if_A_parallel_a2": {
            "M": "MA=A, and M maps N→−N, so same-lab-pulse two-domain map",
            "C2": "C2 A=−A, same N: field reversal, not same pulse",
            "P": "P maps (N,A)→(−N,−A): never same pulse",
        },
        "static_residuals": {
            "M_H_N_k_vs_H_mN_Mk": residual_h(um, h8(geo, R0, k_m, -1), h_p),
            "M_j_polar_N_vs_mN": residual_j_polar(um, j_p, j_m_mk, m),
            "C2_H_N_k_vs_H_N_mMk": residual_h(uc2, h_p_c2, h_p),
            "C2_j_polar_sameN": residual_j_polar(uc2, j_p, j_p_c2, r_c2),
            "note": (
                "Residuals ~3e-5 are the known approximate C2h of the production metric. "
                "They are not a relative error bar on weak Jo or on CJJ d_o."
            ),
        },
        "P_exact_to_1e14": "see static_p_check with sewing; not repeated as a2-specific",
        "x_hat": xh.tolist(),
        "y_hat": yh.tolist(),
        "M_on_xy": {"x": (m @ xh).tolist(), "y": (m @ yh).tolist()},
        "unused_P_matrix_ok": bool(np.allclose(up.conj().T @ up, np.eye(8))),
        "j_p_mk_max": float(np.max(np.abs(j_p_mk))),
        "h_m_mk_max": float(np.max(np.abs(h_m_mk))),
        "h_m_ref_max": float(np.max(np.abs(h_m))),
    }


def selection_tables() -> dict:
    """If M were exact and A(t) stays parallel to a2 for all t."""
    return {
        "assumption": "exact vertical mirror M_a2 containing a2 and z; A(t)=A_parallel(t) a2-hat; finite envelope still OK because MA(t)=A(t) at every t",
        "not_assumed": "strict half-cycle translation; exact C2h of the production lattice; harmonic parity of a finite 8-cycle pulse",
        "mean_current": {
            "Je_parallel": "allowed (Néel-even, field-odd → mostly odd harmonics)",
            "Jo_parallel": "forbidden (would be Néel-odd along the drive)",
            "Je_perp": "forbidden (Néel-even transverse)",
            "Jo_perp": "allowed (Néel-odd, field-even → mostly even harmonics)",
            "plain_language": (
                "The common (two-domain average) current must lie along the drive. "
                "The magnetic-domain difference must lie perpendicular to the drive. "
                "Even-harmonic magnetic readout, if it exists, lives in the n-hat channel, "
                "not in the parallel channel we have been using as Cartesian x."
            ),
        },
        "CJJ": {
            "c_parallel_parallel": "Néel-even (two even currents, or two minuses on two odd currents do not apply here: both ∥ are even)",
            "c_perp_perp": "Néel-even (two perpendicular currents, each Néel-odd, two minus signs cancel)",
            "c_parallel_perp": "Néel-odd (exactly one minus from the perpendicular current)",
            "plain_language": (
                "Diagonal same-polarization source kernels d_nn cannot tell +N from −N "
                "if M is exact. The domain-odd CJJ is the mixed-polarization kernel "
                "c_∥⊥, e.g. a parallel odd-harmonic window with a perpendicular even-harmonic window "
                "(H1_∥–H2_⊥ or H3_∥–H2_⊥). This is the opposite of looking at d_{2x,2x}."
            ),
            "natural_windows": ["H1_parallel with H2_perp", "H3_parallel with H2_perp", "H2_perp with H1_parallel"],
        },
        "why_x_drive_is_dirty": (
            "Cartesian x is not an eigenvector of M, so each of Jx, Jy mixes ∥ and ⊥. "
            "Then d_{2x,2x} mixes even and odd Néel channels and the small leftover is a cancellation."
        ),
        "production_metric": (
            "M is approximate (H residual ~3e-5). Forbidden channels are not guaranteed machine-zero. "
            "The table is a design rule for a future a2-drive protocol, not a claim that a2 jobs have been run."
        ),
        "do_not_run_a2_propagation": True,
    }


def delta_gap_scan(geo, nk: int = 48) -> dict:
    deltas = (0.8, 0.6, 0.4, 0.2, 0.1, 0.0)
    rows = []
    kf, kc, _w = k_grid(geo, nk, shift=0.0)
    omega = 0.05
    for dlt in deltas:
        p = M0Params(delta=float(dlt), label=f"D{dlt}")
        e = np.linalg.eigvalsh(h8(geo, p, kc, +1))
        ev, ec = e[:, 3], e[:, 4]
        eg_dir = float(np.min(ec - ev))
        eg_ind = float(np.min(ec) - np.max(ev))
        rows.append({
            "delta": float(dlt),
            "eg_dir": eg_dir,
            "eg_ind": eg_ind,
            "hbar_omega_over_eg_dir": None if abs(eg_dir) < 1e-15 else omega / eg_dir,
            "two_photon_over_eg_dir": None if abs(eg_dir) < 1e-15 else 2.0 * omega / eg_dir,
            "H_plus_equals_H_minus": bool(dlt == 0.0),
        })
    return {
        "nk": nk,
        "omega_P1": omega,
        "role": "auxiliary_not_main_line",
        "rows": rows,
        "read": (
            "P1 stays in the two-photon-near-threshold window only near R0 Δ=0.8. "
            "By Δ=0 the same pulse is deeply sub-gap. Do not treat a future Δ=0 CJJ "
            "as an equal-resonance magnetic control."
        ),
    }


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    geo = build_geometry()
    rec = {
        "protocol": "derivation_and_static_only",
        "new_polarization_jobs": "not_run",
        "delta0_cjj": "not_run",
        "quantum_input": "not_run",
        "photon_output": "not_run",
        "K_mode_frozen": False,
        "maps": static_maps(geo),
        "selection": selection_tables(),
        "delta_gap_scan": delta_gap_scan(geo),
    }
    dump(OUT / "a2_selection.json", rec)
    dump(OUT / "delta_gap_scan.json", rec["delta_gap_scan"])
    print(json.dumps({
        "wrote": str(OUT / "a2_selection.json"),
        "M_H": rec["maps"]["static_residuals"]["M_H_N_k_vs_H_mN_Mk"],
        "M_j": rec["maps"]["static_residuals"]["M_j_polar_N_vs_mN"],
        "C2_H": rec["maps"]["static_residuals"]["C2_H_N_k_vs_H_N_mMk"],
        "angle_x_to_a2": rec["maps"]["angle_x_to_a2_deg"],
        "M_a2_fix": rec["maps"]["M_a2_minus_a2_abs"],
        "gaps": [(r["delta"], r["eg_dir"], r["hbar_omega_over_eg_dir"]) for r in rec["delta_gap_scan"]["rows"]],
    }, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
