#!/usr/bin/env python3
"""H2 cancellation: exact same-pulse symmetry vs dynamical phase. No new large-grid runs."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from q8_m0_classical import k_grid  # noqa: E402
from q8_m0_h2_qualify import dump  # noqa: E402
from q8_m0_model import (  # noqa: E402
    R0,
    build_geometry,
    current_ops,
    h8,
    mirror_matrix_inplane,
    op_P,
    op_c2_nhat,
    op_mirror_a2,
)

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "results/q_line/p1_harmonics_review_20260916"
NK48 = REPO / "results/q_line/m0_p0p1_20260914/p1_A0.8_nk48_nt400"
NK192 = REPO / "results/q_line/p1_h2_kresolved_diag_20260915/p1_A0.8_nk192_nt400_k2"
TAG48 = "p1_A0.8_nk48_nt400"
TAG192 = "p1_A0.8_nk192_nt400_k2"


def inv_partner_index(nk: int) -> np.ndarray:
    i = np.arange(nk)
    p = (-i) % nk
    ii, jj = np.meshgrid(i, i, indexing="ij")
    return (p[ii] * nk + p[jj]).ravel()


def load_h2(dirpath: Path, tag: str, order: int = 2):
    plus = np.load(dirpath / f"run_{tag}_N+1_A+1.npz")
    minus = np.load(dirpath / f"run_{tag}_N-1_A+1.npz")
    orders = np.asarray(plus["k_orders"], dtype=int)
    i = int(np.where(orders == order)[0][0])
    hp = np.asarray(plus["k_harm_total"])[i]
    hm = np.asarray(minus["k_harm_total"])[i]
    kf = np.asarray(plus["kf"], dtype=float)
    w = np.asarray(plus["w_k"], dtype=float)
    nk = int(round(np.sqrt(kf.shape[0])))
    return kf, w, hp, hm, nk


def residual_report(z, z_partner, w) -> dict:
    diff = z - z_partner
    num = float(np.dot(w, np.abs(diff)))
    den = float(np.dot(w, np.abs(z)))
    return {
        "weighted_abs_residual": num,
        "weighted_abs_signal": den,
        "ratio": None if den <= 1e-30 else num / den,
        "max_abs_residual": float(np.max(np.abs(diff))),
        "median_abs_residual": float(np.median(np.abs(diff))),
    }


def static_p_check(geo, nk: int = 24) -> dict:
    from q8_m0_model import op_sewing

    kf, kc, _w = k_grid(geo, nk, shift=0.0)
    pidx = inv_partner_index(nk)
    kc_p = kc[pidx]
    g_cart = kc + kc_p
    taus = np.asarray(geo.tau_orbitals_cart(), dtype=float)[:, :2]
    ph = np.exp(1j * (kc + kc_p) @ taus.T)
    up = op_P()
    h_plus = h8(geo, R0, kc, +1)
    h_m = h8(geo, R0, kc_p, -1)
    # k_p = -k + G, H(k+G)=U_G† H(k) U_G ⇒ H(-k) = U_G H(k_p) U_G†
    h_minusk = np.einsum("ki,kij,kj->kij", ph, h_m, np.conj(ph))
    mapped = np.einsum("ij,kjl,lm->kim", up, h_minusk, up.conj().T)
    h_res = float(np.max(np.abs(h_plus - mapped)))
    j_plus = current_ops(geo, R0, kc, +1)
    j_m = current_ops(geo, R0, kc_p, -1)
    j_minusk = np.einsum("ki,kaij,kj->kaij", ph, j_m, np.conj(ph))
    j_mapped = -np.einsum("ij,kajl,lm->kaim", up, j_minusk, up.conj().T)
    j_res = float(np.max(np.abs(j_plus - j_mapped)))
    return {
        "nk": nk,
        "P_on_H": "U_P H_{-N}(-k) U_P† vs H_N(k), with sewing for folded -k",
        "max_abs_G": float(np.max(np.linalg.norm(g_cart, axis=1))),
        "H_residual_maxabs": h_res,
        "j_rule": "U_P j_{-N}(-k) U_P† vs -j_N(k), sewing included",
        "j_residual_maxabs": j_res,
        "same_pulse_note": (
            "P relates (N,k,A) to (-N,-k,-A). The two-domain experiment uses the same A, "
            "so this exact map does not force Jo(H2) to vanish."
        ),
    }


def static_c2_check(geo, nk: int = 24) -> dict:
    m = mirror_matrix_inplane(geo)
    kf, kc, w = k_grid(geo, nk, shift=0.0)
    k_c2 = -kc @ m.T
    # nearest grid point in cartesian is awkward; use fractional -M in frac coords
    # M acts in Cartesian. Convert k_c2 to frac via geometry inverse later; compare spectra.
    u = op_c2_nhat(geo)
    h = h8(geo, R0, kc, +1)
    h_c2 = h8(geo, R0, k_c2, +1)
    mapped = np.einsum("ij,kjl,lm->kim", u, h_c2, u.conj().T)
    # C2: U H_N(k) U† = H_N(-M k)
    res = float(np.max(np.abs(h - mapped)))
    a_axis = np.array([1.0, 0.0])
    a_image = m @ a_axis
    return {
        "C2_H_residual_maxabs": res,
        "mirror_normal": geo.mirror_normal_inplane,
        "x_hat_mapped_by_M": a_image.tolist(),
        "pulse_along_x_invariant": bool(np.allclose(a_image, a_axis) or np.allclose(a_image, -a_axis)),
        "if_M_flips_x": "C2 or M also maps Ax, hence is not a same-lab-pulse symmetry unless A→±A is allowed",
        "note": "Approximate C2h is not treated as exact. Residual here is a static Hamiltonian check.",
    }


def driven_pairings(tag_dir, tag, label: str) -> dict:
    kf, w, hp, hm, nk = load_h2(tag_dir, tag)
    pidx = inv_partner_index(nk)
    jo = 0.5 * (hp - hm)
    je = 0.5 * (hp + hm)
    out = {"nk": nk, "label": label, "xy": {}}
    for xy, ax in ((0, "x"), (1, "y")):
        z_o = jo[:, xy]
        z_p = hp[:, xy]
        z_m = hm[:, xy]
        out["xy"][ax] = {
            "Jo_k_minus_Jo_inv": residual_report(z_o, z_o[pidx], w),
            "Jo_k_plus_Jo_inv": residual_report(z_o, -z_o[pidx], w),
            "P_sameA_Jp_k_plus_Jm_inv": residual_report(z_p, -z_m[pidx], w),
            "P_sameA_Jp_k_minus_Jm_inv": residual_report(z_p, z_m[pidx], w),
            "sameN_Jp_k_plus_Jp_inv": residual_report(z_p, -z_p[pidx], w),
            "Je_k_minus_Je_inv": residual_report(je[:, xy], je[pidx, xy], w),
        }
    return out


def quadrature_schemes(pair192: dict) -> dict:
    # tiles with high gross from previous k-diag JSON if present
    tile_path = REPO / "results/q_line/p1_h2_kresolved_diag_20260915/nk192_kresolved.json"
    hot = []
    if tile_path.exists():
        rec = json.loads(tile_path.read_text(encoding="utf-8"))
        for cname in ("Jo_x_H2", "Jo_y_H2"):
            tiles = rec["channels"][cname]["tiles"]
            med = float(np.median([t["gross_sheet"] for t in tiles]))
            hot.extend(t["label"] for t in tiles if t["gross_sheet"] > med)
    hot = sorted(set(hot))
    return {
        "do_not_run_this_round": True,
        "scheme_A_belt_refined_inversion_paired": {
            "idea": (
                "Keep full-BZ coverage and Σw=1. In 6×6 fractional tiles whose nk192 gross "
                "exceeds the median, use local spacing 1/384; elsewhere 1/96. Close the set "
                "under k→−k. Quiet tiles are not dropped."
            ),
            "hot_tiles_from_nk192_median_gross": hot,
            "n_hot_unique": len(hot),
            "cost_estimate": (
                "12–20 hot tiles out of 36, density ×16 in those tiles vs 96-grid, "
                "×1 in quiet tiles at 96: roughly 0.3–0.6 of a uniform nk192 pair. "
                "Stop if |z_A - z_192| does not beat the existing Γ vs shift disagreement."
            ),
            "independent_control": "repeat with the complementary half-step shift on the same composite rule",
        },
        "scheme_B_two_shift_error_bar": {
            "idea": (
                "Do not densify. Use already computed Γ-centred and half-step nk192 totals as "
                "two inversion-paired estimates of the same oscillatory integral. Report "
                "(z_Γ+z_shift)/2 and |z_Γ−z_shift| as a complex error bar. This is not a "
                "new k-export."
            ),
            "uses_existing": True,
            "limitation": "does not resolve filament interiors; only bounds the net",
        },
        "not_chosen": "uniform nk256: previous 96→192 already moved more than the signal; densifying the same rule repeats cancellation",
        "stop_rule": "a scheme stops after one pair if it does not reduce |Δz| below the current empirical Γ–shift scale, or if the user reopens wavelength/protocol review",
    }


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    geo = build_geometry()
    rec = {
        "question": "Is average H2 cancellation a same-pulse symmetry identity, or pulse-dependent phase?",
        "static_P": static_p_check(geo),
        "static_C2_M": static_c2_check(geo),
        "mirror_a2_expected": "U_M H_N(k) U_M† = H_{-N}(M k); maps domains and k, and generally the in-plane drive",
        "driven_nk48": driven_pairings(NK48, TAG48, "nk48"),
        "driven_nk192": driven_pairings(NK192, TAG192, "nk192"),
        "quadrature_design": quadrature_schemes({}),
        "interpretation_for_human": (
            "If a static exact map already forces z(k)=−z(partner) for this same pulse, "
            "the residual ratio would be at floating-point noise. A O(1) residual means "
            "the map is not a same-pulse identity. Filament red/blue is still consistent "
            "with dynamical phase. Cursor does not declare the mechanism."
        ),
        "quantum_input": "not_run",
        "photon_output": "not_run",
        "new_drive_protocol": "not_run",
    }
    dump(OUT / "h2_cancellation_symmetry.json", rec)
    print(json.dumps({
        "P_H_res": rec["static_P"]["H_residual_maxabs"],
        "P_j_res": rec["static_P"]["j_residual_maxabs"],
        "C2_H_res": rec["static_C2_M"]["C2_H_residual_maxabs"],
        "x_mapped": rec["static_C2_M"]["x_hat_mapped_by_M"],
        "nk192_Jo_x_k_minus_inv_ratio": rec["driven_nk192"]["xy"]["x"]["Jo_k_minus_Jo_inv"]["ratio"],
        "nk192_P_sameA_ratio": rec["driven_nk192"]["xy"]["x"]["P_sameA_Jp_k_plus_Jm_inv"]["ratio"],
    }, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
