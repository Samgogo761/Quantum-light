#!/usr/bin/env python3
"""Gates for the static T_perp audit. No tb.dat, no HHG."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from q8_tperp_static_audit import (  # noqa: E402
    CANDIDATE_META,
    DELTA,
    INV_C,
    SCF,
    assemble_h8,
    build_bond_table,
    cr_sites,
    frobenius,
    honeycomb_h,
    integer_R,
    inversion_pairing_proposal,
    k_cart_from_frac,
    named_tab,
    pairing_symmetry_compare,
    parse_scf,
    sample_kpts,
    sewing_check,
    split_ab,
    static_for_candidate,
    tau_y_B_residual_matrix,
    verify_delta_identity,
    wrap_frac,
)


def require(cond: bool, msg: str) -> None:
    if not cond:
        raise RuntimeError(msg)


def test_r_identity() -> None:
    lat, atoms = parse_scf(SCF)
    sites = cr_sites(atoms)
    worst = 0.0
    for top in ("tA", "tB"):
        for bot in ("bA", "bB"):
            rvec = integer_R(sites[top], sites[bot])
            err = verify_delta_identity(lat, sites[top], sites[bot], rvec)
            worst = max(worst, err)
            recon = rvec.astype(float) + sites[bot] - sites[top]
            require(np.max(np.abs(recon - wrap_frac(sites[bot] - sites[top]))) < 1.0e-12, "delta identity")
    require(worst < 1.0e-12, f"R identity {worst}")
    # the previously mis-tagged tA->bB nearest image
    r_ab = integer_R(sites["tA"], sites["bB"])
    require(int(r_ab[1]) == 1, f"tA->bB should be R=(0,1,0), got {r_ab.tolist()}")
    require(int(r_ab[0]) == 0 and int(r_ab[2]) == 0, r_ab.tolist())


def test_v_equals_two_tauy_b() -> None:
    lat, atoms = parse_scf(SCF)
    bonds = build_bond_table(lat, cr_sites(atoms))
    recip = np.array(bonds["reciprocal_2pi_inv_aT"])
    kpts = sample_kpts(recip)
    for name in CANDIDATE_META:
        if name == "C3_EFF_NOT_BUILT":
            continue
        rec = static_for_candidate(name, bonds, kpts)
        require(rec["max_V_minus_2tauyB_F"] < 1.0e-10, f"{name} algebra {rec['max_V_minus_2tauyB_F']}")
        if name in ("C1_SIGMA0", "C2_SIGMA_X", "C3_HERM_KDEP"):
            require(rec["max_V_residual_F"] < 1.0e-10, f"{name} should keep V, got {rec['max_V_residual_F']}")
            require(rec["max_T_antiHerm_F"] < 1.0e-10, f"{name} should be Hermitian")
        if name == "C3_CARTOON_055_045":
            require(rec["max_V_residual_F"] > 1.0e-3, "cartoon should break specified V")


def test_fg_same_nn() -> None:
    lat, atoms = parse_scf(SCF)
    bonds = build_bond_table(lat, cr_sites(atoms))
    require(len(bonds["monolayer_NN_top_A_to_B"]) == 3, bonds["monolayer_NN_top_A_to_B"])
    require(len(bonds["km_nnn_from_same_NN"]) == 3, "KM NNN must come from the same three NN")
    recip = np.array(bonds["reciprocal_2pi_inv_aT"])
    kc = k_cart_from_frac(recip, [0.17, -0.11, 0.0])
    h0 = honeycomb_h(kc, bonds, delta=0.0)
    h0m = honeycomb_h(-kc, bonds, delta=0.0)
    sx = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex)
    p = np.kron(sx, np.eye(2, dtype=complex))
    # monoclinic geometry may leave a small residual; it must not be O(1) from rotated g
    err = frobenius(p @ h0 @ p.conj().T - h0m)
    require(err < 0.05, f"sigma_x h0(k) vs h0(-k) too large ({err}); check NN/NNN orientation")


def test_cartoon_residual_is_im_faa() -> None:
    from q8_tperp_static_audit import historical_cartoon_tab, pauli

    tab = historical_cartoon_tab(0.7, 0.2)
    _a, b = split_ab(tab)
    s0, sx, sy, sz = pauli()
    # B should be proportional to I_2 (Im f_AA σ0), not to σ_y
    b_sigma0 = 0.5 * np.trace(s0 @ b)
    b_sy = 0.5 * np.trace(sy @ b)
    require(abs(b_sigma0) > 10.0 * abs(b_sy), f"cartoon B should be Im(f_AA)σ0, B0={b_sigma0} By={b_sy}")
    require(abs(0.5 * np.trace(sx @ b)) < 1.0e-12, "cartoon B has unexpected σx")
    require(abs(0.5 * np.trace(sz @ b)) < 1.0e-12, "cartoon B has unexpected σz")


def test_geo_h8_sewing_and_pairing() -> None:
    lat, atoms = parse_scf(SCF)
    sites = cr_sites(atoms)
    bonds = build_bond_table(lat, sites)
    recip = np.array(bonds["reciprocal_2pi_inv_aT"])
    kpts = sample_kpts(recip)
    sew = sewing_check(bonds, kpts)
    geo = sew["max_sewing_F_by_candidate"]["GEO_EQUAL_AMP"]
    require(geo["T"] < 1.0e-12, f"GEO T sewing {geo['T']}")
    require(geo["h0"] < 1.0e-12, f"GEO h0 sewing {geo['h0']}")
    require(geo["H8_layer_specific_h0_site_tau"] < 1.0e-12, f"two-layer h0 site sewing {geo['H8_layer_specific_h0_site_tau']}")
    require(geo["H8_site_tau"] > 1.0e-6, "shared h0 + site τ must leave a visible H8 residual")
    # k-independent C1 is not a bond-table Fourier transform
    c1 = sew["max_sewing_F_by_candidate"]["C1_SIGMA0"]
    require(c1["T"] > 1.0e-8, "C1 sewing should not be silently exact")

    rec_geo = static_for_candidate("GEO_EQUAL_AMP", bonds, kpts)
    rec_herm = static_for_candidate("C3_HERM_KDEP", bonds, kpts)
    require(rec_geo["max_PT_residual_F"] > 1.0e-7, "original GEO PT should be the 1e-6-class residual")
    require(rec_herm["max_PT_residual_F"] > 1.0e-7, "original HERM PT is not zero")

    bundle = pairing_symmetry_compare(lat, sites, sample_kpts)
    require(bundle["proposal"]["max_delta_cart_ang"] < 1.0e-3, "pairing must stay a tiny coordinate snap")
    geo_p = bundle["pt_p_compare"]["GEO_EQUAL_AMP"]
    herm_p = bundle["pt_p_compare"]["C3_HERM_KDEP"]
    require(geo_p["paired_max_PT_F"] < 1.0e-10, f"paired GEO PT {geo_p['paired_max_PT_F']}")
    require(geo_p["paired_max_P_F"] < 1.0e-10, f"paired GEO P {geo_p['paired_max_P_F']}")
    require(herm_p["paired_max_PT_F"] < 1.0e-10, f"paired HERM PT {herm_p['paired_max_PT_F']}")
    require(bundle["paired_GEO_T_sewing_max_F"] < 1.0e-12, "paired GEO T sewing")
    _prop, paired = inversion_pairing_proposal(lat, sites)
    for src, dst in (("tA", "bB"), ("tB", "bA")):
        image = (2.0 * INV_C - paired[src]) % 1.0
        require(np.linalg.norm(wrap_frac(image - paired[dst])) < 1.0e-14, f"paired {src}->{dst}")


def main() -> None:
    test_r_identity()
    test_v_equals_two_tauy_b()
    test_fg_same_nn()
    test_cartoon_residual_is_im_faa()
    test_geo_h8_sewing_and_pairing()
    print("test_q8_tperp_static_audit: PASS")


if __name__ == "__main__":
    main()
