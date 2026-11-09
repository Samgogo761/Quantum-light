#!/usr/bin/env python3
"""Gates for the PT-AFM toy. require() so python -O cannot fake PASS."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from pt_afm_toy import (  # noqa: E402
    LABEL,
    N_ORB,
    ToyParams,
    apply_P,
    apply_T,
    h2_symmetry_documented,
    hamiltonian,
    jx_operator,
    je_jo,
    map_minus_operator,
    partner_index,
    k_grid,
    run_toy,
    slater_rho,
    stage1_table,
    wrap_k,
)
from wick import equal_time_wick  # noqa: E402


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def expect_value_error(fn, contains: str) -> None:
    try:
        fn()
    except ValueError as exc:
        require(contains in str(exc), f"wrong error: {exc}")
        return
    raise RuntimeError(f"expected ValueError containing {contains!r}")


def test_p_maps_plus_n_to_minus_n() -> None:
    p = ToyParams()
    kx, ky = 0.7, -0.4
    h_plus = hamiltonian(kx, ky, p.delta_abs, p)
    h_minus = hamiltonian(*wrap_k(-kx, -ky), -p.delta_abs, p)
    require(np.max(np.abs(apply_P(h_plus) - h_minus)) < 1.0e-12, "P H(+N,k) P != H(-N,-k)")


def test_t_maps_plus_n_to_minus_n() -> None:
    p = ToyParams()
    kx, ky = 0.7, -0.4
    h_plus = hamiltonian(kx, ky, p.delta_abs, p)
    h_minus = hamiltonian(*wrap_k(-kx, -ky), -p.delta_abs, p)
    require(np.max(np.abs(apply_T(h_plus) - h_minus)) < 1.0e-12, "T H(+N,k) T^{-1} != H(-N,-k)")


def test_pt_preserves_neel() -> None:
    p = ToyParams()
    kx, ky = 0.35, 1.1
    h_plus = hamiltonian(kx, ky, p.delta_abs, p)
    mapped = apply_P(apply_T(h_plus))
    require(np.max(np.abs(mapped - h_plus)) < 1.0e-12, "PT does not preserve +N Hamiltonian")


def test_jx_matches_finite_difference() -> None:
    p = ToyParams()
    kx, ky = 0.9, 0.2
    dk = 1.0e-6
    dhdk = (
        hamiltonian(kx + dk, ky, p.delta_abs, p) - hamiltonian(kx - dk, ky, p.delta_abs, p)
    ) / (2.0 * dk)
    jx = jx_operator(kx, ky, p.delta_abs, p)
    require(np.max(np.abs(dhdk + jx)) < 1.0e-7, "jx != -dH/dkx")


def test_unmapped_subtraction_is_rejected() -> None:
    p = ToyParams()
    k_plus = (0.4, 0.1)
    j_plus = jx_operator(*k_plus, p.delta_abs, p)
    j_minus_same_k = jx_operator(*k_plus, -p.delta_abs, p)
    expect_value_error(
        lambda: map_minus_operator(
            j_minus_same_k, mapped=False, k_plus=k_plus, k_minus=k_plus
        ),
        "unmapped",
    )
    expect_value_error(
        lambda: map_minus_operator(
            j_minus_same_k, mapped=True, k_plus=k_plus, k_minus=k_plus
        ),
        "inversion partner",
    )
    mapped = map_minus_operator(
        jx_operator(*wrap_k(-k_plus[0], -k_plus[1]), -p.delta_abs, p),
        mapped=True,
        k_plus=k_plus,
        k_minus=wrap_k(-k_plus[0], -k_plus[1]),
    )
    je, jo = je_jo(j_plus, mapped)
    require(np.allclose(je, je.conj().T, atol=1e-12), "Je not Hermitian")
    require(np.allclose(jo, jo.conj().T, atol=1e-12), "Jo not Hermitian")


def test_k_grid_closed_under_inversion() -> None:
    pts = k_grid(4)
    partners = partner_index(pts)
    for i, (kx, ky) in enumerate(pts):
        j = partners[i]
        wx, wy = wrap_k(kx + pts[j][0], ky + pts[j][1])
        require(abs(wx) < 1e-12 and abs(wy) < 1e-12, f"partner {i}->{j}")


def test_equilibrium_wick_and_parity() -> None:
    p = ToyParams(e0=0.0, ncyc=1.0, nt_per_cycle=8, nk1d=4)
    kx, ky = 0.6, -0.25
    h = hamiltonian(kx, ky, p.delta_abs, p)
    rho = slater_rho(h, p.n_occ)
    j = jx_operator(kx, ky, p.delta_abs, p)
    val = equal_time_wick(j, j, rho)
    require(val.imag == 0.0 or abs(val.imag) < 1e-12, "eq Wick not real")
    require(abs(val - equal_time_wick(j, j, rho)) < 1e-15, "idempotent")
    eigs = np.linalg.eigvalsh(rho)
    require(eigs.min() >= -1e-12 and eigs.max() <= 1.0 + 1e-12, "0<=rho<=I")
    require(abs(np.trace(rho) - p.n_occ) < 1e-12, "occupation")


def test_reject_t2_and_qrt() -> None:
    expect_value_error(lambda: run_toy(production_t2=True), "T2")
    expect_value_error(lambda: run_toy(qrt=True), "QRT")
    expect_value_error(lambda: run_toy(dephasing=0.1), "T2")


def test_same_k_isospectrum() -> None:
    p = ToyParams()
    for kx, ky in ((0.0, 0.0), (0.7, -0.4), (1.2, 0.3)):
        ep = np.linalg.eigvalsh(hamiltonian(kx, ky, p.delta_abs, p))
        em = np.linalg.eigvalsh(hamiltonian(kx, ky, -p.delta_abs, p))
        require(np.max(np.abs(np.sort(ep) - np.sort(em))) < 1e-12, f"isospectrum {kx,ky}")


def test_driven_closed_wick_parity() -> None:
    p = ToyParams(nk1d=4, ncyc=3.0, nt_per_cycle=16, e0=0.08, omega=0.25)
    result = run_toy(p)
    require(result["label"] == LABEL, "label")
    require(result["not_a_physics_result"] is True, "physics flag")
    require(result["not_quantum_output_g2"] is True, "g2 flag")
    require(result["absolute_k_normalization_validated"] is False, "k-norm")
    require(result["equal_time_wick_residual"] < 1.0e-12, result["equal_time_wick_residual"])
    require(result["step_unitarity_residual"] < 1.0e-12, result["step_unitarity_residual"])
    require(result["rho_eig_min"] >= -1.0e-10, result["rho_eig_min"])
    require(result["rho_eig_max"] <= 1.0 + 1.0e-10, result["rho_eig_max"])
    require(result["c_oo_even_residual"] < 1.0e-10, f"C_JJ even residual {result['c_oo_even_residual']}")
    require(result["c_odd_abs_mid"] < 1.0e-12, f"C_odd should vanish with isospectrum {result['c_odd_abs_mid']}")
    require(result["neel_jo_max"] < 1.0e-12, f"Neel Jo not zero {result['neel_jo_max']}")
    require(result["jo_h2_hard_test_enabled"] is False, "Jo H2 hard test must stay off")
    require(h2_symmetry_documented(), "H2 symmetry note missing from PT_AFM_TOY.md")
    table = stage1_table(result)
    require("C_JJ_equal_time" in table, table)
    require("Do not call" in table["note"], table["note"])
    require(abs(result["je_h1"]) > 1.0e-4, f"Je(ω) missing {result['je_h1']}")


def main() -> None:
    test_p_maps_plus_n_to_minus_n()
    test_t_maps_plus_n_to_minus_n()
    test_pt_preserves_neel()
    test_jx_matches_finite_difference()
    test_unmapped_subtraction_is_rejected()
    test_k_grid_closed_under_inversion()
    test_equilibrium_wick_and_parity()
    test_same_k_isospectrum()
    test_reject_t2_and_qrt()
    test_driven_closed_wick_parity()
    print("test_pt_afm_toy: PASS")


if __name__ == "__main__":
    main()
