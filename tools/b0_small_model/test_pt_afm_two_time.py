#!/usr/bin/env python3
"""CPU two-time C^>_{JJ}(t,t') gates for the frozen PT-AFM toy (line Q).

Formula source: wick.two_time_wick_greater (theory Eq. 7.10) driven by
b1_dev_closed_wick.closed_wick_reference.  No second Wick implementation.
The Heisenberg-form cross-check below is the same Eq. (7.10) rearranged with
U(t,0); it is used only inside this test to reach t < t'.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from pt_afm_toy import (  # noqa: E402
    FROZEN_SPEC,
    N_ORB,
    ToyParams,
    _two_time_packet,
    assert_frozen_spec,
    evolve_domain,
    jo_case_report,
    k_grid,
    mapped_bundle,
    partner_index,
    time_grid,
    two_time_cjj,
    vector_potential,
)


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


SMALL = ToyParams(nk1d=4, ncyc=2.0, nt_per_cycle=12, e0=0.08, omega=0.25)


def heisenberg_greater(a_t, b_tp, rho0, u_t0, u_tp0) -> complex:
    """Tr[A_H(t) (I-rho0) B_H(t') rho0] with X_H = U(t,0)^dagger X U(t,0).

    Algebraically identical to Eq. (7.10) for every ordering of t, t'.
    """
    eye = np.eye(rho0.shape[0], dtype=complex)
    a_h = u_t0.conj().T @ a_t @ u_t0
    b_h = u_tp0.conj().T @ b_tp @ u_tp0
    return complex(np.trace(a_h @ (eye - rho0) @ b_h @ rho0))


def test_dtype_and_spec() -> None:
    tt = two_time_cjj(SMALL, blocks=False)
    assert_frozen_spec(tt)
    require(np.asarray(tt["c_greater_plus"]).dtype == np.complex128, "C must be complex128")
    require(tt["dtype"] == "complex128" and tt["floating_point"] == "FP64", "spec dtype")
    require(tt["dephasing"] == "none" and tt["open_system"] is False, "closed system")
    require(FROZEN_SPEC["wick_two_time"] == "wick.two_time_wick_greater", "wick source")


def test_equal_time_limit() -> None:
    tt = two_time_cjj(SMALL, blocks=False)
    require(tt["equal_time_limit_residual"] < 1.0e-12, f"t=t' limit {tt['equal_time_limit_residual']}")


def test_pauli_and_unitarity() -> None:
    p = SMALL
    times = time_grid(p)
    ax = vector_potential(times, p)
    rho, _j, steps = evolve_domain(p.delta_abs, p, times, ax)
    eigs = np.linalg.eigvalsh(rho)
    require(eigs.min() >= -1.0e-10 and eigs.max() <= 1.0 + 1.0e-10, "0<=rho<=I violated")
    eye = np.eye(N_ORB, dtype=complex)
    gram = np.matmul(np.swapaxes(steps.conj(), -1, -2), steps)
    require(float(np.max(np.abs(gram - eye))) < 1.0e-12, "steps not unitary")


def test_hermitian_exchange_relation() -> None:
    """C^>_{AB}(t,t') = [C^>_{BA}(t',t)]^*  (theory 7.4 check 3), A != B."""
    p = SMALL
    points = k_grid(p.nk1d)
    partners = partner_index(points)
    nk = len(points)
    weights = np.full(nk, 1.0 / nk)
    times = time_grid(p)
    ax = vector_potential(times, p)
    rho_p, j_p, steps_p = evolve_domain(p.delta_abs, p, times, ax)
    rho_m, j_m, _ = evolve_domain(-p.delta_abs, p, times, ax)
    je, _jo = mapped_bundle(j_p, j_m, points, partners)
    a, b = j_p, je
    nt = times.size
    refs = np.arange(nt)
    c_ab, mask, _ = _two_time_packet(times, weights, rho_p[0], steps_p, a, b, refs)
    c_ba, _, _ = _two_time_packet(times, weights, rho_p[0], steps_p, b, a, refs)

    # U(t,0) per k
    u_t0 = np.empty((nt, nk, N_ORB, N_ORB), dtype=complex)
    u_t0[0] = np.eye(N_ORB, dtype=complex)
    for it in range(1, nt):
        u_t0[it] = steps_p[it - 1] @ u_t0[it - 1]

    worst = 0.0
    magnitude = 0.0
    for it in range(nt):
        for jt in range(it + 1, nt):  # t < t' : not covered by the (7.10) driver
            val = 0.0j
            for ik in range(nk):
                val += weights[ik] ** 2 * heisenberg_greater(
                    a[it, ik], b[jt, ik], rho_p[0, ik], u_t0[it, ik], u_t0[jt, ik]
                )
            # RHS: driver value at (t'=jt >= t=it) with A,B swapped, conjugated
            require(bool(mask[jt, it]), "driver mask")
            rhs = np.conj(c_ba[jt, it])
            worst = max(worst, abs(val - rhs))
            magnitude = max(magnitude, abs(val))
    require(magnitude > 1.0e-6, f"exchange test is trivial, |C|={magnitude}")
    require(worst < 1.0e-12, f"Hermitian exchange relation violated: {worst}")

    # and the driver itself agrees with the Heisenberg form on t >= t'
    worst2 = 0.0
    for it in range(nt):
        for jt in range(0, it + 1):
            val = 0.0j
            for ik in range(nk):
                val += weights[ik] ** 2 * heisenberg_greater(
                    a[it, ik], b[jt, ik], rho_p[0, ik], u_t0[it, ik], u_t0[jt, ik]
                )
            worst2 = max(worst2, abs(val - c_ab[it, jt]))
    require(worst2 < 1.0e-12, f"(7.10) vs Heisenberg form differ: {worst2}")


def test_domain_odd_two_time_is_structural_zero() -> None:
    """Frozen H0 fact: V=tau_x s_z maps +N to -N at the same k and commutes with j_x.

    Hence C_+(t,t') == C_-(t,t') for the lab drive.  If this test fails, H0 changed.
    """
    tt = two_time_cjj(SMALL, blocks=False)
    require(tt["same_k_map_H_residual"] < 1.0e-14, "V H+ V^-1 != H-")
    require(tt["same_k_map_jx_commutator_residual"] < 1.0e-14, "[V, jx] != 0")
    require(tt["state_map_residual_sameKmap"] < 1.0e-12, "rho_- != V rho_+ V^-1")
    scale = max(tt["c_greater_plus_max"], 1.0e-300)
    require(tt["c_greater_domain_odd_max"] / scale < 1.0e-12, f"C+ != C-: {tt['c_greater_domain_odd_max']}")
    require(tt["mean_jo_lab_max"] < 1.0e-12, f"lab Jo not zero {tt['mean_jo_lab_max']}")


def test_parity_blocks_identities() -> None:
    tt = two_time_cjj(SMALL, blocks=True)
    s = tt["blocks"]
    # (10.6)/(10.7) are algebraic identities when evaluated in the matching state
    require(s["identity_10_6_residual_Pmap"] < 1.0e-12, s["identity_10_6_residual_Pmap"])
    require(s["identity_10_7_residual_Pmap"] < 1.0e-12, s["identity_10_7_residual_Pmap"])
    require(s["identity_10_6_residual_sameKmap"] < 1.0e-12, s["identity_10_6_residual_sameKmap"])
    # same-k map: operator Jo vanishes identically -> C_oo=C_eo=C_oe=0 exactly
    v = s["sameKmap_plus_state"]
    require(v["oo"] == 0.0 and v["eo"] == 0.0 and v["oe"] == 0.0, v)
    # case report exists and is one of the documented labels
    rep = jo_case_report(tt)
    require(rep["case"] in {"CASE_A", "CASE_B", "CASE_C_map_dependent", "CASE_D_expectation_nonzero"}, rep)
    # P-map operator Jo is O(1) and equals j_x at A=0: it is not Delta_N-odd
    require(tt["operator_jo_frobenius_max_Pmap"] > 1.0, tt["operator_jo_frobenius_max_Pmap"])
    require(tt["operator_jo_at_A0_equals_jx_Pmap"] < 1.0e-12, tt["operator_jo_at_A0_equals_jx_Pmap"])


def test_reject_open_system() -> None:
    expect_value_error(lambda: two_time_cjj(SMALL, production_t2=True), "T2")
    expect_value_error(lambda: two_time_cjj(SMALL, qrt=True), "QRT")
    expect_value_error(lambda: two_time_cjj(SMALL, dephasing=0.05), "T2")


def main() -> None:
    test_dtype_and_spec()
    test_equal_time_limit()
    test_pauli_and_unitarity()
    test_hermitian_exchange_relation()
    test_domain_odd_two_time_is_structural_zero()
    test_parity_blocks_identities()
    test_reject_open_system()
    print("test_pt_afm_two_time: PASS")


if __name__ == "__main__":
    main()
