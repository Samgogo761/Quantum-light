#!/usr/bin/env python3
"""Six qualification gates for the frozen 4-band zero-response harness. Then STOP.

Jo ≃ 0 is PASS.  H2 is labeled finite_pulse_background, not magnetic H2.
Does not change H0, does not enable 8-band, quantum light, g2, or GPU.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from pt_afm_toy import (  # noqa: E402
    A_RULE_HARNESS,
    A_RULE_LEGACY,
    INITIAL_STATE,
    MODEL_ROLE,
    N_ORB,
    ToyParams,
    U_P,
    U_SAME_K,
    U_W,
    analytic_jx,
    apply_P,
    apply_T,
    bz_mean,
    e_peak_actual,
    equal_time_wick,
    evolve_domain,
    hamiltonian,
    harmonic_complex,
    hidden_w_residuals,
    jx_operator,
    k_grid,
    same_k_map_residuals,
    slater_rho,
    time_grid,
    two_time_wick_greater,
    vector_potential,
    wrap_k,
)
from wick import equal_time_wick as wick_eq  # noqa: E402


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def _lab_currents(p: ToyParams, a_rule: str):
    times = time_grid(p)
    ax = vector_potential(times, p)
    rho_p, j_p, steps_p = evolve_domain(p.delta_abs, p, times, ax, a_rule=a_rule)
    rho_m, j_m, steps_m = evolve_domain(-p.delta_abs, p, times, ax, a_rule=a_rule)
    nk = j_p.shape[1]
    w = np.full(nk, 1.0 / nk)
    jp = bz_mean(j_p, rho_p, w)
    jm = bz_mean(j_m, rho_m, w)
    return times, jp, jm, rho_p, j_p, steps_p, steps_m


def gate1_matrix_symmetry() -> None:
    p = ToyParams()
    r_p = r_t = r_pt = 0.0
    for kx, ky in [(0.7, -0.4), (1.2, 0.3), *k_grid(4)]:
        hp = hamiltonian(kx, ky, p.delta_abs, p)
        h_mapped = hamiltonian(*wrap_k(-kx, -ky), -p.delta_abs, p)
        r_p = max(r_p, float(np.max(np.abs(apply_P(hp) - h_mapped))))
        r_t = max(r_t, float(np.max(np.abs(apply_T(hp) - h_mapped))))
        r_pt = max(r_pt, float(np.max(np.abs(apply_P(apply_T(hp)) - hp))))
    v = same_k_map_residuals(p)
    w = hidden_w_residuals(p)
    require(r_p < 1.0e-12, f"P residual {r_p}")
    require(r_t < 1.0e-12, f"T residual {r_t}")
    require(r_pt < 1.0e-12, f"PT residual {r_pt}")
    require(v["same_k_map_H_residual"] < 1.0e-12, v)
    require(v["same_k_map_jx_commutator_residual"] < 1.0e-12, v)
    require(w["hidden_W_H_residual"] < 1.0e-12, w)
    require(w["hidden_W_jx_polarity_residual"] < 1.0e-12, w)
    require(w["VW_commutator_residual"] < 1.0e-12, w)
    require(not np.allclose(U_P, U_SAME_K), "P accidentally equal V")
    require(not np.allclose(U_P, U_W), "P accidentally equal W")
    require(not np.allclose(U_SAME_K, U_W), "V accidentally equal W")
    require(INITIAL_STATE == "two_lowest_bands_per_k_slater", INITIAL_STATE)
    require(MODEL_ROLE == "zero_response_numerical_harness", MODEL_ROLE)


def gate2_current() -> None:
    p = ToyParams()
    dk = 1.0e-6
    for kx, ky in [(0.9, 0.2), (0.0, 0.0), (-1.1, 0.4)]:
        dhdk = (
            hamiltonian(kx + dk, ky, p.delta_abs, p) - hamiltonian(kx - dk, ky, p.delta_abs, p)
        ) / (2.0 * dk)
        jx = jx_operator(kx, ky, p.delta_abs, p)
        analytic = analytic_jx(kx, p)
        require(np.max(np.abs(dhdk + jx)) < 1.0e-7, "jx != -dH/dkx")
        require(np.max(np.abs(jx - analytic)) < 1.0e-15, "analytic jx mismatch")
        require(np.max(np.abs(jx_operator(kx, ky, -p.delta_abs, p) - jx)) < 1.0e-15, "jx depends on Delta_N")


def gate3_propagation() -> None:
    p = ToyParams(nk1d=4, ncyc=4.0, nt_per_cycle=24, e0=0.08, omega=0.20)
    times, jp, _jm, rho, _j, steps, _sm = _lab_currents(p, A_RULE_HARNESS)
    eigs = np.linalg.eigvalsh(rho)
    require(eigs.min() >= -1.0e-10 and eigs.max() <= 1.0 + 1.0e-10, "0<=rho<=I")
    traces = np.trace(rho, axis1=-2, axis2=-1)
    require(float(np.max(np.abs(traces - traces[0]))) < 1.0e-10, "trace drift")
    require(float(np.max(np.abs(rho - np.swapaxes(rho.conj(), -1, -2)))) < 1.0e-10, "rho not Hermitian")
    eye = np.eye(N_ORB, dtype=complex)
    gram = np.matmul(np.swapaxes(steps.conj(), -1, -2), steps)
    require(float(np.max(np.abs(gram - eye))) < 1.0e-12, "U not unitary")
    curves = {}
    for ntc in (24, 48, 96):
        pn = ToyParams(nk1d=4, ncyc=4.0, nt_per_cycle=ntc, e0=0.08, omega=0.20)
        t, jp, _jm, *_ = _lab_currents(pn, A_RULE_HARNESS)
        curves[ntc] = (t, np.real(jp))
    t_f, j_f = curves[96]
    err_c = float(np.max(np.abs(np.interp(t_f, *curves[24]) - j_f)))
    err_f = float(np.max(np.abs(np.interp(t_f, *curves[48]) - j_f)))
    require(err_f < err_c, f"time-step current not converging: {err_c} -> {err_f}")
    require(err_f / max(float(np.max(np.abs(j_f))), 1.0e-16) < 0.05, f"fine-grid current error {err_f}")
    require(abs(e_peak_actual(p) - p.e0) / p.e0 > 1.0e-4, "e0 accidentally equals E_peak; keep both")


def gate4_domain_zero() -> None:
    for a_rule in (A_RULE_LEGACY, A_RULE_HARNESS):
        p = ToyParams(nk1d=4, ncyc=4.0, nt_per_cycle=24, e0=0.08, omega=0.20)
        _t, jp, jm, *_ = _lab_currents(p, a_rule)
        jo_max = float(np.max(np.abs(0.5 * (jp - jm))))
        require(jo_max < 1.0e-12, f"{a_rule} Jo not structural zero: {jo_max}")


def gate5_finite_pulse_spectrum() -> None:
    p = ToyParams(nk1d=4, ncyc=4.0, nt_per_cycle=24, e0=0.08, omega=0.20)
    t, jp, jm, *_ = _lab_currents(p, A_RULE_HARNESS)
    je = 0.5 * (jp + jm)
    jo = 0.5 * (jp - jm)
    h1 = abs(harmonic_complex(je, t, p.omega, 1, window="hanning"))
    h2 = abs(harmonic_complex(je, t, p.omega, 2, window="hanning"))
    h3 = abs(harmonic_complex(je, t, p.omega, 3, window="hanning"))
    h2_box = abs(harmonic_complex(je, t, p.omega, 2, window="boxcar"))
    require(h1 > 10.0 * h2, f"H2 not a small background vs H1: {h2/max(h1,1e-30)}")
    require(h3 > h2, f"H3 should exceed H2 background: H3={h3} H2={h2}")
    for n in (1, 2, 3):
        require(abs(harmonic_complex(jo, t, p.omega, n)) < 1.0e-12, f"Jo H{n} not floor")
    p_long = ToyParams(nk1d=4, ncyc=8.0, nt_per_cycle=24, e0=0.08, omega=0.20)
    tl, jpl, jml, *_ = _lab_currents(p_long, A_RULE_HARNESS)
    jel = 0.5 * (jpl + jml)
    r_short = h2 / max(h1, 1.0e-16)
    r_long = abs(harmonic_complex(jel, tl, p_long.omega, 2)) / max(
        abs(harmonic_complex(jel, tl, p_long.omega, 1)), 1.0e-16
    )
    require(r_long <= r_short * 1.5 + 1.0e-6, f"H2/H1 grew with longer pulse: {r_short} -> {r_long}")
    require(abs(h2_box - h2) / max(h2, 1.0e-16) > 1.0e-3, "H2 insensitive to window; check label")


def gate6_wick() -> None:
    p = ToyParams(nk1d=4, ncyc=2.0, nt_per_cycle=12, e0=0.08, omega=0.25)
    times = time_grid(p)
    ax = vector_potential(times, p)
    rho, jx, _steps = evolve_domain(p.delta_abs, p, times, ax, a_rule=A_RULE_HARNESS)
    it = times.size // 2
    u_eye = np.eye(N_ORB, dtype=complex)
    worst = 0.0
    for ik in range(rho.shape[1]):
        eq = equal_time_wick(jx[it, ik], jx[it, ik], rho[it, ik])
        tw = two_time_wick_greater(jx[it, ik], jx[it, ik], rho[it, ik], u_eye)
        worst = max(worst, abs(eq - tw), abs(eq - wick_eq(jx[it, ik], jx[it, ik], rho[it, ik])))
        require(abs(eq.imag) < 1.0e-12, "equal-time Wick not real")
    require(worst < 1.0e-12, f"Wick residual {worst}")
    rho0 = slater_rho(hamiltonian(0.6, -0.25, p.delta_abs, p), p.n_occ)
    require(abs(np.trace(rho0) - p.n_occ) < 1.0e-12, "per-k occupation")


def main() -> None:
    gate1_matrix_symmetry()
    gate2_current()
    gate3_propagation()
    gate4_domain_zero()
    gate5_finite_pulse_spectrum()
    gate6_wick()
    print("test_q_harness_gates: PASS")
    print("FOUR_BAND_HARNESS_STATUS=CLOSED")
    print("NEXT=8band_hamiltonian_derivation_hold_for_human")


if __name__ == "__main__":
    main()
