#!/usr/bin/env python3
"""B0 open-fermion three-way: Fock-Lindblad/QRT ↔ Wick on a Gaussian GKSL.

Linear jumps L∝c and L∝c†. Production T2 is not used.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from fock import (  # noqa: E402
    annihilation_matrix,
    bilinear_matrix,
    creation_matrix,
    quadratic_hamiltonian,
    thermal_density,
)
from open_gaussian import (  # noqa: E402
    propagate_onerdm,
    single_particle_g,
    two_time_wick_greater_open,
    two_time_wick_lesser_open,
    two_time_wick_symmetric_open,
)
from qrt import (  # noqa: E402
    connected_qrt_symmetric,
    connected_qrt_two_time,
    connected_qrt_two_time_lesser,
    gksl_superoperator,
    propagate_operator,
)
from wick import equal_time_wick, thermal_onerdm  # noqa: E402


def require(cond: bool, msg: str) -> None:
    if not cond:
        raise RuntimeError(msg)


def _random_hermitian(n: int, rng: np.random.Generator) -> np.ndarray:
    x = rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n))
    return 0.5 * (x + x.conj().T)


def onerdm_from_fock(rho_mb: np.ndarray, n_orb: int) -> np.ndarray:
    rho = np.zeros((n_orb, n_orb), dtype=complex)
    for i in range(n_orb):
        for j in range(n_orb):
            e = np.zeros((n_orb, n_orb), dtype=complex)
            e[j, i] = 1.0  # c_j† c_i  →  ⟨ ⟩ = ρ_ij
            rho[i, j] = np.trace(rho_mb @ bilinear_matrix(e))
    return rho


def fermionic_open_model(rng: np.random.Generator):
    n_orb = 2
    h = _random_hermitian(n_orb, rng)
    gamma_out = np.diag([0.35, 0.12]).astype(complex)
    gamma_in = np.diag([0.08, 0.22]).astype(complex)
    jumps = [
        np.sqrt(gamma_out[0, 0]) * annihilation_matrix(n_orb, 0),
        np.sqrt(gamma_out[1, 1]) * annihilation_matrix(n_orb, 1),
        np.sqrt(gamma_in[0, 0]) * creation_matrix(n_orb, 0),
        np.sqrt(gamma_in[1, 1]) * creation_matrix(n_orb, 1),
    ]
    h_mb = quadratic_hamiltonian(h)
    superop = gksl_superoperator(h_mb, jumps)
    a = _random_hermitian(n_orb, rng)
    b = _random_hermitian(n_orb, rng)
    ja = bilinear_matrix(a)
    jb = bilinear_matrix(b)
    return n_orb, h, gamma_out, gamma_in, superop, a, b, ja, jb, h_mb


def test_onerdm_matches_fock() -> None:
    rng = np.random.default_rng(21)
    n_orb, h, go, gi, superop, _a, _b, _ja, _jb, h_mb = fermionic_open_model(rng)
    rho0 = thermal_onerdm(h, 1.3)
    rho0_mb = thermal_density(h_mb, 1.3)
    require(np.abs(onerdm_from_fock(rho0_mb, n_orb) - rho0).max() < 1e-10, "thermal 1RDM")
    dt = 0.45
    rho_w = propagate_onerdm(h, go, gi, rho0, dt)
    rho_f = onerdm_from_fock(propagate_operator(superop, rho0_mb, dt), n_orb)
    require(np.abs(rho_w - rho_f).max() < 1e-8, f"1RDM Lyapunov vs Fock {np.abs(rho_w-rho_f).max()}")


def test_equal_time_pauli_open() -> None:
    rng = np.random.default_rng(23)
    n_orb, h, go, gi, superop, a, b, ja, jb, h_mb = fermionic_open_model(rng)
    rho0_mb = thermal_density(h_mb, 0.9)
    rho_mb = propagate_operator(superop, rho0_mb, 0.4)
    rho = onerdm_from_fock(rho_mb, n_orb)
    wick = equal_time_wick(a, b, rho)
    exact = np.trace(rho_mb @ ja @ jb) - np.trace(rho_mb @ ja) * np.trace(rho_mb @ jb)
    require(abs(wick - exact) < 1e-8, f"open equal-time {wick} vs {exact}")


def test_two_time_greater_lesser_symmetric() -> None:
    rng = np.random.default_rng(29)
    n_orb, h, go, gi, superop, a, b, ja, jb, h_mb = fermionic_open_model(rng)
    rho0_mb = thermal_density(h_mb, 1.1)
    rho_mb = propagate_operator(superop, rho0_mb, 0.25)
    rho = onerdm_from_fock(rho_mb, n_orb)
    dt = 0.35
    g = single_particle_g(h, go, gi, dt)
    wick_g = two_time_wick_greater_open(a, b, rho, g)
    wick_l = two_time_wick_lesser_open(a, b, rho, g)
    wick_s = two_time_wick_symmetric_open(a, b, rho, g)
    qrt_g = connected_qrt_two_time(superop, rho_mb, ja, jb, dt)
    qrt_l = connected_qrt_two_time_lesser(superop, rho_mb, ja, jb, dt)
    qrt_s = connected_qrt_symmetric(superop, rho_mb, ja, jb, dt)
    require(abs(wick_g - qrt_g) < 1e-7, f"C> {wick_g} vs {qrt_g}")
    require(abs(wick_l - qrt_l) < 1e-7, f"C< {wick_l} vs {qrt_l}")
    require(abs(wick_s - qrt_s) < 1e-7, f"CS {wick_s} vs {qrt_s}")


def test_hermitianize_mutation_fails() -> None:
    rng = np.random.default_rng(31)
    _n, h, go, gi, superop, a, b, ja, jb, h_mb = fermionic_open_model(rng)
    rho_mb = propagate_operator(superop, thermal_density(h_mb, 1.0), 0.2)
    dt = 0.4
    right = connected_qrt_two_time(superop, rho_mb, ja, jb, dt)
    aux = jb @ rho_mb
    aux_h = 0.5 * (aux + aux.conj().T)
    wrong_aux = propagate_operator(superop, aux_h, dt)
    rho_t = propagate_operator(superop, rho_mb, dt)
    wrong = np.trace(ja @ wrong_aux) - np.trace(ja @ rho_t) * np.trace(jb @ rho_mb)
    require(abs(wrong - right) > 1e-6, "Hermitianized Bρ must not match QRT")


def test_missing_pauli_mutation_fails() -> None:
    rng = np.random.default_rng(37)
    n_orb, h, go, gi, superop, a, b, ja, jb, h_mb = fermionic_open_model(rng)
    rho_mb = propagate_operator(superop, thermal_density(h_mb, 1.4), 0.3)
    rho = onerdm_from_fock(rho_mb, n_orb)
    dt = 0.3
    g = single_particle_g(h, go, gi, dt)
    right = two_time_wick_greater_open(a, b, rho, g)
    wrong = np.trace(a @ g @ b @ rho @ g.conj().T)
    require(abs(wrong - right) > 1e-6, "dropping (I-ρ) must fail")


def test_wrong_heisenberg_sign_fails() -> None:
    rng = np.random.default_rng(41)
    n_orb, h, go, gi, superop, a, b, ja, jb, h_mb = fermionic_open_model(rng)
    rho_mb = thermal_density(h_mb, 0.8)
    dt = 0.5
    right = connected_qrt_two_time(superop, rho_mb, ja, jb, dt)
    wrong_super = -superop.conj().T  # not a substitute; flip unitary part only
    # Flip only the Hamiltonian sign: L vec = +i[H,X] + dissipator.
    h_mb_flip = -h_mb
    jumps = [
        np.sqrt(go[0, 0]) * annihilation_matrix(n_orb, 0),
        np.sqrt(go[1, 1]) * annihilation_matrix(n_orb, 1),
        np.sqrt(gi[0, 0]) * creation_matrix(n_orb, 0),
        np.sqrt(gi[1, 1]) * creation_matrix(n_orb, 1),
    ]
    flipped = gksl_superoperator(h_mb_flip, jumps)
    wrong = connected_qrt_two_time(flipped, rho_mb, ja, jb, dt)
    require(abs(wrong - right) > 1e-6, "wrong Heisenberg/Fourier sign must fail")
    del wrong_super


def main() -> int:
    tests = [
        test_onerdm_matches_fock,
        test_equal_time_pauli_open,
        test_two_time_greater_lesser_symmetric,
        test_hermitianize_mutation_fails,
        test_missing_pauli_mutation_fails,
        test_wrong_heisenberg_sign_fails,
    ]
    for fn in tests:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"B0 open-Gaussian: {len(tests)}/{len(tests)} PASS")
    print("B0_SMALL_MODEL_PROTOTYPE=PRELIMINARY_PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
