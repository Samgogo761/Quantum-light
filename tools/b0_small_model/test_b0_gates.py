"""B0 readiness gates: exact Fock ↔ Wick ↔ explicit GKSL-QRT.

Status: B0_SMALL_MODEL_PROTOTYPE=PRELIMINARY_PASS
Theory source: 完整理论 20260729 第二部分 §7.2–7.6.
Does not reuse production T2, and does not call any output quantum statistics.
Checks use require(), not assert, so python -O cannot fake PASS.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from fock import (  # noqa: E402
    bilinear_matrix,
    connected_equal_time,
    quadratic_hamiltonian,
    thermal_density,
    two_time_greater,
)
from qrt import (  # noqa: E402
    SM,
    SP,
    amplitude_damping_superop,
    connected_qrt_two_time,
    propagate_operator,
)
from wick import (  # noqa: E402
    equal_time_wick,
    single_particle_u,
    thermal_onerdm,
    two_time_wick_greater,
)


def _random_hermitian(n: int, rng: np.random.Generator) -> np.ndarray:
    x = rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n))
    return 0.5 * (x + x.conj().T)


def require(cond: bool, msg: str) -> None:
    """Explicit check; not stripped by python -O."""
    if not cond:
        raise RuntimeError(msg)


def test_equal_time_empty_and_filled() -> None:
    a = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex)
    rho0 = np.zeros((2, 2), dtype=complex)
    rho1 = np.eye(2, dtype=complex)
    require(abs(equal_time_wick(a, a, rho0)) < 1e-14, "empty Wick noise")
    require(abs(equal_time_wick(a, a, rho1)) < 1e-14, "filled Wick noise")


def test_equal_time_slater() -> None:
    a = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex)
    rho = np.diag([1.0, 0.0]).astype(complex)
    val = equal_time_wick(a, a, rho)
    require(abs(val - 1.0) < 1e-12, f"Slater Wick {val}")


def test_wick_matches_fock_thermal() -> None:
    rng = np.random.default_rng(7)
    n_orb = 3
    h = _random_hermitian(n_orb, rng)
    a = _random_hermitian(n_orb, rng)
    b = _random_hermitian(n_orb, rng)
    beta = 1.7
    rho = thermal_onerdm(h, beta)
    wick = equal_time_wick(a, b, rho)
    h_mb = quadratic_hamiltonian(h)
    rho_mb = thermal_density(h_mb, beta)
    ja = bilinear_matrix(a)
    jb = bilinear_matrix(b)
    exact = connected_equal_time(rho_mb, ja, jb)
    require(abs(wick - exact) < 1e-10, f"thermal Fock-Wick {wick} vs {exact}")


def test_two_time_equal_time_limit() -> None:
    rng = np.random.default_rng(11)
    n_orb = 3
    h = _random_hermitian(n_orb, rng)
    a = _random_hermitian(n_orb, rng)
    b = _random_hermitian(n_orb, rng)
    rho = thermal_onerdm(h, 1.2)
    eye = np.eye(n_orb, dtype=complex)
    c_tt = two_time_wick_greater(a, b, rho, eye)
    c_eq = equal_time_wick(a, b, rho)
    require(abs(c_tt - c_eq) < 1e-12, "two-time equal-time limit")


def test_two_time_hermiticity() -> None:
    rng = np.random.default_rng(13)
    n_orb = 3
    h = _random_hermitian(n_orb, rng)
    a = _random_hermitian(n_orb, rng)
    b = _random_hermitian(n_orb, rng)
    rho = thermal_onerdm(h, 0.9)
    dt = 0.35
    u = single_particle_u(h, dt)
    c_ab = two_time_wick_greater(a, b, rho, u)
    rho_t = u @ rho @ u.conj().T
    u_back = single_particle_u(h, -dt)
    c_ba = two_time_wick_greater(b, a, rho_t, u_back)
    require(abs(c_ab - np.conjugate(c_ba)) < 1e-10, "two-time Hermiticity")


def test_two_time_wick_matches_fock() -> None:
    rng = np.random.default_rng(17)
    n_orb = 3
    h = _random_hermitian(n_orb, rng)
    a = _random_hermitian(n_orb, rng)
    b = _random_hermitian(n_orb, rng)
    beta = 1.1
    rho = thermal_onerdm(h, beta)
    dt = 0.4
    u_sp = single_particle_u(h, dt)
    wick = two_time_wick_greater(a, b, rho, u_sp)

    h_mb = quadratic_hamiltonian(h)
    rho_mb = thermal_density(h_mb, beta)
    ja = bilinear_matrix(a)
    jb = bilinear_matrix(b)
    evals, vecs = np.linalg.eigh(h_mb)
    u_mb = (vecs * np.exp(-1j * evals * dt)) @ vecs.conj().T
    exact = two_time_greater(rho_mb, ja, jb, u_mb)
    require(abs(wick - exact) < 1e-9, f"two-time Fock-Wick {wick} vs {exact}")


def test_qrt_amplitude_damping_analytic() -> None:
    """Ground-state start: ⟨σ+(t) σ-(0)⟩ = 0. Excited start: e^{-γt/2} e^{iωt}.

    |e⟩, B=σ-, A=σ+: ⟨σ+(t) σ-(0)⟩ = e^{-γ t / 2} e^{+i ω t}.
    """
    omega = 1.3
    gamma = 0.4
    dt = 0.55
    superop = amplitude_damping_superop(omega, gamma)
    rho_e = np.array([[1.0, 0.0], [0.0, 0.0]], dtype=complex)
    corr = connected_qrt_two_time(superop, rho_e, SP, SM, dt)
    expected = np.exp(-0.5 * gamma * dt) * np.exp(1j * omega * dt)
    require(abs(corr - expected) < 1e-8, f"QRT analytic {corr} vs {expected}")


def test_qrt_does_not_hermitianize_auxiliary() -> None:
    omega = 0.8
    gamma = 0.25
    superop = amplitude_damping_superop(omega, gamma)
    rho = np.array([[0.2, 0.1 + 0.05j], [0.1 - 0.05j, 0.8]], dtype=complex)
    rho = 0.5 * (rho + rho.conj().T)
    rho /= np.trace(rho)
    aux = SM @ rho
    require(abs(aux - aux.conj().T).max() > 1e-8, "B rho should start non-Hermitian")
    evolved = propagate_operator(superop, aux, 0.3)
    require(abs(evolved - evolved.conj().T).max() > 1e-10, "QRT aux must not be Hermitianized")
    require(
        abs(np.trace(evolved) - np.trace(aux)) > 0.0 or abs(np.trace(aux)) > 0.0,
        "aux trace diagnostic",
    )


def test_missing_pauli_factor_fails() -> None:
    """Tr(A B ρ) is not the connected current noise (理论 §7.2).

    Slater occupation of orbital 0 has zero number fluctuation, but Tr(A²ρ)=1
    if A = n0 − n1.
    """
    a = np.diag([1.0, -1.0]).astype(complex)
    rho = np.diag([1.0, 0.0]).astype(complex)
    wrong = np.trace(a @ a @ rho)
    right = equal_time_wick(a, a, rho)
    require(abs(right) < 1e-14, "Pauli-blocked number noise")
    require(abs(wrong - 1.0) < 1e-14, "naive Tr(A A rho)")


def main() -> int:
    tests = [
        test_equal_time_empty_and_filled,
        test_equal_time_slater,
        test_wick_matches_fock_thermal,
        test_two_time_equal_time_limit,
        test_two_time_hermiticity,
        test_two_time_wick_matches_fock,
        test_qrt_amplitude_damping_analytic,
        test_qrt_does_not_hermitianize_auxiliary,
        test_missing_pauli_factor_fails,
    ]
    for fn in tests:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"B0 gates: {len(tests)}/{len(tests)} PASS")
    print("B0_SMALL_MODEL_PROTOTYPE=PRELIMINARY_PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
