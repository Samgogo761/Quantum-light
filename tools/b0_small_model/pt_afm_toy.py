#!/usr/bin/env python3
"""4-band PT-AFM toy: Houston drive, P-mapped Je/Jo, closed-system Wick C_JJ.

Line Q only.  Reuses wick.py Eqs. (7.6)/(7.10).  Not CrI3, not g^(2), not QRT.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import numpy as np

try:
    from .b1_dev_closed_wick import ClosedWickPacket, closed_wick_reference
    from .wick import equal_time_wick, mean_current, single_particle_u, two_time_wick_greater
except ImportError:
    from b1_dev_closed_wick import ClosedWickPacket, closed_wick_reference
    from wick import equal_time_wick, mean_current, single_particle_u, two_time_wick_greater

Array = np.ndarray

LABEL = "PT_AFM_TOY_CANDIDATE"
STATUS = "development_only / not_cri3"
H2_TOKEN = "H2_DYNAMIC_SYMMETRY_ALLOWED"
JO_STATUS = "STRUCTURAL_ZERO_CANDIDATE"
INITIAL_STATE = "two_lowest_bands_per_k_slater"
MODEL_ROLE = "zero_response_numerical_harness"
N_ORB = 4
N_OCC = 2
A_RULE_LEGACY = "sampled_A_arithmetic_mean"
A_RULE_HARNESS = "evaluate_A_at_t_mid"


def model_version() -> str:
    """Local identifier of the frozen physics: sha256 of this file (untracked in git)."""
    return "sha256:" + hashlib.sha256(Path(__file__).read_bytes()).hexdigest()[:16]


# Frozen physics spec (line Q).  Any backend must reproduce these keys verbatim.
# Values are compared by test_q_frozen_spec.py against Q_THEORY_FROZEN_SPEC.md.
FROZEN_SPEC: dict[str, object] = {
    "line": "Q_THEORY",
    "model": "current_4band_pt_afm_toy",
    "gauge_or_representation": "Houston_characteristics",
    "k_evolution": "k0 + A(t)",
    "equation": "i drho/dt = [H(k(t)), rho]",
    "integrator": "midpoint_unitary_exponential",
    "dephasing": "none",
    "open_system": False,
    "current_operator": "-dH/dkx",
    "wick_equal_time": "wick.equal_time_wick",
    "wick_two_time": "wick.two_time_wick_greater",
    "two_time_driver": "b1_dev_closed_wick.closed_wick_reference",
    "dtype": "complex128",
    "floating_point": "FP64",
    "k_points": "current_existing_small_grid",
    "k_points_are_independent": True,
    "quantum_light_ensemble": "none",
    "pump": "classical_coherent_prescribed_field",
    "absolute_k_normalization_validated": False,
    "k_normalization_provenance": "discrete_weight_squared",
    "material_interpretation": False,
    "cri3_claim": False,
    "current_4band_jo_status": JO_STATUS,
    "jo_h2_hard_test_enabled": False,
    "model_role": MODEL_ROLE,
    "initial_state": INITIAL_STATE,
    "light_matter_coupling": "peierls_k_shift",
    "propagation_coordinate": "houston_characteristic",
    "default_run_toy_a_rule": A_RULE_LEGACY,
    "h2_label": "finite_pulse_background",
}

FORBIDDEN_BACKEND_KNOBS = (
    "T2",
    "t2",
    "dephasing",
    "gauge",
    "integrator",
    "dipole_mode",
    "lg_cov",
    "dtype",
    "hamiltonian",
)


def assert_frozen_spec(metadata: Mapping[str, object]) -> None:
    """Raise if a result's metadata contradicts the frozen spec."""
    bad = {}
    for key, value in FROZEN_SPEC.items():
        if key in metadata and metadata[key] != value:
            bad[key] = (metadata[key], value)
    if bad:
        raise ValueError(f"frozen spec violated: {bad}")

I2 = np.eye(2, dtype=complex)
SX = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex)
SY = np.array([[0.0, -1.0j], [1.0j, 0.0]], dtype=complex)
SZ = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=complex)
TAU0 = np.kron(I2, I2)
TAUX = np.kron(SX, I2)
TAUZ = np.kron(SZ, I2)
SX_F = np.kron(I2, SX)
SY_F = np.kron(I2, SY)
SZ_F = np.kron(I2, SZ)
U_P = TAUX.copy()
U_T = 1.0j * SY_F
# Same-k internal map found 2026-09-03: V H(+D,k) V^-1 = H(-D,k) exactly, [V, j_x] = 0.
# Layer exchange combined with a pi spin rotation about z.  It does NOT flip k or A(t).
# Whether this is the physically correct U_N of theory Eq. (10.4) is a human-review item.
U_SAME_K = TAUX @ SZ_F
# Hidden single-domain map (2026-09-07): W H_N(k) W^-1 = H_N(-k), W j_x(k) W^-1 = -j_x(-k).
# Not crystal inversion P=tau_x, and not the domain map V.  P ≠ V ≠ W.
U_W = SZ_F.copy()


@dataclass(frozen=True)
class ToyParams:
    t: float = 1.0
    t_perp: float = 0.30
    lam: float = 0.40
    delta_abs: float = 0.50
    n_occ: int = N_OCC
    nk1d: int = 4
    e0: float = 0.08
    omega: float = 0.20
    ncyc: float = 4.0
    nt_per_cycle: int = 24

    def metadata(self) -> dict[str, object]:
        return {
            **FROZEN_SPEC,
            "schema": "pt_afm_toy_v2",
            "model_version": model_version(),
            "label": LABEL,
            "status": STATUS,
            "not_a_physics_result": True,
            "not_quantum_output_g2": True,
            "dynamics": "closed_unitary_houston",
            "polarization": "E_parallel_x",
            "u_p": "tau_x",
            "u_v": "tau_x s_z",
            "u_w": "s_z",
            "k_normalization_provenance_note": (
                "uniform 1/Nk weights on a finite Gamma-centered torus; "
                "connected C uses sum_k w_k^2 C_k as in B1-dev discrete operator"
            ),
            "production_t2_used": False,
            "qrt_used": False,
            "params": {
                "t": self.t,
                "t_perp": self.t_perp,
                "lambda": self.lam,
                "delta_abs": self.delta_abs,
                "n_occ": self.n_occ,
                "nk1d": self.nk1d,
                "e0": self.e0,
                "omega": self.omega,
                "ncyc": self.ncyc,
                "nt_per_cycle": self.nt_per_cycle,
            },
        }


def h2_symmetry_documented() -> bool:
    text = Path(__file__).with_name("PT_AFM_TOY.md").read_text(encoding="utf-8")
    return H2_TOKEN in text


def _reject_open_or_qrt(*, production_t2: bool, qrt: bool, dephasing) -> None:
    if production_t2 or dephasing is not None:
        raise ValueError("production T2 / open dynamics rejected on line Q")
    if qrt:
        raise ValueError("QRT is not a line-Q primary result")


def wrap_k(kx: float, ky: float) -> tuple[float, float]:
    return (
        float(np.mod(kx + np.pi, 2.0 * np.pi) - np.pi),
        float(np.mod(ky + np.pi, 2.0 * np.pi) - np.pi),
    )


def k_grid(nk1d: int) -> list[tuple[float, float]]:
    vals = 2.0 * np.pi * np.arange(nk1d) / nk1d
    return [(float(kx), float(ky)) for kx in vals for ky in vals]


def _kkey(kx: float, ky: float) -> tuple[float, float]:
    """Rounded, wrapped key so 2*pi*n/N grids (N=6, 12, ...) pair exactly."""
    wx, wy = wrap_k(kx, ky)
    wx = round(wx, 9)
    wy = round(wy, 9)
    if abs(wx - np.pi) < 1e-9 or abs(wx + np.pi) < 1e-9:
        wx = -round(np.pi, 9)
    if abs(wy - np.pi) < 1e-9 or abs(wy + np.pi) < 1e-9:
        wy = -round(np.pi, 9)
    return (float(wx), float(wy))


def partner_index(points: list[tuple[float, float]]) -> list[int]:
    lookup = {_kkey(kx, ky): i for i, (kx, ky) in enumerate(points)}
    out = []
    for kx, ky in points:
        key = _kkey(-kx, -ky)
        if key not in lookup:
            raise ValueError("k-grid is not closed under inversion; refuse +/-N map")
        out.append(lookup[key])
    return out


def hamiltonian(kx: float, ky: float, delta_n: float, p: ToyParams) -> Array:
    eps = -2.0 * p.t * (np.cos(kx) + np.cos(ky))
    rashba = p.lam * TAUZ @ (np.sin(ky) * SX_F - np.sin(kx) * SY_F)
    return eps * TAU0 + p.t_perp * TAUX + rashba + delta_n * (TAUZ @ SZ_F)


def jx_operator(kx: float, ky: float, delta_n: float, p: ToyParams) -> Array:
    """j_x = -dH/dk_x in e=hbar=1 Houston units."""
    del ky  # Rashba ky term does not enter dH/dk_x
    deps = 2.0 * p.t * np.sin(kx)
    return -deps * TAU0 + p.lam * TAUZ @ (np.cos(kx) * SY_F)


def analytic_jx(kx: float, p: ToyParams) -> Array:
    """Same as jx_operator; written out for the current Gate."""
    return -2.0 * p.t * np.sin(kx) * TAU0 + p.lam * TAUZ @ (np.cos(kx) * SY_F)


def apply_T(mat: Array) -> Array:
    return U_T @ mat.conj() @ U_T.conj().T


def apply_P(mat: Array) -> Array:
    return U_P @ mat @ U_P.conj().T


def inversion_partners(k_plus: tuple[float, float], k_minus: tuple[float, float]) -> bool:
    wx, wy = wrap_k(k_plus[0] + k_minus[0], k_plus[1] + k_minus[1])
    return abs(wx) < 1.0e-12 and abs(wy) < 1.0e-12


def map_minus_operator(
    op_minus_at_minus_k: Array,
    *,
    mapped: bool,
    k_plus: tuple[float, float],
    k_minus: tuple[float, float],
) -> Array:
    if not mapped:
        raise ValueError("unmapped +/-N subtraction is forbidden; apply U_P first")
    if not inversion_partners(k_plus, k_minus):
        raise ValueError("k_minus is not the inversion partner of k_plus")
    return U_P.conj().T @ op_minus_at_minus_k @ U_P


def je_jo(j_plus: Array, j_minus_mapped: Array) -> tuple[Array, Array]:
    je = 0.5 * (j_plus + j_minus_mapped)
    jo = 0.5 * (j_plus - j_minus_mapped)
    return je, jo


def slater_rho(h: Array, n_occ: int) -> Array:
    evals, vec = np.linalg.eigh(h)
    occ = np.zeros(h.shape[0], dtype=float)
    occ[:n_occ] = 1.0
    rho = (vec * occ) @ vec.conj().T
    del evals
    return rho


def vector_potential_at(t: float, p: ToyParams) -> float:
    period = 2.0 * np.pi / p.omega
    t_pulse = p.ncyc * period
    if 0.0 <= t <= t_pulse:
        env = np.sin(np.pi * t / t_pulse) ** 2
        return float((p.e0 / p.omega) * env * np.cos(p.omega * t))
    return 0.0


def electric_field_at(t: float, p: ToyParams) -> float:
    """E = -dA/dt.  e0 is a scale parameter, not automatically max|E|."""
    period = 2.0 * np.pi / p.omega
    t_pulse = p.ncyc * period
    if t < 0.0 or t > t_pulse:
        return 0.0
    phase = np.pi * t / t_pulse
    env = np.sin(phase) ** 2
    env_dot = (np.pi / t_pulse) * np.sin(2.0 * phase)
    dadt = (p.e0 / p.omega) * (env_dot * np.cos(p.omega * t) - p.omega * env * np.sin(p.omega * t))
    return float(-dadt)


def e_peak_actual(p: ToyParams, n_sample: int = 4000) -> float:
    times = np.linspace(0.0, p.ncyc * 2.0 * np.pi / p.omega, n_sample)
    return float(np.max(np.abs([electric_field_at(float(t), p) for t in times])))


def vector_potential(times: Array, p: ToyParams) -> Array:
    return np.array([vector_potential_at(float(t), p) for t in times], dtype=float)


def time_grid(p: ToyParams) -> Array:
    period = 2.0 * np.pi / p.omega
    nt = int(p.ncyc * p.nt_per_cycle) + 1
    return np.linspace(0.0, p.ncyc * period, nt)


def evolve_domain(
    delta_n: float,
    p: ToyParams,
    times: Array,
    ax: Array,
    *,
    a_rule: str = A_RULE_LEGACY,
) -> tuple[Array, Array, Array]:
    if a_rule not in (A_RULE_LEGACY, A_RULE_HARNESS):
        raise ValueError(f"unknown a_rule {a_rule!r}")
    points = k_grid(p.nk1d)
    nk = len(points)
    nt = times.size
    rho = np.empty((nt, nk, N_ORB, N_ORB), dtype=complex)
    jx = np.empty((nt, nk, N_ORB, N_ORB), dtype=complex)
    steps = np.empty((max(nt - 1, 0), nk, N_ORB, N_ORB), dtype=complex)
    for ik, (kx0, ky0) in enumerate(points):
        h0 = hamiltonian(kx0, ky0, delta_n, p)
        rho[0, ik] = slater_rho(h0, p.n_occ)
        jx[0, ik] = jx_operator(kx0 + ax[0], ky0, delta_n, p)
        rho_now = rho[0, ik]
        for it in range(1, nt):
            dt = float(times[it] - times[it - 1])
            if a_rule == A_RULE_LEGACY:
                a_mid = 0.5 * (ax[it] + ax[it - 1])
            else:
                t_mid = 0.5 * (float(times[it]) + float(times[it - 1]))
                a_mid = vector_potential_at(t_mid, p)
            h_mid = hamiltonian(kx0 + a_mid, ky0, delta_n, p)
            u = single_particle_u(h_mid, dt)
            steps[it - 1, ik] = u
            rho_now = u @ rho_now @ u.conj().T
            rho[it, ik] = rho_now
            jx[it, ik] = jx_operator(kx0 + ax[it], ky0, delta_n, p)
    return rho, jx, steps


def mapped_bundle(j_plus: Array, j_minus: Array, points: list[tuple[float, float]], partners: list[int]) -> tuple[Array, Array]:
    je = np.empty_like(j_plus)
    jo = np.empty_like(j_plus)
    for it in range(j_plus.shape[0]):
        for ik, k_plus in enumerate(points):
            ikm = partners[ik]
            k_minus = points[ikm]
            mapped = map_minus_operator(
                j_minus[it, ikm],
                mapped=True,
                k_plus=k_plus,
                k_minus=k_minus,
            )
            je[it, ik], jo[it, ik] = je_jo(j_plus[it, ik], mapped)
    return je, jo


def mapped_rho(rho_minus: Array, points: list[tuple[float, float]], partners: list[int]) -> Array:
    out = np.empty_like(rho_minus)
    for it in range(rho_minus.shape[0]):
        for ik, k_plus in enumerate(points):
            ikm = partners[ik]
            out[it, ik] = map_minus_operator(
                rho_minus[it, ikm],
                mapped=True,
                k_plus=k_plus,
                k_minus=points[ikm],
            )
    return out


def bz_mean(op: Array, rho: Array, weights: Array) -> Array:
    nt, nk = rho.shape[:2]
    out = np.zeros(nt, dtype=complex)
    for it in range(nt):
        for ik in range(nk):
            out[it] += weights[ik] * mean_current(op[it, ik], rho[it, ik])
    return out


def bz_equal_time(a: Array, b: Array, rho: Array, weights: Array) -> Array:
    nt, nk = rho.shape[:2]
    out = np.zeros(nt, dtype=complex)
    for it in range(nt):
        for ik in range(nk):
            out[it] += weights[ik] ** 2 * equal_time_wick(a[it, ik], b[it, ik], rho[it, ik])
    return out


def harmonic_complex(
    signal: Array,
    times: Array,
    omega: float,
    order: int,
    *,
    window: str = "hanning",
) -> complex:
    if window == "hanning":
        w = np.hanning(times.size)
    elif window == "boxcar":
        w = np.ones(times.size)
    else:
        raise ValueError(f"unknown window {window!r}")
    dt = float(times[1] - times[0])
    spec = np.fft.rfft(w * np.real(signal))
    freq = np.fft.rfftfreq(times.size, dt)
    target = order * omega / (2.0 * np.pi)
    idx = int(np.argmin(np.abs(freq - target)))
    return complex(spec[idx])


def relative_even_residual(a: complex, b: complex, floor: float = 1.0e-16) -> float:
    return float(abs(a - b) / (abs(a) + abs(b) + floor))


def run_toy(
    p: ToyParams | None = None,
    *,
    production_t2: bool = False,
    qrt: bool = False,
    dephasing=None,
) -> dict[str, object]:
    _reject_open_or_qrt(production_t2=production_t2, qrt=qrt, dephasing=dephasing)
    p = p or ToyParams()
    points = k_grid(p.nk1d)
    partners = partner_index(points)
    nk = len(points)
    weights = np.full(nk, 1.0 / nk)
    times = time_grid(p)
    ax = vector_potential(times, p)
    rho_p, j_p, steps_p = evolve_domain(p.delta_abs, p, times, ax)
    rho_m, j_m, _steps_m = evolve_domain(-p.delta_abs, p, times, ax)
    je, jo = mapped_bundle(j_p, j_m, points, partners)

    mean_j_plus = bz_mean(j_p, rho_p, weights)
    mean_j_minus = bz_mean(j_m, rho_m, weights)
    mean_je = 0.5 * (mean_j_plus + mean_j_minus)
    mean_jo = 0.5 * (mean_j_plus - mean_j_minus)
    c_full_p = bz_equal_time(j_p, j_p, rho_p, weights)
    c_full_m = bz_equal_time(j_m, j_m, rho_m, weights)
    c_oo_p = c_full_p
    c_oo_m = c_full_m
    c_ee = bz_equal_time(je, je, rho_p, weights)
    c_eo = bz_equal_time(je, jo, rho_p, weights)
    c_oe = bz_equal_time(jo, je, rho_p, weights)
    c_odd = 0.5 * (c_full_p - c_full_m)
    c_odd_from_full = c_odd

    eq_res = 0.0
    it_mid = times.size // 2
    u_eye = np.eye(N_ORB, dtype=complex)
    for ik in range(nk):
        tw = two_time_wick_greater(j_p[it_mid, ik], j_p[it_mid, ik], rho_p[it_mid, ik], u_eye)
        eq = equal_time_wick(j_p[it_mid, ik], j_p[it_mid, ik], rho_p[it_mid, ik])
        eq_res = max(eq_res, abs(tw - eq))

    eigs = np.linalg.eigvalsh(rho_p)
    gram = np.matmul(np.swapaxes(steps_p.conj(), -1, -2), steps_p)
    eye = np.eye(N_ORB, dtype=complex)
    jo_h2 = harmonic_complex(mean_jo, times, p.omega, 2)
    je_h1 = harmonic_complex(mean_je, times, p.omega, 1)
    coo_h2 = harmonic_complex(c_oo_p, times, p.omega, 2)

    result = {
        **p.metadata(),
        "times": times,
        "mean_je": mean_je,
        "mean_jo": mean_jo,
        "c_oo_plus": c_oo_p,
        "c_oo_minus": c_oo_m,
        "c_oo_fields_are_legacy_full_C_JJ": True,
        "c_ee": c_ee,
        "c_eo": c_eo,
        "c_oe": c_oe,
        "c_odd": c_odd,
        "c_odd_from_full_pm": c_odd_from_full,
        "jo_h2": jo_h2,
        "je_h1": je_h1,
        "c_oo_h2h2_proxy": coo_h2,
        "c_oo_even_residual": relative_even_residual(c_oo_p[it_mid], c_oo_m[it_mid]),
        "c_odd_abs_mid": float(abs(c_odd[it_mid])),
        "neel_jo_max": float(np.max(np.abs(mean_jo))),
        "accidental_same_k_isospectrum": True,
        "jo_h2_hard_test_enabled": False,
        "equal_time_wick_residual": float(eq_res),
        "rho_eig_min": float(eigs.min()),
        "rho_eig_max": float(eigs.max()),
        "step_unitarity_residual": float(np.max(np.abs(gram - eye))),
        "h2_symmetry_documented": h2_symmetry_documented(),
        "k_normalization_provenance": p.metadata()["k_normalization_provenance"],
        "a_rule": A_RULE_LEGACY,
        "e0_parameter": p.e0,
        "E_peak_actual": e_peak_actual(p),
        **same_k_map_residuals(p),
        **hidden_w_residuals(p),
    }
    return result


def hidden_w_residuals(p: ToyParams | None = None) -> dict[str, float]:
    """Algebraic facts about W = s_z (single-domain k -> -k). Not P, not V."""
    p = p or ToyParams()
    w = U_W
    r_h = 0.0
    r_j = 0.0
    for kx, ky in k_grid(p.nk1d):
        h = hamiltonian(kx, ky, p.delta_abs, p)
        h_inv = hamiltonian(*wrap_k(-kx, -ky), p.delta_abs, p)
        r_h = max(r_h, float(np.max(np.abs(w @ h @ w.conj().T - h_inv))))
        j = jx_operator(kx, ky, p.delta_abs, p)
        j_inv = jx_operator(*wrap_k(-kx, -ky), p.delta_abs, p)
        r_j = max(r_j, float(np.max(np.abs(w @ j @ w.conj().T + j_inv))))
    return {
        "hidden_W_H_residual": r_h,
        "hidden_W_jx_polarity_residual": r_j,
        "VW_commutator_residual": float(np.max(np.abs(U_SAME_K @ U_W - U_W @ U_SAME_K))),
    }


def same_k_map_residuals(p: ToyParams | None = None) -> dict[str, float]:
    """Algebraic facts about V = tau_x s_z on the current k grid (no interpretation)."""
    p = p or ToyParams()
    v = U_SAME_K
    r_h = 0.0
    r_j = 0.0
    for kx, ky in k_grid(p.nk1d):
        h_plus = hamiltonian(kx, ky, p.delta_abs, p)
        h_minus = hamiltonian(kx, ky, -p.delta_abs, p)
        r_h = max(r_h, float(np.max(np.abs(v @ h_plus @ v.conj().T - h_minus))))
        j = jx_operator(kx, ky, p.delta_abs, p)
        r_j = max(r_j, float(np.max(np.abs(v @ j @ v.conj().T - j))))
    eye = np.eye(N_ORB, dtype=complex)
    return {
        "same_k_map_H_residual": r_h,
        "same_k_map_jx_commutator_residual": r_j,
        "same_k_map_unitarity_residual": float(np.max(np.abs(v @ v.conj().T - eye))),
    }


def _map_same_k(mats_minus: Array) -> Array:
    v = U_SAME_K
    return np.einsum("ab,...bc,dc->...ad", v.conj().T, mats_minus, v.conj())


def _map_p_steps(steps_minus: Array, points: list[tuple[float, float]], partners: list[int]) -> Array:
    out = np.empty_like(steps_minus)
    for ik, k_plus in enumerate(points):
        ikm = partners[ik]
        for it in range(steps_minus.shape[0]):
            out[it, ik] = map_minus_operator(
                steps_minus[it, ikm], mapped=True, k_plus=k_plus, k_minus=points[ikm]
            )
    return out


def _two_time_packet(times, weights, rho0, steps, cur_a, cur_b, refs) -> Array:
    """Full lower-triangular C^>(t,t') via the existing B1-dev driver (Eq. 7.10)."""
    packet = ClosedWickPacket(
        times=times,
        k_weights=weights,
        rho0=rho0,
        step_unitaries=steps,
        current_a=cur_a,
        current_b=cur_b,
        reference_indices=tuple(int(r) for r in refs),
    )
    res = closed_wick_reference(packet, validation_tolerance=1.0e-9)
    return res.c_greater, res.valid_mask, float(res.metadata["equal_time_residual"])


def two_time_cjj(
    p: ToyParams | None = None,
    *,
    production_t2: bool = False,
    qrt: bool = False,
    dephasing=None,
    blocks: bool = True,
) -> dict[str, object]:
    """CPU reference for the two-time greater correlator on the frozen toy.

    Returns lab-domain C^>_{JJ}(t,t') for +N and -N (basis-independent traces),
    plus the operator-level Je/Jo block decomposition under two candidate
    common-basis maps: the inversion map U_P (k -> -k) and the same-k map V.
    The block decompositions are diagnostics for human review, not pass/fail.
    """
    _reject_open_or_qrt(production_t2=production_t2, qrt=qrt, dephasing=dephasing)
    p = p or ToyParams()
    points = k_grid(p.nk1d)
    partners = partner_index(points)
    nk = len(points)
    weights = np.full(nk, 1.0 / nk)
    times = time_grid(p)
    ax = vector_potential(times, p)
    nt = times.size
    refs = np.arange(nt)

    rho_p, j_p, steps_p = evolve_domain(p.delta_abs, p, times, ax)
    rho_m, j_m, steps_m = evolve_domain(-p.delta_abs, p, times, ax)

    c_plus, mask, eq_res_p = _two_time_packet(times, weights, rho_p[0], steps_p, j_p, j_p, refs)
    c_minus, _, eq_res_m = _two_time_packet(times, weights, rho_m[0], steps_m, j_m, j_m, refs)

    mean_j_plus = bz_mean(j_p, rho_p, weights)
    mean_j_minus = bz_mean(j_m, rho_m, weights)
    mean_jo = 0.5 * (mean_j_plus - mean_j_minus)

    out: dict[str, object] = {
        **p.metadata(),
        "times": times,
        "valid_mask": mask,
        "c_greater_plus": c_plus,
        "c_greater_minus": c_minus,
        "c_greater_domain_odd": 0.5 * (c_plus - c_minus),
        "c_greater_domain_odd_max": float(np.max(np.abs(c_plus - c_minus)) / 2.0),
        "c_greater_plus_max": float(np.max(np.abs(c_plus))),
        "equal_time_limit_residual": max(eq_res_p, eq_res_m),
        "mean_jo_lab_max": float(np.max(np.abs(mean_jo))),
        "mean_je_lab_h1": harmonic_complex(0.5 * (mean_j_plus + mean_j_minus), times, p.omega, 1),
        **same_k_map_residuals(p),
    }

    # operator-level Jo under the two candidate maps
    je_p, jo_p = mapped_bundle(j_p, j_m, points, partners)
    jo_v = 0.5 * (j_p - _map_same_k(j_m))
    je_v = 0.5 * (j_p + _map_same_k(j_m))
    out["operator_jo_frobenius_max_Pmap"] = float(np.max(np.linalg.norm(jo_p, axis=(-2, -1))))
    out["operator_je_frobenius_max_Pmap"] = float(np.max(np.linalg.norm(je_p, axis=(-2, -1))))
    out["operator_jo_frobenius_max_sameKmap"] = float(np.max(np.linalg.norm(jo_v, axis=(-2, -1))))
    out["operator_je_frobenius_max_sameKmap"] = float(np.max(np.linalg.norm(je_v, axis=(-2, -1))))
    out["operator_jo_at_A0_equals_jx_Pmap"] = float(np.max(np.abs(jo_p[0] - j_p[0])))
    rho_m_same_k = _map_same_k(rho_m)
    out["state_map_residual_sameKmap"] = float(np.max(np.abs(rho_m_same_k - rho_p)))
    rho_m_p = mapped_rho(rho_m, points, partners)
    out["state_map_residual_Pmap"] = float(np.max(np.abs(rho_m_p - rho_p)))

    if blocks:
        out["blocks"] = _parity_blocks(
            times, weights, points, partners,
            rho_p, steps_p, rho_m, steps_m, j_p, j_m, je_p, jo_p, je_v, jo_v, refs,
            c_plus, c_minus,
        )
    return out


def _parity_blocks(times, weights, points, partners, rho_p, steps_p, rho_m, steps_m,
                   j_p, j_m, je_p, jo_p, je_v, jo_v, refs, c_plus, c_minus) -> dict[str, object]:
    """C_ee, C_eo, C_oe, C_oo (theory 10.5) in the +N state and in the mapped -N state.

    Theory (10.6)-(10.9) assume a single expectation state.  Here both states are
    reported so the residual of (10.9) shows how much the identity depends on
    rho~_- == rho_+ .  Nothing is asserted; numbers only.
    """
    def four(rho0, steps, je, jo):
        res = {}
        for name, (a, b) in {"ee": (je, je), "eo": (je, jo), "oe": (jo, je), "oo": (jo, jo)}.items():
            c, _, _ = _two_time_packet(times, weights, rho0, steps, a, b, refs)
            res[name] = c
        return res

    steps_m_p = _map_p_steps(steps_m, points, partners)
    rho0_m_p = mapped_rho(rho_m[:1], points, partners)[0]
    steps_m_v = _map_same_k(steps_m)
    rho0_m_v = _map_same_k(rho_m[0])

    blk = {
        "Pmap_plus_state": four(rho_p[0], steps_p, je_p, jo_p),
        "Pmap_minus_state": four(rho0_m_p, steps_m_p, je_p, jo_p),
        "sameKmap_plus_state": four(rho_p[0], steps_p, je_v, jo_v),
    }
    summary: dict[str, object] = {}
    for key, b in blk.items():
        summary[key] = {name: float(np.max(np.abs(c))) for name, c in b.items()}
    # algebraic identities (10.6)/(10.7): exact when evaluated in the matching state
    bp = blk["Pmap_plus_state"]
    bm = blk["Pmap_minus_state"]
    summary["identity_10_6_residual_Pmap"] = float(
        np.max(np.abs(c_plus - (bp["ee"] + bp["eo"] + bp["oe"] + bp["oo"])))
    )
    summary["identity_10_7_residual_Pmap"] = float(
        np.max(np.abs(c_minus - (bm["ee"] - bm["eo"] - bm["oe"] + bm["oo"])))
    )
    # (10.9) with a single state: (C+ - C-)/2 vs C_eo + C_oe from the +N state
    summary["identity_10_9_residual_Pmap_plus_state"] = float(
        np.max(np.abs(0.5 * (c_plus - c_minus) - (bp["eo"] + bp["oe"])))
    )
    summary["identity_10_9_residual_Pmap_minus_state"] = float(
        np.max(np.abs(0.5 * (c_plus - c_minus) - (bm["eo"] + bm["oe"])))
    )
    bv = blk["sameKmap_plus_state"]
    summary["identity_10_6_residual_sameKmap"] = float(
        np.max(np.abs(c_plus - (bv["ee"] + bv["eo"] + bv["oe"] + bv["oo"])))
    )
    summary["arrays"] = blk
    return summary


def jo_case_report(tt: Mapping[str, object], floor: float = 1.0e-12) -> dict[str, object]:
    """Classify operator-level vs expectation-level Jo.  Facts only."""
    op_p = float(tt["operator_jo_frobenius_max_Pmap"])
    op_v = float(tt["operator_jo_frobenius_max_sameKmap"])
    ev = float(tt["mean_jo_lab_max"])
    scale = max(float(tt["operator_je_frobenius_max_Pmap"]), 1.0)
    if ev > floor * scale:
        case = "CASE_D_expectation_nonzero"
    elif op_v <= floor * scale and op_p > floor * scale:
        case = "CASE_C_map_dependent"
    elif op_v <= floor * scale and op_p <= floor * scale:
        case = "CASE_A"
    else:
        case = "CASE_B"
    return {
        "case": case,
        "operator_jo_max_Pmap": op_p,
        "operator_jo_max_sameKmap": op_v,
        "expectation_jo_lab_max": ev,
        "same_k_map_H_residual": tt["same_k_map_H_residual"],
        "same_k_map_jx_commutator_residual": tt["same_k_map_jx_commutator_residual"],
        "state_map_residual_sameKmap": tt["state_map_residual_sameKmap"],
        "state_map_residual_Pmap": tt["state_map_residual_Pmap"],
        "note": (
            "Under the inversion map U_P (k->-k, field-reversing) the operator Jo is O(1) "
            "and equals j_x at A=0; under the same-k map V=tau_x s_z it vanishes identically "
            "and rho_- = V rho_+ V^-1.  Which map defines the physical J_o is not decided here."
        ),
    }


def stage1_table(result: Mapping[str, object], it: int | None = None) -> dict[str, object]:
    times = np.asarray(result["times"])
    if it is None:
        it = times.size // 2
    return {
        "label": LABEL,
        "status": STATUS,
        "not_a_physics_result": True,
        "not_quantum_output_g2": True,
        "t_index": int(it),
        "t": float(times[it]),
        "Jo_2omega": result["jo_h2"],
        "Jo_2omega_hard_test_enabled": False,
        "accidental_same_k_isospectrum": True,
        "C_JJ_equal_time": complex(np.asarray(result["c_oo_plus"])[it]),
        "C_JJ_N_even_residual": result["c_oo_even_residual"],
        "C_eo_plus_C_oe_legacy_name": complex(np.asarray(result["c_odd"])[it]),
        "C_eo_plus_C_oe": complex(np.asarray(result["c_odd"])[it]),
        "note": (
            "C_JJ(+N)~C_JJ(-N) is N-even, not a failed magnet test. "
            "Jo(2w) hard test is off: V and W structural zeros. "
            "Do not call (C(+N)-C(-N))/2 C_oo. "
            "c_oo_* and C_eo_plus_C_oe keys are legacy labels; not for physics figures."
        ),
    }
