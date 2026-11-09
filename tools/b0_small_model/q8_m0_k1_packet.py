#!/usr/bin/env python3
"""K1 electromagnetic frequency packet. Not a photon-number evaluation.

b is assembled from a 1D continuum of beam operators a(ω) with
∫_0^∞ |φ(ω)|² dω = 1, hence [b, b†] = 1 for that beam. The radiation
kernel χ(ω) stays inside unset g. Q9 is not L2-normalized into a photon.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from q8_m0_cjj import window_fourier
from q8_m0_classical import p1_a2_protocol
from q8_m0_k1_chi_interface import chi_spec
from q8_m0_s1_kernel import pack_complex, s1_d_mu_nu

PACKET_ID = "K1_R0_reference_perp_H2_pulse_matched"
PACKET_STATUS = "adopted_as_reference_output_mode_not_experimental_optimum"
ROUTE_STATUS = "adopted_em_first_option2_20260917"

FOURIER_CONVENTION = {
    "analysis_Phi": "Phi(omega)=∫_0^{Tp} W(t) exp(-i(omega-omega_c)t) dt = Tp * I(omega_c-omega)",
    "window_fourier_I": "I(Omega)=∫_0^{Tp} (W/Tp) exp(+i Omega t) dt  [q8_m0_cjj.window_fourier]",
    "synthesis_psi": "psi(t)=∫_0^∞ d omega phi(omega) exp(+i omega t)",
    "q9_current_filter": "f_Q9(t)=(W/Tp) exp(+i n omega_1 t)",
    "plancherel": "∫ dν |Ŵ(ν)|² = 2π ∫ dt |W(t)|² with Ŵ(ν)=∫ W exp(-i ν t) dt",
    "positive_frequency_only": "phi and b live on omega>0; omega<0 is Plancherel check only",
}


@dataclass(frozen=True)
class ReferencePhysicalPacket:
    ident: str = PACKET_ID
    status: str = PACKET_STATUS
    role: str = "first_reference_output_mode_not_final_experimental_optimum"
    polarization: str = "e_perp_mirror_normal"
    collection_direction: str = "sample_normal_one_spatial_beam"
    inplane_momentum: str = "q0_uniform_sheet"
    numerical_aperture: str = "unset_absorbed_in_chi"
    center_harmonic: int = 2
    bandwidth: str = "pulse_matched_P1_sin2_FT_around_2omega"
    frequency_measure: str = "int_0^inf |phi|^2 d omega = 1"
    commutator: str = "[b,bdagger]=1 on this beam packet"
    coupling_chi: str = "unset_awaiting_theory"
    n_cell: str = "unset"
    photon_output: str = "not_run"
    not_q9_renormalized_current_filter: bool = True


def phi_unnormalized(omega, omega_c: float, tp: float) -> np.ndarray:
    """Φ(ω) = ∫_0^{Tp} W(t) exp(-i(ω-ω_c)t) dt = Tp * I(ω_c-ω)."""
    om = np.asarray(omega, dtype=float)
    return tp * window_fourier(omega_c - om, tp)


def omega_grid(omega0: float, tp: float, *, n_width: int = 64, harm_max: float = 12.0) -> np.ndarray:
    dw = (2.0 * np.pi / tp) / n_width
    wmax = harm_max * omega0
    return np.arange(0.0, wmax + 0.5 * dw, dw)


def normalize_phi(omega, phi_raw) -> tuple[np.ndarray, float]:
    raw = np.asarray(phi_raw, dtype=complex)
    mass = float(np.trapezoid(np.abs(raw) ** 2, omega))
    if mass <= 0.0:
        raise RuntimeError("phi has no L2 mass on ω>0")
    norm = float(np.sqrt(mass))
    return raw / norm, norm


def packet_on_grid(omega0: float, tp: float, *, n_harm: int = 2) -> dict:
    omega_c = float(n_harm) * omega0
    omega = omega_grid(omega0, tp)
    raw = phi_unnormalized(omega, omega_c, tp)
    phi, nrm = normalize_phi(omega, raw)
    mass = float(np.trapezoid(np.abs(phi) ** 2, omega))
    i_c = int(np.argmin(np.abs(omega - omega_c)))
    peaks = {}
    for n in (0, 1, 2, 3, 4):
        i = int(np.argmin(np.abs(omega - n * omega0)))
        peaks[f"abs_phi_H{n}"] = float(np.abs(phi[i]))
    plancherel_all = _plancherel_check(omega0, tp, n_harm=n_harm)
    return {
        "omega": omega,
        "phi": phi,
        "omega_c": omega_c,
        "l2_norm_raw": nrm,
        "commutator_integral": mass,
        "abs_phi_at_center": float(np.abs(phi[i_c])),
        "peaks": peaks,
        "plancherel": plancherel_all,
        "packet": asdict(ReferencePhysicalPacket()),
    }


def _plancherel_check(omega0: float, tp: float, *, n_harm: int) -> dict:
    """∫_{-∞}^{∞} |Φ|² dω  ?=?  2π ∫ |W|² dt = 2π (3/8) Tp."""
    omega_c = float(n_harm) * omega0
    dw = (2.0 * np.pi / tp) / 64.0
    wmax = 12.0 * omega0
    omega = np.arange(-wmax, wmax + 0.5 * dw, dw)
    ph = phi_unnormalized(omega, omega_c, tp)
    lhs = float(np.trapezoid(np.abs(ph) ** 2, omega))
    rhs = 2.0 * np.pi * (3.0 / 8.0) * tp
    return {
        "int_all_omega_abs_Phi2": lhs,
        "two_pi_int_W2": rhs,
        "rel_diff": abs(lhs - rhs) / rhs,
        "note": "Plancherel for FT[W] with Ŵ(ν)=∫ W e^{-iνt} dt",
    }


def scale_psi_over_q9(omega0: float, tp: float, *, n_harm: int = 2) -> dict:
    """If χ=1, ψ(t)≈α f_Q9(t) with f_Q9=(W/Tp) e^{i n ω t}.

    Inverse FT gives α ≈ 2π Tp / ||Φ||_L2(ω>0) when the packet sits on ω>0.
    """
    rec = packet_on_grid(omega0, tp, n_harm=n_harm)
    nrm = rec["l2_norm_raw"]
    alpha_plancherel = 2.0 * np.pi * tp / nrm
    # Restrict Plancherel to ω>0 only: the inverse uses ∫_0^∞ Φ e^{iωt} dω
    # which equals 2π W e^{i ω_c t} only after adding ω<0. Report both.
    omega_c = rec["omega_c"]
    omega = rec["omega"]
    phi = rec["phi"]
    t_mid = 0.5 * tp
    psi_mid = complex(np.trapezoid(phi * np.exp(1j * omega * t_mid), omega))
    f_q9_mid = (np.sin(np.pi * t_mid / tp) ** 2) / tp * np.exp(1j * omega_c * t_mid)
    alpha_mid = psi_mid / f_q9_mid if abs(f_q9_mid) > 1e-18 else np.nan
    return {
        "alpha_plancherel_full_line": float(alpha_plancherel),
        "alpha_from_t_mid": pack_complex(alpha_mid),
        "d_scale_from_t_mid": float(abs(alpha_mid) ** 2),
        "q9_not_a_photon": True,
        "packet": rec["packet"],
        "commutator_integral": rec["commutator_integral"],
        "plancherel": rec["plancherel"],
        "peaks": rec["peaks"],
    }


def y_perp_h2(y, geo):
    from q8_m0_s1_kernel import y_cartesian_to_parallel_perp

    _y_par, y_perp = y_cartesian_to_parallel_perp(y, geo)
    return y_perp[:, 1]


def reproject_d_perpperp(y, w, geo, alpha: complex) -> dict:
    """d_φ = |α|² d_Q9 for a pulse-matched packet (same envelope, EM L2 in ω)."""
    d_q9 = s1_d_mu_nu(y, w, geo)
    d_pp = complex(d_q9[1, 1])
    return {
        "d_Q9_perpperp_H2": pack_complex(d_pp),
        "d_phi_perpperp_H2": pack_complex((alpha * np.conj(alpha)) * d_pp),
        "alpha": pack_complex(alpha),
        "note": (
            "narrowband shape-only source kernel (d_φ ≈ |α|² d_Q9); "
            "not a strict identity; not a photon number; chi_rad unset"
        ),
        "photon_output": "not_run",
        "kind": "narrowband_shape_only_source_projection",
    }


def spec_scale(omega0: float | None = None, tp: float | None = None) -> dict:
    prot = p1_a2_protocol(0.8)
    omega0 = prot.omega0 if omega0 is None else float(omega0)
    tp = prot.t_pulse if tp is None else float(tp)
    return {
        "route": ROUTE_STATUS,
        "packet_id": PACKET_ID,
        "packet_status": PACKET_STATUS,
        "protocol_id": prot.protocol_id,
        "omega0": omega0,
        "t_pulse": tp,
        "scale": scale_psi_over_q9(omega0, tp),
        "fourier_convention": FOURIER_CONVENTION,
        "chi": chi_spec(),
        "b_out_language": "reference_mode_operator_defined; absolute_n_not_evaluated",
        "photon_output": "not_run",
        "g2": "not_run",
        "quantum_pump": "not_run",
        "d_parperp": "unresolved_magnetic_candidate_not_reprojected_to_light",
        "role": "first_reference_output_mode_not_final_experimental_optimum",
    }
