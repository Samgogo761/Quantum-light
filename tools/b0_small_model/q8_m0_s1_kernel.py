#!/usr/bin/env python3
"""S1-v0 reduced collection kernel. Not a frozen photon mode.

Adopted as a source-filter / collection coordinate (not photon number).
g_μ and N_cell remain unset; [b, b†]=1 is not imposed.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from q8_m0_classical import _sin2_window
from q8_m0_model import M0Geometry, build_geometry

S1_STATUS = "adopted_as_source_filter_not_photon_mode"
S1_ID = "S1_v0_normal_q0_H1parallel_H2perp"
S1_NOT = (
    "not a bosonic photon wavepacket, not a photon number, not g2, "
    "not cavity output, not commutator-normalized K"
)


@dataclass(frozen=True)
class S1ReducedKernel:
    """Polarization-aligned source filter. Physical K = g_μ * this, g_μ unset."""

    ident: str = S1_ID
    status: str = S1_STATUS
    collection: str = "sample_normal_q0_farfield_sheet"
    window: str = "Q9_sin2_over_Tp"
    parallel_order: int = 1
    perp_order: int = 2
    g_scale: str = "unset"
    n_cell: str = "unset"
    numerical_aperture: str = "unset"
    commutator_normalized: bool = False


def parallel_perp_basis(geo: M0Geometry) -> tuple[np.ndarray, np.ndarray]:
    """ê_∥ along a2, ê_⊥ along the in-plane mirror normal n̂."""
    e_par = np.asarray(geo.a2_cart[:2], dtype=float)
    e_par = e_par / np.linalg.norm(e_par)
    e_perp = np.asarray(geo.mirror_normal_inplane, dtype=float)
    e_perp = e_perp / np.linalg.norm(e_perp)
    return e_par, e_perp


def cartesian_from_parallel_perp(j_par, j_perp, geo: M0Geometry) -> np.ndarray:
    e_par, e_perp = parallel_perp_basis(geo)
    return np.asarray(j_par)[..., None] * e_par + np.asarray(j_perp)[..., None] * e_perp


def parallel_perp_from_cartesian(j_xy, geo: M0Geometry) -> tuple[np.ndarray, np.ndarray]:
    """j_xy[..., 2] → (j_∥, j_⊥)."""
    e_par, e_perp = parallel_perp_basis(geo)
    j = np.asarray(j_xy)
    return j[..., 0] * e_par[0] + j[..., 1] * e_par[1], j[..., 0] * e_perp[0] + j[..., 1] * e_perp[1]


def reduced_k_mu_a(times, omega: float, tp: float, geo: M0Geometry) -> dict:
    """K̃_{μ a}(t) for μ ∈ {parallel, perp}, a ∈ {x, y}.

    K̃_μa(t) = ê_{μ,a} (W(t)/T_p) exp(+i n_μ ω t), n_∥=1, n_⊥=2.
    """
    t = np.asarray(times, dtype=float)
    w = _sin2_window(t, tp)
    phase = np.exp(1j * np.array([1, 2])[:, None] * omega * t[None, :])  # (2, nt)
    filt = (w / tp) * phase
    e_par, e_perp = parallel_perp_basis(geo)
    k = np.zeros((2, 2, t.size), dtype=complex)  # μ, a, t
    k[0, :, :] = e_par[:, None] * filt[0]
    k[1, :, :] = e_perp[:, None] * filt[1]
    return {
        "mu": ("parallel_H1", "perp_H2"),
        "cartesian": ("x", "y"),
        "K_tilde": k,
        "e_parallel": e_par,
        "e_perp": e_perp,
        "spec": asdict(S1ReducedKernel()),
    }


def y_cartesian_to_parallel_perp(y, geo: M0Geometry) -> tuple[np.ndarray, np.ndarray]:
    """Y[k,n,a,c,v] cartesian a=x,y → (Y_∥, Y_⊥) each [k,n,c,v]."""
    e_par, e_perp = parallel_perp_basis(geo)
    y = np.asarray(y)
    y_par = y[:, :, 0] * e_par[0] + y[:, :, 1] * e_par[1]
    y_perp = y[:, :, 0] * e_perp[0] + y[:, :, 1] * e_perp[1]
    return y_par, y_perp


def d_dir_from_y(y_dir, w) -> np.ndarray:
    """d[n,m] = Σ_k w Tr[Y_n† Y_m] for a single in-plane direction."""
    return np.einsum("k,kncv,kmcv->nm", w, np.conj(y_dir), y_dir)


def s1_reduced_y(y, geo: M0Geometry) -> np.ndarray:
    """Stack H1∥ and H2⊥ as μ=0,1. Shape (nk, 2, 4, 4)."""
    y_par, y_perp = y_cartesian_to_parallel_perp(y, geo)
    return np.stack([y_par[:, 0], y_perp[:, 1]], axis=1)


def s1_d_mu_nu(y, w, geo: M0Geometry) -> np.ndarray:
    """2×2 reduced Gram: μ,ν ∈ {∥@H1, ⊥@H2}."""
    ys = s1_reduced_y(y, geo)
    return np.einsum("k,kacv,kbcv->ab", w, np.conj(ys), ys)


def pack_complex(z) -> dict:
    z = complex(z)
    return {"re": float(z.real), "im": float(z.imag), "abs": float(abs(z))}


def s1_report_from_y(y, w, geo: M0Geometry) -> dict:
    y_par, y_perp = y_cartesian_to_parallel_perp(y, geo)
    d_s1 = s1_d_mu_nu(y, w, geo)
    d_pp = d_dir_from_y(y_par, w)
    d_nn = d_dir_from_y(y_perp, w)
    d_pn = np.einsum("k,kncv,kmcv->nm", w, np.conj(y_par), y_perp)
    return {
        "d_s1": {
            "d_parpar_H1H1": pack_complex(d_s1[0, 0]),
            "d_parperp_H1H2": pack_complex(d_s1[0, 1]),
            "d_perppar_H2H1": pack_complex(d_s1[1, 0]),
            "d_perpperp_H2H2": pack_complex(d_s1[1, 1]),
        },
        "d_parallel_H1H1": pack_complex(d_pp[0, 0]),
        "d_perp_H2H2": pack_complex(d_nn[1, 1]),
        "d_parperp_all_H1H2": pack_complex(d_pn[0, 1]),
        "d_parperp_H2H2": pack_complex(d_pn[1, 1]),
        "d_parpar_H2H2": pack_complex(d_pp[1, 1]),
        "d_perpperp_H1H1": pack_complex(d_nn[0, 0]),
    }


def spec_dict(geo: M0Geometry | None = None) -> dict:
    geo = geo or build_geometry()
    e_par, e_perp = parallel_perp_basis(geo)
    return {
        "s1_id": S1_ID,
        "status": S1_STATUS,
        "not": S1_NOT,
        "kernel": asdict(S1ReducedKernel()),
        "e_parallel": e_par.tolist(),
        "e_perp": e_perp.tolist(),
        "observables": {
            "mean_magnetic": "Jo_perp windowed at 2ω (same W/Tp)",
            "mean_common": "Je_parallel windowed at ω",
            "cjj_magnetic": "d_parallel_perp = Σ_k w Tr(Y_∥† Y_⊥)",
            "cjj_even_controls": ["d_parallel_parallel", "d_perp_perp"],
            "forbidden_if_M_exact": ["Jo_parallel", "Je_perp", "d_o of diagonal kernels"],
        },
        "channel_qualification": {
            "d_parperp_H1H2_odd": "unresolved_magnetic_candidate",
            "d_perpperp_H2_even": "retained_algorithm_baseline",
            "Jo_perp_H2": "unresolved",
            "decision": "physics3_execution2_20260917",
        },
        "a2_propagation": "engineering_P1_a2_same_envelope",
        "nk96": "not_run",
        "nk192": "not_run",
        "quantum_input": "not_run",
        "photon_output": "not_run",
    }
