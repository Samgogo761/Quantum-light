#!/usr/bin/env python3
"""Two-port free-space sheet self-field G_self. Distinct from collection K.

Frozen 2026-09-21 (human node, not an engineering menu):

1. P1 is the given local classical drive at the sample. Do not re-solve
   classical Maxwell for A_cl. Quantum fluctuations ride on top:
   A_sheet = A_cl + δA, and only δI sources G_self.
2. Geometry: infinite uniform 2D sheet, vacuum on both sides, q_parallel=0.
   Ports: L_in, R_in → sheet → L_out, R_out. The unused-side vacuum is kept
   because [a, a†]=1 even when ⟨a† a⟩=0.
3. Feedback is solved on the continuum time / frequency field. H2 is a
   detection packet applied to a_R_out after the field is known.

G_self ≠ K. K collects radiation onto a normalized photon packet (~1/√ω).
G_self is the local radiation reaction on the sheet (~∫ dt, or 1/ω).

Not a cavity, not far-field photon statistics, not a BZ claim.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from q8_m0_mode_io import ALPHA_FS, L0_ANG, eta2_ncell
from q8_m0_s1_kernel import pack_complex

GSHEET_ID = "two_port_vacuum_sheet_Gself_20260921"
GSHEET_STATUS = "frozen_two_port_P1_local_continuum_feedback"
FOURIER_NOTE = (
    "Code synthesis uses e^{+i omega t}. Then d/dt → +i omega and E = -dA/dt "
    "gives E(omega) = -i omega A(omega). Do not write a bare 1/omega formula "
    "without the finite-t0 initial value."
)


@dataclass(frozen=True)
class TwoPortSheetSelfField:
    ident: str = GSHEET_ID
    status: str = GSHEET_STATUS
    geometry: str = "infinite_uniform_2d_sheet_vacuum_both_sides_q0_normal"
    p1_meaning: str = "given_local_classical_A_at_sample_not_re_solved_Maxwell"
    quantum_split: str = "A_sheet = A_cl + delta_A; only delta_I sources G_self"
    ports: str = "L_in,R_in -> sheet -> L_out,R_out"
    polarizations: str = "two_inplane_cartesian_kept"
    feedback_space: str = "continuum_time_field_not_H1_H2_H3_port_list"
    detection: str = "project_H2_on_a_R_out_after_feedback"
    a_self_initial: str = "A_self(t0)=0 at the start of the stored electronic window"
    not_collection_K: bool = True
    not_cavity: bool = True
    classical_current_reenters_G: bool = False
    finite_q_parallel: str = "not_this_version"
    dissipation_added: str = "none"


def gamma_self(geo) -> float:
    """dκ_self / dt = γ I_cell, I_cell = -∂H/∂A per cell.

    SI: dA_SI/dt = (Z0/2) j_sheet = (Z0/(2 A_uc)) I_cell_SI.
    Code ħ=e=1 with the same convention already used for η² N_cell = α_fs/A_uc
    takes Z0 = 4π α_fs, hence γ = 2π α_fs / A_uc = 2π η² N_cell.

    Length unit Å sits in A_uc. Energy unit t_h=1. c is not restored as
    retardation: at the infinitely thin q=0 sheet the local reaction is
    instantaneous, E_self(t)=-(Z0/2) j(t).
    """
    return float(2.0 * np.pi * eta2_ncell(geo))


def gamma_vs_eta_table(geo) -> dict:
    pref = eta2_ncell(geo)
    gamma = gamma_self(geo)
    return {
        "A_uc_ang2": float(geo.a_cell_ang2),
        "L0_ang": L0_ANG,
        "alpha_fs": ALPHA_FS,
        "eta2_Ncell": pref,
        "gamma_self": gamma,
        "gamma_over_eta2_Ncell": float(gamma / pref),
        "expect_gamma_over_eta2": 2.0 * np.pi,
        "K_frequency_weight": "1/sqrt(omega)",
        "G_self_time": "theta(t-s) integrator, A_self(t0)=0",
        "G_self_not_equal_K": True,
        "S_cancels_like_connected_moments": True,
        "nk_is_not_N_cell": True,
    }


def a_self_from_current(i_cell, dt: float, gamma: float, a0=None) -> np.ndarray:
    """A_self(t) = A_self(t0) + γ ∫_{t0}^t I_cell(s) ds, left Riemann.

    i_cell: (nt, 2). A[n] uses I[0]..I[n-1] only, matching prefix R.
    """
    i_cell = np.asarray(i_cell, dtype=float)
    nt = i_cell.shape[0]
    a = np.zeros_like(i_cell)
    if a0 is not None:
        a[0] = np.asarray(a0, dtype=float)
    acc = np.array(a[0], dtype=float, copy=True)
    for n in range(nt):
        a[n] = acc
        acc = acc + gamma * dt * i_cell[n]
    return a


def two_port_from_current(i_cell, a_L_in, a_R_in, dt: float, gamma: float) -> dict:
    """Fluctuation ports. Classical P1 current is not added to i_cell here."""
    i_cell = np.asarray(i_cell, dtype=float)
    a_L_in = np.asarray(a_L_in, dtype=float)
    a_R_in = np.asarray(a_R_in, dtype=float)
    a_self = a_self_from_current(i_cell, dt, gamma)
    a_sheet = a_L_in + a_R_in + a_self
    a_R_out = a_L_in + a_self
    a_L_out = a_R_in + a_self
    return {
        "A_self": a_self,
        "A_sheet": a_sheet,
        "A_R_out": a_R_out,
        "A_L_out": a_L_out,
        "E_self_leftrect": -gamma * i_cell,
        "note": (
            "E = -dA/dt. With left-rect G, E_self(n) = -γ I(n) is the "
            "consistent difference of the integrator that uses past I for A."
        ),
    }


def j_zero_identities(a_L_in, a_R_in, dt: float, gamma: float) -> dict:
    """I=0 ⇒ A_R_out=A_L_in, A_L_out=A_R_in, A_sheet=A_L_in+A_R_in."""
    i = np.zeros_like(a_L_in, dtype=float)
    rec = two_port_from_current(i, a_L_in, a_R_in, dt, gamma)
    err_R = float(np.max(np.abs(rec["A_R_out"] - a_L_in)))
    err_L = float(np.max(np.abs(rec["A_L_out"] - a_R_in)))
    err_s = float(np.max(np.abs(rec["A_sheet"] - a_L_in - a_R_in)))
    err_self = float(np.max(np.abs(rec["A_self"])))
    return {
        "maxabs_A_R_out_minus_A_L_in": err_R,
        "maxabs_A_L_out_minus_A_R_in": err_L,
        "maxabs_A_sheet_minus_A_in_sum": err_s,
        "maxabs_A_self": err_self,
        "pass": bool(max(err_R, err_L, err_s, err_self) < 1e-14),
    }


def prescribed_current_e_self_check(i_cell, dt: float, gamma: float) -> dict:
    """For prescribed I, A_self is the integrator and E_self = -γ I."""
    i_cell = np.asarray(i_cell, dtype=float)
    zeros = np.zeros_like(i_cell)
    rec = two_port_from_current(i_cell, zeros, zeros, dt, gamma)
    err = float(np.max(np.abs(rec["E_self_leftrect"] + gamma * i_cell)))
    a_end = rec["A_self"][-1]
    integ = gamma * dt * np.sum(i_cell[:-1], axis=0)
    err_a = float(np.max(np.abs(a_end - integ)))
    return {
        "maxabs_E_self_plus_gamma_I": err,
        "maxabs_A_end_minus_gamma_sum_past_I": err_a,
        "pass": bool(err < 1e-14 and err_a < 1e-12),
    }


def gsheet_spec(geo=None) -> dict:
    rec = asdict(TwoPortSheetSelfField())
    rec["fourier_convention"] = FOURIER_NOTE
    rec["maxwell_si"] = {
        "E_self": "-(Z0/2) j_sheet",
        "E_sheet": "E_L_in + E_R_in - (Z0/2) j_sheet",
        "E_R_out": "E_L_in - (Z0/2) j_sheet",
        "E_L_out": "E_R_in - (Z0/2) j_sheet",
        "no_double_count_two_sides_at_the_sheet": True,
        "A_self": "A_self(t0) + (Z0/(2 S)) ∫ I_total = A_self(t0) + (Z0/(2 A_uc)) ∫ I_cell",
    }
    rec["code_units"] = {
        "hbar_e_t_h": 1,
        "length": "Angstrom",
        "kappa": "k + A, A in Angstrom^{-1}",
        "I_cell": "-dH/dA per cell",
        "gamma_self": "dA_self/dt = gamma I_cell, gamma = 2 pi alpha_fs / A_uc",
    }
    rec["vs_K"] = {
        "K": "i eta F, F=∫ phi/sqrt(omega) e^{+i omega t}, collection packet",
        "G_self": "theta(t-s) integrator on the full two-component current",
        "same_Maxwell_two_uses": True,
        "not_the_same_function": True,
    }
    rec["p1"] = {
        "A_cl": "already inside the stored electronic trajectory",
        "classical_J_does_not_source_G_self": True,
        "quantum_delta_A_does": True,
    }
    rec["photon_output"] = "not_production"
    rec["g2"] = "not_run"
    rec["nk96"] = "not_run"
    if geo is not None:
        rec["gamma_table"] = gamma_vs_eta_table(geo)
    return rec


def ohmic_sheet_transmission(sigma_z0_over_2: float) -> dict:
    """Static conducting-sheet check of signs, not the electronic R^A.

    J = σ E_loc, E_loc = E_L+E_R-(Z0/2)J.
    t = 1 / (1 + σ Z0/2), r = - (σ Z0/2) / (1 + σ Z0/2), t = 1+r.
    """
    x = float(sigma_z0_over_2)
    den = 1.0 + x
    t = 1.0 / den
    r = -x / den
    return {
        "sigma_Z0_over_2": x,
        "T": t,
        "R": r,
        "T_minus_1_minus_R": float(t - (1.0 + r)),
        "pass_T_eq_1_plus_R": bool(abs(t - (1.0 + r)) < 1e-15),
        "kind": "static_ohmic_sheet_sign_check_not_electronic_RA",
    }
