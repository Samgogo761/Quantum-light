#!/usr/bin/env python3
"""Shared-mode k,-k static construction. No dynamics, not HHG, not a BSV pulse.

H = omega0 n + He(k, X; N) + He(-k, X; N)
Two electronic subsystems, one projected photon mode, same Peierls X = a+a†.
Independent archive. Does not rewrite trial.json / preflight.json / diag.json.
Does not run G1/H2, long evolution, full BZ, SV, or semiclassical trajectories.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

import q8_m0_qinput_dynamics_20261026 as dyn
import q8_m0_qinput_minimal_20261020 as qm
from q8_m0_model import R0, build_geometry, op_P

MODEL_ID = "M0_shared_mode_k_minus_k_static_v1"
G_ABS = 0.005
OMEGA0 = 0.05
K_FRAC = [0.173, 0.271]
NMAX_INTERFACE = 12
NMAX_APPLY = 2
NEEL = 1


def _n_up(det) -> int:
    return sum(1 for orb in det if orb % 2 == 0)


def sector36_tables():
    idx36 = [i for i, det in enumerate(qm.DETS) if _n_up(det) == 2]
    to36 = -np.ones(qm.ELECTRON_DIM, dtype=int)
    to36[idx36] = np.arange(len(idx36))
    return np.array(idx36, dtype=int), to36


IDX36, TO36 = sector36_tables()
EL36 = int(IDX36.size)


def restrict36(mat70: np.ndarray) -> np.ndarray:
    if mat70.ndim == 2:
        return mat70[np.ix_(IDX36, IDX36)]
    if mat70.ndim == 3:
        return mat70[np.ix_(np.arange(mat70.shape[0]), IDX36, IDX36)]
    raise ValueError("expected 2d or 3d array")


def embed36(vec36: np.ndarray) -> np.ndarray:
    out = np.zeros(qm.ELECTRON_DIM, dtype=complex)
    out[IDX36] = vec36
    return out


def spin_closure_report(geo, p):
    bonds = qm.hopping_bonds(geo, p)
    spin_flip = [b for b in bonds if (b.row % 2) != (b.col % 2)]
    h70 = qm.lift_one_body(qm.onsite(p, NEEL))
    for b in bonds:
        h70 = h70 + b.amplitude * qm.fermion_bilinear(b.row, b.col)
    j70 = np.zeros((2, qm.ELECTRON_DIM, qm.ELECTRON_DIM), complex)
    for b in bonds:
        F = b.amplitude * qm.fermion_bilinear(b.row, b.col)
        j70 += -1j * np.asarray(b.delta)[:, None, None] * F
    comp = np.array([i for i in range(qm.ELECTRON_DIM) if i not in set(IDX36.tolist())], int)
    leak_h = float(np.max(np.abs(h70[np.ix_(IDX36, comp)]))) if comp.size else 0.0
    leak_j = float(np.max(np.abs(j70[np.ix_(np.arange(2), IDX36, comp)]))) if comp.size else 0.0
    return {
        "n_dets_70": qm.ELECTRON_DIM,
        "n_sector_36": EL36,
        "n_complement": int(comp.size),
        "n_bonds": len(bonds),
        "n_spin_flip_bonds": len(spin_flip),
        "max_abs_H_sector_to_complement": leak_h,
        "max_abs_J_sector_to_complement": leak_j,
        "closed_under_onsite_and_hopping": bool(leak_h < 1e-14 and leak_j < 1e-14 and not spin_flip),
        "peierls_note": "Peierls tensors photon D(q s X) onto the same spin-conserving bilinears; it cannot open the sector.",
    }


def slater36(occupied: np.ndarray):
    el70 = qm.slater_state(occupied)
    el = el70[IDX36].copy()
    nrm = np.linalg.norm(el)
    leak = float(np.linalg.norm(el70) ** 2 - nrm ** 2)
    if nrm <= 0.0:
        raise RuntimeError("Slater has no mass in the 36-sector")
    return el / nrm, float(nrm ** 2), leak


def apply_photon_parity(v: np.ndarray, el_dim: int, d1: int) -> np.ndarray:
    V = np.asarray(v, complex).reshape(el_dim, el_dim, d1).copy()
    V *= np.array([(-1.0) ** n for n in range(d1)])[None, None, :]
    return V.ravel()


def swap_electronic_factors(v: np.ndarray, el_dim: int, d1: int) -> np.ndarray:
    return np.transpose(np.asarray(v, complex).reshape(el_dim, el_dim, d1), (1, 0, 2)).ravel()


def swap_pair_matrix(h: np.ndarray) -> np.ndarray:
    d = EL36
    return np.transpose(np.asarray(h, complex).reshape(d, d, d, d), (1, 0, 3, 2)).reshape(d * d, d * d)


def classical_x_energy(model: SharedModel, ek, emk, x: float) -> complex:
    """Electronic energy with c-number X=x at s=1. Not a photon expectation."""
    e = np.vdot(ek, model.onsite_k * ek) + np.vdot(emk, model.onsite_mk * emk)
    for Fk, Fmk, q in zip(model.f_k, model.f_mk, model.q_units):
        phase = np.exp(1j * float(q) * float(x))
        e = e + np.vdot(ek, Fk @ ek) * phase + np.vdot(emk, Fmk @ emk) * phase
    return complex(e)


@dataclass(frozen=True)
class SharedModel:
    k: np.ndarray
    neel: int
    n_max: int
    omega0: float
    g: float
    e: np.ndarray
    onsite_k: np.ndarray
    onsite_mk: np.ndarray
    q_units: np.ndarray
    f_k: tuple
    f_mk: tuple
    j_k: tuple
    j_mk: tuple

    @property
    def d1(self) -> int:
        return self.n_max + 1

    @property
    def el_dim(self) -> int:
        return EL36

    @property
    def dim(self) -> int:
        return EL36 * EL36 * self.d1

    @property
    def n_electronic_subsystems(self) -> int:
        return 2


def _accumulate(geo, p, k, g, e):
    groups, jgroups = {}, {}
    for bond in qm.hopping_bonds(geo, p):
        qu = round(float(g * np.dot(e, bond.delta)), 12)
        if qu not in groups:
            groups[qu] = np.zeros((EL36, EL36), complex)
            jgroups[qu] = np.zeros((2, EL36, EL36), complex)
        phase = bond.amplitude * np.exp(1j * np.dot(k, bond.delta))
        F = restrict36(phase * qm.fermion_bilinear(bond.row, bond.col))
        groups[qu] += F
        jgroups[qu] += -1j * np.asarray(bond.delta)[:, None, None] * F
    keys = np.array(sorted(groups), float)
    return keys, tuple(groups[q] for q in keys), tuple(jgroups[q] for q in keys)


def build_shared(geo, p, k, neel: int, *, n_max: int, omega0: float, g: float) -> SharedModel:
    e = qm.a2_polarization(geo)
    onsite = dyn.onsite_determinant_energy(p, neel)[IDX36]
    qk, fk, jk = _accumulate(geo, p, k, g, e)
    qmk, fmk, jmk = _accumulate(geo, p, -np.asarray(k, float), g, e)
    if qk.shape != qmk.shape or np.max(np.abs(qk - qmk)) > 1e-15:
        raise RuntimeError("k and -k produced different Peierls q-groups; grouping must be shared")
    return SharedModel(np.asarray(k, float), int(neel), int(n_max), float(omega0), float(g), e,
                       onsite, onsite.copy(), qk, fk, fmk, jk, jmk)


def apply_h(model: SharedModel, v: np.ndarray, s: float, Ds=None) -> np.ndarray:
    d, d1 = model.el_dim, model.d1
    V = v.reshape(d, d, d1)
    nph = np.arange(d1, dtype=float)
    out = (model.onsite_k[:, None, None] + model.onsite_mk[None, :, None]
           + model.omega0 * nph[None, None, :]) * V
    if model.q_units.size:
        Ds = dyn.peierls_at(model.q_units, s, model.n_max, Ds)
        for Fk, Fmk, D in zip(model.f_k, model.f_mk, Ds):
            vd = np.tensordot(V, D, axes=(2, 0))
            out = out + np.einsum("ai,ijn->ajn", Fk, vd, optimize=True)
            out = out + np.einsum("bj,ijn->ibn", Fmk, vd, optimize=True)
    return out.ravel()


def apply_current(model: SharedModel, v: np.ndarray, s: float, Ds=None) -> np.ndarray:
    d, d1 = model.el_dim, model.d1
    V = v.reshape(d, d, d1)
    out = np.zeros((2, d, d, d1), complex)
    if model.q_units.size:
        Ds = dyn.peierls_at(model.q_units, s, model.n_max, Ds)
        for Jk, Jmk, D in zip(model.j_k, model.j_mk, Ds):
            vd = np.tensordot(V, D, axes=(2, 0))
            for a in range(2):
                out[a] += np.einsum("ai,ijn->ajn", Jk[a], vd, optimize=True)
                out[a] += np.einsum("bj,ijn->ibn", Jmk[a], vd, optimize=True)
    return out.reshape(2, -1)


def product_state(ek, emk, photon) -> np.ndarray:
    return np.kron(np.kron(np.asarray(ek, complex), np.asarray(emk, complex)),
                   np.asarray(photon, complex))


def apply_h_product_reference(model: SharedModel, ek, emk, c, s: float) -> np.ndarray:
    """Independent product-state action. Not two independent photon modes."""
    ek = np.asarray(ek, complex)
    emk = np.asarray(emk, complex)
    c = np.asarray(c, complex)
    Ds = dyn.peierls_at(model.q_units, s, model.n_max) if model.q_units.size else np.zeros((0, model.d1, model.d1), complex)
    n_c = model.omega0 * (np.arange(model.d1, dtype=float) * c)
    acc_k = model.onsite_k * ek
    acc_mk = model.onsite_mk * emk
    pieces = [np.kron(np.kron(acc_k, emk), c),
              np.kron(np.kron(ek, acc_mk), c),
              np.kron(np.kron(ek, emk), n_c)]
    for Fk, Fmk, D in zip(model.f_k, model.f_mk, Ds):
        dc = D @ c
        pieces.append(np.kron(np.kron(Fk @ ek, emk), dc))
        pieces.append(np.kron(np.kron(ek, Fmk @ emk), dc))
    return sum(pieces)


def electronic_mean(mat36, el36):
    return complex(np.vdot(el36, mat36 @ el36))


def fk_from_groups(f_groups, q_units, el):
    acc = 0.0j
    for F, q in zip(f_groups, q_units):
        acc = acc + 1j * float(q) * np.vdot(el, F @ el)
    return complex(acc)


def connected_current_cov(model: SharedModel, ek, emk):
    """s=0 product state: J_tot = Jk⊗I + I⊗Jmk. Photon is idle."""
    jk = sum(J for J in model.j_k)
    jmk = sum(J for J in model.j_mk)
    mean = np.zeros(2, complex)
    raw = np.zeros((2, 2), complex)
    for a in range(2):
        mean[a] = electronic_mean(jk[a], ek) + electronic_mean(jmk[a], emk)
    for a in range(2):
        for b in range(2):
            raw[a, b] = (electronic_mean(jk[a] @ jk[b], ek)
                         + electronic_mean(jmk[a] @ jmk[b], emk)
                         + electronic_mean(jk[a], ek) * electronic_mean(jmk[b], emk)
                         + electronic_mean(jmk[a], emk) * electronic_mean(jk[b], ek))
    conn = raw - np.outer(np.conj(mean), mean)
    e = model.e
    jpar_var = float(np.real(e @ conn.real @ e))
    return mean, conn, jpar_var


def photon_observables(psi: np.ndarray, el_dim: int, d1: int) -> dict:
    V = np.asarray(psi, complex).reshape(el_dim, el_dim, d1)
    rho = np.tensordot(V, np.conj(V), axes=([0, 1], [0, 1]))
    tr = float(np.trace(rho).real)
    if abs(tr) < 1e-30:
        raise RuntimeError("photon reduced state has vanishing trace")
    rho = rho / tr
    a = np.diag(np.sqrt(np.arange(1, d1, dtype=float)), 1).astype(complex)
    n_op = np.diag(np.arange(d1, dtype=float))
    mean_a = complex(np.trace(rho @ a))
    n_ph = float(np.trace(rho @ n_op).real)
    mean_aa = complex(np.trace(rho @ (a @ a)))
    purity = float(np.trace(rho @ rho).real)
    return {
        "trace": tr,
        "mean_a": {"re": mean_a.real, "im": mean_a.imag},
        "n_ph": n_ph,
        "n_incoherent": n_ph - abs(mean_a) ** 2,
        "mean_aa": {"re": mean_aa.real, "im": mean_aa.imag},
        "purity": purity,
    }


def wedge_p36():
    w70 = qm.wedge_unitary(op_P())
    return restrict36(w70)


def dense_electronic_pair(model: SharedModel) -> np.ndarray:
    """He(k)+He(-k) at s=0, no photon. dim = 36^2. Used only as a tiny reference."""
    hk = np.diag(model.onsite_k).astype(complex)
    hmk = np.diag(model.onsite_mk).astype(complex)
    for Fk, Fmk in zip(model.f_k, model.f_mk):
        hk = hk + Fk
        hmk = hmk + Fmk
    eye = np.eye(EL36, dtype=complex)
    return np.kron(hk, eye) + np.kron(eye, hmk)


def cpack(z):
    z = complex(z)
    return {"re": float(z.real), "im": float(z.imag)}


def matpack(a):
    a = np.asarray(a, complex)
    return [[{"re": float(z.real), "im": float(z.imag)} for z in row] for row in a]


def g_normalization_note(n_sub: int, g: float) -> dict:
    return {
        "g": g,
        "units": "Angstrom^-1 abstract, not experimental V_eff",
        "n_electronic_subsystems": n_sub,
        "linear_force": "F_shared = sum_alpha -g e·<J_alpha> on a product state",
        "if_currents_cancel": "linear mean force can vanish while g stays finite",
        "if_currents_do_not_cancel": "F scales with subsystem count at fixed g",
        "adding_k_without_rescaling_g": "not the thermodynamic limit of one cavity mode",
        "not_macroscopic_BSV_intensity": True,
        "shared_mode_is_not_sum_of_independent_photon_runs": True,
    }


def resource_estimate(model12: SharedModel, n_steps_ref=2019, wall_ref_s=32.15, dim_ref=910) -> dict:
    dim = model12.dim
    n_group = int(model12.q_units.size)
    flop_ratio = (2.0 * (EL36 ** 3)) / (70.0 ** 2)
    state_ratio = dim / float(dim_ref)
    return {
        "n_max": model12.n_max,
        "el_dim_each": EL36,
        "photon_dim": model12.d1,
        "joint_dim": dim,
        "joint_dim_70x70_would_be": 70 * 70 * model12.d1,
        "n_group": n_group,
        "dense_H_bytes_complex128": dim * dim * 16,
        "state_bytes_complex128": dim * 16,
        "do_not_form_dense_H": True,
        "ref_single_k_vacuum_ramp": {"dim": dim_ref, "steps": n_steps_ref, "wall_s": wall_ref_s},
        "flop_ratio_vs_single_k_matvec": flop_ratio,
        "state_ratio_vs_single_k": state_ratio,
        "scaled_wall_s_one_ramp_order_of_magnitude": wall_ref_s * max(flop_ratio, state_ratio),
        "note": "order-of-magnitude only; same n_group assumed; this archive did not propagate",
    }


def run_static():
    geo = build_geometry()
    k = geo.k_cart(K_FRAC)
    closure = spin_closure_report(geo, R0)
    model = build_shared(geo, R0, k, NEEL, n_max=NMAX_APPLY, omega0=OMEGA0, g=G_ABS)
    model12 = build_shared(geo, R0, k, NEEL, n_max=NMAX_INTERFACE, omega0=OMEGA0, g=G_ABS)
    occ_k = dyn.electron_ground_state(geo, R0, k, NEEL)
    occ_mk = dyn.electron_ground_state(geo, R0, -k, NEEL)
    ek, mass_k, leak_k = slater36(occ_k)
    emk, mass_mk, leak_mk = slater36(occ_mk)

    vac = np.zeros(model.d1, complex)
    vac[0] = 1.0
    psi = product_state(ek, emk, vac)
    rng = np.random.default_rng(20261029)
    vrand = rng.normal(size=model.dim) + 1j * rng.normal(size=model.dim)
    vrand = vrand / np.linalg.norm(vrand)

    apply_checks = {}
    for s in (0.0, 0.4, 1.0):
        hv = apply_h(model, psi, s)
        href = apply_h_product_reference(model, ek, emk, vac, s)
        apply_checks[f"product_s_{s}"] = float(np.max(np.abs(hv - href)))
        hr = apply_h(model, vrand, s)
        apply_checks[f"hermiticity_s_{s}"] = float(abs(np.vdot(vrand, hr) - np.vdot(hr, vrand).conjugate()))
    H0 = dense_electronic_pair(model)
    psi_el = np.kron(ek, emk)
    apply_checks["dense_s0_no_photon"] = float(np.max(np.abs(
        apply_h(build_shared(geo, R0, k, NEEL, n_max=0, omega0=OMEGA0, g=G_ABS), psi_el, 0.0) - H0 @ psi_el)))

    single70 = dyn.build_joint(geo, R0, k, NEEL, n_max=1, omega0=OMEGA0, g=G_ABS)
    f70 = sum(single70.f_group)
    f36 = sum(model.f_k)
    block_err = float(np.max(np.abs(restrict36(f70) - f36)))

    mean, conn, jpar_var = connected_current_cov(model, ek, emk)
    jk = np.array([electronic_mean(sum(J[a] for J in model.j_k), ek) for a in range(2)])
    jmk = np.array([electronic_mean(sum(J[a] for J in model.j_mk), emk) for a in range(2)])
    Fk = fk_from_groups(model.f_k, model.q_units, ek)
    Fmk = fk_from_groups(model.f_mk, model.q_units, emk)
    F_from_j = -G_ABS * np.dot(model.e, (jk + jmk).real)
    Jv = apply_current(model, psi, 0.0)
    mean_from_apply = np.array([complex(np.vdot(psi, Jv[a])) for a in range(2)])

    wp = wedge_p36()
    p_resid = {}
    model_mN = build_shared(geo, R0, k, -NEEL, n_max=0, omega0=OMEGA0, g=G_ABS)
    hk = np.diag(model.onsite_k).astype(complex) + sum(model.f_k)
    hmk = np.diag(model.onsite_mk).astype(complex) + sum(model.f_mk)
    hk_m = np.diag(model_mN.onsite_k).astype(complex) + sum(model_mN.f_k)
    hmk_m = np.diag(model_mN.onsite_mk).astype(complex) + sum(model_mN.f_mk)
    p_resid["P36_unitarity"] = float(np.max(np.abs(wp.conj().T @ wp - np.eye(EL36))))
    p_resid["P_He_k_N_vs_He_mk_minusN"] = float(np.max(np.abs(wp @ hk @ wp.conj().T - hmk_m)))
    p_resid["P_He_mk_N_vs_He_k_minusN"] = float(np.max(np.abs(wp @ hmk @ wp.conj().T - hk_m)))
    H_N = dense_electronic_pair(model)
    H_mN = dense_electronic_pair(model_mN)
    Pp = np.kron(wp, wp)
    p_resid["PoxP_Hshared_N_vs_Hshared_minusN_naive"] = float(np.max(np.abs(Pp @ H_N @ Pp.conj().T - H_mN)))
    p_resid["PoxP_Hshared_N_vs_swap_Hshared_minusN"] = float(np.max(np.abs(
        Pp @ H_N @ Pp.conj().T - swap_pair_matrix(H_mN))))

    model_swap = SharedModel(model.k, model.neel, model.n_max, model.omega0, model.g, model.e,
                             model.onsite_mk, model.onsite_k, model.q_units,
                             model.f_mk, model.f_k, model.j_mk, model.j_k)
    # Photon parity Π X Π = -X maps q → -q, i.e. g → -g, not N → -N by itself.
    model_gneg = build_shared(geo, R0, k, NEEL, n_max=model.n_max, omega0=OMEGA0, g=-G_ABS)
    psi_p = apply_photon_parity(psi, EL36, model.d1)
    lhs = apply_photon_parity(apply_h(model, psi_p, 1.0), EL36, model.d1)
    rhs = apply_h(model_gneg, psi, 1.0)
    lhs_r = apply_photon_parity(apply_h(model, apply_photon_parity(vrand, EL36, model.d1), 1.0), EL36, model.d1)
    rhs_r = apply_h(model_gneg, vrand, 1.0)
    eps = 1e-6
    fd_x = (classical_x_energy(model, ek, emk, eps) - classical_x_energy(model, ek, emk, -eps)) / (2.0 * eps)
    h_orig = apply_h(model, product_state(ek, emk, vac), 0.0)
    h_swap = apply_h(model_swap, product_state(emk, ek, vac), 0.0)
    parity = {
        "do_not_reuse_single_k_selection_rule": True,
        "approx_mirror_is_not_a_theorem": True,
        "PoxP_naive_is_not_a_symmetry": p_resid["PoxP_Hshared_N_vs_Hshared_minusN_naive"],
        "PoxP_equals_factor_swap_of_H_minusN": p_resid["PoxP_Hshared_N_vs_swap_Hshared_minusN"],
        "Pi_H_g_Pi_minus_H_minusg_product": float(np.max(np.abs(lhs - rhs))),
        "Pi_H_g_Pi_minus_H_minusg_random": float(np.max(np.abs(lhs_r - rhs_r))),
        "swap_k_minusk_with_photon_parity_is_not_installed_as_a_rule": True,
        "relabel_k_minusk_after_factor_swap": float(np.max(np.abs(
            swap_electronic_factors(h_orig, EL36, model.d1) - h_swap))),
    }

    ph_vac = photon_observables(psi, EL36, model.d1)
    coh = np.zeros(model.d1, complex)
    coh[0] = 1.0
    alpha = 0.5
    for n in range(1, model.d1):
        coh[n] = alpha * coh[n - 1] / np.sqrt(n)
    coh = coh / np.linalg.norm(coh)
    ph_coh = photon_observables(product_state(ek, emk, coh), EL36, model.d1)

    return {
        "record_date": "2026-10-29",
        "model_id": MODEL_ID,
        "claim_level": "static_shared_mode_construction_not_dynamics_not_HHG",
        "not_free_space_BSV_pulse": True,
        "single_k_retained_as_propagator_benchmark_only": True,
        "did_not": [
            "rewrite trial.json", "rewrite preflight.json", "rewrite diag.json",
            "G1", "H2", "long evolution", "HHG spectra", "SV scan",
            "semiclassical trajectories", "full BZ", "AI inverse design",
            "sum two independent single-k photon runs",
        ],
        "definition": {
            "H": "omega0 n + He(k, X; N) + He(-k, X; N)",
            "X": "a + a^dagger, shared",
            "Peierls": "P exp(i q s X) P on each bond, q = g e·delta, same s and X on both k",
            "h8_frozen": True,
            "neel": NEEL,
            "k_fractional": K_FRAC,
            "g": G_ABS,
            "omega0": OMEGA0,
        },
        "spin_sector": {
            **closure,
            "ground_mass_in_36_k": mass_k,
            "ground_mass_in_36_minusk": mass_mk,
            "ground_leak_out_of_36_k": leak_k,
            "ground_leak_out_of_36_minusk": leak_mk,
            "used_36": bool(closure["closed_under_onsite_and_hopping"] and mass_k > 1 - 1e-12 and mass_mk > 1 - 1e-12),
            "max_abs_70block_minus_36_hopping": block_err,
        },
        "dimensions": {
            "el_each": EL36,
            "photon_n_max_interface": NMAX_INTERFACE,
            "joint_n_max_12": model12.dim,
            "joint_n_max_apply_tested": model.dim,
            "not_2_times_single_k": 2 * EL36 * (NMAX_INTERFACE + 1),
            "not_two_independent_photon_modes": (EL36 * (NMAX_INTERFACE + 1)) ** 2,
            "shared_is_not_tensor_of_two_joint_single_k": True,
        },
        "matrix_free": {
            "apply_h_product_max_abs": apply_checks,
            "n_group": int(model.q_units.size),
            "formed_full_joint_dense_H": False,
            "tiny_dense_reference": "36x36 electronic pair at n_max=0 only",
        },
        "zero_field_product_ground": {
            "mean_J_k": [cpack(z) for z in jk],
            "mean_J_minusk": [cpack(z) for z in jmk],
            "mean_J_total": [cpack(z) for z in mean],
            "mean_J_total_from_apply": [cpack(z) for z in mean_from_apply],
            "max_abs_mean_J_total": float(np.max(np.abs(mean))),
            "F_k": cpack(Fk),
            "F_minusk": cpack(Fmk),
            "F_shared": cpack(Fk + Fmk),
            "F_from_minus_g_e_dot_Jtot": float(F_from_j),
            "max_abs_F_groups_minus_current_identity": float(abs((Fk + Fmk).real - F_from_j)),
            "max_abs_F_minus_dH_dX_classical": float(abs((Fk + Fmk) - fd_x)),
            "connected_cov_J_ab": matpack(conn),
            "connected_var_J_parallel": jpar_var,
            "connected_var_is_not_zero": bool(abs(jpar_var) > 1e-8),
        },
        "symmetry": {
            **p_resid,
            **parity,
        },
        "g_normalization": g_normalization_note(2, G_ABS),
        "photon_observables_interface": {
            "operators": ["mean_a", "n_ph", "n_incoherent=<Δa†Δa>", "mean_aa=<a a>", "purity"],
            "no_propagation_this_round": True,
            "vacuum_product": ph_vac,
            "coherent_alpha_0p5_product": ph_coh,
            "n_ph_equals_abs_mean_a_sq_plus_incoherent": True,
        },
        "vacuum_excitation_leftover": {
            "linear_mean_force_on_this_product_state": cpack(Fk + Fmk),
            "do_not_preset_vacuum_growth_to_zero": True,
            "possible_sources_even_if_F_shared_is_zero": [
                "Peierls X^2 and higher at finite s",
                "connected current fluctuations coupling to X",
                "envelope ds/dt doing work",
            ],
            "next_stage_control": "vacuum input |0>, same s(t), shared-mode short ramp; not run here",
        },
        "resource_short_ramp_estimate": resource_estimate(model12),
        "files": {
            Path(p).name: hashlib.sha256(Path(p).read_bytes()).hexdigest()
            for p in (__file__, dyn.__file__, qm.__file__)
        },
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", type=Path,
                    default=Path("results/q_line/qinput_a_shared_kk_static_20261029/static.json"))
    args = ap.parse_args(argv)
    if args.report.exists():
        print(f"report already exists: {args.report}", file=__import__("sys").stderr)
        return 2
    t0 = time.perf_counter()
    report = run_static()
    report["wall_seconds_total"] = time.perf_counter() - t0
    report["stopped"] = True
    args.report.parent.mkdir(parents=True, exist_ok=True)
    with args.report.open("x", encoding="utf-8") as fh:
        json.dump(report, fh, ensure_ascii=False, indent=2, allow_nan=False)
        fh.write("\n")
    print(json.dumps({
        "model_id": report["model_id"],
        "joint_dim_12": report["dimensions"]["joint_n_max_12"],
        "spin_closed": report["spin_sector"]["used_36"],
        "max_abs_Jtot": report["zero_field_product_ground"]["max_abs_mean_J_total"],
        "F_shared_re": report["zero_field_product_ground"]["F_shared"]["re"],
        "Jpar_connected_var": report["zero_field_product_ground"]["connected_var_J_parallel"],
        "PoxP_swap": report["symmetry"]["PoxP_Hshared_N_vs_swap_Hshared_minusN"],
        "wall": report["wall_seconds_total"],
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
