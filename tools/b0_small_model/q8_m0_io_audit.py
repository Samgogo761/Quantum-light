#!/usr/bin/env python3
"""Canonical linear quantum I/O audit. No new physics.

Field analysis of A_out via Q, not the two-mode K-collection identity.
H2 is one output packet. Inputs are an orthonormal positive-frequency
basis on both ports. Target:

    Σ_ν (|S_{H2,ν}|^2 − |T_{H2,ν}|^2) + C_el = 1

C_el is the dressed electron-noise commutator in the same Q units.
The two-mode current-K number (~152) is a labeled comparison only.
Vacuum RMS is not computed here and is not a fail gate.
"""
from __future__ import annotations

import numpy as np

from q8_m0_feedback_op import apply_Gadj
from q8_m0_k1_packet import normalize_phi, phi_unnormalized
from q8_m0_photon_op import (
    ANALYSIS_PREFACTOR,
    analyze_phi,
    h2_phi,
    inner_phi,
    norm_phi,
    omega_weights,
    photon_omega_grid,
    qs_reconstruction,
    synthesize_A,
)
from q8_m0_s1_kernel import pack_complex, parallel_perp_basis
from q8_m0_volterra import (
    filtered_source_moments,
    h2_kernels,
    project_with_g,
    two_port_output_moments,
    volterra_adjoint,
    volterra_forward,
    volterra_response_to_complex_F,
    collect_A2_B2,
)

AUDIT_ID = "canonical_field_io_audit_20260923"
CANONICAL_GATE = "SS_dagger_minus_TT_dagger_plus_C_el_equals_I"
K_TWO_MODE_LABEL = "current_K_two_H2_ports_labeled_comparison_not_canonical"


def inner_c(p, q, dw: float) -> complex:
    """Complex photon overlap dw Σ φ* ψ. inner_phi is its real part."""
    p = np.asarray(p)
    q = np.asarray(q)
    return complex(float(dw) * np.sum(np.conj(p) * q))


def F_from_phi(phi, omega, times, dw: float) -> np.ndarray:
    """Complex F(t,a) = ∫ dω [φ(ω,a)/√ω] e^{+iωt}. synthesize_A = Re F."""
    phi = np.asarray(phi)
    omega = np.asarray(omega, dtype=float)
    wo = np.sqrt(omega)
    g = phi / wo[:, None]
    phase = np.exp(1j * np.asarray(times, dtype=float)[:, None] * omega[None, :])
    return float(dw) * (phase @ g)


def st_from_z(z_re, z_im) -> tuple[complex, complex]:
    """Invert Q-overlaps of Re F and Im F drives.

    Q is real-linear and QS≈I on the real synthesis, so
    z_re = ⟨φ, Q(Re F)⟩ ≈ S+T
    z_im = ⟨φ, Q(Im F)⟩ ≈ −i(S−T)
    (the Im sign is the one measured on the empty sheet).
    """
    z_re = complex(z_re)
    z_im = complex(z_im)
    return 0.5 * (z_re + 1j * z_im), 0.5 * (z_re - 1j * z_im)


def _unit(e) -> np.ndarray:
    e = np.asarray(e, dtype=float).reshape(-1)
    return e / max(float(np.linalg.norm(e)), 1e-30)


def harmonic_packet(omega, n_harm, omega0: float, tp: float, ehat) -> np.ndarray:
    raw = phi_unnormalized(omega, float(n_harm) * float(omega0), tp)
    phi, _ = normalize_phi(omega, raw)
    return phi[:, None] * _unit(ehat)[None, :]


def bin_packet(omega, dw: float, j: int, ehat) -> np.ndarray:
    phi = np.zeros((int(np.asarray(omega).size), 2), dtype=complex)
    phi[int(j)] = _unit(ehat) / np.sqrt(float(dw))
    return phi


def gram_schmidt_packets(items, dw: float, *, drop: float = 1e-10):
    """items: list[(phi, label)]. Real photon inner product."""
    out_phi = []
    out_lab = []
    for phi, lab in items:
        v = np.asarray(phi, dtype=complex).copy()
        for u in out_phi:
            v = v - inner_phi(u, v, dw) * u
        nrm = norm_phi(v, dw)
        if nrm < drop:
            continue
        out_phi.append(v / nrm)
        out_lab.append(str(lab))
    return out_phi, out_lab


def packet_basis(omega, dw: float, omega0: float, tp: float, e_perp, e_par, *, include_par: bool = False):
    """H2 first, then harmonics, then IR/remaining bins. Optional e_par."""
    omega = np.asarray(omega, dtype=float)
    items = []
    items.append((h2_phi(omega, omega0, tp, e_perp), "H2_perp"))
    for n, tag in ((1, "H1"), (3, "H3"), (4, "H4"), (5, "H5"), (6, "H6")):
        items.append((harmonic_packet(omega, n, omega0, tp, e_perp), f"{tag}_perp"))
    for j in range(int(omega.size)):
        items.append((bin_packet(omega, dw, j, e_perp), f"bin{j}_perp"))
    if include_par:
        items.append((h2_phi(omega, omega0, tp, e_par), "H2_par"))
        for n, tag in ((1, "H1"), (3, "H3")):
            items.append((harmonic_packet(omega, n, omega0, tp, e_par), f"{tag}_par"))
        for j in range(int(omega.size)):
            items.append((bin_packet(omega, dw, j, e_par), f"bin{j}_par"))
    phis, labs = gram_schmidt_packets(items, dw)
    return phis, labs


def basis_gram(phis, dw: float) -> dict:
    n = len(phis)
    g = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(i, n):
            gij = inner_phi(phis[i], phis[j], dw)
            g[i, j] = gij
            g[j, i] = gij
    off = g - np.eye(n)
    return {
        "n": n,
        "max_offdiag": float(np.max(np.abs(off - np.diag(np.diag(off))))),
        "max_diag_minus_1": float(np.max(np.abs(np.diag(g) - 1.0))) if n else 0.0,
        "orthonormal": bool(n > 0 and np.max(np.abs(off)) < 1e-8),
    }


def empty_matter(nt: int):
    """I=0 sheet: A_R_out = A_L_in, A_self = 0."""
    x = np.zeros((nt, 1, 2, 4, 4), dtype=complex)
    w = np.array([1.0], dtype=float)
    md = np.zeros((nt, 2, 2), dtype=float)
    return x, w, md


def analyze_overlap_kernel(phi, omega, times, dw: float) -> np.ndarray:
    """k(t,a) such that inner_c(φ, Q(A)) = dt Σ A[t,a] k[t,a]."""
    omega = np.asarray(omega, dtype=float)
    phi = np.asarray(phi)
    times = np.asarray(times, dtype=float)
    phase = np.exp(-1j * times[:, None] * omega[None, :])
    coef = (np.sqrt(omega) * float(ANALYSIS_PREFACTOR))[:, None]
    g = np.conj(phi) * coef
    return float(dw) * (phase @ g)


def field_noise_kernel(x, w, mean_d, dt: float, gamma: float, phi, omega, times, dw: float):
    """g_ξ for N = inner_c(φ, Q(A_self[ξ])). Packed Euclidean, same X language."""
    k = analyze_overlap_kernel(phi, omega, times, dw)
    g_re = apply_Gadj(np.real(k), dt, gamma)
    g_im = apply_Gadj(np.imag(k), dt, gamma)
    adj_r = volterra_adjoint(x, w, mean_d, dt, gamma, float(dt) * g_re)
    adj_i = volterra_adjoint(x, w, mean_d, dt, gamma, float(dt) * g_im)
    return adj_r["g_xi"] + 1j * adj_i["g_xi"]


def c_el_field(x, w, mean_d, dt: float, gamma: float, phi, omega, times, dw: float) -> dict:
    """[N, N†] of the H2 field-analysis of A_self[ξ]. No η² prefactor."""
    k_eff = field_noise_kernel(x, w, mean_d, dt, gamma, phi, omega, times, dw)
    yk = project_with_g(x, k_eff)
    yks = project_with_g(x, np.conj(k_eff))
    n_f = complex(np.einsum("k,kcv,kcv->", w, np.conj(yk), yk))
    n_fs = complex(np.einsum("k,kcv,kcv->", w, np.conj(yks), yks))
    m_f = complex(np.einsum("k,kcv,kcv->", w, np.conj(yks), yk))
    comm = n_fs - n_f
    return {
        "n_F": n_f,
        "n_Fstar": n_fs,
        "m_F": m_f,
        "C_el": comm,
        "C_el_real": float(np.real(comm)),
        "kind": "field_Q_Aself_xi_commutator_dimensionless",
    }


def a_self_re_im(x, w, mean_d, dt: float, gamma: float, F):
    """A_self for A_in = Re F and Im F (total in = L+R). Same A_self for either port."""
    F = np.asarray(F)
    hr = np.real(F)
    hi = np.imag(F)
    pr = volterra_forward(x, w, mean_d, dt, gamma, a_L_in=hr)
    pi = volterra_forward(x, w, mean_d, dt, gamma, a_L_in=hi)
    return {
        "A_self_re": np.asarray(pr["A_self"], dtype=float),
        "A_self_im": np.asarray(pi["A_self"], dtype=float),
        "finite": bool(pr["finite"] and pi["finite"]),
        "maxabs_A_self": max(pr["maxabs_A_self"], pi["maxabs_A_self"]),
    }


def z_of(phi, a_t, omega, times, dt: float, dw: float) -> complex:
    return inner_c(phi, analyze_phi(a_t, omega, times, dt), dw)


def h2_st_from_self(phi_h2, F_nu, a_self_re, a_self_im, omega, times, dt, dw):
    """S,T for one input packet: L→R_out (feedthrough+self) and R→R_out (self)."""
    F = np.asarray(F_nu)
    a_re = np.real(F)
    a_im = np.imag(F)
    z_re_rl = z_of(phi_h2, a_re + a_self_re, omega, times, dt, dw)
    z_im_rl = z_of(phi_h2, a_im + a_self_im, omega, times, dt, dw)
    z_re_rr = z_of(phi_h2, a_self_re, omega, times, dt, dw)
    z_im_rr = z_of(phi_h2, a_self_im, omega, times, dt, dw)
    s_rl, t_rl = st_from_z(z_re_rl, z_im_rl)
    s_rr, t_rr = st_from_z(z_re_rr, z_im_rr)
    return {
        "S_RL": s_rl,
        "T_RL": t_rl,
        "S_RR": s_rr,
        "T_RR": t_rr,
        "vac_piece": (
            abs(s_rl) ** 2 - abs(t_rl) ** 2
            + abs(s_rr) ** 2 - abs(t_rr) ** 2
        ),
    }


def h2_row_identity(phis, drives, phi_h2, omega, times, dt, dw, c_el: complex) -> dict:
    """Sum vacuum pieces over the given packets + C_el. Detection on R_out."""
    vac = 0.0
    absS2 = 0.0
    absT2 = 0.0
    first = None
    for phi, rec in zip(phis, drives):
        row = h2_st_from_self(
            phi_h2, rec["F"], rec["A_self_re"], rec["A_self_im"],
            omega, times, dt, dw,
        )
        vac += row["vac_piece"]
        absS2 += abs(row["S_RL"]) ** 2 + abs(row["S_RR"]) ** 2
        absT2 += abs(row["T_RL"]) ** 2 + abs(row["T_RR"]) ** 2
        if first is None:
            first = row
    comm = vac + complex(c_el)
    return {
        "n_packets": len(phis),
        "n_inputs": 2 * len(phis),
        "sum_absS2": float(absS2),
        "sum_absT2": float(absT2),
        "vac_piece": pack_complex(vac),
        "vac_piece_real": float(np.real(vac)),
        "C_el": pack_complex(c_el),
        "commutator": pack_complex(comm),
        "commutator_real": float(np.real(comm)),
        "abs_minus_1": float(abs(comm - 1.0)),
        "S_H2_RL": None if first is None else pack_complex(first["S_RL"]),
        "T_H2_RL": None if first is None else pack_complex(first["T_RL"]),
        "S_H2_RR": None if first is None else pack_complex(first["S_RR"]),
        "T_H2_RR": None if first is None else pack_complex(first["T_RR"]),
    }


def prepare_drives(x, w, mean_d, dt, gamma, phis, omega, times, dw, *, log=None):
    out = []
    for i, phi in enumerate(phis):
        F = F_from_phi(phi, omega, times, dw)
        rec = a_self_re_im(x, w, mean_d, dt, gamma, F)
        rec["F"] = F
        rec["phi"] = phi
        out.append(rec)
        if log is not None and ((i + 1) % 4 == 0 or i + 1 == len(phis)):
            log(f"  drives {i + 1}/{len(phis)} finite={rec['finite']}")
    return out


def identity_scan(
    x, w, mean_d, times, dt, gamma, omega, dw, phis, labs, phi_h2, n_list, *, log=None,
):
    n_max = max(int(n) for n in n_list)
    n_max = min(n_max, len(phis))
    if log is not None:
        log(f"prepare drives n_max={n_max}")
    drives = prepare_drives(
        x, w, mean_d, dt, gamma, phis[:n_max], omega, times, dw, log=log,
    )
    if log is not None:
        log("C_el field")
    cel = c_el_field(x, w, mean_d, dt, gamma, phi_h2, omega, times, dw)
    rows = []
    for n in n_list:
        n = min(int(n), n_max)
        rec = h2_row_identity(
            phis[:n], drives[:n], phi_h2, omega, times, dt, dw, cel["C_el"],
        )
        rec["labels"] = labs[:n]
        rec["n_used"] = n
        rows.append(rec)
        if log is not None:
            log(
                f"  n={n} inputs={rec['n_inputs']} "
                f"comm={rec['commutator_real']:.6g} |·-1|={rec['abs_minus_1']:.6g} "
                f"|S|^2={rec['sum_absS2']:.6g} |T|^2={rec['sum_absT2']:.6g} "
                f"C_el={cel['C_el_real']:.6g}"
            )
    return {"rows": rows, "C_el": cel, "n_prepared": n_max}


def empty_sheet_identity(times, dt, omega, dw, omega0, tp, e_perp, e_par, n_list, gamma: float = 0.05):
    """X=0 control. Field I/O must stay near the QS floor, independent of γ."""
    nt = int(np.asarray(times).shape[0])
    x, w, md = empty_matter(nt)
    phis, labs = packet_basis(omega, dw, omega0, tp, e_perp, e_par, include_par=False)
    phi_h2 = phis[0]
    qs = qs_reconstruction(phi_h2, omega, times, dt, dw)
    scan = identity_scan(
        x, w, md, times, dt, gamma, omega, dw, phis, labs, phi_h2, n_list,
    )
    return {"qs": qs, "gram": basis_gram(phis, dw), "scan": scan, "labels": labs}


def gamma0_field_identity(x, w, mean_d, times, dt, omega, dw, omega0, tp, e_perp, e_par, n_list):
    """γ=0 ⇒ A_self=0 even with matter. Same algebra as the empty sheet."""
    phis, labs = packet_basis(omega, dw, omega0, tp, e_perp, e_par, include_par=False)
    phi_h2 = phis[0]
    scan = identity_scan(
        x, w, mean_d, times, dt, 0.0, omega, dw, phis, labs, phi_h2, n_list,
    )
    return {"scan": scan, "labels": labs}


def kubo_born_current(packed, geo, prot) -> dict:
    """Existing 2 Re A2 + comm_s. State piece should match C^>−C^<; contact may not."""
    from q8_m0_mode_io import close_output_moments

    rec = close_output_moments(
        packed["X_full"], packed["w_k"], packed["mu_per_cell"], packed["mean_D_t"],
        packed["times"], packed["dt"], geo, prot,
    )
    chk = rec["commutator_check"]
    return {
        "two_Re_A2": chk["two_Re_A2"],
        "two_Re_A2_state": chk["two_Re_A2_state_only"],
        "two_Re_A2_contact": chk["two_Re_A2_contact_only"],
        "s1_commutator": chk["s1_commutator"],
        "residual_full_abs": chk["residual_full_plus_commutator_abs"],
        "residual_state_abs": chk["residual_state_plus_commutator_abs"],
        "kept_order_identity_closed": chk["kept_order_identity_closed"],
        "note": (
            "Born current-K Kubo: 2 Re A2_state + comm_s should vanish. "
            "Contact D is a c-number kernel and may have no noise counterpart."
        ),
    }


def k_two_mode_comparison(packed, geo, prot, gamma, pref) -> dict:
    """Labeled 2-mode H2 L+R current-K commutator. Not the canonical gate."""
    hk = h2_kernels(packed["times"], prot.omega0, prot.t_pulse, geo)
    vol = volterra_response_to_complex_F(
        packed["X_full"], packed["w_k"], packed["mean_D_t"],
        packed["dt"], gamma, hk["F"], hk["e_perp"],
    )
    c = collect_A2_B2(hk["k_a"], vol["total"], packed["dt"], pref)
    noise = filtered_source_moments(
        packed["X_full"], packed["w_k"], packed["mean_D_t"],
        packed["dt"], gamma, hk["k_a"],
    )
    n_src = pref * noise["n_F"]
    m_src = -pref * noise["m_F"]
    comm_s = pref * noise["commutator_F"]
    mom = two_port_output_moments(c["A2"], c["B2"], n_src, m_src, comm_s)
    return {
        "kind": K_TWO_MODE_LABEL,
        "A2": pack_complex(c["A2"]),
        "B2": pack_complex(c["B2"]),
        "n_src": pack_complex(n_src),
        "comm_s": pack_complex(comm_s),
        "commutator": mom["commutator"],
        "commutator_abs_minus_1": mom["commutator_abs_minus_1"],
        "not_canonical_gate": True,
        "not_production_n_m": True,
    }


def audit_grid(prot, *, harm_max: float = 12.0, t_obs=None):
    omega = photon_omega_grid(prot.t_pulse, prot.omega0, harm_max=harm_max, t_obs=t_obs)
    dw = omega_weights(omega)
    return omega, dw


def slim_cel(cel: dict) -> dict:
    return {
        "n_F": pack_complex(cel["n_F"]),
        "n_Fstar": pack_complex(cel["n_Fstar"]),
        "m_F": pack_complex(cel["m_F"]),
        "C_el": pack_complex(cel["C_el"]),
        "C_el_real": float(cel["C_el_real"]),
        "kind": cel["kind"],
    }


def slim_row(rec: dict) -> dict:
    return {k: v for k, v in rec.items() if k != "labels"}
