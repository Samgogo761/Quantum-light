#!/usr/bin/env python3
"""P1-dressed particle-hole + two-port photon canonical Gaussian theory.

Quadratic Hamiltonian on the Slater background. No hand-made G_self.
k-weights are BZ quadrature, not N_cell. η² N_cell stays the SI layer.

Prototype: symplectic evolution of explicit p-h and packet oscillators.
Weak coupling (prescribed δA) must recover C^>, R^A, A2/B2, Kubo.
"""
from __future__ import annotations

import numpy as np

from q8_m0_io_audit import F_from_phi
from q8_m0_photon_op import h2_phi
from q8_m0_s1_kernel import pack_complex
from q8_m0_volterra import collect_A2_B2, h2_kernels

PHGAUSS_ID = "P1_dressed_ph_photon_canonical_Gaussian_20260923"
PHGAUSS_STATUS = "approved_quantum_main_Gself_emergence_S_not_fixed"
N_C = 4
N_V = 4
N_CHAN = N_C * N_V
NO_GSELF_NOTE = "G_self_is_not_an_input_kernel_it_must_emerge_from_photon_continuum"


def n_ph_osc(nk: int) -> int:
    return int(nk) * N_CHAN


def sqrt_w(w) -> np.ndarray:
    w = np.asarray(w, dtype=float)
    if np.any(w < -1e-18):
        raise ValueError("negative BZ weight")
    return np.sqrt(np.maximum(w, 0.0))


def i_state_from_B(x_n, sw, br, bi) -> np.ndarray:
    """Per-cell δI^{ph} from B=x+iy. I_a = 2 Σ_k √w (ReX x + ImX y)."""
    x_n = np.asarray(x_n)
    br = np.asarray(br).reshape(x_n.shape[0], N_C, N_V)
    bi = np.asarray(bi).reshape(x_n.shape[0], N_C, N_V)
    xr = np.real(x_n)
    xi = np.imag(x_n)
    return (
        2.0 * np.einsum("k,kacv,kcv->a", sw, xr, br)
        + 2.0 * np.einsum("k,kacv,kcv->a", sw, xi, bi)
    )


def response_from_ph_prefix(x, w, h, dt, mean_d) -> dict:
    """Interaction-picture p-h with prescribed δA. Left-Riemann = prefix R^A."""
    x = np.asarray(x)
    w = np.asarray(w, dtype=float)
    h = np.asarray(h, dtype=float)
    md = np.asarray(mean_d, dtype=float)
    nt, nk = x.shape[0], x.shape[1]
    sw = sqrt_w(w)
    br = np.zeros((nk, N_C, N_V), dtype=float)
    bi = np.zeros((nk, N_C, N_V), dtype=float)
    i_state = np.zeros((nt, 2), dtype=float)
    i_contact = np.zeros((nt, 2), dtype=float)
    for n in range(nt):
        i_state[n] = i_state_from_B(x[n], sw, br, bi)
        i_contact[n] = -md[n] @ h[n]
        xr = np.real(x[n])
        xi = np.imag(x[n])
        br = br + dt * np.einsum("k,kacv,a->kcv", -sw, xi, h[n])
        bi = bi + dt * np.einsum("k,kacv,a->kcv", sw, xr, h[n])
    return {
        "state": i_state,
        "contact": i_contact,
        "total": i_state + i_contact,
    }


def c_greater_from_channels(x, w) -> np.ndarray:
    """Σ_k w_k Tr[X_a† X_b]. Same CJJ Gram. Vertex carries √w so ⟨II⟩ matches."""
    x = np.asarray(x)
    w = np.asarray(w, dtype=float)
    return np.einsum("k,kacv,kbcv->ab", w, np.conj(x), x)


def symplectic_J(n: int) -> np.ndarray:
    """ξ=(q,p), [q_i,p_j]=i δ_ij. J = [[0,I],[-I,0]]."""
    j = np.zeros((2 * n, 2 * n), dtype=float)
    j[:n, n:] = np.eye(n)
    j[n:, :n] = -np.eye(n)
    return j


def _C_A(n_ph, n_b, f_r, f_i, eta: float) -> np.ndarray:
    """A_a = η√2 Σ_μ (Re F_{μa} q_μ + Im F_{μa} p_μ)."""
    n = n_ph + n_b
    c = np.zeros((2, 2 * n), dtype=float)
    if n_ph == 0:
        return c
    coef = float(eta) * np.sqrt(2.0)
    for mu in range(n_ph):
        for a in range(2):
            c[a, mu] = coef * f_r[mu, a]
            c[a, n + mu] = coef * f_i[mu, a]
    return c


def _C_Iph(n_ph, n_b, x_r, x_i, sw) -> np.ndarray:
    """I^{ph}_a = √2 Σ_α √w (ReX q + ImX p)."""
    n = n_ph + n_b
    c = np.zeros((2, 2 * n), dtype=float)
    nk = int(sw.size)
    for k in range(nk):
        for cv in range(N_CHAN):
            alpha = k * N_CHAN + cv
            c4, v4 = divmod(cv, N_V)
            for a in range(2):
                coef = np.sqrt(2.0) * float(sw[k])
                c[a, n_ph + alpha] = coef * x_r[k, a, c4, v4]
                c[a, n + n_ph + alpha] = coef * x_i[k, a, c4, v4]
    return c


def linear_K(n_ph, sw, x_n, f_r, f_i, d_ab, eta: float) -> np.ndarray:
    """ξ̇ = K ξ on (q,p). Photons first, then p-h. No hand-made G_self."""
    nk = int(sw.size)
    n_b = nk * N_CHAN
    n = n_ph + n_b
    x_r = np.real(np.asarray(x_n))
    x_i = np.imag(np.asarray(x_n))
    d_ab = np.asarray(d_ab, dtype=float)
    c_a = _C_A(n_ph, n_b, f_r, f_i, eta)
    c_iph = _C_Iph(n_ph, n_b, x_r, x_i, sw)
    c_i = c_iph - d_ab @ c_a
    kmat = np.zeros((2 * n, 2 * n), dtype=float)
    for k in range(nk):
        for cv in range(N_CHAN):
            alpha = k * N_CHAN + cv
            c4, v4 = divmod(cv, N_V)
            row_q = n_ph + alpha
            row_p = n + n_ph + alpha
            coef = np.sqrt(2.0) * float(sw[k])
            kmat[row_q] = -coef * (x_i[k, 0, c4, v4] * c_a[0] + x_i[k, 1, c4, v4] * c_a[1])
            kmat[row_p] = coef * (x_r[k, 0, c4, v4] * c_a[0] + x_r[k, 1, c4, v4] * c_a[1])
    for mu in range(n_ph):
        coef = np.sqrt(2.0) * float(eta)
        kmat[mu] = -coef * (f_i[mu, 0] * c_i[0] + f_i[mu, 1] * c_i[1])
        kmat[n + mu] = coef * (f_r[mu, 0] * c_i[0] + f_r[mu, 1] * c_i[1])
    return kmat


def is_hamiltonian(kmat, j, *, atol: float = 1e-9) -> float:
    """||J K + K^T J||. Zero iff K = J Hess with Hess symmetric."""
    kmat = np.asarray(kmat, dtype=float)
    return float(np.max(np.abs(j @ kmat + kmat.T @ j)))


def cayley_step(phi, kmat, dt: float) -> np.ndarray:
    eye = np.eye(kmat.shape[0])
    hdt = 0.5 * float(dt)
    left = eye - hdt * kmat
    right = (eye + hdt * kmat) @ phi
    return np.linalg.solve(left, right)


def pack_waveforms(f_list) -> tuple[np.ndarray, np.ndarray]:
    """f_list: list of (nt, 2) complex F. Returns Fr, Fi shaped (n_ph, nt, 2)."""
    if not f_list:
        return np.zeros((0, 1, 2)), np.zeros((0, 1, 2))
    f = np.stack([np.asarray(f, dtype=complex) for f in f_list], axis=0)
    return np.real(f), np.imag(f)


def vacuum_moment_matrix(n_osc: int) -> np.ndarray:
    """⟨ξ ξ^T⟩ on the bosonic vacuum. ξ=(q,p), ⟨q p⟩=i/2."""
    n = int(n_osc)
    return 0.5 * np.eye(2 * n, dtype=complex) + 0.5j * symplectic_J(n)


def moments_from_phi(phi, n_osc: int) -> np.ndarray:
    m0 = vacuum_moment_matrix(n_osc)
    phi = np.asarray(phi, dtype=float)
    return phi @ m0 @ phi.T


def bdag_b_from_moments(m, n_ph: int, n_b: int) -> np.ndarray:
    """⟨B_α† B_β⟩ for the p-h oscillators. Shape (n_b, n_b)."""
    n = int(n_ph) + int(n_b)
    qa = int(n_ph) + np.arange(int(n_b))
    pa = n + qa
    return 0.5 * (
        m[np.ix_(qa, qa)]
        + 1j * m[np.ix_(qa, pa)]
        - 1j * m[np.ix_(pa, qa)]
        + m[np.ix_(pa, pa)]
    )


PAULI_NORM = {
    "B_num_is": "B_coll",
    "B_coll_is": "M_k^{-1/2} sum_{j in k-patch} B_j^{micro}",
    "why_not_B_coll_over_sqrt_Ncell": (
        "that identification would give [B_num, B_num^dag] = 1/N_cell, "
        "but the Gaussian oscillator is canonical with [B,B^dag]=1"
    ),
    "M_k": "N_cell * w_k",
    "N_cell": "unset_not_equal_to_nk",
    "never_N_cell_eq_nk": True,
    "I_cell_vertex": "sqrt(w_k) X  (frozen CJJ Gram; eta^2 N_cell sits in A)",
    "H_int": "- A · I_cell",
    "n_coll": "N_B_num",
    "f_micro": "N_B_num / (N_cell * w_k)",
    "delta_rho_micro": "delta_rho_coll / (N_cell * w_k)",
    "eps_Pauli": "max_k max(opnorm(delta_rho_coll), opnorm(h_coll)) / (N_cell * w_k)",
    "reported_without_N_cell": "N_cell * eps_Pauli and N_cell * f_micro",
    "N_cell_cancels": False,
    "raw_N_B_is_not_a_pass_fail": True,
    "S_needed_for_this_gate": True,
    "S_node": "not_fixed_report_as_area_requirement_curve",
    "S_is_physical_mode_area": True,
    "S_is_not_a_normalization_box": True,
}

# Stage-1Q two-H2 prototype. Replace after continuum Pauli.
PAULI_C_P = 28.020752514399163
PAULI_C_C = 55.744789878664065


def pauli_area_curve(a_uc: float, *, c_p: float = PAULI_C_P, c_c: float = PAULI_C_C) -> dict:
    """ε_Pauli(S)=C_P A_uc/S. S is the physical mode area, not a box."""
    a_uc = float(a_uc)
    out = {
        "C_P": float(c_p),
        "C_C": float(c_c),
        "A_uc_ang2": a_uc,
        "eps_Pauli_of_S": "C_P * A_uc / S",
        "C_minus_I_of_S": "C_C * A_uc / S",
        "S_not_fixed": True,
        "source": "two_H2_prototype_replace_after_continuum",
        "S_min_ang2": {},
        "S_min_side_nm": {},
        "N_cell_min": {},
    }
    for tag, thr in (("1pct", 0.01), ("5pct", 0.05), ("10pct", 0.10)):
        n_p = float(c_p) / thr
        n_c = float(c_c) / thr
        s_p = n_p * a_uc
        s_c = n_c * a_uc
        out["N_cell_min"][f"eps_{tag}"] = n_p
        out["N_cell_min"][f"CminusI_{tag}"] = n_c
        out["S_min_ang2"][f"eps_{tag}"] = s_p
        out["S_min_ang2"][f"CminusI_{tag}"] = s_c
        out["S_min_side_nm"][f"eps_{tag}"] = float(np.sqrt(s_p) * 0.1)
        out["S_min_side_nm"][f"CminusI_{tag}"] = float(np.sqrt(s_c) * 0.1)
    return out


def _opnorm_herm(mat) -> float:
    h = 0.5 * (np.asarray(mat) + np.conjugate(np.asarray(mat)).T)
    return float(np.linalg.norm(h, 2))


def pauli_from_bdag_b(bb, w) -> dict:
    """Three-layer Pauli report from ⟨B†B⟩. Does not pass/fail on max N_B."""
    bb = np.asarray(bb)
    w = np.asarray(w, dtype=float)
    n_b = bb.shape[0]
    nk = int(w.size)
    if n_b != nk * N_CHAN:
        raise ValueError("B block does not match nk * 16")
    n_b_k = np.zeros((nk, N_C, N_V), dtype=float)
    drho = np.zeros((nk, N_C, N_C), dtype=complex)
    hv = np.zeros((nk, N_V, N_V), dtype=complex)
    c_def = np.zeros((nk, N_CHAN, N_CHAN), dtype=complex)
    eye_c = np.eye(N_C)
    eye_v = np.eye(N_V)
    op_drho = np.zeros(nk)
    op_hv = np.zeros(nk)
    op_c = np.zeros(nk)
    for k in range(nk):
        sl = slice(k * N_CHAN, (k + 1) * N_CHAN)
        blk = bb[sl, sl].reshape(N_C, N_V, N_C, N_V)
        n_b_k[k] = np.real(np.einsum("abab->ab", blk)).astype(float)
        drho[k] = np.einsum("abcb->ac", blk)
        hv[k] = np.einsum("abac->bc", blk)
        d1 = np.einsum("ac,bd->abcd", eye_c, hv[k])
        d2 = np.einsum("ca,bd->abcd", drho[k], eye_v)
        c_def[k] = (d1 + d2).reshape(N_CHAN, N_CHAN)
        op_drho[k] = _opnorm_herm(drho[k])
        op_hv[k] = _opnorm_herm(hv[k])
        op_c[k] = _opnorm_herm(c_def[k])
    wk = np.maximum(w, 1e-30)
    eps_coll = float(np.max(np.maximum(op_drho, op_hv)))
    ncell_eps = float(np.max(np.maximum(op_drho / wk, op_hv / wk)))
    ncell_cmi = float(np.max(op_c / wk))
    nmax = float(np.max(n_b_k))
    loc = np.unravel_index(int(np.argmax(n_b_k)), n_b_k.shape)
    ncell_f = n_b_k / wk[:, None, None]
    return {
        "norm": dict(PAULI_NORM),
        "N_B": n_b_k,
        "N_B_max": nmax,
        "N_B_argmax": {"k": int(loc[0]), "c": int(loc[1]), "v": int(loc[2])},
        "delta_rho_c_coll": drho,
        "h_v_coll": hv,
        "Ncell_times_delta_rho_micro": drho / wk[:, None, None],
        "Ncell_times_h_micro": hv / wk[:, None, None],
        "Ncell_times_f_micro": ncell_f,
        "Ncell_times_f_micro_max": float(np.max(ncell_f)),
        "opnorm_delta_rho_coll": op_drho,
        "opnorm_h_coll": op_hv,
        "eps_Pauli_coll": eps_coll,
        "Ncell_times_eps_Pauli": ncell_eps,
        "Ncell_times_C_minus_I": ncell_cmi,
        "C_defect_op_coll": op_c,
        "eps_Pauli": "Ncell_times_eps_Pauli / N_cell",
        "C_minus_I_op": "Ncell_times_C_minus_I / N_cell",
        "f_micro": "N_B / (N_cell * w_k)",
        "w_k": w,
        "N_cell": "unset_not_equal_to_nk",
        "note": "raw N_B is not a gate; divide Ncell_times_* by N_cell after a human S",
    }


def pauli_from_phi(phi, n_ph: int, n_b: int, w) -> dict:
    n = int(n_ph) + int(n_b)
    m = moments_from_phi(phi, n)
    bb = bdag_b_from_moments(m, n_ph, n_b)
    return pauli_from_bdag_b(bb, w)


def slim_pauli(p: dict) -> dict:
    """JSON-safe snapshot. Full matrices kept as re/im."""
    def cpack(arr):
        arr = np.asarray(arr)
        return {"re": np.real(arr).tolist(), "im": np.imag(arr).tolist()}

    return {
        "norm": p["norm"],
        "N_B": np.asarray(p["N_B"]).tolist(),
        "N_B_max": p["N_B_max"],
        "N_B_argmax": p["N_B_argmax"],
        "delta_rho_c_coll": cpack(p["delta_rho_c_coll"]),
        "h_v_coll": cpack(p["h_v_coll"]),
        "Ncell_times_delta_rho_micro": cpack(p["Ncell_times_delta_rho_micro"]),
        "Ncell_times_h_micro": cpack(p["Ncell_times_h_micro"]),
        "Ncell_times_f_micro": np.asarray(p["Ncell_times_f_micro"]).tolist(),
        "Ncell_times_f_micro_max": p["Ncell_times_f_micro_max"],
        "opnorm_delta_rho_coll": np.asarray(p["opnorm_delta_rho_coll"]).tolist(),
        "opnorm_h_coll": np.asarray(p["opnorm_h_coll"]).tolist(),
        "eps_Pauli_coll": p["eps_Pauli_coll"],
        "Ncell_times_eps_Pauli": p["Ncell_times_eps_Pauli"],
        "Ncell_times_C_minus_I": p["Ncell_times_C_minus_I"],
        "C_defect_op_coll": np.asarray(p["C_defect_op_coll"]).tolist(),
        "eps_Pauli": p["eps_Pauli"],
        "C_minus_I_op": p["C_minus_I_op"],
        "f_micro": p["f_micro"],
        "w_k": np.asarray(p["w_k"]).tolist(),
        "N_cell": p["N_cell"],
        "note": p["note"],
    }


def propagate_phi(x, w, mean_d, f_list, dt, eta, *, times=None, save_pauli: bool = False):
    """Symplectic Φ(T←0) with midpoint K. Photons in the interaction picture."""
    x = np.asarray(x)
    w = np.asarray(w, dtype=float)
    md = np.asarray(mean_d, dtype=float)
    nt, nk = x.shape[0], x.shape[1]
    sw = sqrt_w(w)
    fr, fi = pack_waveforms(f_list)
    n_ph = int(fr.shape[0])
    n_b = n_ph_osc(nk)
    n = n_ph + n_b
    j = symplectic_J(n)
    phi = np.eye(2 * n, dtype=float)
    ham_res = 0.0
    n_b_t = np.zeros(nt) if save_pauli else None
    eps_t = np.zeros(nt) if save_pauli else None
    cmi_t = np.zeros(nt) if save_pauli else None
    if save_pauli:
        p0 = pauli_from_phi(phi, n_ph, n_b, w)
        n_b_t[0] = p0["N_B_max"]
        eps_t[0] = p0["Ncell_times_eps_Pauli"]
        cmi_t[0] = p0["Ncell_times_C_minus_I"]
    for nstep in range(nt - 1):
        xmid = 0.5 * (x[nstep] + x[nstep + 1])
        dmid = 0.5 * (md[nstep] + md[nstep + 1])
        if n_ph:
            frm = 0.5 * (fr[:, nstep] + fr[:, nstep + 1])
            fim = 0.5 * (fi[:, nstep] + fi[:, nstep + 1])
        else:
            frm = np.zeros((0, 2))
            fim = np.zeros((0, 2))
        kmat = linear_K(n_ph, sw, xmid, frm, fim, dmid, eta)
        ham_res = max(ham_res, is_hamiltonian(kmat, j))
        phi = cayley_step(phi, kmat, dt)
        if save_pauli:
            pt = pauli_from_phi(phi, n_ph, n_b, w)
            n_b_t[nstep + 1] = pt["N_B_max"]
            eps_t[nstep + 1] = pt["Ncell_times_eps_Pauli"]
            cmi_t[nstep + 1] = pt["Ncell_times_C_minus_I"]
    resid = phi.T @ j @ phi - j
    out = {
        "Phi": phi,
        "n_ph": n_ph,
        "n_B": n_b,
        "n": n,
        "J": j,
        "symplectic_maxabs": float(np.max(np.abs(resid))),
        "hamiltonian_K_maxabs": ham_res,
        "eta": float(eta),
        "no_G_self": True,
    }
    if save_pauli:
        out["N_B_max_t"] = n_b_t
        out["Ncell_times_eps_Pauli_t"] = eps_t
        out["Ncell_times_C_minus_I_t"] = cmi_t
        out["pauli_final"] = pauli_from_phi(phi, n_ph, n_b, w)
    return out


def io_from_phi(phi, n_ph: int, n_b: int) -> dict:
    """b_out = S b_in + T b_in† + U B + V B† from the symplectic map."""
    n = n_ph + n_b
    phi = np.asarray(phi, dtype=float)
    s = np.zeros((n_ph, n_ph), dtype=complex)
    t = np.zeros((n_ph, n_ph), dtype=complex)
    u = np.zeros((n_ph, n_b), dtype=complex)
    v = np.zeros((n_ph, n_b), dtype=complex)

    def st_from_cols(mu, idx):
        a = (phi[mu, idx] + 1j * phi[n + mu, idx]) / np.sqrt(2.0)
        c = (phi[mu, n + idx] + 1j * phi[n + mu, n + idx]) / np.sqrt(2.0)
        return (a - 1j * c) / np.sqrt(2.0), (a + 1j * c) / np.sqrt(2.0)

    for mu in range(n_ph):
        for nu in range(n_ph):
            s[mu, nu], t[mu, nu] = st_from_cols(mu, nu)
        for alpha in range(n_b):
            u[mu, alpha], v[mu, alpha] = st_from_cols(mu, n_ph + alpha)
    cel = u @ u.conj().T - v @ v.conj().T
    ident = s @ s.conj().T - t @ t.conj().T + cel
    eye = np.eye(n_ph, dtype=complex)
    return {
        "S": s,
        "T": t,
        "U": u,
        "V": v,
        "C_el": cel,
        "ident": ident,
        "ident_minus_I_maxabs": float(np.max(np.abs(ident - eye))) if n_ph else 0.0,
        "diag_ident": np.real(np.diag(ident)).astype(float) if n_ph else np.zeros(0),
    }


def h2_two_port_packets(times, omega0, tp, e_perp, dw, omega):
    """H2 ⊥ on L and on R. Same F, two oscillators. No G_self."""
    phi = h2_phi(omega, omega0, tp, e_perp)
    f = F_from_phi(phi, omega, times, dw)
    return [f, f], ["H2_L", "H2_R"]


def born_channel_from_ph(x, w, mean_d, dt, geo, prot, pref) -> dict:
    """A2,B2 and Kubo from p-h prefix response. Must match stored R^A."""
    from q8_m0_mode_io import response_to_complex_scalar, sigma_commutator_per_cell
    from q8_m0_response import predict_delta_mu

    times = np.arange(x.shape[0], dtype=float) * float(dt)
    hk = h2_kernels(times, prot.omega0, prot.t_pulse, geo)
    f, ehat, k_a = hk["F"], hk["e_perp"], hk["k_a"]
    ref = response_to_complex_scalar(x, w, f, ehat, dt, mean_d)
    hr = np.real(f)[:, None] * ehat[None, :]
    hi = np.imag(f)[:, None] * ehat[None, :]
    pr = response_from_ph_prefix(x, w, hr, dt, mean_d)
    pi = response_from_ph_prefix(x, w, hi, dt, mean_d)
    i_ph = pr["total"] + 1j * pi["total"]
    i_ref = ref["total"]
    c_ph = collect_A2_B2(k_a, i_ph, dt, pref)
    c_ref = collect_A2_B2(k_a, i_ref, dt, pref)
    comm = pref * sigma_commutator_per_cell(x, w, k_a, dt)["commutator"]
    a2 = complex(c_ph["A2"])
    a2_ref = complex(c_ref["A2"])
    identity = 2.0 * np.real(a2) + comm
    identity_ra = 2.0 * np.real(a2_ref) + comm
    pred = predict_delta_mu(x, w, hr, dt, mean_d)
    return {
        "maxabs_I_ph_minus_RA": float(np.max(np.abs(i_ph - i_ref))),
        "maxabs_state_ph_minus_prefix": float(np.max(np.abs(pr["state"] - pred["state"]))),
        "A2_ph": pack_complex(c_ph["A2"]),
        "B2_ph": pack_complex(c_ph["B2"]),
        "A2_RA": pack_complex(c_ref["A2"]),
        "B2_RA": pack_complex(c_ref["B2"]),
        "kubo_residual_abs": float(abs(identity)),
        "kubo_RA_residual_abs": float(abs(identity_ra)),
        "kubo_ph_minus_RA": float(abs(identity - identity_ra)),
        "two_Re_A2": float(2.0 * np.real(a2)),
        "comm_s": pack_complex(comm),
        "kept_order_closed": bool(abs(identity - identity_ra) < 1e-12),
        "kept_order_note": "ph Kubo equals R^A Kubo on the same grid; grid floor is not a fail",
    }


def c_greater_match(x, w) -> dict:
    from q8_m0_response import c_greater_from_x

    x0 = np.asarray(x[0])
    w = np.asarray(w, dtype=float)
    a = c_greater_from_x(x0, x0, w)
    b = c_greater_from_channels(x0, w)
    return {
        "maxabs": float(np.max(np.abs(a - b))),
        "note": "equal_time_C_greater_from_X_is_the_ph_vacuum_gram",
    }
