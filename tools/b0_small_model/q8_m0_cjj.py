#!/usr/bin/env python3
"""Classical-P1 electronic CJJ (greater connected, q=0 per cell).

Definition: classical_P1_q0_percell_greater_connected_v1
Authorized 2026-09-16 for small-scale algorithm validation only.
Not a photon number, g2, BSV, or output-mode calculation.
"""
from __future__ import annotations

import hashlib
import json
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np

from q8_m0_classical import (
    ALGORITHM,
    DriveProtocol,
    _sin2_window,
    e_field,
    initial_rho,
    k_grid,
    kappa_shift,
    project_harmonic,
    sha256_file,
    trace_current,
)
from q8_m0_model import (  # noqa: E402
    MODEL_ID,
    N_OCC,
    M0Params,
    current_ops,
    g_and_grad,
    h8,
    h8_and_current,
)

REPO = Path(__file__).resolve().parents[2]
CJJ_MODULE = Path(__file__).resolve()
CJJ_DEFINITION = "classical_P1_q0_percell_greater_connected_v1"
CJJ_ALGORITHM = "midpoint_unitary_XvVc_gram_v1"
DEFINITION_REVIEW = "user_accepted_limited_validation_20260916"
APPROVAL_SCOPE = "small_scale_classical_pump_electronic_source_validation"
ORDERS = tuple(range(1, 10))
N_SNAP_MAX = 25
# Envelope stations plus non-integer-cycle phases, recorded before any production run.
SNAPSHOT_FRAC = (
    0.0, 0.07, 0.13, 0.19, 0.26, 0.31, 0.39, 0.44, 0.47, 0.50,
    0.53, 0.58, 0.64, 0.71, 0.78, 0.84, 0.91, 0.97, 1.0,
    1.3 / 8.0, 2.6 / 8.0, 3.4 / 8.0, 4.7 / 8.0, 5.25 / 8.0, 6.8 / 8.0,
)


def dump(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    def _default(o):
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, (np.floating, np.integer)):
            return o.item()
        raise TypeError(type(o))

    path.write_text(json.dumps(obj, indent=2, default=_default, ensure_ascii=False), encoding="utf-8")


def occup_empty_bases(h0: np.ndarray, mu: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    e, w = np.linalg.eigh(h0)
    n_occ = (e < mu).sum(axis=1)
    if not np.all(n_occ == N_OCC):
        raise RuntimeError(f"μ does not occupy {N_OCC} bands at every k: {np.unique(n_occ)}")
    return w[:, :, :N_OCC], w[:, :, N_OCC:], e


def x_from_bases(j: np.ndarray, vv: np.ndarray, vc: np.ndarray) -> np.ndarray:
    """X[k,a,c,v] = (Vc† j_a Vv)_{cv}."""
    jv = np.einsum("kaij,kjv->kaiv", j, vv)
    return np.einsum("kic,kaiv->kacv", vc.conj(), jv)


def mean_from_bases(j: np.ndarray, vv: np.ndarray) -> np.ndarray:
    return np.real(np.einsum("kiv,kaij,kjv->ka", vv.conj(), j, vv))


def c_equal_from_x(x: np.ndarray) -> np.ndarray:
    """Per-k equal-time c^>_ab = Tr[X_a† X_b], shape (Nk,2,2)."""
    return np.einsum("kacv,kbcv->kab", np.conj(x), x)


def d_from_y(y: np.ndarray, w: np.ndarray) -> np.ndarray:
    """d[n,a,m,b] = Σ_k w_k Tr[Y_{n a}† Y_{m b}]."""
    return np.einsum("k,knacv,kmbcv->namb", w, np.conj(y), y)


def snapshot_indices(nt: int) -> np.ndarray:
    idxs = sorted({int(round(frac * nt)) for frac in SNAPSHOT_FRAC})
    idxs = [i for i in idxs if 0 <= i <= nt]
    if len(idxs) > N_SNAP_MAX:
        sel = np.linspace(0, len(idxs) - 1, N_SNAP_MAX).round().astype(int)
        idxs = [idxs[i] for i in sel]
    return np.array(idxs, dtype=int)


def _int_exp(alpha, tp: float) -> np.ndarray:
    """∫_0^{Tp} exp(i α t) dt."""
    a = np.asarray(alpha, dtype=complex)
    out = np.empty(a.shape, dtype=complex)
    small = np.abs(a) < 1e-12
    out[small] = tp
    aa = a[~small]
    out[~small] = (np.exp(1j * aa * tp) - 1.0) / (1j * aa)
    return out


def window_fourier(omega, tp: float) -> np.ndarray:
    """I(Ω)=∫_0^{Tp} sin²(πt/Tp)/Tp exp(iΩt) dt, closed form."""
    om = np.asarray(omega, dtype=float)
    two = 2.0 * np.pi / tp
    return _int_exp(om, tp) / (2.0 * tp) - _int_exp(om + two, tp) / (4.0 * tp) - _int_exp(om - two, tp) / (4.0 * tp)


def choose_probe_k(geo, p: M0Params, mu: float) -> dict:
    """Γ, K/−K, min-direct-gap inversion pair, one generic off-symmetry point."""
    kf48, kc48, _ = k_grid(geo, 48, shift=0.0)
    g, _ = g_and_grad(geo, kc48)
    i_k = int(np.argmax(np.abs(g)))
    k_frac = np.mod(kf48[i_k], 1.0)
    mk_frac = np.mod(-k_frac, 1.0)
    h = h8(geo, p, kc48, +1)
    e = np.linalg.eigvalsh(h)
    gap = e[:, N_OCC] - e[:, N_OCC - 1]
    i_gap = int(np.argmin(gap))
    gap_frac = np.mod(kf48[i_gap], 1.0)
    gap_inv = np.mod(-gap_frac, 1.0)
    generic = np.array([0.17, 0.41])
    pts = [
        ("Gamma", np.array([0.0, 0.0])),
        ("K", k_frac),
        ("mK", mk_frac),
        ("min_direct_gap", gap_frac),
        ("min_direct_gap_inv", gap_inv),
        ("generic", generic),
    ]
    used = []
    labels = []
    for name, vec in pts:
        key = tuple(np.round(np.mod(vec, 1.0), 8))
        if key in used:
            continue
        used.append(key)
        labels.append(name)
    extras = [np.array([0.22, 0.07]), np.array([0.31, 0.58]), np.array([0.09, 0.73])]
    ei = 0
    while len(used) < 6:
        key = tuple(np.round(np.mod(extras[ei], 1.0), 8))
        ei += 1
        if key in used:
            continue
        used.append(key)
        labels.append(f"extra{ei}")
    kf = np.mod(np.array([np.array(x) for x in used[:6]], dtype=float), 1.0)
    rec = {
        "labels": labels[:6],
        "kf": kf,
        "note": "Cursor-chosen probe set recorded before B0; not a user-selected mesh",
        "K_abs_g": float(np.max(np.abs(g))),
        "min_direct_gap_on_nk48": float(gap[i_gap]),
        "min_direct_gap_k_frac": gap_frac.tolist(),
    }
    return rec


def cjj_fingerprint(geo, p: M0Params, prot: DriveProtocol, *, nk, nt_per_cycle, mu, grid_shift,
                    neel, sign, kf_kind: str) -> dict:
    body = {
        "model_id": MODEL_ID,
        "algorithm": ALGORITHM,
        "cjj_algorithm": CJJ_ALGORITHM,
        "cjj_definition": CJJ_DEFINITION,
        "definition_review": DEFINITION_REVIEW,
        "approval_scope": APPROVAL_SCOPE,
        "dtype": "complex128",
        "params": p.as_dict(),
        "protocol": asdict(prot),
        "protocol_id": prot.protocol_id,
        "nk": int(nk),
        "nt_per_cycle": int(nt_per_cycle),
        "mu": float(mu),
        "grid_shift": float(grid_shift) if isinstance(grid_shift, (int, float)) else str(grid_shift),
        "kf_kind": kf_kind,
        "neel": int(neel),
        "field_sign": int(sign),
        "cjj_orders": list(ORDERS),
        "a_nn_ang": float(geo.a_nn_mean_ang),
        "a_cell_ang2": float(geo.a_cell_ang2),
        "source_hashes": geo.source_hashes,
        "quantum_input": "not_run",
        "photon_output": "not_run",
    }
    blob = json.dumps(body, sort_keys=True, ensure_ascii=True)
    body["cjj_identity_sha256"] = hashlib.sha256(blob.encode("utf-8")).hexdigest()
    body["cjj_module_sha256"] = sha256_file(CJJ_MODULE)
    return body


def propagate_cjj(
    geo,
    p: M0Params,
    prot: DriveProtocol,
    *,
    nk: int | None = None,
    nt_per_cycle: int,
    neel: int,
    sign: int,
    mu: float,
    verbose: bool = True,
    grid_shift: float = 0.0,
    kf_frac=None,
    save_full_x: bool = False,
    a_extra=None,
    save_mean_D: bool = False,
    save_perk_D: bool = False,
    w_k=None,
    n_hold_cycles: int = 0,
) -> dict:
    if kf_frac is None:
        nk_lin = int(nk)
        kf, kc, w = k_grid(geo, nk_lin, shift=grid_shift)
        kf_kind = f"mp_nk{nk_lin}_shift{float(grid_shift)}"
        n_k = int(kc.shape[0])
        if w_k is not None:
            w = np.asarray(w_k, dtype=float).reshape(n_k)
            w = w / float(np.sum(w))
    else:
        kf = np.mod(np.asarray(kf_frac, dtype=float), 1.0)
        kc = geo.k_cart(kf)
        nk_lin = int(kf.shape[0])
        kf_kind = "custom"
        grid_shift = float("nan")
        n_k = int(kc.shape[0])
        if w_k is None:
            w = np.full(n_k, 1.0 / n_k)
        else:
            w = np.asarray(w_k, dtype=float).reshape(n_k)
            wsum = float(np.sum(w))
            if wsum <= 0.0:
                raise ValueError("w_k must sum to a positive weight")
            w = w / wsum
    rho, e0 = initial_rho(geo, p, kc, neel, mu)
    h0 = h8(geo, p, kc, neel)
    vv, vc, e_bands = occup_empty_bases(h0, mu)
    vv0 = vv.copy()
    vc0 = vc.copy()
    n_hold = max(0, int(n_hold_cycles))
    nt = (prot.n_cycles + n_hold) * nt_per_cycle
    dt = prot.period / nt_per_cycle
    times = np.arange(nt + 1) * dt
    j_t = np.zeros((nt + 1, 2))
    mu_t = np.zeros((nt + 1, 2))
    n_ord = len(ORDERS)
    y_acc = np.zeros((n_k, n_ord, 2, 4, 4), dtype=complex)
    y_prev = None
    snap_idx = snapshot_indices(nt)
    snap_x = np.zeros((snap_idx.size, n_k, 2, 4, 4), dtype=complex)
    snap_c = np.zeros((snap_idx.size, n_k, 2, 2), dtype=complex)
    snap_vv = np.zeros((snap_idx.size, n_k, 8, 4), dtype=complex)
    snap_vc = np.zeros((snap_idx.size, n_k, 8, 4), dtype=complex)
    snap_pos = {int(i): p for p, i in enumerate(snap_idx)}
    full_x = np.zeros((nt + 1, n_k, 2, 4, 4), dtype=complex) if save_full_x else None
    need_D = bool(save_mean_D or save_perk_D)
    mean_D_t = np.zeros((nt + 1, 2, 2), dtype=float) if need_D else None
    perk_D_t = np.zeros((nt + 1, n_k, 2, 2), dtype=float) if save_perk_D else None
    perk_mu_t = np.zeros((nt + 1, n_k, 2), dtype=float) if save_perk_D else None
    omegas = np.array(ORDERS, dtype=float) * prot.omega0

    def _a_ex(t_now, idx):
        if a_extra is None:
            return 0.0
        if callable(a_extra):
            return np.asarray(a_extra(float(t_now)), dtype=float)
        return np.asarray(a_extra[idx], dtype=float)
    diag_every = max(1, nt // 64)
    diag = {"t": [], "tr_rho": [], "herm": [], "idem": [], "mean_vs_rho_maxabs": []}
    t0 = time.perf_counter()
    work = 0.0
    e_prev = None
    for n in range(nt + 1):
        t = times[n]
        kap = kc + kappa_shift(t, geo, prot, sign) + _a_ex(t, n)
        need_h = n % diag_every == 0 or n == nt or n in snap_pos
        if need_h:
            h, j = h8_and_current(geo, p, kap, neel)
        else:
            h = None
            j = current_ops(geo, p, kap, neel)
        jk_rho = trace_current(rho, j)
        jk_vv = mean_from_bases(j, vv)
        j_t[n] = w @ jk_rho
        mu_t[n] = w @ jk_vv
        x = x_from_bases(j, vv, vc)
        if save_full_x:
            full_x[n] = x
        if need_D:
            from q8_m0_response import d_ops, mean_d_from_bases

            d_ab = d_ops(geo, p, kap, neel)
            d_k = mean_d_from_bases(d_ab, vv)
            mean_D_t[n] = np.einsum("k,kab->ab", w, d_k)
            if save_perk_D:
                perk_D_t[n] = d_k
                perk_mu_t[n] = jk_vv
        if n in snap_pos:
            s = snap_pos[n]
            snap_x[s] = x
            snap_c[s] = c_equal_from_x(x)
            snap_vv[s] = vv
            snap_vc[s] = vc
        wgt = _sin2_window(t, prot.t_pulse) / prot.t_pulse
        phases = np.exp(1j * omegas * t)
        term = (wgt * phases)[None, :, None, None, None] * x[:, None, :, :, :]
        if n > 0:
            y_acc += 0.5 * dt * (y_prev + term)
        y_prev = term
        e_now = np.asarray(e_field(np.array([t]), geo, prot, sign), dtype=float)[0]
        if n > 0:
            work += 0.5 * dt * (float(np.dot(j_t[n - 1], e_prev)) + float(np.dot(j_t[n], e_now)))
        e_prev = e_now
        if need_h:
            diag["t"].append(float(t))
            diag["tr_rho"].append(float(np.real(np.einsum("k,kii->", w, rho))))
            diag["herm"].append(float(np.max(np.abs(rho - np.conj(np.swapaxes(rho, 1, 2))))))
            diag["idem"].append(float(np.max(np.abs(rho @ rho - rho))))
            diag["mean_vs_rho_maxabs"].append(float(np.max(np.abs(jk_rho - jk_vv))))
        if n == nt:
            break
        kmid = kc + kappa_shift(t + 0.5 * dt, geo, prot, sign) + _a_ex(t + 0.5 * dt, n)
        hm = h8(geo, p, kmid, neel)
        em, wm = np.linalg.eigh(hm)
        ph = np.exp(-1j * em * dt)
        u = (wm * ph[:, None, :]) @ np.conj(np.swapaxes(wm, 1, 2))
        rho = u @ rho @ np.conj(np.swapaxes(u, 1, 2))
        vv = u @ vv
        vc = u @ vc
        if verbose and n % max(1, nt // 8) == 0:
            print(f"  cjj step {n}/{nt}  t={t:.1f}  elapsed {time.perf_counter() - t0:.1f}s", flush=True)
    d = d_from_y(y_acc, w)
    b = np.array([project_harmonic(times, mu_t, n * prot.omega0, "sin2") for n in ORDERS])
    out = {
        "neel": neel,
        "field_sign": sign,
        "nk": int(nk_lin),
        "n_k": int(n_k),
        "nt_per_cycle": int(nt_per_cycle),
        "grid_shift": None if kf_kind == "custom" else float(grid_shift),
        "kf_kind": kf_kind,
        "dt": dt,
        "n_hold_cycles": n_hold,
        "times": times,
        "kf": kf,
        "w_k": w,
        "J_sheet": j_t / geo.a_cell_ang2,
        "mu_per_cell": mu_t,
        "Y": y_acc,
        "d_namb": d,
        "b_n": b,
        "orders": np.array(ORDERS),
        "snapshot_index": snap_idx,
        "snapshot_t": times[snap_idx],
        "snapshot_X": snap_x,
        "snapshot_c_ab": snap_c,
        "snapshot_Vv": snap_vv,
        "snapshot_Vc": snap_vc,
        "diag": {k: np.array(v) for k, v in diag.items()},
        "wall_seconds": time.perf_counter() - t0,
        "work_integral": work,
        "vv0": vv0,
        "vc0": vc0,
        "e_bands0": e_bands,
        "cjj_definition": CJJ_DEFINITION,
        "mean_vs_rho_maxabs": float(np.max(np.abs(j_t - mu_t))),
        "a_cell_ang2": float(geo.a_cell_ang2),
    }
    if save_full_x:
        out["X_full"] = full_x
        out["U_from_bases_note"] = "U(t)=[Vv(t) Vc(t)][Vv0 Vc0]†"
    if need_D:
        out["mean_D_t"] = mean_D_t
        out["a_extra"] = "applied" if a_extra is not None else None
    if save_perk_D:
        out["perk_D_t"] = perk_D_t
        out["perk_mu_t"] = perk_mu_t
    return out


def gram_psd_residual(d: np.ndarray) -> dict:
    """Flatten (n,a) into a Gram matrix over source filters."""
    n_ord, n_dir = d.shape[0], d.shape[1]
    g = d.reshape(n_ord * n_dir, n_ord * n_dir)
    g_h = 0.5 * (g + g.conj().T)
    evals = np.linalg.eigvalsh(g_h)
    return {
        "herm_residual": float(np.max(np.abs(g - g.conj().T))),
        "min_eig": float(np.min(evals.real)),
        "neg_eig_floor": float(min(0.0, np.min(evals.real))),
        "psd_ok": bool(np.min(evals.real) >= -1e-10 * max(1.0, np.max(np.abs(evals)))),
    }


def undriven_y(geo, p: M0Params, prot: DriveProtocol, kf, mu: float, neel: int) -> dict:
    """Static H0, analytic X(t)=e^{i(Ec-Ev)t} X0, same Q9 window. Both Neels needed."""
    kf = np.mod(np.asarray(kf, dtype=float), 1.0)
    kc = geo.k_cart(kf)
    w = np.full(kc.shape[0], 1.0 / kc.shape[0])
    h0 = h8(geo, p, kc, neel)
    vv, vc, e = occup_empty_bases(h0, mu)
    j0 = current_ops(geo, p, kc, neel)
    x0 = x_from_bases(j0, vv, vc)
    omega_cv = e[:, N_OCC:, None] - e[:, None, :N_OCC]
    y = np.zeros((kf.shape[0], len(ORDERS), 2, 4, 4), dtype=complex)
    for i_n, n_h in enumerate(ORDERS):
        om = n_h * prot.omega0 + omega_cv
        iw = window_fourier(om, prot.t_pulse)
        y[:, i_n, :, :, :] = iw[:, None, :, :] * x0
    return {
        "Y": y,
        "d_namb": d_from_y(y, w),
        "w_k": w,
        "kf": kf,
        "neel": neel,
        "omega_cv_minmax": [float(omega_cv.min()), float(omega_cv.max())],
        "note": "finite-window ground-state source filter; not spontaneous emission",
    }


def undriven_trapz_check(geo, p: M0Params, prot: DriveProtocol, kf, mu: float, neel: int, nt_per_cycle: int) -> dict:
    """Trapezoid of analytic X(t) vs closed-form window Fourier; for small k sets."""
    kf = np.mod(np.asarray(kf, dtype=float), 1.0)
    rec = undriven_y(geo, p, prot, kf, mu, neel)
    kc = geo.k_cart(kf)
    h0 = h8(geo, p, kc, neel)
    vv, vc, e = occup_empty_bases(h0, mu)
    x0 = x_from_bases(current_ops(geo, p, kc, neel), vv, vc)
    omega_cv = e[:, N_OCC:, None] - e[:, None, :N_OCC]
    nt = prot.n_cycles * nt_per_cycle
    dt = prot.period / nt_per_cycle
    times = np.arange(nt + 1) * dt
    y_trap = np.zeros_like(rec["Y"])
    prev = None
    for n, t in enumerate(times):
        x = x0 * np.exp(1j * omega_cv * t)[:, None, :, :]
        wgt = _sin2_window(t, prot.t_pulse) / prot.t_pulse
        term = (wgt * np.exp(1j * np.array(ORDERS) * prot.omega0 * t))[None, :, None, None, None] * x[:, None]
        if n > 0:
            y_trap += 0.5 * dt * (prev + term)
        prev = term
    w = rec["w_k"]
    return {
        "max_abs_Y_diff": float(np.max(np.abs(y_trap - rec["Y"]))),
        "max_abs_d_diff": float(np.max(np.abs(d_from_y(y_trap, w) - rec["d_namb"]))),
        "nt_per_cycle": nt_per_cycle,
        "note": "trapz vs closed-form I(Ω); finite-dt disagreement is expected and is the dt check",
    }


def q4_from_schrodinger(ja, jb, u_t_tp, rho_tp, q_tp) -> np.ndarray:
    """Tr[j_a(t) U(t,t') (I-ρ') j_b(t') ρ' U†(t,t')], per k. ja,jb (Nk,8,8)."""
    mid = q_tp @ jb @ rho_tp
    mapped = u_t_tp @ mid @ np.conj(np.swapaxes(u_t_tp, 1, 2))
    return np.einsum("kij,kji->k", ja, mapped)


def domain_combine(d_plus: np.ndarray, d_minus: np.ndarray) -> dict:
    de = 0.5 * (d_plus + d_minus)
    do = 0.5 * (d_plus - d_minus)
    return {"d_dom_e": de, "d_dom_o": do, "psd_plus": gram_psd_residual(d_plus), "psd_minus": gram_psd_residual(d_minus)}
