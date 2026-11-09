"""Replace the whole [0.0001, 0.02] tail panel and compare quadratures at 4096.

The three infrared nodes are part of one 24-node sinh panel. Removing only
those three and inserting another integral leaves the other 21 nodes in place.
"""
from pathlib import Path
import json
import os
import time

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np

import q8_m0_gamma12_transpose_input_refine_20261008 as base
import q8_m0_p1_h2_finite_kcv_input_close_20261005 as q

OUT = base.OUT
A, B = 1.0e-4, 0.02


def pack(m):
    m = np.asarray(m)
    return {"real": np.real(m).tolist(), "imag": np.imag(m).tolist(),
            "maxabs": float(np.max(np.abs(m)))}


def log_panel(a, b, n):
    xa, xb = np.log(a), np.log(b)
    x, w = np.polynomial.legendre.leggauss(int(n))
    xx = 0.5 * (xb - xa) * x + 0.5 * (xa + xb)
    om = np.exp(xx)
    return om, 0.5 * (xb - xa) * w * om


def panel_nodes(gaps, u, gamma):
    bright, widths, groups = q.grouped_bright_gaps(gaps, u, gamma)
    n_g = len(groups)
    g_left = np.array([float(bright[grp][0]) for grp in groups])
    g_right = np.array([float(bright[grp][-1]) for grp in groups])
    cuts = np.empty(n_g + 1)
    cuts[0], cuts[-1] = A, 12.0
    for i in range(n_g - 1):
        cuts[i + 1] = 0.5 * (g_right[i] + g_left[i + 1])
    for i, grp in enumerate(groups):
        dgs, gs = bright[grp], widths[grp]
        cell_lo, cell_hi = float(cuts[i]), float(cuts[i + 1])
        pole_cuts = (np.concatenate(([cell_lo], 0.5 * (dgs[:-1] + dgs[1:]), [cell_hi]))
                     if dgs.size > 1 else np.array([cell_lo, cell_hi]))
        for j, dg in enumerate(dgs):
            aa, bb = float(pole_cuts[j]), float(pole_cuts[j + 1])
            gwid = max(float(gs[j]), 1e-16)
            t_left = float(np.clip(np.arcsinh((aa - float(dg)) / gwid), -q.T_TAIL_CAP, q.T_TAIL_CAP))
            t_right = float(np.clip(np.arcsinh((bb - float(dg)) / gwid), -q.T_TAIL_CAP, q.T_TAIL_CAP))
            _rank, all_w = q.cluster_rank_width(gaps, u, gamma, float(dg))
            g_max = float(np.max(all_w)) if getattr(all_w, "size", 0) else gwid
            half = max(min(float(dg) - aa, bb - float(dg)), 8.0 * gwid)
            t_cell = float(np.arcsinh(half / gwid))
            t_wide = float(np.arcsinh(8.0 * g_max / gwid))
            t_core = min(q.T_MAX_CAP, max(q.T_MAX_MIN, t_wide, min(t_cell, q.T_MAX_CAP)))
            t_core = min(t_core, max(-t_left, 0.0), max(t_right, 0.0))
            o0 = float(dg) + gwid * np.sinh(t_left)
            if abs(o0 - A) > 1e-8:
                continue
            ta = float(np.arcsinh((A - float(dg)) / gwid))
            tb = float(np.arcsinh((B - float(dg)) / gwid))
            n = min(q.even_count(int(q.N_TAIL_MIN + 2.0 * (tb - ta)), q.N_TAIL_MIN), q.N_TAIL_CAP) * 2
            return q.t_gauss_panel(float(dg), gwid, ta, tb, n)
    raise RuntimeError("Infrared panel not found")


def contract_frequencies(cache, rows, gamma, pol, phi_of, eta, oms, wts):
    normal = np.zeros((2, 2), complex)
    anomalous = np.zeros((2, 2), complex)
    for first in range(0, len(oms), 128):
        sl = slice(first, min(first + 128, len(oms)))
        ph, u0 = base.generated_photon_block(cache, gamma, oms[sl], wts[sl], pol, phi_of, eta)
        got, _tt = base._contract(rows, (ph.initial.b, ph.initial.bdag, ph.initial.A),
                                  ph.omega, ph.free_A0, 0.0, 128, 2048)
        du = got["driven_raw_u"] - got["undriven_raw_u"]
        v = got["driven_raw_v"] - got["undriven_raw_v"]
        u = u0 + du
        normal += u @ u.conj().T - v @ v.conj().T
        anomalous += u @ v.T - v @ u.T
        if first == 0 or sl.stop == len(oms) or sl.stop % 2048 == 0:
            print(f"  contracted {sl.stop}/{len(oms)}", flush=True)
    return normal, anomalous


def residuals(n_all, a_all, n_old, a_old, n_new, a_new):
    rn = n_all - n_old + n_new - np.eye(2)
    ra = a_all - a_old + a_new
    return rn, ra


def main():
    started = time.perf_counter()
    manifest, cache, rows4096, _info, _ident = base.load_frozen()
    gamma = float(cache["gamma"])
    pol = np.array(cache["pol"], copy=True)
    prot = base.p1_a2_protocol(base.q.A_STAR)
    op = base.q.omega_grid(prot.omega0, prot.t_pulse)
    op = op[op > 0]
    _, norm = base.q.normalize_phi(op, base.q.phi_unnormalized(op, 2 * prot.omega0, prot.t_pulse))
    phi_of = lambda nu: base.q.phi_at(nu, prot, norm)
    eta = base.q.eta_of(gamma)
    saved = np.load(OUT / "endpoint_low.npz")
    om0, wt0 = panel_nodes(cache["gaps"], cache["u0"], gamma)
    if abs(float(wt0.sum()) - (B - A)) > 1e-6:
        raise AssertionError(f"panel weight {float(wt0.sum())} != {B-A}")
    saved_key = np.round(saved["oms"], 12)
    if not np.all(np.isin(np.round(om0, 12), saved_key)):
        raise AssertionError("Rebuilt panel is not inside the saved nodes")
    outside = ~np.isin(saved_key, np.round(om0, 12))
    if np.any((saved["oms"][outside] >= A) & (saved["oms"][outside] < B)):
        raise AssertionError("A node inside the panel was left unmatched")
    idx = np.flatnonzero(np.isin(saved_key, np.round(om0, 12)))
    cols = (4 * idx[:, None] + np.arange(4)).ravel()
    u_old, v_old = saved["U"][:, cols], saved["V"][:, cols]
    n_old = u_old @ u_old.conj().T - v_old @ v_old.conj().T
    a_old = u_old @ v_old.T - v_old @ u_old.T
    n_all = saved["U"] @ saved["U"].conj().T - saved["V"] @ saved["V"].conj().T
    a_all = saved["U"] @ saved["V"].T - saved["V"] @ saved["U"].T
    catalog = {"original_24": (om0, wt0)}
    for n in (24, 48, 96):
        catalog[f"gauss_{n}"] = q.gauss_segment(A, B, n)
        catalog[f"log_{n}"] = log_panel(A, B, n)
    methods = {}
    for name, (om, wt) in catalog.items():
        nu, au = contract_frequencies(cache, rows4096, gamma, pol, phi_of, eta, om, wt)
        rn, ra = residuals(n_all, a_all, n_old, a_old, nu, au)
        methods[name] = {"n_frequencies": int(len(om)), "weight_sum": float(np.sum(wt)),
                         "R": pack(rn), "A": pack(ra),
                         "Rmax": float(np.max(np.abs(rn))), "Amax": float(np.max(np.abs(ra)))}
        print(name, "Rmax", methods[name]["Rmax"], "R00", float(rn[0, 0].real), flush=True)
    # Do not contract 1024/2048 from photon frequencies alone. That omits the
    # dark columns and is not an output residual.
    report = {
        "bug_in_previous_ir_swap": "Removed 3 of 24 nodes from the panel [0.0001, 0.02] and inserted a separate integral of [1e-4, 1e-3].",
        "panel": [A, B], "original_nodes": int(len(om0)),
        "original_weight": float(wt0.sum()), "interval_length": B - A,
        "methods_at_4096": methods,
        "elapsed_s": time.perf_counter() - started,
        "production_qualification": False,
        "gauss96_timesteps_invalid": "1024 and 2048 contractions omitted the 1976 dark columns, so those Rmax values are not residuals.",
    }
    path = OUT / "ir_panel_crosscheck.json"
    path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print("methods", len(methods))


if __name__ == "__main__":
    main()
