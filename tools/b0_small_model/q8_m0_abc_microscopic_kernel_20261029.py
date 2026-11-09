#!/usr/bin/env python3
"""Eight-band M0 microscopic analysis of A/B/C. Not chi^(2), not photon output.

Reads the 2026-10-28 Fourier-repaired BAC fits. Builds the complete Peierls
two-frequency retarded current kernel at the Q6 k/-k pair. Compares the
lowest-order map J ~ K (alpha^2 + M) with the measured coefficients.
Does not propagate electrons, does not overwrite Q6 archives, and does not
start server jobs.
"""
from __future__ import annotations

import json
import math
import sys
import time
from itertools import product
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))

from q8_m0_model import (  # noqa: E402
    M0Params,
    R0,
    _interlayer_block,
    _layer_block,
    _phase,
    assemble,
    build_geometry,
    current_ops,
    dh8,
    h8,
)
from q8_m0_qinput_minimal_20261020 import a2_polarization  # noqa: E402
from q8_m0_qinput_shared_kk_static_20261029 import K_FRAC, OMEGA0  # noqa: E402
from q8_m0_response import d2h8, d_ops_finite_difference  # noqa: E402

OUT = ROOT / "results" / "q_line" / "abc_microscopic_20261029"
FITS = ROOT / "results" / "q_line" / "qinput_spectrum_reanalysis_20261028" / "fits_BAC.json"
KEY = ROOT / "results" / "q_line" / "qinput_spectrum_reanalysis_20261028" / "key_results.json"
PHYS1600 = ROOT / "results" / "q_line" / "q1600_bz_stat_production_20261028" / "physics_i1e+10_nk21_gh5.json"

ALPHA = 0.5 + 0.0j
R_SQUEEZE = 0.3
OMEGA_1600 = 1.23984193 / 1.6


def cpack(z):
    z = complex(z)
    return {
        "re": float(z.real),
        "im": float(z.imag),
        "abs": float(abs(z)),
        "phase_deg": float(np.degrees(np.angle(z))),
    }


def from_fit(rec):
    return rec["re"] + 1j * rec["im"]


def load_fit(rows, **sel):
    hits = [r for r in rows if all(r[k] == v for k, v in sel.items())]
    if len(hits) != 1:
        raise RuntimeError(f"expected 1 fit for {sel}, got {len(hits)}")
    h = hits[0]
    return {
        "B": from_fit(h["B"]),
        "A": from_fit(h["A"]),
        "C": from_fit(h["C"]),
        "abs_C_over_A": float(h["abs_C_over_A"]),
        "relative_L2_residual": float(h["relative_L2_residual"]),
        "meta": {k: h[k] for k in ("solver", "eta", "window", "estimator", "channel", "axis")},
    }


def ds_moments(alpha=ALPHA, r=R_SQUEEZE):
    a2 = abs(alpha) ** 2
    mabs = 0.5 * math.sinh(2.0 * r)
    nbar = a2 + math.sinh(r) ** 2
    return {
        "alpha": cpack(alpha),
        "r": float(r),
        "alpha2": float(a2),
        "M_abs": float(mabs),
        "M_over_alpha2": float(mabs / a2),
        "nbar": float(nbar),
        "field_pi_over_0": float((a2 + mabs) / (a2 - mabs)),
        "note": "Ideal D(alpha)S(r e^{i theta})|0> with real alpha. M = -0.5 sinh(2r) e^{i theta}.",
    }


def geometry_from_abc(B, A, C=0.0):
    rho = abs(A) / abs(B)
    arg_ab = float(np.degrees(np.angle(A / B)))
    j0 = B + A + C
    jpi = B - A + C
    jh = B + 1j * A - 1j * C
    jq = B - 1j * A + 1j * C
    return {
        "rho_absA_over_absB": float(rho),
        "arg_A_over_B_deg": arg_ab,
        "two_arctan_rho_deg": float(np.degrees(2.0 * math.atan(rho))),
        "pi_over_0_amp_ratio": float(abs(jpi) / abs(j0)),
        "covflip_phase_diff_deg": float(np.degrees(np.angle(jq) - np.angle(jh))),
        "theta0": cpack(j0),
        "theta_pi": cpack(jpi),
        "theta_pi2": cpack(jh),
        "theta_3pi2": cpack(jq),
    }


def d3h8(geo, p, k2, neel: int) -> np.ndarray:
    """d^3 H / d k_a d k_b d k_c, shape (..., 2, 2, 2, 8, 8). Exchange drops."""
    k2 = np.asarray(k2, dtype=float)
    shp = k2.shape[:-1]
    d3f = np.zeros(shp + (2, 2, 2), dtype=complex)
    d3g = np.zeros(shp + (2, 2, 2), dtype=float)
    d3t0 = np.zeros(shp + (2, 2, 2, 2, 2), dtype=complex)
    for delta in geo.nn_deltas_cart:
        ph = _phase(k2, delta)
        for a, b, c in product(range(2), repeat=3):
            d3f[..., a, b, c] += (1j * delta[a]) * (1j * delta[b]) * (1j * delta[c]) * ph
    for nu, bvec in geo.nnn_terms:
        arg = k2[..., 0] * bvec[0] + k2[..., 1] * bvec[1]
        cc = np.cos(arg)
        for a, b, c in product(range(2), repeat=3):
            d3g[..., a, b, c] += -2.0 * nu * bvec[a] * bvec[b] * bvec[c] * cc
    idx = {"A": 0, "B": 1}
    amp = {"u": p.u, "v": p.v}
    for bond in geo.interlayer_bonds:
        i, j = idx[bond["from"][1]], idx[bond["to"][1]]
        ph = amp[bond["amplitude"]] * _phase(k2, bond["delta_cart"])
        dlt = bond["delta_cart"]
        for a, b, c in product(range(2), repeat=3):
            d3t0[..., a, b, c, i, j] += (1j * dlt[a]) * (1j * dlt[b]) * (1j * dlt[c]) * ph
    out = np.zeros(shp + (2, 2, 2, 8, 8), dtype=complex)
    for a, b, c in product(range(2), repeat=3):
        ht = _layer_block(d3f[..., a, b, c], d3g[..., a, b, c], 0.0, p.t_h, p.lam_so)
        out[..., a, b, c, :, :] = assemble(ht, ht, _interlayer_block(d3t0[..., a, b, c, :, :]))
    del neel
    return out


def d3h8_finite_difference(geo, p, k2, neel: int, eps: float = 1e-4) -> np.ndarray:
    k2 = np.asarray(k2, dtype=float)
    out = np.zeros(k2.shape[:-1] + (2, 2, 2, 8, 8), dtype=complex)
    eye = np.zeros(k2.shape[:-1] + (2,), dtype=float)
    for c in range(2):
        shift = eye.copy()
        shift[..., c] = eps
        dp = d2h8(geo, p, k2 + shift, neel)
        dm = d2h8(geo, p, k2 - shift, neel)
        out[..., :, :, c, :, :] = (dp - dm) / (2.0 * eps)
    return out


def derivative_checks(geo, p, kcart, neel=1):
    k2 = np.asarray(kcart, dtype=float)
    d2_an = d2h8(geo, p, k2, neel)
    d2_fd = d_ops_finite_difference(geo, p, k2, neel, eps=1e-4)
    d3_an = d3h8(geo, p, k2, neel)
    d3_fd = d3h8_finite_difference(geo, p, k2, neel, eps=1e-4)
    j = current_ops(geo, p, k2, neel)
    dh = dh8(geo, p, k2, neel)
    return {
        "max_abs_j_plus_dh": float(np.max(np.abs(j + dh))),
        "max_rel_d2_vs_fd": float(np.max(np.abs(d2_an - d2_fd)) / max(np.max(np.abs(d2_an)), 1e-30)),
        "max_rel_d3_vs_fd": float(np.max(np.abs(d3_an - d3_fd)) / max(np.max(np.abs(d3_an)), 1e-30)),
        "d2_hermiticity": float(np.max(np.abs(d2_an - np.conj(np.swapaxes(d2_an, -1, -2))))),
        "d3_hermiticity": float(np.max(np.abs(d3_an - np.conj(np.swapaxes(d3_an, -1, -2))))),
    }


def peierls_kernel_one_k(H, dH, d2H, d3H, ehat, nhat, z1, z2):
    evals, U = np.linalg.eigh(H)

    def rot(A):
        return U.conj().T @ A @ U

    P = np.zeros((8, 8), dtype=complex)
    P[:4, :4] = np.eye(4)
    V = rot(np.einsum("a,aij->ij", ehat, dH))
    V2 = rot(np.einsum("a,b,abij->ij", ehat, ehat, d2H))
    J0 = rot(-np.einsum("a,aij->ij", nhat, dH))
    J1 = rot(-np.einsum("a,b,abij->ij", nhat, ehat, d2H))
    J2 = rot(-np.einsum("a,b,c,abcij->ij", nhat, ehat, ehat, d3H))
    gaps = evals[:, None] - evals[None, :]

    def comm(A, B):
        return A @ B - B @ A

    R1 = comm(V, P) / (z1 - gaps)
    R2 = comm(V, P) / (z2 - gaps)
    R12 = (comm(V, R1) + comm(V, R2) + comm(V2, P)) / (z1 + z2 - gaps)
    t0 = np.trace(R12 @ J0)
    t1 = np.trace((R1 + R2) @ J1)
    t2 = np.trace(P @ J2)
    tot = t0 + t1 + t2
    return tot, (t0, t1, t2), evals


def pair_kernel(geo, p, kcart, neel, ehat, nhat, z1, z2):
    acc = 0.0 + 0.0j
    terms = np.zeros(3, dtype=complex)
    evals_keep = None
    for sign in (1.0, -1.0):
        kk = sign * np.asarray(kcart, dtype=float)
        H = h8(geo, p, kk, neel)
        dH = dh8(geo, p, kk, neel)
        d2H = d2h8(geo, p, kk, neel)
        d3H = d3h8(geo, p, kk, neel)
        tot, t, evals = peierls_kernel_one_k(H, dH, d2H, d3H, ehat, nhat, z1, z2)
        acc += tot
        terms += np.array(t)
        if sign > 0:
            evals_keep = evals
    return acc, terms, evals_keep


def odd_even_kernel(geo, p, kcart, ehat, nhat, z1, z2):
    kp, tp, ep = pair_kernel(geo, p, kcart, +1, ehat, nhat, z1, z2)
    km, tm, em = pair_kernel(geo, p, kcart, -1, ehat, nhat, z1, z2)
    return {
        "odd": kp - km,
        "even": 0.5 * (kp + km),
        "plus": kp,
        "minus": km,
        "terms_odd": tp - tm,
        "evals_plus": ep,
        "evals_minus": em,
    }


def gap_report(evals):
    occ = evals[:4]
    virt = evals[4:]
    gap = float(np.min(virt) - np.max(occ))
    return {
        "evals": [float(x) for x in evals],
        "vbm": float(np.max(occ)),
        "cbm": float(np.min(virt)),
        "min_gap": gap,
        "gap_over_omega0": gap / OMEGA0,
        "gap_over_2omega0": gap / (2.0 * OMEGA0),
    }


def kernel_record(geo, p, kcart, ehat, omega, gamma, label):
    z = omega + 1j * gamma
    out = {"label": label, "omega": float(omega), "gamma": float(gamma), "params": p.as_dict(), "axes": {}}
    for ax, nhat in (("x", np.array([1.0, 0.0])), ("y", np.array([0.0, 1.0]))):
        rec = odd_even_kernel(geo, p, kcart, ehat, nhat, z, z)
        out["axes"][ax] = {
            "odd": cpack(0.5 * rec["odd"]),
            "even": cpack(rec["even"]),
            "plus": cpack(rec["plus"]),
            "minus": cpack(rec["minus"]),
            "terms_odd": [cpack(0.5 * z) for z in rec["terms_odd"]],
        }
        if ax == "x":
            out["gap_plus"] = gap_report(rec["evals_plus"])
            out["gap_minus"] = gap_report(rec["evals_minus"])
    return out


def permutation_check(geo, p, kcart, ehat, omega, gamma):
    z1 = omega + 1j * gamma
    z2 = 1.25 * omega + 1j * gamma
    nhat = np.array([1.0, 0.0])
    a = odd_even_kernel(geo, p, kcart, ehat, nhat, z1, z2)
    b = odd_even_kernel(geo, p, kcart, ehat, nhat, z2, z1)
    return {
        "odd_perm_abs": float(abs(a["odd"] - b["odd"])),
        "even_perm_abs": float(abs(a["even"] - b["even"])),
    }


def svg_escape(s):
    return str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def write_svg_complex_plane(path, pts, title):
    w, h = 640, 480
    xs = [p["z"].real for p in pts] + [0.0]
    ys = [p["z"].imag for p in pts] + [0.0]
    mx = max(max(abs(x) for x in xs), max(abs(y) for y in ys), 1e-30)
    pad = 40

    def xy(z):
        x = pad + (w - 2 * pad) * (0.5 + 0.45 * z.real / mx)
        y = pad + (h - 2 * pad) * (0.5 - 0.45 * z.imag / mx)
        return x, y

    ox, oy = xy(0j)
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}">',
        '<rect width="100%" height="100%" fill="#fffaf3"/>',
        f'<line x1="{pad}" y1="{oy}" x2="{w-pad}" y2="{oy}" stroke="#bbb"/>',
        f'<line x1="{ox}" y1="{pad}" x2="{ox}" y2="{h-pad}" stroke="#bbb"/>',
        f'<text x="{w/2}" y="24" text-anchor="middle" font-size="16">{svg_escape(title)}</text>',
    ]
    colors = ["#1d4e89", "#c44", "#2a9d8f", "#e9c46a", "#6d597a", "#222"]
    for i, p in enumerate(pts):
        x, y = xy(p["z"])
        col = colors[i % len(colors)]
        parts.append(f'<line x1="{ox}" y1="{oy}" x2="{x}" y2="{y}" stroke="{col}" stroke-width="2"/>')
        parts.append(f'<circle cx="{x}" cy="{y}" r="4" fill="{col}"/>')
        parts.append(f'<text x="{x+6}" y="{y-6}" font-size="12" fill="{col}">{svg_escape(p["name"])}</text>')
    parts.append("</svg>")
    path.write_text("\n".join(parts), encoding="utf-8")


def write_svg_bars(path, groups, title, ylabel):
    w, h = 720, 420
    pad = 50
    ymax = max(max(v for _, v in g["vals"]) for g in groups) * 1.15
    n = len(groups)
    gw = (w - 2 * pad) / n
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}">',
        '<rect width="100%" height="100%" fill="#fffaf3"/>',
        f'<text x="{w/2}" y="24" text-anchor="middle" font-size="16">{svg_escape(title)}</text>',
        f'<text x="16" y="{h/2}" font-size="12" transform="rotate(-90 16 {h/2})">{svg_escape(ylabel)}</text>',
    ]
    colors = ["#1d4e89", "#c44", "#2a9d8f", "#6d597a"]
    for gi, g in enumerate(groups):
        m = len(g["vals"])
        bw = gw / (m + 1)
        x0 = pad + gi * gw
        for j, (name, val) in enumerate(g["vals"]):
            bh = (h - 2 * pad) * val / ymax
            x = x0 + (j + 0.5) * bw
            y = h - pad - bh
            col = colors[j % len(colors)]
            parts.append(f'<rect x="{x}" y="{y}" width="{bw*0.8}" height="{bh}" fill="{col}"/>')
            parts.append(f'<text x="{x+bw*0.4}" y="{y-4}" text-anchor="middle" font-size="10">{val:.3g}</text>')
            if gi == 0:
                parts.append(f'<text x="{pad}" y="{30+14*j}" font-size="12" fill="{col}">{svg_escape(name)}</text>')
        parts.append(
            f'<text x="{x0+gw/2}" y="{h-18}" text-anchor="middle" font-size="12">{svg_escape(g["label"])}</text>'
        )
    parts.append("</svg>")
    path.write_text("\n".join(parts), encoding="utf-8")


def main():
    t0 = time.monotonic()
    OUT.mkdir(parents=True, exist_ok=True)
    fits = json.loads(FITS.read_text(encoding="utf-8"))
    key = json.loads(KEY.read_text(encoding="utf-8"))
    p1600 = json.loads(PHYS1600.read_text(encoding="utf-8"))
    geo = build_geometry()
    kcart = geo.k_cart(K_FRAC)
    ehat = a2_polarization(geo)
    mom = ds_moments()

    bac = {}
    # QED archive only has eta=1 (main+front). GH5 has eta=0.5/1/1.5.
    wanted = []
    for ch in ("odd", "even"):
        for ax in ("x", "y"):
            wanted.append(("QED", 1.0, ch, ax))
            for eta in (0.5, 1.0, 1.5):
                wanted.append(("GH5", eta, ch, ax))
    for solver, eta, ch, ax in wanted:
        rec = load_fit(
            fits,
            solver=solver,
            eta=eta,
            window="main",
            estimator="linear_exact",
            channel=ch,
            axis=ax,
        )
        bac[f"{solver}_eta{eta}_{ch}_{ax}"] = {
            **{k: cpack(rec[k]) for k in ("B", "A", "C")},
            "abs_C_over_A": rec["abs_C_over_A"],
            "relative_L2_residual": rec["relative_L2_residual"],
            "geometry": geometry_from_abc(rec["B"], rec["A"], rec["C"]),
            "rho_over_field": float((abs(rec["A"]) / abs(rec["B"])) / mom["M_over_alpha2"]),
        }

    qed_x = bac["QED_eta1.0_odd_x"]
    geom_pred = geometry_from_abc(from_fit(qed_x["B"]), from_fit(qed_x["A"]), from_fit(qed_x["C"]))

    deriv = derivative_checks(geo, R0, kcart, neel=1)
    perm = permutation_check(geo, R0, kcart, ehat, OMEGA0, 0.03)
    p_delta0 = M0Params(t_h=R0.t_h, lam_so=R0.lam_so, delta=0.0, u=R0.u, v=R0.v, label="delta0")
    z = OMEGA0 + 1j * 0.03
    d0 = odd_even_kernel(geo, p_delta0, kcart, ehat, np.array([1.0, 0.0]), z, z)
    p_soc0 = M0Params(t_h=R0.t_h, lam_so=0.0, delta=R0.delta, u=R0.u, v=R0.v, label="soc0")
    s0 = odd_even_kernel(geo, p_soc0, kcart, ehat, np.array([1.0, 0.0]), z, z)

    kernels = []
    for eta in (0.5, 1.0, 1.5):
        p = M0Params(delta=eta * R0.delta, label=f"eta{eta}")
        for omega, glabel in ((OMEGA0, "q6"), (OMEGA_1600, "w1600")):
            for gamma in (0.08, 0.03, 0.01):
                kernels.append(kernel_record(geo, p, kcart, ehat, omega, gamma, f"{glabel}_eta{eta}_g{gamma}"))
    for lam in (0.0, 0.05, 0.2):
        p = M0Params(lam_so=lam, label=f"lam{lam}")
        kernels.append(kernel_record(geo, p, kcart, ehat, OMEGA0, 0.03, f"q6_lam{lam}_g0.03"))

    def pick(label):
        hits = [k for k in kernels if k["label"] == label]
        if len(hits) != 1:
            raise RuntimeError(label)
        return hits[0]

    k_q6 = pick("q6_eta1.0_g0.03")
    k_1600 = pick("w1600_eta1.0_g0.03")
    pred_rho = mom["M_over_alpha2"]
    meas_rho = abs(from_fit(qed_x["A"])) / abs(from_fit(qed_x["B"]))

    proto = p1600.get("protocol", p1600)
    alpha1600 = float(proto["alpha_ds"])
    r1600 = float(proto["r_ds"])
    mom1600 = ds_moments(alpha1600, r1600)

    def wrap_deg(x):
        return float((x + 180.0) % 360.0 - 180.0)

    eta_scan = []
    for eta in (0.5, 1.0, 1.5):
        krec = pick(f"q6_eta{eta}_g0.03")
        kphase = krec["axes"]["x"]["odd"]["phase_deg"]
        row = {
            "eta": eta,
            "kernel_odd_x": krec["axes"]["x"]["odd"],
            "kernel_odd_y": krec["axes"]["y"]["odd"],
            "gap_plus": krec["gap_plus"],
        }
        for solver in (("GH5",) if eta != 1.0 else ("GH5", "QED")):
            rec = bac[f"{solver}_eta{eta}_odd_x"]
            B = from_fit(rec["B"])
            A = from_fit(rec["A"])
            row[solver] = {
                "rho": abs(A) / abs(B),
                "rho_over_field": (abs(A) / abs(B)) / pred_rho,
                "arg_A_over_B_deg": float(np.degrees(np.angle(A / B))),
                "arg_B_deg": float(np.degrees(np.angle(B))),
                "arg_A_deg": float(np.degrees(np.angle(A))),
                "B_minus_kernel_phase_deg": wrap_deg(float(np.degrees(np.angle(B))) - kphase),
                "A_minus_kernel_phase_deg": wrap_deg(float(np.degrees(np.angle(A))) - kphase),
                "abs_C_over_A": rec["abs_C_over_A"],
            }
        eta_scan.append(row)
    even_qed = bac["QED_eta1.0_even_x"]
    even_rho = abs(from_fit(even_qed["A"])) / abs(from_fit(even_qed["B"]))

    payload = {
        "scope": (
            "Eight-band M0 A/B/C microscopic kernel at the Q6 k/-k pair. "
            "Classical retarded Peierls kernel, not a quantum output S-matrix, not chi^(2)=A."
        ),
        "frozen_qed": {
            "pi_over_0_odd_2d": key["QED_main_odd_2D_pi_over_0"],
            "covflip_phase_x_deg": key["QED_covariance_flip_phase_x_deg"],
            "q6_holdout": key["Q6_corrected"],
        },
        "input_moments_q6": mom,
        "input_moments_1600_ds": mom1600,
        "bac": bac,
        "geometry_qed_eta1_odd_x": geom_pred,
        "derivative_checks": deriv,
        "symmetry_checks": {
            "permutation": perm,
            "delta0_odd_abs": float(abs(d0["odd"])),
            "soc0_odd_abs": float(abs(s0["odd"])),
            "soc0_even_abs": float(abs(s0["even"])),
        },
        "lowest_order_test": {
            "statement": "If one complex K multiplies (alpha^2 + M), then |A|/|B|=|M|/alpha^2 and arg(A/B)=180 deg.",
            "predicted_rho": pred_rho,
            "measured_rho_qed_eta1_odd_x": meas_rho,
            "measured_over_predicted": float(meas_rho / pred_rho),
            "predicted_two_arctan_deg": float(np.degrees(2.0 * math.atan(pred_rho))),
            "measured_covflip_deg": key["QED_covariance_flip_phase_x_deg"],
            "predicted_pi_over_0_from_meas_rho": geom_pred["pi_over_0_amp_ratio"],
            "even_rho_qed_eta1_x": even_rho,
            "eta_scan": eta_scan,
            "naive_A_is_not_chi2": True,
            "alpha_is_0.5_not_1.41": True,
            "qin_a2_141_means_a2_polarization": True,
        },
        "kernel_selected": {
            "q6_eta1_g0.03": k_q6,
            "w1600_eta1_g0.03": k_1600,
        },
        "kernels": kernels,
        "k_frac": list(K_FRAC),
        "k_cart": [float(x) for x in kcart],
        "ehat_a2": [float(x) for x in ehat],
        "elapsed_s": round(time.monotonic() - t0, 3),
    }
    (OUT / "abc_microscopic.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    write_svg_complex_plane(
        OUT / "qed_odd_x_abc_plane.svg",
        [
            {"name": "B", "z": from_fit(qed_x["B"])},
            {"name": "A", "z": from_fit(qed_x["A"])},
            {"name": "B+A (theta=0)", "z": from_fit(qed_x["B"]) + from_fit(qed_x["A"])},
            {"name": "B-A (theta=pi)", "z": from_fit(qed_x["B"]) - from_fit(qed_x["A"])},
            {"name": "B+iA", "z": from_fit(qed_x["B"]) + 1j * from_fit(qed_x["A"])},
            {"name": "B-iA", "z": from_fit(qed_x["B"]) - 1j * from_fit(qed_x["A"])},
        ],
        "QED eta=1 odd Jx: A/B/C geometry (main window)",
    )
    write_svg_bars(
        OUT / "rho_odd_even_vs_field.svg",
        [
            {
                "label": f"eta={eta}",
                "vals": [
                    ("field |M|/a2", mom["M_over_alpha2"]),
                    (
                        "odd |A|/|B|",
                        abs(from_fit(bac[f"GH5_eta{eta}_odd_x"]["A"]))
                        / abs(from_fit(bac[f"GH5_eta{eta}_odd_x"]["B"])),
                    ),
                    (
                        "even |A|/|B|",
                        abs(from_fit(bac[f"GH5_eta{eta}_even_x"]["A"]))
                        / abs(from_fit(bac[f"GH5_eta{eta}_even_x"]["B"])),
                    ),
                ],
            }
            for eta in (0.5, 1.0, 1.5)
        ],
        "Lowest-order field ratio versus measured |A|/|B|",
        "|A|/|B|",
    )
    print(
        json.dumps(
            {
                "out": str(OUT),
                "elapsed_s": payload["elapsed_s"],
                "deriv": deriv,
                "perm_odd": perm["odd_perm_abs"],
                "delta0_odd": payload["symmetry_checks"]["delta0_odd_abs"],
                "rho_meas": meas_rho,
                "rho_field": pred_rho,
                "argAB": geom_pred["arg_A_over_B_deg"],
                "two_arctan": geom_pred["two_arctan_rho_deg"],
                "k_q6_odd_x": k_q6["axes"]["x"]["odd"],
                "k_1600_odd_x": k_1600["axes"]["x"]["odd"],
                "gap_q6": k_q6["gap_plus"]["min_gap"],
                "eta_scan": eta_scan,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
