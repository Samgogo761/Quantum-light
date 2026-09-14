#!/usr/bin/env python3
"""M0 stage A: static acceptance and full-BZ gap for the eight-band PT-AFM model.

Writes results/q_line/m0_v1_20260912/static/*.json and figures. No propagation here.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from q8_m0_model import (
    MODEL_ID,
    N_OCC,
    M0Params,
    R0,
    build_geometry,
    dh8,
    gamma_energies,
    h8,
    mirror_matrix_inplane,
    op_P,
    op_V_specified,
    op_c2_nhat,
    op_mirror_a2,
    op_sewing,
    op_sz,
    op_theta_unitary,
    t0_and_grad,
    write_spec,
)

REPO = Path(__file__).resolve().parents[2]
OUT_ROOT = REPO / "results/q_line/m0_v1_20260912"


def frob(a) -> float:
    return float(np.linalg.norm(np.asarray(a)))


def sample_kfrac() -> list:
    return [
        ("Gamma", [0.0, 0.0]),
        ("M1", [0.5, 0.0]),
        ("M2", [0.0, 0.5]),
        ("M3", [0.5, 0.5]),
        ("K", [1.0 / 3.0, 1.0 / 3.0]),
        ("Kp", [-1.0 / 3.0, -1.0 / 3.0]),
        ("g1", [0.17, -0.11]),
        ("g2", [0.31, 0.22]),
        ("g3", [-0.08, 0.41]),
        ("g4", [0.05, 0.37]),
        ("g5", [0.44, -0.29]),
        ("g6", [0.61, 0.13]),  # outside first BZ on purpose (sewing)
        ("g7", [-0.52, 0.77]),
    ]


def symmetry_report(geo, p: M0Params) -> dict:
    P = op_P()
    UT = op_theta_unitary()
    PT = P @ UT
    SZ = op_sz()
    V = op_V_specified()
    UM = op_mirror_a2(geo)
    UC = op_c2_nhat(geo)
    Mm = mirror_matrix_inplane(geo)
    escale = 3.0 * p.t_h + p.delta + abs(p.lam_so) * 3.0 * math.sqrt(3.0) + abs(p.u) + abs(p.v)
    gs = [geo.k_cart(gf) for gf in ([1, 0], [0, 1], [1, 1], [1, -1])]
    rows = []
    worst = {k: 0.0 for k in ("herm", "P", "Theta", "PTheta", "sz", "sewing", "kramers", "mirror", "C2", "fd_dx", "fd_dy", "V_minus_2tauyY")}
    v_worst = 0.0
    for name, kf in sample_kfrac():
        k = geo.k_cart(kf)
        hp = h8(geo, p, k, +1)
        hm = h8(geo, p, k, -1)
        hp_mk = h8(geo, p, -k, +1)
        hm_mk = h8(geo, p, -k, -1)
        r = {"k": name, "k_frac": kf}
        r["herm"] = frob(hp - hp.conj().T)
        r["P"] = frob(P @ hp @ P.conj().T - hm_mk)
        r["Theta"] = frob(UT @ hp.conj() @ UT.conj().T - hm_mk)
        r["PTheta"] = frob(PT @ hp.conj() @ PT.conj().T - hp)
        r["sz"] = frob(hp @ SZ - SZ @ hp)
        r["sewing"] = max(frob(h8(geo, p, k + g, +1) - op_sewing(geo, g).conj().T @ hp @ op_sewing(geo, g)) for g in gs)
        e = np.linalg.eigvalsh(hp)
        r["kramers"] = float(np.max(np.abs(e[0::2] - e[1::2])))
        r["mirror"] = frob(UM @ hp @ UM.conj().T - h8(geo, p, Mm @ k, -1))
        r["C2"] = frob(UC @ hp @ UC.conj().T - h8(geo, p, -(Mm @ k), +1))
        # V = tau_x s_z and 2 tau_y Y identity
        t0, _ = t0_and_grad(geo, p, k)
        y = (t0 - t0.conj().T) / (2.0j)
        y4 = np.kron(y, np.eye(2))
        pred = np.zeros((8, 8), dtype=complex)
        pred[:4, 4:] = -2.0j * y4
        pred[4:, :4] = 2.0j * y4
        got = V @ hp @ V.conj().T - hm
        r["V_residual"] = frob(got)
        r["V_minus_2tauyY"] = frob(got - pred)
        r["T0_antihermitian_part"] = frob(t0 - t0.conj().T)
        r["T0_normality"] = frob(t0 @ t0.conj().T - t0.conj().T @ t0)
        # finite differences
        eps = 1.0e-5
        D = dh8(geo, p, k, +1)
        for a, key in ((0, "fd_dx"), (1, "fd_dy")):
            dk = np.zeros(2)
            dk[a] = eps
            fd = (h8(geo, p, k + dk, +1) - h8(geo, p, k - dk, +1)) / (2.0 * eps)
            r[key] = frob(fd - D[a]) / max(frob(D[a]), 1e-300)
        r["E_N_vs_E_minusN"] = float(np.max(np.abs(e - np.linalg.eigvalsh(hm))))
        r["E_k_vs_E_minusk"] = float(np.max(np.abs(e - np.linalg.eigvalsh(hp_mk))))
        rows.append(r)
        for key in worst:
            worst[key] = max(worst[key], r[key])
        v_worst = max(v_worst, r["V_residual"])
    # Gamma analytic
    eg = np.linalg.eigvalsh(h8(geo, p, geo.k_cart([0, 0]), +1))
    ana = np.repeat(np.array(gamma_energies(p)), 2)
    # K analytic: f = 0, g = -3sqrt3
    from q8_m0_model import f_and_grad, g_and_grad

    fK, _ = f_and_grad(geo, geo.k_cart([1 / 3, 1 / 3]))
    gK, _ = g_and_grad(geo, geo.k_cart([1 / 3, 1 / 3]))
    gKp, _ = g_and_grad(geo, geo.k_cart([-1 / 3, -1 / 3]))
    return {
        "energy_scale_for_normalisation": escale,
        "sample_points": rows,
        "worst_absolute": worst,
        "worst_normalised": {k: v / escale for k, v in worst.items()},
        "V_specified_max_residual": v_worst,
        "PTheta_squared_plus_identity": frob(PT @ PT.conj() + np.eye(8)),
        "gamma_numeric": eg.tolist(),
        "gamma_analytic": ana.tolist(),
        "gamma_max_abs_diff": float(np.max(np.abs(eg - ana))),
        "K_check": {"abs_f": float(abs(fK)), "g": float(gK), "minus_3sqrt3": -3.0 * math.sqrt(3.0), "g_Kprime": float(gKp)},
        "mirror_C2_note": (
            "mirror(a2) and C2(n̂) are symmetries of the ideal 120° honeycomb bilayer with shift along a2. "
            "The production metric deviates from 120° by ~0.005°, so their residual is O(1e-6 t_h), not zero. "
            "P/Θ/PΘ/sewing are exact by construction (pairing)."
        ),
    }


def intertwiner_report(geo, p: M0Params) -> dict:
    rows = []
    for _name, kf in sample_kfrac():
        k = geo.k_cart(kf)
        hp = h8(geo, p, k, +1)
        hm = h8(geo, p, k, -1)
        rows.append(np.kron(np.eye(8), hm) - np.kron(hp.T, np.eye(8)))
    big = np.vstack(rows)
    s = np.linalg.svd(big, compute_uv=False)
    tiny = [float(x) for x in s if x < 1.0e-8]
    return {
        "unknowns": 64,
        "equations": int(big.shape[0]),
        "smallest_singular_values": [float(x) for x in s[-8:]],
        "n_below_1e-8": len(tiny),
        "conclusion": (
            "no constant same-k intertwiner U with U H_+N(k) = H_-N(k) U on the sample; "
            "the 64-unknown SVD already spans all constant linear combinations, so the Pauli list is not enlarged"
            if not tiny
            else "nullspace found; unitarity must be checked before any claim"
        ),
        "isospectral_note": (
            "H_+N(k) and H_-N(k) are exactly isospectral at every k (spec even in Δ) because T0(k) is a normal matrix "
            "and both layers share h; the relating map is k-dependent and involves transposition, hence it is neither a "
            "constant unitary nor a same-drive dynamical symmetry."
        ),
    }


def _refine_extremum(func, geo, k0_frac, half0, levels=5, n=21, sign=+1):
    """Nested-grid refinement of func(kfrac_array)->values; sign=+1 minimise, -1 maximise."""
    k0 = np.array(k0_frac, dtype=float)
    half = half0
    history = []
    for _ in range(levels):
        g1 = np.linspace(-half, half, n)
        kk = np.stack(np.meshgrid(k0[0] + g1, k0[1] + g1, indexing="ij"), axis=-1).reshape(-1, 2)
        vals = func(geo, kk)
        idx = int(np.argmin(sign * vals))
        k0 = kk[idx]
        history.append({"value": float(vals[idx]), "k_frac": k0.tolist(), "half_width_frac": half})
        half = half / 5.0
    return history


def _bands(geo, p, kfrac_arr, neel=+1):
    return np.linalg.eigvalsh(h8(geo, p, geo.k_cart(kfrac_arr), neel))


def gap_report(geo, p: M0Params, n_coarse: int = 240) -> dict:
    g1 = np.arange(n_coarse) / n_coarse
    kk = np.stack(np.meshgrid(g1, g1, indexing="ij"), axis=-1).reshape(-1, 2)
    e = _bands(geo, p, kk)
    e4 = e[:, N_OCC - 1]
    e5 = e[:, N_OCC]
    direct = e5 - e4
    # candidates: distinct local minima of the direct gap (top 6 lowest, separated)
    order = np.argsort(direct)
    cands = []
    for idx in order:
        kf = kk[idx]
        if all(np.linalg.norm(((kf - c) + 0.5) % 1.0 - 0.5) > 0.06 for c in cands):
            cands.append(kf)
        if len(cands) >= 6:
            break

    def f_direct(geo_, kf):
        b = _bands(geo_, p, kf)
        return b[:, N_OCC] - b[:, N_OCC - 1]

    def f_e4(geo_, kf):
        return _bands(geo_, p, kf)[:, N_OCC - 1]

    def f_e5(geo_, kf):
        return _bands(geo_, p, kf)[:, N_OCC]

    half0 = 1.5 / n_coarse
    direct_ref = [_refine_extremum(f_direct, geo, c, half0) for c in cands]
    best_direct = min(direct_ref, key=lambda h: h[-1]["value"])
    # VBM / CBM
    i4 = int(np.argmax(e4))
    i5 = int(np.argmin(e5))
    vbm_ref = _refine_extremum(f_e4, geo, kk[i4], half0, sign=-1)
    cbm_ref = _refine_extremum(f_e5, geo, kk[i5], half0, sign=+1)
    # also refine VBM/CBM from a few more candidates
    vbm_cands = []
    for idx in np.argsort(-e4):
        kf = kk[idx]
        if all(np.linalg.norm(((kf - c) + 0.5) % 1.0 - 0.5) > 0.06 for c in vbm_cands):
            vbm_cands.append(kf)
        if len(vbm_cands) >= 4:
            break
    cbm_cands = []
    for idx in np.argsort(e5):
        kf = kk[idx]
        if all(np.linalg.norm(((kf - c) + 0.5) % 1.0 - 0.5) > 0.06 for c in cbm_cands):
            cbm_cands.append(kf)
        if len(cbm_cands) >= 4:
            break
    vbm_all = [_refine_extremum(f_e4, geo, c, half0, sign=-1) for c in vbm_cands]
    cbm_all = [_refine_extremum(f_e5, geo, c, half0, sign=+1) for c in cbm_cands]
    vbm_best = max(vbm_all, key=lambda h: h[-1]["value"])
    cbm_best = min(cbm_all, key=lambda h: h[-1]["value"])
    e_vbm = vbm_best[-1]["value"]
    e_cbm = cbm_best[-1]["value"]
    eg_ind = e_cbm - e_vbm
    eg_dir = best_direct[-1]["value"]
    unc_dir = abs(best_direct[-1]["value"] - best_direct[-2]["value"])
    unc_ind = abs(vbm_best[-1]["value"] - vbm_best[-2]["value"]) + abs(cbm_best[-1]["value"] - cbm_best[-2]["value"])
    mu = 0.5 * (e_vbm + e_cbm)
    band_ranges = [[float(e[:, n].min()), float(e[:, n].max())] for n in range(8)]
    # Neel independence of the gap (spec even in Delta)
    e_m = _bands(geo, p, kk[::37], -1)
    neel_diff = float(np.max(np.abs(e[::37] - e_m)))
    del vbm_ref, cbm_ref
    return {
        "coarse_grid": f"{n_coarse}x{n_coarse} fractional, Γ-centred",
        "E_g_direct": eg_dir,
        "E_g_direct_uncertainty": unc_dir,
        "E_g_direct_k_frac": best_direct[-1]["k_frac"],
        "direct_gap_candidates_refined": [h[-1] for h in direct_ref],
        "E_g_indirect": eg_ind,
        "E_g_indirect_uncertainty": unc_ind,
        "VBM": {"E": e_vbm, "k_frac": vbm_best[-1]["k_frac"], "candidates": [h[-1] for h in vbm_all]},
        "CBM": {"E": e_cbm, "k_frac": cbm_best[-1]["k_frac"], "candidates": [h[-1] for h in cbm_all]},
        "mu_half_filling": mu,
        "band_ranges_coarse": band_ranges,
        "coarse_min_direct_gap": float(direct.min()),
        "spectrum_neel_independent_max_diff": neel_diff,
        "insulating_initial_state_supported": bool(eg_ind > 20.0 * max(unc_ind, 1e-12) and eg_ind > 0.0),
        "refinement_history_direct": best_direct,
        "refinement_history_vbm": vbm_best,
        "refinement_history_cbm": cbm_best,
        "note": "finite nested grids; not a proof over the continuous BZ. Uncertainty = change over the last refinement level.",
    }


def band_path(geo, p: M0Params, n_seg: int = 120) -> dict:
    pts = [("Gamma", [0, 0]), ("M", [0.5, 0]), ("K", [1 / 3, 1 / 3]), ("Gamma", [0, 0]), ("Kp", [-1 / 3, -1 / 3]), ("M2", [0, -0.5]), ("Gamma", [0, 0])]
    ks, labels, dist = [], [], [0.0]
    acc = 0.0
    for (la, a), (lb, b) in zip(pts[:-1], pts[1:]):
        a = np.array(a, float)
        b = np.array(b, float)
        seg = [a + (b - a) * t for t in np.linspace(0, 1, n_seg, endpoint=False)]
        ks.extend(seg)
        labels.append((la, acc))
        acc += float(np.linalg.norm(geo.k_cart(b) - geo.k_cart(a)))
    ks.append(np.array(pts[-1][1], float))
    labels.append((pts[-1][0], acc))
    ks = np.array(ks)
    kc = geo.k_cart(ks)
    x = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(kc, axis=0), axis=1))])
    e = np.linalg.eigvalsh(h8(geo, p, kc, +1))
    return {"x": x.tolist(), "E": e.tolist(), "labels": labels, "k_frac": ks.tolist()}


def make_figures(geo, p, gap, bands, outdir: Path) -> list:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    files = []
    fig, ax = plt.subplots(figsize=(7, 4.2))
    x = np.array(bands["x"])
    e = np.array(bands["E"])
    for n in range(8):
        ax.plot(x, e[:, n], color="C0" if n < 4 else "C3", lw=1.2)
    for lab, pos in bands["labels"]:
        ax.axvline(pos, color="k", lw=0.5, alpha=0.4)
    ax.set_xticks([pos for _, pos in bands["labels"]])
    ax.set_xticklabels([lab for lab, _ in bands["labels"]])
    ax.axhline(gap["mu_half_filling"], color="gray", ls="--", lw=0.8, label=r"$\mu$")
    ax.set_ylabel(r"$E/t_h$")
    ax.set_title(f"M0 {p.label}: E_g^dir={gap['E_g_direct']:.4f}, E_g^ind={gap['E_g_indirect']:.4f} (t_h)")
    ax.legend(loc="upper right")
    f1 = outdir / "m0_bands_path.png"
    fig.tight_layout()
    fig.savefig(f1, dpi=140)
    plt.close(fig)
    files.append(str(f1))
    # direct gap map
    n = 160
    g1 = np.arange(n) / n - 0.5
    kk = np.stack(np.meshgrid(g1, g1, indexing="ij"), axis=-1).reshape(-1, 2)
    eb = _bands(geo, p, kk)
    dg = (eb[:, N_OCC] - eb[:, N_OCC - 1]).reshape(n, n)
    fig, ax = plt.subplots(figsize=(5.2, 4.4))
    im = ax.imshow(dg.T, origin="lower", extent=[-0.5, 0.5, -0.5, 0.5], cmap="viridis")
    ax.set_xlabel("k1 (frac)")
    ax.set_ylabel("k2 (frac)")
    ax.set_title(r"$E_5-E_4$ (t_h), fractional BZ")
    fig.colorbar(im, ax=ax)
    kf = np.array(gap["E_g_direct_k_frac"])
    kf = (kf + 0.5) % 1.0 - 0.5
    ax.plot([kf[0]], [kf[1]], "r+", ms=12)
    f2 = outdir / "m0_direct_gap_map.png"
    fig.tight_layout()
    fig.savefig(f2, dpi=140)
    plt.close(fig)
    files.append(str(f2))
    return files


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--coarse", type=int, default=240)
    args = ap.parse_args(argv)
    outdir = OUT_ROOT / "static"
    outdir.mkdir(parents=True, exist_ok=True)
    geo = build_geometry()
    p = R0
    write_spec(outdir / "m0_spec.json", geo, p)
    sym = symmetry_report(geo, p)
    inter = intertwiner_report(geo, p)
    gap = gap_report(geo, p, args.coarse)
    bands = band_path(geo, p)
    figs = make_figures(geo, p, gap, bands, outdir)
    (outdir / "symmetry_report.json").write_text(json.dumps(sym, indent=2), encoding="utf-8")
    (outdir / "intertwiner_report.json").write_text(json.dumps(inter, indent=2), encoding="utf-8")
    (outdir / "gap_report.json").write_text(json.dumps(gap, indent=2), encoding="utf-8")
    (outdir / "band_path.json").write_text(json.dumps(bands), encoding="utf-8")
    index = {
        "model_id": MODEL_ID,
        "stage": "A_static",
        "params": p.as_dict(),
        "figures": figs,
        "insulating_initial_state_supported": gap["insulating_initial_state_supported"],
        "E_g_direct": gap["E_g_direct"],
        "E_g_indirect": gap["E_g_indirect"],
        "mu": gap["mu_half_filling"],
        "worst_normalised_residuals": sym["worst_normalised"],
        "constant_intertwiner_nullspace": inter["n_below_1e-8"],
        "quantum_input": "not_run",
        "quantum_output": "not_run",
    }
    (outdir / "static_index.json").write_text(json.dumps(index, indent=2), encoding="utf-8")
    print(json.dumps(index, indent=2))
    print("worst abs", json.dumps(sym["worst_absolute"], indent=1))
    print("gamma diff", sym["gamma_max_abs_diff"], "K", sym["K_check"])
    print("intertwiner sv", inter["smallest_singular_values"][-4:])
    print("VBM", gap["VBM"]["E"], gap["VBM"]["k_frac"], "CBM", gap["CBM"]["E"], gap["CBM"]["k_frac"])
    print("direct candidates", [(round(c["value"], 5), [round(x, 4) for x in c["k_frac"]]) for c in gap["direct_gap_candidates_refined"]])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
