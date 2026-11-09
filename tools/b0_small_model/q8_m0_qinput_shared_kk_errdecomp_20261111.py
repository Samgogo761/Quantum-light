#!/usr/bin/env python3
"""Archive-side QED–MTEF error channels on the frozen shared-mode object.

Time-aligned ε_C, ε_DS, ε_common, ε_Δ from 20261102 + 20261110.
Checkpoint inventory and endpoint reduced purities; no new propagation.
Does not rewrite 20261026-20261110 archives. Not HHG, not a mechanism claim.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import time
from pathlib import Path

import numpy as np

import q8_m0_qinput_shared_kk_ds_dt_20261104 as dth
import q8_m0_qinput_shared_kk_dyn_preflight_20261030 as pf
import q8_m0_qinput_shared_kk_mtef_compare_20261109 as cmp
import q8_m0_qinput_shared_kk_mtef_gh5_20261110 as gh5
import q8_m0_qinput_shared_kk_mtef_map_20261108 as mp
import q8_m0_qinput_shared_kk_static_20261029 as sh
from q8_m0_model import R0, build_geometry

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "results/q_line/qinput_a_shared_kk_errdecomp_20261111"
MODEL_ID = "M0_shared_mode_k_minus_k_errdecomp_v1"
RECORD_DATE = "2026-11-11"
ARCHIVE_QED = ROOT / "results/q_line/qinput_a_shared_kk_ds_odd_20261102/compare.json"
ARCHIVE_GH3 = ROOT / "results/q_line/qinput_a_shared_kk_mtef_compare_20261109/compare.json"
ARCHIVE_GH5 = ROOT / "results/q_line/qinput_a_shared_kk_mtef_gh5_20261110/gh5.json"
DIR_DS8 = ROOT / "results/q_line/qinput_a_shared_kk_ds_dt_20261104"
DIR_COH8 = ROOT / "results/q_line/qinput_a_shared_kk_coh_dt_20261105"
DIR_DS10 = ROOT / "results/q_line/qinput_a_shared_kk_ds_nmax10_20261106"
OLD_ARCHIVES = gh5.OLD_ARCHIVES + (ARCHIVE_GH5,)
EL = sh.EL36
CKPT_SPEC = (
    ("qed_nmax8_dt2_ds_p", DIR_DS8, "ds_p", 8, 1, 2570, "DS +N nmax=8 dt/2; not 20261102 ψ"),
    ("qed_nmax8_dt2_ds_m", DIR_DS8, "ds_m", 8, -1, 2570, "DS -N nmax=8 dt/2"),
    ("qed_nmax8_dt2_coh_p", DIR_COH8, "coh_p", 8, 1, 2570, "Cα +N nmax=8 dt/2"),
    ("qed_nmax8_dt2_coh_m", DIR_COH8, "coh_m", 8, -1, 2570, "Cα -N nmax=8 dt/2"),
    ("qed_nmax10_dt2_ds_p", DIR_DS10, "ds_p", 10, 1, 2570, "DS +N nmax=10; not the error-series object"),
    ("qed_nmax10_dt2_ds_m", DIR_DS10, "ds_m", 10, -1, 2570, "DS -N nmax=10; not the error-series object"),
)


def _progress(progress, msg):
    mp._progress(progress, msg)


def corr_real(a, b):
    ar, br = np.real(np.asarray(a)), np.real(np.asarray(b))
    if np.std(ar) < 1e-30 or np.std(br) < 1e-30:
        return 0.0
    return float(np.corrcoef(ar, br)[0, 1])


def fit_scale(q, m):
    """Real least-squares scale through zero; do not infer it from peak ratios."""
    q, m = np.real(q), np.real(m)
    den = float(q @ q)
    scale = float(q @ m) / den if den > 0 else 0.0
    return {"scale_through_zero": scale,
            "pearson": corr_real(q, m),
            "relative_rms_after_scale_fit": float(np.linalg.norm(m-scale*q) / max(np.linalg.norm(q), 1e-30))}


def xy_stats(a, name):
    a = np.asarray(a, complex)
    return {
        "qty": name,
        "max_abs_x": float(np.max(np.abs(a[:, 0]))),
        "max_abs_y": float(np.max(np.abs(a[:, 1]))),
        "rms_x": float(np.sqrt(np.mean(np.abs(a[:, 0]) ** 2))),
        "rms_y": float(np.sqrt(np.mean(np.abs(a[:, 1]) ** 2))),
    }


def channels(mtef_c, mtef_d, qed_c, qed_d):
    e_c = np.asarray(mtef_c, complex) - np.asarray(qed_c, complex)
    e_ds = np.asarray(mtef_d, complex) - np.asarray(qed_d, complex)
    e_com = 0.5 * (e_c + e_ds)
    e_del = e_ds - e_c
    return {"e_c": e_c, "e_ds": e_ds, "e_common": e_com, "e_delta": e_del}


def channel_identities_ok(ch, atol=1e-12):
    rec_c = ch["e_common"] - 0.5 * ch["e_delta"]
    rec_ds = ch["e_common"] + 0.5 * ch["e_delta"]
    return bool(np.allclose(rec_c, ch["e_c"], atol=atol, rtol=0)
                and np.allclose(rec_ds, ch["e_ds"], atol=atol, rtol=0))


def summarize(ch, qed_c, qed_d, d_qed, d_mtef):
    e_c, e_ds, e_com, e_del = ch["e_c"], ch["e_ds"], ch["e_common"], ch["e_delta"]
    return {
        "n_times": int(e_c.shape[0]),
        "identities_ok": channel_identities_ok(ch),
        "e_c": xy_stats(e_c, "eps_C"),
        "e_ds": xy_stats(e_ds, "eps_DS"),
        "e_common": xy_stats(e_com, "eps_common"),
        "e_delta": xy_stats(e_del, "eps_Delta"),
        "qed_J_odd_coh": xy_stats(qed_c, "QED_J_odd_Calpha"),
        "qed_J_odd_ds": xy_stats(qed_d, "QED_J_odd_DS"),
        "qed_dJ_odd": xy_stats(d_qed, "QED_Delta_J_odd"),
        "mtef_dJ_odd": xy_stats(d_mtef, "MTEF_Delta_J_odd"),
        "corr": {
            "eC_eDS_x": corr_real(e_c[:, 0], e_ds[:, 0]),
            "eC_eDS_y": corr_real(e_c[:, 1], e_ds[:, 1]),
            "eC_qedC_x": corr_real(e_c[:, 0], qed_c[:, 0]),
            "eC_qedC_y": corr_real(e_c[:, 1], qed_c[:, 1]),
            "eDS_qedDS_x": corr_real(e_ds[:, 0], qed_d[:, 0]),
            "eDS_qedDS_y": corr_real(e_ds[:, 1], qed_d[:, 1]),
            "eCom_eC_x": corr_real(e_com[:, 0], e_c[:, 0]),
            "eDelta_dJqed_x": corr_real(e_del[:, 0], d_qed[:, 0]),
            "dJmtef_dJqed_x": corr_real(d_mtef[:, 0], d_qed[:, 0]),
            "dJmtef_dJqed_y": corr_real(d_mtef[:, 1], d_qed[:, 1]),
        },
        "scale": {
            "note": "Ratios of sampled norms are descriptive, not causal percentages. Peak ratios alone do not establish proportional waveforms.",
            "rms_eC_over_rms_qedC_x": float(
                xy_stats(e_c, "_")["rms_x"] / max(xy_stats(qed_c, "_")["rms_x"], 1e-30)),
            "rms_eDS_over_rms_qedDS_x": float(
                xy_stats(e_ds, "_")["rms_x"] / max(xy_stats(qed_d, "_")["rms_x"], 1e-30)),
            "max_eCommon_over_max_eC_x": float(
                xy_stats(e_com, "_")["max_abs_x"] / max(xy_stats(e_c, "_")["max_abs_x"], 1e-30)),
            "max_eDelta_over_max_eC_x": float(
                xy_stats(e_del, "_")["max_abs_x"] / max(xy_stats(e_c, "_")["max_abs_x"], 1e-30)),
            "max_eDelta_over_max_dJqed_x": float(
                xy_stats(e_del, "_")["max_abs_x"] / max(xy_stats(d_qed, "_")["max_abs_x"], 1e-30)),
        },
        "waveform_fit": {state: {axis: fit_scale(q[:, a], m[:, a])
                                     for a, axis in enumerate(("x", "y"))}
                         for state, q, m in (("C", qed_c, qed_c+e_c),
                                              ("DS", qed_d, qed_d+e_ds),
                                              ("Delta", d_qed, d_mtef))},
        "relative_delta_error": {
            axis: {"max_error_over_max_qed": float(np.max(np.abs(e_del[:, a])) / max(np.max(np.abs(d_qed[:, a])), 1e-30)),
                   "rms_error_over_rms_qed": float(np.linalg.norm(e_del[:, a]) / max(np.linalg.norm(d_qed[:, a]), 1e-30))}
            for a, axis in enumerate(("x", "y"))},
    }


def pack_ch(t, ch):
    return {
        "t": np.asarray(t, float).tolist(),
        "eps_C": cmp.pack_j(ch["e_c"]),
        "eps_DS": cmp.pack_j(ch["e_ds"]),
        "eps_common": cmp.pack_j(ch["e_common"]),
        "eps_Delta": cmp.pack_j(ch["e_delta"]),
    }


def load_gh5_series():
    with ARCHIVE_GH5.open(encoding="utf-8") as fh:
        o = json.load(fh)
    ser = o["gh5_meta"]["series"]
    odd_c = cmp.unpack_rows(ser["J_odd_coh"])
    odd_d = cmp.unpack_rows(ser["J_odd_ds"])
    d_odd = cmp.unpack_rows(ser["dJ_odd"])
    if not np.allclose(d_odd, odd_d - odd_c, atol=1e-18, rtol=1e-10):
        raise RuntimeError("GH5 dJ_odd is not J_odd_ds - J_odd_coh")
    return {
        "t": np.asarray(ser["t"], float),
        "J_odd_coh": odd_c,
        "J_odd_ds": odd_d,
        "dJ_odd": d_odd,
        "hash": mp.sha256_file(ARCHIVE_GH5),
        "cancellation_gh5": o.get("cancellation_gh5"),
        "verdict_gate": o.get("verdict_gate"),
    }


def load_gh3_series():
    g3 = gh5.load_gh3()
    return {
        "t": g3["t"],
        "J_odd_coh": g3["odd_c"],
        "J_odd_ds": g3["odd_d"],
        "dJ_odd": g3["d_odd"],
        "hash": g3["hash"],
    }


def purity_from_samples(path: Path):
    if not path.is_file():
        return None
    rows = json.loads(path.read_text(encoding="utf-8"))
    t = np.array([float(r["t"]) for r in rows], float)
    p = np.array([float(r["purity"]) for r in rows], float)
    nph = np.array([float(r["n_ph"]) for r in rows], float)
    return {
        "n": int(len(t)),
        "t": t.tolist(),
        "purity": p.tolist(),
        "n_ph": nph.tolist(),
        "purity_min": float(np.min(p)),
        "purity_max": float(np.max(p)),
        "purity_end": float(p[-1]),
        "one_minus_purity_max": float(np.max(1.0 - p)),
        "one_minus_purity_end": float(1.0 - p[-1]),
    }


def reduced_purities(psi, el=EL):
    psi = np.asarray(psi, complex).reshape(-1)
    d1 = psi.size // (el * el)
    if d1 * el * el != psi.size:
        raise RuntimeError(f"psi size {psi.size} is not {el}x{el}x d1")
    V = psi.reshape(el, el, d1)
    n2 = float(np.vdot(psi, psi).real)
    if not np.isfinite(n2) or n2 <= 0:
        raise ValueError("non-finite or zero checkpoint state")
    rho_ph = np.tensordot(V, np.conj(V), axes=([0, 1], [0, 1])) / n2
    rho_k = np.tensordot(V, np.conj(V), axes=([1, 2], [1, 2])) / n2
    rho_mk = np.tensordot(V, np.conj(V), axes=([0, 2], [0, 2])) / n2
    # Schmidt singular values give Tr(rho_el^2) without a 1296x1296 dense product.
    schmidt = np.linalg.svd(V.reshape(el*el, d1) / np.sqrt(n2), compute_uv=False)
    purity_el = float(np.sum(schmidt**4))

    def pur(rho):
        return float(np.trace(rho @ rho).real)

    return {
        "d1": int(d1),
        "norm2": n2,
        "purity_photon": pur(rho_ph),
        "purity_k": pur(rho_k),
        "purity_mk": pur(rho_mk),
        "purity_el_joint": purity_el,
        "photon_trace_error": float(abs(np.trace(rho_ph)-1)),
        "photon_min_eigenvalue": float(np.linalg.eigvalsh(rho_ph).min()),
        "schmidt_purity_difference": float(abs(purity_el-pur(rho_ph))),
        "one_minus_purity_photon": 1.0 - pur(rho_ph),
        "one_minus_purity_k": 1.0 - pur(rho_k),
        "one_minus_purity_el_joint": 1.0 - purity_el,
    }


def factorized_current(model, psi, s=1.0):
    """Current in rho_e tensor rho_ph with the same exact endpoint marginals.

    This is an instantaneous connected Peierls-current diagnostic, not MTEF:
    a MTEF ensemble can retain statistical electron-field correlations.
    """
    V = np.asarray(psi, complex).reshape(EL, EL, model.d1)
    norm2 = float(np.vdot(psi, psi).real)
    rk = np.tensordot(V, V.conj(), axes=([1, 2], [1, 2])) / norm2
    rm = np.tensordot(V, V.conj(), axes=([0, 2], [0, 2])) / norm2
    rp = np.tensordot(V, V.conj(), axes=([0, 1], [0, 1])) / norm2
    Ds = sh.dyn.peierls_at(model.q_units, s, model.n_max)
    out = np.zeros(2, complex)
    for Jk, Jm, D in zip(model.j_k, model.j_mk, Ds):
        # apply_current acts on the last axis as V @ D; D is symmetric here.
        if not np.allclose(D, D.T, atol=1e-13, rtol=0):
            raise ValueError("Peierls photon matrix symmetry convention changed")
        for a in range(2):
            out[a] += (np.trace(rk @ Jk[a]) + np.trace(rm @ Jm[a])) * np.trace(rp @ D.T)
    return out


def connected_from_psi(model, psi, s=1.0):
    psi = np.asarray(psi, complex).reshape(-1)
    n2 = float(np.vdot(psi, psi).real) + 1e-30
    jv = sh.apply_current(model, psi, s)
    mean = np.array([complex(np.vdot(psi, jv[a]) / n2) for a in range(2)])
    raw = np.zeros((2, 2), complex)
    for a in range(2):
        for b in range(2):
            raw[a, b] = complex(np.vdot(jv[a], jv[b]) / n2)
    conn = raw - np.outer(np.conj(mean), mean)
    v = psi.reshape(EL, EL, model.d1)
    n_psi = (v * np.arange(model.d1, dtype=float)[None, None, :]).reshape(-1)
    n_ph = float(np.vdot(psi, n_psi).real / n2)
    nj = np.array([complex(np.vdot(n_psi, jv[a]) / n2) for a in range(2)])
    fac = factorized_current(model, psi, s)
    return {
        "s": float(s),
        "mean_J": [pf.cpack(z) for z in mean],
        "connected_JJ": [[pf.cpack(z) for z in row] for row in conn],
        "connected_JJ_xx_re": float(conn[0, 0].real),
        "connected_JJ_yy_re": float(conn[1, 1].real),
        "n_ph": n_ph,
        "connected_nJ": [pf.cpack(z) for z in (nj - n_ph * mean)],
        "current_product_marginals": [pf.cpack(z) for z in fac],
        "current_minus_product_marginals": [pf.cpack(z) for z in mean-fac],
        "interpretation": "JJ is a total-current covariance; J itself contains photon Peierls factors. nJ is ordered and need not be real. Neither is by itself an electron-photon entanglement witness or a causal explanation of MTEF error.",
    }


def inventory_one(key, folder, tag, n_max, neel, n_steps_expect, note):
    npz, js = dth.ckpt_paths(folder, tag)
    item = {
        "key": key, "tag": tag, "n_max": int(n_max), "neel": int(neel),
        "note": note,
        "npz": str(npz.as_posix()) if npz.is_file() else None,
        "samples": str(js.as_posix()) if js.is_file() else None,
        "npz_exists": bool(npz.is_file()),
        "samples_exist": bool(js.is_file()),
        "recoverable_joint_psi": False,
        "complete": False,
    }
    if npz.is_file():
        z = np.load(npz)
        nxt = int(np.asarray(z["next_j"]))
        shp = tuple(int(x) for x in np.asarray(z["psi"]).shape)
        item["next_j"] = nxt
        item["psi_shape"] = shp
        item["n_steps_expect"] = int(n_steps_expect)
        item["complete"] = bool(nxt == n_steps_expect)
        item["recoverable_joint_psi"] = True
        item["hash"] = mp.sha256_file(npz)
    if js.is_file():
        item["photon_purity_series"] = purity_from_samples(js)
        item["samples_hash"] = mp.sha256_file(js)
    return item


def product_state_purity_gate(n_max=2):
    rng = np.random.default_rng(7)
    ek = rng.normal(size=EL) + 1j * rng.normal(size=EL)
    emk = rng.normal(size=EL) + 1j * rng.normal(size=EL)
    ph = rng.normal(size=n_max + 1) + 1j * rng.normal(size=n_max + 1)
    ek, emk, ph = ek / np.linalg.norm(ek), emk / np.linalg.norm(emk), ph / np.linalg.norm(ph)
    psi = np.kron(np.kron(ek, emk), ph)
    p = reduced_purities(psi, el=EL)
    ok = (abs(p["purity_photon"] - 1.0) < 1e-12
          and abs(p["purity_k"] - 1.0) < 1e-12
          and abs(p["purity_mk"] - 1.0) < 1e-12
          and abs(p["purity_el_joint"] - 1.0) < 1e-12)
    return ok, p


def write_plots(out_dir: Path, t, ch5, ch3):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(9.4, 6.6), sharex=True)
    specs = (
        (axes[0, 0], "e_c", r"$\varepsilon_C=J_{\mathrm{odd}}^{\mathrm{C}\alpha,\mathrm{MTEF}}-J_{\mathrm{odd}}^{\mathrm{C}\alpha,\mathrm{QED}}$"),
        (axes[0, 1], "e_ds", r"$\varepsilon_{DS}$"),
        (axes[1, 0], "e_common", r"$\varepsilon_{\mathrm{common}}=(\varepsilon_C+\varepsilon_{DS})/2$"),
        (axes[1, 1], "e_delta", r"$\varepsilon_\Delta=\varepsilon_{DS}-\varepsilon_C$"),
    )
    for ax, key, title in specs:
        ax.plot(t, ch5[key][:, 0].real, color="#1d4e89", lw=1.5, label="GH5 x")
        ax.plot(t, ch5[key][:, 1].real, color="#c45c26", lw=1.2, label="GH5 y")
        ax.plot(t, ch3[key][:, 0].real, color="#1d4e89", lw=0.8, ls="--", alpha=0.7, label="GH3 x")
        ax.set_title(title, fontsize=9)
        ax.axhline(0.0, color="#aaa", lw=0.5)
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=7)
    axes[1, 0].set_xlabel("t")
    axes[1, 1].set_xlabel("t")
    fig.tight_layout()
    png = out_dir / "eps_channels.png"
    svg = out_dir / "eps_channels.svg"
    fig.savefig(png, dpi=140)
    fig.savefig(svg)
    plt.close(fig)
    return {"png": str(png.as_posix()), "svg": str(svg.as_posix())}


def write_tsv(path: Path, t, ch):
    lines = ["t\teps_C_x\teps_C_y\teps_DS_x\teps_DS_y\teps_common_x\teps_common_y\teps_Delta_x\teps_Delta_y"]
    for i, ti in enumerate(t):
        row = [f"{ti:.16e}"]
        for key in ("e_c", "e_ds", "e_common", "e_delta"):
            row.append(f"{ch[key][i, 0].real:.16e}")
            row.append(f"{ch[key][i, 1].real:.16e}")
        lines.append("\t".join(row))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def endpoint_from_ckpt(folder, tag, n_max, neel, models, do_connected, n_steps_expect=2570):
    loaded = dth.load_ckpt(folder, tag)
    if loaded is None:
        return {"status": "missing"}
    psi, next_j, _samples = loaded
    if next_j != n_steps_expect or psi.size != EL*EL*(n_max+1):
        raise RuntimeError(f"incomplete or incompatible endpoint {folder}/{tag}")
    expected_t = 2*np.pi/pf.OMEGA0
    if not _samples or abs(_samples[-1]["t"]-expected_t) > 1e-10:
        raise RuntimeError(f"endpoint time not at s=1 for {tag}")
    out = {"status": "ok", "next_j": int(next_j), "reduced": reduced_purities(psi)}
    if do_connected:
        model = models[(n_max, neel)]
        out["connected_s1"] = connected_from_psi(model, psi, s=1.0)
        archived_j = np.array([complex(z["re"], z["im"]) for z in _samples[-1]["mean_J"]])
        recomputed = np.array([complex(z["re"], z["im"]) for z in out["connected_s1"]["mean_J"]])
        out["current_matches_saved_endpoint_max_abs"] = float(np.max(np.abs(archived_j-recomputed)))
    out["purity_matches_saved_endpoint_abs"] = float(abs(out["reduced"]["purity_photon"]-_samples[-1]["purity"]))
    return out


def build_models(needed):
    geo = build_geometry()
    k = geo.k_cart(sh.K_FRAC)
    models = {}
    for n_max, neel in needed:
        models[(n_max, neel)] = sh.build_shared(
            geo, R0, k, neel, n_max=n_max, omega0=pf.OMEGA0, g=pf.G_ABS)
    return models


def run(progress=None, do_endpoint=True, do_connected=True):
    t0 = time.perf_counter()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    hashes_before = {str(p.as_posix()): mp.sha256_file(p) for p in OLD_ARCHIVES if p.is_file()}
    dest = OUT_DIR / "errdecomp.json"
    if dest.resolve() in {p.resolve() for p in OLD_ARCHIVES}:
        raise RuntimeError("refusing to overwrite an old archive")

    qed = cmp.load_qed()
    g5 = load_gh5_series()
    g3 = load_gh3_series()
    if not np.allclose(qed["t"], g5["t"], rtol=0, atol=1e-12):
        raise RuntimeError("GH5 times are not locked to 20261102")
    if not np.allclose(qed["t"], g3["t"], rtol=0, atol=1e-12):
        raise RuntimeError("GH3 times are not locked to 20261102")

    ch5 = channels(g5["J_odd_coh"], g5["J_odd_ds"], qed["J_odd_coh"], qed["J_odd_ds"])
    ch3 = channels(g3["J_odd_coh"], g3["J_odd_ds"], qed["J_odd_coh"], qed["J_odd_ds"])
    sum5 = summarize(ch5, qed["J_odd_coh"], qed["J_odd_ds"], qed["dJ_odd"], g5["dJ_odd"])
    sum3 = summarize(ch3, qed["J_odd_coh"], qed["J_odd_ds"], qed["dJ_odd"], g3["dJ_odd"])

    inv = [inventory_one(*row) for row in CKPT_SPEC]
    snapshot_inputs = [Path(item[field]) for item in inv for field in ("npz", "samples") if item.get(field)]
    snapshot_hashes = {str(p.as_posix()): mp.sha256_file(p) for p in snapshot_inputs}
    psi_files = list(ARCHIVE_QED.parent.glob("*.npz")) + list(ARCHIVE_QED.parent.glob("*.npy"))
    qed_20261102_psi = {
        "folder": str(ARCHIVE_QED.parent.as_posix()),
        "recoverable_joint_psi": bool(psi_files),
        "npy_npz_present": bool(psi_files),
        "note": "20261102 compare.json is current series only",
    }

    gate_ok, gate_p = product_state_purity_gate()
    endpoint = {}
    if do_endpoint:
        needed = {(n_max, neel) for _, _, _, n_max, neel, _, _ in CKPT_SPEC if do_connected}
        models = build_models(needed) if do_connected else {}
        _progress(progress, f"endpoint snapshots connected={do_connected}")
        for key, folder, tag, n_max, neel, n_exp, _note in CKPT_SPEC:
            if not (folder / f"ckpt_{tag}.npz").is_file():
                endpoint[key] = {"status": "missing"}
                continue
            endpoint[key] = endpoint_from_ckpt(
                folder, tag, n_max, neel, models, do_connected=do_connected)
            _progress(progress, f"endpoint {key} status={endpoint[key]['status']}")

    try:
        figs = write_plots(OUT_DIR, qed["t"], ch5, ch3)
    except ModuleNotFoundError as exc:
        if exc.name != "matplotlib":
            raise
        figs = {"status": "optional_plot_unavailable", "reason": "matplotlib not installed; full complex series and TSV retained"}
    tsv = OUT_DIR / "eps_channels.tsv"
    write_tsv(tsv, qed["t"], ch5)
    hashes_after = {str(p.as_posix()): mp.sha256_file(p) for p in OLD_ARCHIVES if p.is_file()}

    report = {
        "record_date": RECORD_DATE,
        "record_date_is_task_label": True,
        "executed_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_sha256": mp.sha256_file(Path(__file__)),
        "model_id": MODEL_ID,
        "claim_level": "archive_error_decomposition_not_HHG_not_mechanism",
        "odd_current_precision_campaign_closed": True,
        "mtef_quadrature_campaign_closed": True,
        "vacuum_squeeze_branch_closed": True,
        "did_not": [
            "rewrite any 20261026-20261110 json",
            "new QED or MTEF propagation",
            "G1", "H2", "HHG spectra", "vacuum", "theta scan", "Cn",
            "nmax scan", "GH densification", "output photon statistics",
            "change g/omega0/alpha/r",
        ],
        "definition": {
            "eps_C": "J_odd_Calpha_MTEF - J_odd_Calpha_QED",
            "eps_DS": "J_odd_DS_MTEF - J_odd_DS_QED",
            "eps_common": "0.5 (eps_C + eps_DS)",
            "eps_Delta": "eps_DS - eps_C = Delta_MTEF - Delta_QED",
            "identity": "eps_C = eps_common - eps_Delta/2 ; eps_DS = eps_common + eps_Delta/2",
            "time_lock": "42-point 20261102 mesh, no interpolation",
            "statistics_scope": "sampled maxima, unweighted sampled RMS, real-part Pearson; not certified continuous-time extrema",
            "qed_psi_note": "error series uses 20261102 currents; purity uses 20261104/05 dt/2 ψ",
        },
        "qed_ref": {"path": str(ARCHIVE_QED.as_posix()), "hash": qed["hash"]},
        "gh3_ref": {"path": str(ARCHIVE_GH3.as_posix()), "hash": g3["hash"]},
        "gh5_ref": {"path": str(ARCHIVE_GH5.as_posix()), "hash": g5["hash"],
                    "verdict_gate": g5.get("verdict_gate")},
        "status": "ok",
        "product_state_purity_gate": {"ok": bool(gate_ok), "sample": gate_p},
        "gh5_vs_qed": sum5,
        "gh3_vs_qed": sum3,
        "series_gh5": pack_ch(qed["t"], ch5),
        "checkpoint_inventory": {
            "qed_20261102": qed_20261102_psi,
            "items": inv,
        },
        "endpoint_reduced": endpoint,
        "figures": figs,
        "tsv": str(tsv.as_posix()),
        "old_archive_hashes_before": hashes_before,
        "old_archive_hashes_unchanged": bool(hashes_before == hashes_after),
        "checkpoint_hashes_before": snapshot_hashes,
        "checkpoint_hashes_unchanged": all(mp.sha256_file(Path(p)) == h for p, h in snapshot_hashes.items()),
        "wall_seconds": float(time.perf_counter() - t0),
    }
    report["all_checks_pass"] = bool(gate_ok and sum5["identities_ok"] and sum3["identities_ok"]
        and report["old_archive_hashes_unchanged"] and report["checkpoint_hashes_unchanged"]
        and all(item["complete"] for item in inv)
        and all(e["status"] == "ok" and e["reduced"]["schmidt_purity_difference"] < 1e-12
                and e["purity_matches_saved_endpoint_abs"] < 1e-12
                and e.get("current_matches_saved_endpoint_max_abs", 0) < 1e-12 for e in endpoint.values()))
    report["status"] = "ok" if report["all_checks_pass"] else "check_failed"
    return report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--progress", default="")
    ap.add_argument("--out", default="")
    ap.add_argument("--skip-endpoint", action="store_true")
    ap.add_argument("--skip-connected", action="store_true")
    args = ap.parse_args()
    progress = Path(args.progress) if args.progress else OUT_DIR / "progress.txt"
    out = Path(args.out) if args.out else OUT_DIR / "errdecomp.json"
    if out.exists() or out.resolve() in {p.resolve() for p in OLD_ARCHIVES}:
        raise FileExistsError(f"refusing to overwrite an existing or protected result: {out}")
    report = run(progress=progress, do_endpoint=not args.skip_endpoint,
                 do_connected=not args.skip_connected)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as fh:
        json.dump(mp.jsonable(report), fh, ensure_ascii=False, indent=1)
    _progress(progress, f"wrote {out} status={report['status']} wall={report['wall_seconds']:.2f}s")
    print(f"status={report['status']} wall={report['wall_seconds']:.3f}s out={out}")


if __name__ == "__main__":
    main()
