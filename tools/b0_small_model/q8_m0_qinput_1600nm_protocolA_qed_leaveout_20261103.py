# -*- coding: utf-8 -*-
"""Independent 1600 nm protocol-A exact-QED holdout analysis.

Does not import the four-angle runner or the postprocess module, does not
propagate, and does not rewrite fourangle_fit.json. Run only after p_q and
m_q have both finished.
"""
from __future__ import annotations

import hashlib
import json
import os
import platform
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
import sys
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from qinput_spectrum_reanalysis_20261028 import packed, spectrum, unpack

ROOT = Path(__file__).resolve().parents[2]
QED = ROOT / "results/q_line/qinput_1600nm_protocolA_qed_fourangle_20261102"
GH5 = ROOT / "results/q_line/qinput_1600nm_protocolA_gh5_20261102"
LOW = ROOT / "results/q_line/qinput_spectrum_reanalysis_20261028"
ANALYSIS = QED / "analysis_endpoint_v1"
FIT = QED / "fourangle_fit.json"
FIT_SHA = QED / "fourangle_fit.sha256"
EXPECTED_FIT_SHA = "17a8e37ec3e0ab54096ff4edd1440c5a3f0412b09450c2fd8960cbc00d8e69db"

OMEGA = 0.775
OMEGA_LOW = 0.05
PERIOD = 2.0 * np.pi / OMEGA
HBAR_EV_FS = 0.6582119568564649
T_H_EV = 1.0
N_FULL = 11154
N_MAX = 8
G_ABS = 0.005
ALPHA = 0.5
HOLD = 0.25 * np.pi
ESYM_MAX = 0.02
WINDOWS = {"main": (2.0, 4.0), "front": (1.0, 3.0), "back": (3.0, 5.0)}


def sha256_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dump_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(
        json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(tmp, path)


def cz(z) -> complex:
    if isinstance(z, dict):
        return unpack(z)
    return complex(z)


def principal_deg(rad: float) -> float:
    return float(np.degrees(np.angle(np.exp(1j * float(rad)))))


def arg_ab_deg(A: complex, B: complex):
    if abs(B) == 0.0:
        return None
    return principal_deg(np.angle(A * np.conj(B)))


def dist_to_pi_deg(arg_deg):
    if arg_deg is None:
        return None
    return abs(principal_deg(np.radians(arg_deg) - np.pi))


def fs_from_model(t_model: float, omega: float) -> dict:
    period = 2.0 * np.pi / float(omega)
    return {
        "omega_eV": float(omega),
        "t_h_eV": T_H_EV,
        "period_model": period,
        "period_fs": period * HBAR_EV_FS / T_H_EV,
        "six_period_fs": 6.0 * period * HBAR_EV_FS / T_H_EV,
        "main_window_fs": 2.0 * period * HBAR_EV_FS / T_H_EV,
        "note": "same cycle count is not the same laboratory duration",
    }


def load_frozen() -> tuple[dict, str]:
    if not FIT.is_file() or not FIT_SHA.is_file():
        raise RuntimeError("missing frozen fit")
    raw = FIT.read_bytes()
    got = hashlib.sha256(raw).hexdigest()
    want = FIT_SHA.read_text(encoding="utf-8").strip().split()[0]
    if got != want or got != EXPECTED_FIT_SHA:
        raise RuntimeError(f"freeze sha mismatch got={got} file={want}")
    payload = json.loads(raw.decode("utf-8"))
    if payload.get("holdout_used_in_fit") is not False:
        raise RuntimeError("frozen fit claims holdout was used")
    if abs(float(payload["frequency"]) - OMEGA) > 1e-12:
        raise RuntimeError("frozen frequency is not 0.775")
    if list(payload.get("training_tags") or [])[-2:] == ["p_q", "m_q"]:
        raise RuntimeError("training tags include holdout")
    return payload, got


def load_job(folder: Path, tag: str) -> dict:
    return json.loads((folder / f"job_{tag}.json").read_text(encoding="utf-8"))


def load_series(folder: Path, tag: str):
    path = folder / f"series_{tag}.npz"
    with np.load(path, allow_pickle=False) as z:
        t = np.asarray(z["t"], float)
        j = np.column_stack([np.asarray(z["Jx"]), np.asarray(z["Jy"])]).astype(complex)
    return t, j


def qed_axis_ok(tag: str, neel: int) -> tuple[bool, list[str], dict]:
    path = QED / f"job_{tag}.json"
    series = QED / f"series_{tag}.npz"
    problems = []
    if not path.is_file() or not series.is_file():
        return False, [f"missing {tag}"], {}
    rec = load_job(QED, tag)
    defn = rec.get("definition") or {}
    ledger = rec.get("ledger") or {}
    checks = {
        "status": rec.get("status") == "ok",
        "alarm": rec.get("fock_alarm_0p05") is False,
        "finite": rec.get("finite_state") is True,
        "cover": rec.get("covers_requested") is True,
        "error": rec.get("error") is None,
        "n_full": int(defn.get("n_full") or -1) == N_FULL,
        "steps": int(ledger.get("steps_completed") or -1) == N_FULL,
        "timed_out": ledger.get("timed_out") is False,
        "nmax": int(rec.get("n_max") or -1) == N_MAX and int(defn.get("n_max") or -1) == N_MAX,
        "omega": abs(float(rec.get("omega0")) - OMEGA) < 1e-12,
        "g": abs(float(rec.get("g")) - G_ABS) < 1e-12,
        "theta": abs(float(rec.get("theta")) - HOLD) < 1e-12,
        "neel": int(rec.get("neel")) == neel,
        "did_not_scale_g": rec.get("did_not_scale_g") is True,
        "used_pf": rec.get("used_pf_OMEGA0") is False,
    }
    problems.extend([k for k, ok in checks.items() if not ok])
    t, j = load_series(QED, tag)
    if t.shape != (350,) or j.shape != (350, 2):
        problems.append(f"shape {t.shape} {j.shape}")
    if not np.all(np.isfinite(t)) or not np.all(np.isfinite(j)):
        problems.append("nonfinite")
    if np.any(np.diff(t) <= 0) or t[-1] < 6.0 * PERIOD - 1e-8:
        problems.append("time coverage")
    return (not problems), problems, rec


def h2_pair(t, jp, jm, method: str, window: str):
    if not np.array_equal(t, t):
        raise RuntimeError("time mismatch")
    jo = 0.5 * (jp - jm)
    a, b = WINDOWS[window]
    lo, hi = a * PERIOD, b * PERIOD
    qs = np.arange(1, 7, dtype=float)
    odd = spectrum(t, jo, lo, hi, qs, method=method, omega=OMEGA)
    zx = complex(odd[1, 0])
    zy = complex(odd[1, 1])
    return zx, zy


def holdout_metrics(z: complex, pred: complex, A: complex) -> dict:
    delta = z - pred
    denom = max(abs(z), abs(pred))
    esym = None if denom == 0.0 else float(abs(delta) / denom)
    ea = None if abs(A) == 0.0 else float(abs(delta) / abs(A))
    dphi = float(np.degrees(np.angle(z * np.conj(pred)))) if abs(z) and abs(pred) else None
    return {
        "z": packed(z),
        "z_pred": packed(pred),
        "delta": packed(delta),
        "abs_delta": float(abs(delta)),
        "Esym": esym,
        "EA": ea,
        "phase_error_deg": dphi,
        "Esym_pass_0p02": None if esym is None else bool(esym <= ESYM_MAX),
    }


def low_frequency() -> dict:
    fits = json.loads((LOW / "fits_BAC.json").read_text(encoding="utf-8"))
    rows = [
        r for r in fits
        if r.get("solver") == "QED" and r.get("channel") == "odd"
        and r.get("axis") == "x" and r.get("eta") == 1.0
        and r.get("estimator") == "linear_exact"
    ]
    by_window = {r["window"]: r for r in rows}
    main = by_window["main"]
    A, B, C = cz(main["A"]), cz(main["B"]), cz(main["C"])
    key = json.loads((LOW / "key_results.json").read_text(encoding="utf-8"))
    qrows = {r["case"]: r for r in key["main_QED_rows"]}
    z0 = cz(qrows["QED_theta0"]["odd_x"])
    zpi = cz(qrows["QED_theta_pi"]["odd_x"])
    n0 = float(qrows["QED_theta0"]["odd_2d"])
    npi = float(qrows["QED_theta_pi"]["odd_2d"])
    out = {
        "source": "results/q_line/qinput_spectrum_reanalysis_20261028",
        "estimator": "linear_exact",
        "window": "main",
        "component": "Jx",
        "domain_parity": "Neel-odd",
        "omega_eV": OMEGA_LOW,
        "time": fs_from_model(0.0, OMEGA_LOW),
        "B": packed(B),
        "A": packed(A),
        "C": packed(C),
        "rho": float(abs(A) / abs(B)),
        "arg_A_over_B_deg": arg_ab_deg(A, B),
        "arg_distance_to_pi_deg": dist_to_pi_deg(arg_ab_deg(A, B)),
        "abs_C_over_A": float(abs(C) / abs(A)),
        "relative_L2_residual": main.get("relative_L2_residual"),
        "x_amp_ratio_pi_over_0": float(abs(zpi) / abs(z0)),
        "amp2d_ratio_pi_over_0": float(npi / n0),
        "key_results_2d_ratio": key.get("QED_main_odd_2D_pi_over_0"),
        "archived_2d_matches_recomputed": abs(float(npi / n0) - float(key["QED_main_odd_2D_pi_over_0"])) < 1e-12,
        "do_not_compare_2d_4p98_with_1600_x_ratio": True,
        "retired_5p80_106p9_not_used": True,
    }
    sens = {}
    for name in ("front", "back"):
        if name not in by_window:
            continue
        Af, Bf = cz(by_window[name]["A"]), cz(by_window[name]["B"])
        sens[name] = {
            "rho": float(abs(Af) / abs(Bf)),
            "arg_A_over_B_deg": arg_ab_deg(Af, Bf),
            "arg_distance_to_pi_deg": dist_to_pi_deg(arg_ab_deg(Af, Bf)),
        }
    out["neighbor_windows_not_error_bars"] = sens
    return out


def gh5_fourangle_and_holdout(pred: complex, A: complex) -> dict:
    sources = {}
    for tag in ("p_q", "m_q"):
        for kind in ("job", "series"):
            path = GH5 / f"{kind}_{tag}.json" if kind == "job" else GH5 / f"{kind}_{tag}.npz"
            sources[str(path.relative_to(ROOT))] = sha256_file(path)
    jp_job = load_job(GH5, "p_q")
    jm_job = load_job(GH5, "m_q")
    problems = []
    for rec, neel in ((jp_job, 1), (jm_job, -1)):
        if rec.get("status") != "ok":
            problems.append(f"gh5 status {rec.get('tag')}")
        if abs(float(rec.get("theta")) - HOLD) > 1e-12 or int(rec.get("neel")) != neel:
            problems.append(f"gh5 identity {rec.get('tag')}")
        if abs(float(rec.get("omega0")) - OMEGA) > 1e-12:
            problems.append("gh5 omega")
        if int(rec.get("n_steps") or -1) != N_FULL:
            problems.append("gh5 steps")
    tp, jp = load_series(GH5, "p_q")
    tm, jm = load_series(GH5, "m_q")
    if not np.array_equal(tp, tm):
        problems.append("gh5 time mismatch")
    if problems:
        raise RuntimeError("GH5 holdout preflight failed: " + "; ".join(problems))
    variants = {}
    for method in ("linear_exact", "quadratic_exact"):
        for window in WINDOWS:
            zx, zy = h2_pair(tp, jp, jm, method, window)
            variants[f"{method}:{window}"] = holdout_metrics(zx, pred, A) | {
                "y": packed(zy),
                "amp2d": float(np.hypot(abs(zx), abs(zy))),
            }
    primary = variants["linear_exact:main"]
    calls = {k: v["Esym_pass_0p02"] for k, v in variants.items()}
    return {
        "method": "local_GH5_not_Cohen_MTEF",
        "did_not_propagate": True,
        "did_not_call_analyze_fourangle": True,
        "did_not_refit": True,
        "sources": sources,
        "primary": primary,
        "variants": variants,
        "sensitivity_changes_2pct_call": len(set(calls.values())) > 1,
        "historical_legacy_holdout_0p0115pct_not_used": True,
    }


def make_figures(fit: dict, z_train, z_curve, z_pred, z_actual, gh5_obs, gh5_pred, gh5_actual, low, qed_metrics, gh5_block) -> dict:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "font.sans-serif": ["Microsoft YaHei", "SimHei", "DejaVu Sans"],
        "axes.unicode_minus": False,
        "font.size": 10,
        "axes.labelsize": 11,
        "axes.titlesize": 12,
        "figure.dpi": 140,
        "savefig.dpi": 180,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })
    figdir = ANALYSIS / "figures"
    figdir.mkdir(parents=True, exist_ok=True)
    scale = 1e5

    fig, ax = plt.subplots(figsize=(7.4, 6.2))
    th = np.linspace(0.0, 2.0 * np.pi, 361)
    B, A, C = [cz(fit["coefficients"][k]) for k in ("B", "A", "C")]
    curve = B + A * np.exp(1j * th) + C * np.exp(-1j * th)
    ax.plot(curve.real * scale, curve.imag * scale, color="#1f4e79", lw=1.4,
            label="QED 冻结三项曲线（不是已测连续角）")
    labels = [r"$0$", r"$\pi/2$", r"$\pi$", r"$3\pi/2$"]
    ax.scatter(z_train.real * scale, z_train.imag * scale, s=46, c="#1f4e79",
               zorder=4, label="QED 四角实测")
    for lab, z in zip(labels, z_train):
        ax.annotate(lab, (z.real * scale, z.imag * scale), textcoords="offset points",
                    xytext=(6, 5), fontsize=9, color="#1f4e79")
    ax.scatter([z_pred.real * scale], [z_pred.imag * scale], s=80, facecolors="none",
               edgecolors="#c0392b", linewidths=1.6, marker="D", zorder=5,
               label=r"QED $\pi/4$ 冻结预测")
    ax.scatter([z_actual.real * scale], [z_actual.imag * scale], s=110, c="#c0392b",
               marker="*", zorder=6, label=r"QED $\pi/4$ 独立实测")
    if gh5_obs is not None:
        ax.scatter(gh5_obs.real * scale, gh5_obs.imag * scale, s=36,
                   facecolors="none", edgecolors="#7f8c8d", marker="s",
                   label="GH5 四角（同口径，不是 MTEF）")
    if gh5_pred is not None:
        ax.scatter([gh5_pred.real * scale], [gh5_pred.imag * scale], s=54,
                   c="#7f8c8d", marker="+", linewidths=1.4, label=r"GH5 $\pi/4$ 预测")
    if gh5_actual is not None:
        ax.scatter([gh5_actual.real * scale], [gh5_actual.imag * scale], s=54,
                   c="#7f8c8d", marker="x", linewidths=1.4, label=r"GH5 $\pi/4$ 实测")
    ax.axhline(0, color="#d0d0d0", lw=0.6)
    ax.axvline(0, color="#d0d0d0", lw=0.6)
    ax.set_xlabel(r"Re $J_2^{o,x}$  ($\times 10^{-5}$)")
    ax.set_ylabel(r"Im $J_2^{o,x}$  ($\times 10^{-5}$)")
    ax.set_title("1600 nm 主窗 Néel-odd $H_2$，$J_x$")
    ax.set_aspect("equal", adjustable="datalim")
    ax.legend(frameon=False, fontsize=8, loc="best")
    fig.tight_layout()
    p1 = figdir / "fig1_complex_plane.png"
    fig.savefig(p1)
    fig.savefig(figdir / "fig1_complex_plane.pdf")
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(8.6, 6.6))
    qed_rho = float(fit["derived_metrics"]["rho_absA_over_absB"])
    qed_arg = float(fit["derived_metrics"]["arg_A_conjB_deg"])
    qed_x = float(fit["derived_metrics"]["x_amp_ratio_pi_over_0"])
    qed_2d = float(fit["derived_metrics"]["amp2d_ratio_pi_over_0"])
    gh5 = json.loads((ANALYSIS / "gh5_reference_corrected.json").read_text(encoding="utf-8"))
    gm = gh5["derived_metrics"]
    freqs = ["24.8 μm\n0.05 eV", "1600 nm\n0.775 eV"]
    ax = axes[0, 0]
    ax.plot([0, 1], [low["rho"], qed_rho], "o-", color="#1f4e79", label="精确 QED")
    ax.plot([1], [gm["rho_absA_over_absB"]], "s", color="#7f8c8d", label="1600 nm GH5")
    ax.axhline(1.2733071642964826, color="#b9770e", ls="--", lw=1, label=r"理想 $R=1.273$")
    ax.set_xticks([0, 1], freqs)
    ax.set_ylabel(r"$\rho=|A|/|B|$")
    ax.set_title("同口径 rho，不是 $K_A/K_B$")
    ax.legend(frameon=False, fontsize=8)

    ax = axes[0, 1]
    ax.plot([0, 1], [low["arg_A_over_B_deg"], qed_arg], "o-", color="#1f4e79")
    ax.plot([1], [gm["arg_A_conjB_deg"]], "s", color="#7f8c8d")
    ax.axhline(180.0, color="#b9770e", ls="--", lw=1)
    ax.set_xticks([0, 1], freqs)
    ax.set_ylabel(r"$\arg(A/B)$ (deg)")
    ax.set_title("距 $\\pi$ 的相位，不是远场相位")

    ax = axes[1, 0]
    labels = ["x 比", "二维比"]
    x = np.arange(2)
    ax.bar(x - 0.16, [low["x_amp_ratio_pi_over_0"], low["amp2d_ratio_pi_over_0"]],
           width=0.32, color="#5d6d7e", label="24.8 μm QED")
    ax.bar(x + 0.16, [qed_x, qed_2d], width=0.32, color="#1f4e79", label="1600 nm QED")
    ax.set_xticks(x, labels)
    ax.set_ylabel(r"$|H_2(\pi)|/|H_2(0)|$")
    ax.set_title("两种振幅比，禁止混栏")
    ax.legend(frameon=False, fontsize=8)

    ax = axes[1, 1]
    names = ["QED\n主窗", "QED\n二次", "QED\n前窗", "QED\n后窗", "GH5\n主窗"]
    # filled by caller via attributes attached below; placeholder replaced after
    ax.set_visible(False)
    fig.tight_layout()
    # redraw panel 4 with actual sensitivity after the first save skeleton
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(8.6, 6.6))
    ax = axes[0, 0]
    ax.plot([0, 1], [low["rho"], qed_rho], "o-", color="#1f4e79", ms=7, label="精确 QED")
    ax.plot([1], [gm["rho_absA_over_absB"]], "s", color="#7f8c8d", ms=7, label="1600 nm GH5")
    ax.axhline(1.2733071642964826, color="#b9770e", ls="--", lw=1, label="理想 R=1.273")
    ax.set_xticks([0, 1], freqs)
    ax.set_ylabel(r"$\rho=|A|/|B|$")
    ax.set_title("同口径 rho，不是 KA/KB")
    ax.legend(frameon=False, fontsize=8)

    ax = axes[0, 1]
    ax.plot([0, 1], [low["arg_A_over_B_deg"], qed_arg], "o-", color="#1f4e79", ms=7)
    ax.plot([1], [gm["arg_A_conjB_deg"]], "s", color="#7f8c8d", ms=7)
    ax.axhline(180.0, color="#b9770e", ls="--", lw=1, label=r"180°")
    ax.set_xticks([0, 1], freqs)
    ax.set_ylabel("arg(A/B) (deg)")
    ax.set_title("近反相；不是远场谐波相位")
    ax.legend(frameon=False, fontsize=8)

    ax = axes[1, 0]
    x = np.arange(2)
    ax.bar(x - 0.16, [low["x_amp_ratio_pi_over_0"], low["amp2d_ratio_pi_over_0"]],
           width=0.32, color="#5d6d7e", label="24.8 μm QED")
    ax.bar(x + 0.16, [qed_x, qed_2d], width=0.32, color="#1f4e79", label="1600 nm QED")
    ax.set_xticks(x, ["x 分量比", "二维振幅比"])
    ax.set_ylabel("|H2(pi)| / |H2(0)|")
    ax.set_title("4.98 是二维比，8.53 是 x 比")
    ax.legend(frameon=False, fontsize=8)

    ax = axes[1, 1]
    qvars = qed_metrics["variants"]
    vals = [
        qvars["linear_exact:main"]["Esym"] * 100,
        qvars["quadratic_exact:main"]["Esym"] * 100,
        qvars["linear_exact:front"]["Esym"] * 100,
        qvars["linear_exact:back"]["Esym"] * 100,
        gh5_block["primary"]["Esym"] * 100,
    ]
    colors = ["#1f4e79", "#5dade2", "#85929e", "#85929e", "#7f8c8d"]
    ax.bar(np.arange(5), vals, color=colors)
    ax.axhline(2.0, color="#c0392b", ls="--", lw=1, label="2% 工作线")
    ax.set_xticks(np.arange(5), ["QED\n主窗", "QED\n二次插值", "QED\n前窗", "QED\n后窗", "GH5\n主窗"])
    ax.set_ylabel("Esym (%)")
    ax.set_title("跨度不是误差棒")
    ax.legend(frameon=False, fontsize=8)
    fig.suptitle("频率迁移对照：有限时间协议，不是稳态 χ(ω)", y=1.02, fontsize=12)
    fig.tight_layout()
    p2 = figdir / "fig2_frequency_comparison.png"
    fig.savefig(p2, bbox_inches="tight")
    fig.savefig(figdir / "fig2_frequency_comparison.pdf", bbox_inches="tight")
    plt.close(fig)
    return {
        "fig1": str(p1.relative_to(ROOT)),
        "fig2": str(p2.relative_to(ROOT)),
        "matplotlib": matplotlib.__version__,
        "no_invented_error_bars": True,
    }


def main() -> int:
    fit, sha_before = load_frozen()
    ok_p, prob_p, job_p = qed_axis_ok("p_q", 1)
    ok_m, prob_m, job_m = qed_axis_ok("m_q", -1)
    if not ok_p or not ok_m:
        raise SystemExit(f"holdout axes not ready p={prob_p} m={prob_m}")
    tp, jp = load_series(QED, "p_q")
    tm, jm = load_series(QED, "m_q")
    if not np.array_equal(tp, tm):
        raise RuntimeError("QED pi/4 time arrays differ; no interpolation")
    # constant leak check on this grid
    qs = np.arange(1, 7, dtype=float)
    leak = spectrum(tp, np.ones_like(tp), 2 * PERIOD, 4 * PERIOD, qs,
                    method="linear_exact", omega=OMEGA)
    if np.max(np.abs(leak)) > 1e-12:
        raise RuntimeError("linear_exact constant leak on holdout grid")

    pred = cz(fit["holdout_prediction"])
    A = cz(fit["coefficients"]["A"])
    B = cz(fit["coefficients"]["B"])
    variants = {}
    y_main = None
    for method in ("linear_exact", "quadratic_exact"):
        for window in WINDOWS:
            zx, zy = h2_pair(tp, jp, jm, method, window)
            rec = holdout_metrics(zx, pred, A)
            rec["y"] = packed(zy)
            rec["amp2d"] = float(abs(complex(zx, zy))) if False else float(np.hypot(abs(zx), abs(zy)))
            variants[f"{method}:{window}"] = rec
            if method == "linear_exact" and window == "main":
                y_main = zy
                z_actual = zx
    primary = variants["linear_exact:main"]
    calls = {k: v["Esym_pass_0p02"] for k, v in variants.items()}
    sens_changes = len(set(calls.values())) > 1
    abs_delta = primary["abs_delta"]
    interp = abs(cz(variants["linear_exact:main"]["z"]) - cz(variants["quadratic_exact:main"]["z"]))
    near_sensitivity = interp > 0.5 * abs_delta or interp > 0.5 * ESYM_MAX * max(abs(z_actual), abs(pred))
    if sens_changes or near_sensitivity:
        verdict = "当前保存数据不足以稳健判断"
        robust = False
    elif primary["Esym_pass_0p02"]:
        verdict = "主窗三项模型在该留出点达到 2% 工作判据"
        robust = True
    else:
        verdict = "三项模型在该留出点预测不足"
        robust = True

    gh5_ref = json.loads((ANALYSIS / "gh5_reference_corrected.json").read_text(encoding="utf-8"))
    gh5_pred = cz(gh5_ref["holdout_prediction"])
    gh5_A = cz(gh5_ref["A"])
    gh5_block = gh5_fourangle_and_holdout(gh5_pred, gh5_A)
    low = low_frequency()

    z_train = np.array([cz(fit["observations"][k]) for k in
                        ("theta_0", "theta_h", "theta_pi", "theta_t")], complex)
    gh5_train = np.array([cz(gh5_ref["observations"][k]) for k in
                          ("theta_0", "theta_h", "theta_pi", "theta_t")], complex)
    figures = make_figures(
        fit, z_train, None, pred, z_actual, gh5_train, gh5_pred,
        cz(gh5_block["primary"]["z"]), low,
        {"variants": variants}, gh5_block,
    )

    sources = {
        str((QED / "job_p_q.json").relative_to(ROOT)): sha256_file(QED / "job_p_q.json"),
        str((QED / "series_p_q.npz").relative_to(ROOT)): sha256_file(QED / "series_p_q.npz"),
        str((QED / "job_m_q.json").relative_to(ROOT)): sha256_file(QED / "job_m_q.json"),
        str((QED / "series_m_q.npz").relative_to(ROOT)): sha256_file(QED / "series_m_q.npz"),
    }
    training_unchanged = all(
        sha256_file(ROOT / rel) == h for rel, h in fit["sources"].items()
    )
    sha_after = sha256_file(FIT)
    if sha_after != sha_before:
        raise RuntimeError("freeze file changed during analysis")

    qed_rho = float(fit["derived_metrics"]["rho_absA_over_absB"])
    ideal_R = float(fit["protocol"]["ideal_R"])
    leaveout = {
        "schema": "1600nm_protocolA_qed_leaveout_v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "fit_sha256_before_and_after": sha_after,
        "fit_not_rewritten": True,
        "training_inputs_unchanged": training_unchanged,
        "n8_p_q_not_used": True,
        "did_not_refit": True,
        "did_not_propagate": True,
        "axes": {
            "p_q": {
                "status": job_p.get("status"),
                "neel": job_p.get("neel"),
                "theta": job_p.get("theta"),
                "steps": (job_p.get("ledger") or {}).get("steps_completed"),
                "max_top_fock": job_p.get("max_top_fock"),
                "max_n_ph": job_p.get("max_n_ph"),
                "max_abs_norm_minus_1": job_p.get("max_abs_norm_minus_1"),
                "fock_alarm_0p05": job_p.get("fock_alarm_0p05"),
            },
            "m_q": {
                "status": job_m.get("status"),
                "neel": job_m.get("neel"),
                "theta": job_m.get("theta"),
                "steps": (job_m.get("ledger") or {}).get("steps_completed"),
                "max_top_fock": job_m.get("max_top_fock"),
                "max_n_ph": job_m.get("max_n_ph"),
                "max_abs_norm_minus_1": job_m.get("max_abs_norm_minus_1"),
                "fock_alarm_0p05": job_m.get("fock_alarm_0p05"),
            },
        },
        "sources": sources,
        "integrator": "linear_exact",
        "window": "[2T,4T]",
        "omega": OMEGA,
        "component": "Jx Neel-odd",
        "constant_leak_abs_max": float(np.max(np.abs(leak))),
        "primary": primary,
        "y_main": packed(y_main),
        "variants": variants,
        "interpolation_abs_main": float(interp),
        "sensitivity_changes_2pct_call": sens_changes,
        "interpolation_comparable_to_residual": bool(near_sensitivity),
        "robust_against_recorded_sensitivity": robust and not sens_changes and not near_sensitivity,
        "verdict": verdict,
        "criterion": "Esym<=0.02 on linear_exact main; EA reported separately",
        "not_nmax_convergence": True,
        "not_proof_higher_delta_n_absent": True,
        "failure_would_not_identify_delta_n_pm4": True,
        "python": platform.python_version(),
        "numpy": np.__version__,
    }
    comparison = {
        "schema": "1600nm_protocolA_qed_comparison_v1",
        "fit_sha256": sha_after,
        "time": {
            "low": fs_from_model(0.0, OMEGA_LOW),
            "high": fs_from_model(0.0, OMEGA),
            "same_cycle_count_changes_laboratory_duration": True,
            "not_steady_state_chi": True,
            "not_25fs_laboratory_BSV": True,
        },
        "qed_1600": {
            "rho": qed_rho,
            "rho_over_ideal_R": qed_rho / ideal_R,
            "ideal_R": ideal_R,
            "finite_truncation_input_R": fit.get("input_moments"),
            "arg_A_over_B_deg": fit["derived_metrics"]["arg_A_conjB_deg"],
            "arg_distance_to_pi_deg": fit["derived_metrics"]["arg_distance_to_pi_deg"],
            "abs_C_over_A": fit["derived_metrics"]["abs_C_over_A"],
            "x_amp_ratio_pi_over_0": fit["derived_metrics"]["x_amp_ratio_pi_over_0"],
            "amp2d_ratio_pi_over_0": fit["derived_metrics"]["amp2d_ratio_pi_over_0"],
            "relative_L2_residual": fit["relative_L2_residual"],
            "holdout": primary,
            "holdout_verdict": verdict,
        },
        "gh5_1600_corrected": {
            "rho": gh5_ref["derived_metrics"]["rho_absA_over_absB"],
            "arg_A_over_B_deg": gh5_ref["derived_metrics"]["arg_A_conjB_deg"],
            "arg_distance_to_pi_deg": gh5_ref["derived_metrics"]["arg_distance_to_pi_deg"],
            "abs_C_over_A": gh5_ref["derived_metrics"]["abs_C_over_A"],
            "x_amp_ratio_pi_over_0": gh5_ref["derived_metrics"]["x_amp_ratio_pi_over_0"],
            "amp2d_ratio_pi_over_0": gh5_ref["derived_metrics"]["amp2d_ratio_pi_over_0"],
            "holdout": gh5_block["primary"],
            "sensitivity_changes_2pct_call": gh5_block["sensitivity_changes_2pct_call"],
            "not_cohen_mtef": True,
        },
        "qed_minus_gh5_relative": {
            "rho": float(qed_rho - gh5_ref["derived_metrics"]["rho_absA_over_absB"]),
            "arg_deg": float(fit["derived_metrics"]["arg_A_conjB_deg"] - gh5_ref["derived_metrics"]["arg_A_conjB_deg"]),
            "x_ratio": float(fit["derived_metrics"]["x_amp_ratio_pi_over_0"] - gh5_ref["derived_metrics"]["x_amp_ratio_pi_over_0"]),
            "coefficient_abs_diff_over_qed": {
                "B": float(abs(cz(fit["coefficients"]["B"]) - cz(gh5_ref["B"])) / abs(cz(fit["coefficients"]["B"]))),
                "A": float(abs(cz(fit["coefficients"]["A"]) - cz(gh5_ref["A"])) / abs(cz(fit["coefficients"]["A"]))),
                "C": float(abs(cz(fit["coefficients"]["C"]) - cz(gh5_ref["C"])) / max(abs(cz(fit["coefficients"]["C"])), 1e-30)),
            },
            "not_interpreted_as_entanglement": True,
        },
        "low_frequency_linear_exact": low,
        "frequency_change": {
            "rho_1600_minus_low": float(qed_rho - low["rho"]),
            "arg_1600_minus_low_deg": float(fit["derived_metrics"]["arg_A_conjB_deg"] - low["arg_A_over_B_deg"]),
            "x_ratio_not_compared_to_low_2d": True,
            "x_ratio_1600": fit["derived_metrics"]["x_amp_ratio_pi_over_0"],
            "x_ratio_low": low["x_amp_ratio_pi_over_0"],
            "amp2d_1600": fit["derived_metrics"]["amp2d_ratio_pi_over_0"],
            "amp2d_low": low["amp2d_ratio_pi_over_0"],
        },
        "boundaries": [
            "finite-time protocol frequency move, not steady-state chi(omega)",
            "abstract g=0.005, not calibrated W/cm2",
            "single k/-k M0, not full BZ or CrI3",
            "displaced squeezed state, not Rasputnyi pure BSV",
            "top Fock occupation low is not nmax=8 versus 10 H2 convergence",
            "holdout success is not absence of higher Delta-n sectors",
            "rho near R is not a KA/KB theorem",
            "fitted B/A/C are not independently reconstructed microscopic kernels",
            "GH5 is not Cohen MTEF",
        ],
        "figures": figures,
    }
    dump_json(ANALYSIS / "leaveout_validation.json", leaveout)
    dump_json(ANALYSIS / "comparison.json", comparison)
    if sha256_file(FIT) != EXPECTED_FIT_SHA:
        raise RuntimeError("freeze hash changed after write")
    print(json.dumps({
        "verdict": verdict,
        "Esym": primary["Esym"],
        "EA": primary["EA"],
        "phase_error_deg": primary["phase_error_deg"],
        "abs_delta": primary["abs_delta"],
        "sensitivity_changes_2pct_call": sens_changes,
        "gh5_Esym": gh5_block["primary"]["Esym"],
        "fit_sha": sha_after,
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
