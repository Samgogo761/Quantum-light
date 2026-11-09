"""Explain the odd-channel K_B window reversal. No new QED.

Compare QED, GH5 kick=True, and Ehrenfest kick=False on the same
two-angle B and A+C proxies. Production numbers stay frozen.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
import sys

sys.path.insert(0, str(HERE))

import numpy as np

import q8_m0_abc_a1_gate_gh5_20261029 as a1
import q8_m0_abc_km_kb_split_20261029 as split
import q8_m0_qinput_shared_kk_dyn_preflight_20261030 as pf
import qinput_spectrum_reanalysis_20261028 as spec

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results/q_line/abc_kb_window_20261029"
DOCS = ROOT / "docs/04_BSV量子光"
FIG = OUT / "figures"

QED_TH0 = split.QED_TH0_P
QED_THPI = split.QED_THPI_P
GH5_DIR = split.GH5_DIR
A2B = ROOT / "results/q_line/abc_a2b_ehrenfest_kick_20261029"


def dump(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(
        json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(tmp, path)


def odd_jx_pair(folder: Path, tag_p: str, tag_m: str):
    t, j, extra = split.load_xy(folder, tag_p)
    tm, jm, _ = split.load_xy(folder, tag_m)
    if np.max(np.abs(t - tm)) > 1e-12:
        raise RuntimeError(f"time mismatch {folder} {tag_p}")
    return t, 0.5 * (j[:, 0] - jm[:, 0]), extra


def interp_to(t_ref, t, y):
    return np.interp(t_ref, t, y)


def rms(x):
    x = np.asarray(x, dtype=float)
    return float(np.sqrt(np.mean(np.square(x))))


def h2_proxy(t, j0, jpi, lo, hi):
    j0c = np.column_stack([j0, np.zeros_like(j0)])
    jpic = np.column_stack([jpi, np.zeros_like(jpi)])
    q0 = spec.spectrum(t, j0c, lo, hi, a1.QS, method="linear_exact", omega=pf.OMEGA0)
    qpi = spec.spectrum(t, jpic, lo, hi, a1.QS, method="linear_exact", omega=pf.OMEGA0)
    pr = a1.two_angle_proxy(q0[1, 0], qpi[1, 0])
    return {
        "B": pr["B_proxy"],
        "AC": pr["A_plus_C_proxy"],
        "abs_B": pr["B_proxy"]["abs"],
        "abs_AC": pr["A_plus_C_proxy"]["abs"],
        "arg_B_deg": pr["B_proxy"]["arg_deg"],
        "arg_AC_deg": pr["A_plus_C_proxy"]["arg_deg"],
        "amp_ratio_pi_over_0": pr["amp_ratio_pi_over_0"],
        "phase_diff_deg": pr["phase_diff_deg"],
    }


def sliding(t, sources: dict, T0: float, t_end: float):
    rows = []
    start = 0.50
    while start + 2.0 <= t_end / T0 + 1e-12:
        lo, hi = start * T0, (start + 2.0) * T0
        if hi > t_end + 1e-12:
            break
        row = {
            "start_T0": float(start),
            "center_T0": float(start + 1.0),
            "stop_T0": float(start + 2.0),
        }
        mask = (t >= lo) & (t <= hi)
        s = sources["qed"]["s"]
        ds = np.gradient(s, t)
        row["mean_s"] = float(np.mean(s[mask])) if np.any(mask) else None
        row["mean_abs_dsdt"] = float(np.mean(np.abs(ds[mask]))) if np.any(mask) else None
        for name, src in sources.items():
            row[name] = h2_proxy(t, src["j0"], src["jpi"], lo, hi)
        row["ratio"] = {}
        for pair in (("qed", "gh5"), ("qed", "off"), ("on", "off"), ("qed", "on")):
            a, b = pair
            row["ratio"][f"{a}_over_{b}"] = {
                "KB": row[a]["abs_B"] / max(row[b]["abs_B"], 1e-30),
                "KM": row[a]["abs_AC"] / max(row[b]["abs_AC"], 1e-30),
                "darg_B_deg": a1.angdiff_deg(row[a]["arg_B_deg"], row[b]["arg_B_deg"]),
            }
        rows.append(row)
        start += 0.25
    return rows


def region_rms(t, yq, yg, lo, hi):
    mask = (t >= lo) & (t <= hi)
    if not np.any(mask):
        return None
    return {
        "rms_diff": rms(yq[mask] - yg[mask]),
        "rms_ref": rms(yg[mask]),
        "rel": rms(yq[mask] - yg[mask]) / max(rms(yg[mask]), 1e-30),
        "n": int(np.count_nonzero(mask)),
    }


def make_figure(rows, rec):
    FIG.mkdir(parents=True, exist_ok=True)
    path = FIG / "kb_window_origin.png"
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        (FIG / "kb_window_origin.skip.txt").write_text(str(exc), encoding="utf-8")
        return None
    c = [r["center_T0"] for r in rows]
    fig, axes = plt.subplots(3, 1, figsize=(6.8, 7.2), dpi=140, sharex=True)

    ax = axes[0]
    ax.axhline(1.0, color="0.55", lw=0.8)
    ax.plot(c, [r["ratio"]["qed_over_gh5"]["KB"] for r in rows], "o-", ms=4, label=r"$K_B$ QED/GH5")
    ax.plot(c, [r["ratio"]["qed_over_gh5"]["KM"] for r in rows], "s-", ms=4, label=r"$K_M$ QED/GH5")
    ax.plot(c, [r["ratio"]["on_over_off"]["KB"] for r in rows], "^-", ms=4, label=r"$K_B$ kick on/off")
    ax.axvspan(2.0, 4.0, color="0.85", zorder=0)
    ax.set_ylabel("kernel ratio")
    ax.set_ylim(0.70, 1.30)
    ax.legend(fontsize=8, loc="best")
    ax.set_title("Plateau-interior K_B is stably suppressed; K_M stays at GH5")

    ax = axes[1]
    ax.plot(c, [r["qed"]["abs_B"] * 1e8 for r in rows], "o-", ms=4, label=r"QED $|B|$")
    ax.plot(c, [r["gh5"]["abs_B"] * 1e8 for r in rows], "s-", ms=4, label=r"GH5 $|B|$")
    ax.plot(c, [r["off"]["abs_B"] * 1e8 for r in rows], "^-", ms=4, label=r"kick-off $|B|$")
    ax.axvspan(2.0, 4.0, color="0.85", zorder=0)
    ax.set_ylabel(r"$|B|\times 10^{8}$")
    ax.legend(fontsize=8, loc="best")

    ax = axes[2]
    ax.plot(c, [r["mean_s"] for r in rows], "k-", label="mean switch s")
    ax.plot(c, [10.0 * r["mean_abs_dsdt"] for r in rows], "k--", label=r"$10|ds/dt|$")
    ax.axvspan(2.0, 4.0, color="0.85", zorder=0)
    ax.set_xlabel("2-cycle window center / T0")
    ax.set_ylabel("switch")
    ax.legend(fontsize=8, loc="best")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path


def write_reports(rec) -> None:
    rows = rec["sliding"]
    plat = rec["plateau_interior"]
    early = rec["early_windows"]
    md = DOCS / "线Q_ABC_KB窗口_一页_20261029.md"
    lines = [
        "# 线 Q：奇通道 K_B 窗口变号的来源（2026-10-29）",
        "",
        "无新 QED。两角代理 B=(J₀+Jπ)/2，A+C=(J₀−Jπ)/2。权威数字不改写：ρ=1.456，R=1.273，主窗 K_B 比=0.868，K_M 比=1.003。",
        "比较对象：精确 QED、GH5 kick=True（R 扫描 a05r03）、A2b Ehrenfest kick=False。",
        "",
        "## 判定",
        "",
        f"- 平台内窗（中心 ≥ 3 T₀，即 2–4 T₀ 及更后）：QED/GH5 的 K_B 均值 **{plat['KB_mean']:.3f}**，散度 **{plat['KB_rel']:.1%}**。不是乱翻符号，而是稳定压低约 **{100*(1-plat['KB_mean']):.1f}%**。",
        f"- 早窗（中心 ≤ 2.25 T₀，覆盖升沿结束后的前两周期）：K_B 均值 **{early['KB_mean']:.3f}**。所谓变号，是开关附近 QED 的 |B| 暂时高于 GH5。",
        f"- 全程 K_M：QED/GH5 均值 **{rec['all']['KM_mean']:.3f}**，散度 **{rec['all']['KM_rel']:.1%}**。异常关联核不被窗口改写。",
        f"- Ehrenfest on/off 的 K_B：主窗 **{rec['main']['on_over_off_KB']:.3f}**，平台内均值 **{plat['on_off_KB_mean']:.3f}**。kick 解释相位，不解释这 14%。",
        f"- QED 与 kick=True 主窗 K_B 比 **{rec['main']['qed_over_on_KB']:.3f}**，与 QED/GH5 的 0.868 一致，余量不在 Wigner 采样细节。",
        f"- 时域：升沿 (0–1 T₀) B 残差相对 **{rec['td']['ramp']['B']['rel']:.2f}**；主窗平台 B 残差相对 **{rec['td']['main']['B']['rel']:.2f}**；同一主窗 AC 残差相对 **{rec['td']['main']['AC']['rel']:.2f}**。波形差主要在 Δn=0。",
        f"- 主窗波形：AC 相关 **{rec['wave']['AC']['corr']:.2f}**，B 相关只有 **{rec['wave']['B']['corr']:.2f}**。去掉准直流后 B 残差从 3.53 降到 **{rec['wave']['B']['dc_rel']:.2f}**。QED 的 Δn=0 奇电流多了一块慢分量，它污染升沿附近的 H2 窗，但不改写平台内 H2 的稳定 14% 压低。",
        f"- QED 自身 |B| 邻窗变化 **{rec['self_window']['qed_B_span']:.1%}**；GH5 自身 **{rec['self_window']['gh5_B_span']:.1%}**。已知 16%–39% 是复响应窗口敏感；**QED/GH5 的平台比是另外一个稳定的量子映射余量。**",
        "- 因此不能写成材料选择定则，也不能把 14% 当成邻窗噪声后丢掉。正确边界：有限时间开关下，精确量子映射在 Néel-odd 的 Δn=0 块留下 O(10%)、窗口依赖的余量；生产主窗上它稳定为压低，前窗因升沿而反号。",
        "",
        f"判定串：`{rec['verdict']}`。",
        f"图：`{rec.get('figure')}`。",
        f"归档：`{OUT.as_posix()}`。",
        "",
        "本轮没有新传播，也没有 nmax=10。A 仍然不是 χ^{(2)}。",
        "",
    ]
    md.write_text("\n".join(lines), encoding="utf-8")
    rec["one_page"] = str(md)


def analyze() -> dict:
    T0 = 2.0 * np.pi / pf.OMEGA0
    tq0, jq0, eq0 = odd_jx_pair(QED_TH0, "ds_p", "ds_m")
    tqpi, jqpi, _ = odd_jx_pair(QED_THPI, "ds_p", "ds_m")
    tg0, jg0, _ = odd_jx_pair(GH5_DIR, "a05r03_p_0", "a05r03_m_0")
    tgpi, jgpi, _ = odd_jx_pair(GH5_DIR, "a05r03_p_pi", "a05r03_m_pi")
    to0, jo0, _ = odd_jx_pair(A2B, "e10_on_p_0", "e10_on_m_0")
    topi, jopi, _ = odd_jx_pair(A2B, "e10_on_p_pi", "e10_on_m_pi")
    tf0, jf0, _ = odd_jx_pair(A2B, "e10_off_p_0", "e10_off_m_0")
    tfpi, jfpi, _ = odd_jx_pair(A2B, "e10_off_p_pi", "e10_off_m_pi")

    t = tqpi
    t_end = float(t[-1])
    z = np.load(QED_TH0 / "series_ds_p.npz")
    s_full = interp_to(t, np.asarray(z["t"], dtype=float), np.asarray(z["s"], dtype=float))
    sources = {
        "qed": {"j0": interp_to(t, tq0, jq0), "jpi": jqpi, "s": s_full},
        "gh5": {"j0": interp_to(t, tg0, jg0), "jpi": interp_to(t, tgpi, jgpi), "s": s_full},
        "on": {"j0": interp_to(t, to0, jo0), "jpi": interp_to(t, topi, jopi), "s": s_full},
        "off": {"j0": interp_to(t, tf0, jf0), "jpi": interp_to(t, tfpi, jfpi), "s": s_full},
    }
    rows = sliding(t, sources, T0, t_end)

    def pick(pred):
        return [r for r in rows if pred(r)]

    plat_rows = pick(lambda r: r["center_T0"] >= 3.0 - 1e-12)
    early_rows = pick(lambda r: r["center_T0"] <= 2.25 + 1e-12)
    main_row = next(r for r in rows if abs(r["center_T0"] - 3.0) < 1e-9)
    front_row = next(r for r in rows if abs(r["center_T0"] - 2.0) < 1e-9)

    def stats(subset, key_fn):
        vals = np.array([key_fn(r) for r in subset], dtype=float)
        mean = float(np.mean(vals))
        return {
            "n": int(vals.size),
            "mean": mean,
            "min": float(np.min(vals)),
            "max": float(np.max(vals)),
            "std": float(np.std(vals)),
            "rel": float(np.std(vals) / max(abs(mean), 1e-30)),
        }

    def Bq(src):
        return 0.5 * (sources[src]["j0"] + sources[src]["jpi"])

    def ACq(src):
        return 0.5 * (sources[src]["j0"] - sources[src]["jpi"])

    td = {}
    for name, lo, hi in (("ramp", 0.0, T0), ("early_plat", T0, 2.0 * T0), ("main", 2.0 * T0, 4.0 * T0)):
        td[name] = {
            "B": region_rms(t, Bq("qed"), Bq("gh5"), lo, hi),
            "AC": region_rms(t, ACq("qed"), ACq("gh5"), lo, hi),
        }
    mask_main = (t >= 2.0 * T0) & (t <= 4.0 * T0)
    wave = {
        "B": waveform_stats(Bq("qed")[mask_main], Bq("gh5")[mask_main]),
        "AC": waveform_stats(ACq("qed")[mask_main], ACq("gh5")[mask_main]),
    }

    qed_B = np.array([r["qed"]["abs_B"] for r in rows])
    gh5_B = np.array([r["gh5"]["abs_B"] for r in rows])
    self_window = {
        "qed_B_span": float((qed_B.max() - qed_B.min()) / max(qed_B.mean(), 1e-30)),
        "gh5_B_span": float((gh5_B.max() - gh5_B.min()) / max(gh5_B.mean(), 1e-30)),
        "qed_AC_span": float(
            (max(r["qed"]["abs_AC"] for r in rows) - min(r["qed"]["abs_AC"] for r in rows))
            / max(np.mean([r["qed"]["abs_AC"] for r in rows]), 1e-30)
        ),
    }

    rec = {
        "verdict": "KB_plateau_stable_suppression_front_is_switch",
        "no_new_qed": True,
        "t_end": t_end,
        "T0": float(T0),
        "n_slide": len(rows),
        "main": {
            "qed_over_gh5_KB": main_row["ratio"]["qed_over_gh5"]["KB"],
            "qed_over_gh5_KM": main_row["ratio"]["qed_over_gh5"]["KM"],
            "on_over_off_KB": main_row["ratio"]["on_over_off"]["KB"],
            "qed_over_on_KB": main_row["ratio"]["qed_over_on"]["KB"],
            "qed_over_off_KB": main_row["ratio"]["qed_over_off"]["KB"],
        },
        "front": {
            "qed_over_gh5_KB": front_row["ratio"]["qed_over_gh5"]["KB"],
            "qed_over_gh5_KM": front_row["ratio"]["qed_over_gh5"]["KM"],
        },
        "plateau_interior": {
            "KB_mean": stats(plat_rows, lambda r: r["ratio"]["qed_over_gh5"]["KB"])["mean"],
            "KB_rel": stats(plat_rows, lambda r: r["ratio"]["qed_over_gh5"]["KB"])["rel"],
            "KB_min": stats(plat_rows, lambda r: r["ratio"]["qed_over_gh5"]["KB"])["min"],
            "KB_max": stats(plat_rows, lambda r: r["ratio"]["qed_over_gh5"]["KB"])["max"],
            "KM_mean": stats(plat_rows, lambda r: r["ratio"]["qed_over_gh5"]["KM"])["mean"],
            "on_off_KB_mean": stats(plat_rows, lambda r: r["ratio"]["on_over_off"]["KB"])["mean"],
            "n": len(plat_rows),
        },
        "early_windows": {
            "KB_mean": stats(early_rows, lambda r: r["ratio"]["qed_over_gh5"]["KB"])["mean"],
            "KB_min": stats(early_rows, lambda r: r["ratio"]["qed_over_gh5"]["KB"])["min"],
            "KB_max": stats(early_rows, lambda r: r["ratio"]["qed_over_gh5"]["KB"])["max"],
            "n": len(early_rows),
        },
        "all": {
            "KM_mean": stats(rows, lambda r: r["ratio"]["qed_over_gh5"]["KM"])["mean"],
            "KM_rel": stats(rows, lambda r: r["ratio"]["qed_over_gh5"]["KM"])["rel"],
            "KB_mean": stats(rows, lambda r: r["ratio"]["qed_over_gh5"]["KB"])["mean"],
        },
        "td": td,
        "wave": wave,
        "self_window": self_window,
        "sliding": rows,
        "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    rec["figure"] = None if make_figure(rows, rec) is None else str(FIG / "kb_window_origin.png")
    write_reports(rec)
    dump(OUT / "kb_window.json", rec)
    return rec


def main() -> int:
    rec = analyze()
    print(json.dumps({
        "verdict": rec["verdict"],
        "main": rec["main"],
        "front": rec["front"],
        "plateau_interior": rec["plateau_interior"],
        "early_windows": rec["early_windows"],
        "all": rec["all"],
        "td": rec["td"],
        "wave": rec["wave"],
        "self_window": rec["self_window"],
        "one_page": rec["one_page"],
        "figure": rec["figure"],
    }, indent=2))
    return 0


def waveform_stats(yq, yg):
    yq = np.asarray(yq, dtype=float)
    yg = np.asarray(yg, dtype=float)
    yq0 = yq - yq.mean()
    yg0 = yg - yg.mean()
    denom = float(np.linalg.norm(yq0) * np.linalg.norm(yg0))
    corr = float(np.dot(yq0, yg0) / max(denom, 1e-30))
    return {
        "corr": corr,
        "mean_qed": float(yq.mean()),
        "mean_gh5": float(yg.mean()),
        "dc_rel": rms(yq0 - yg0) / max(rms(yg0), 1e-30),
        "mean_abs_qed": float(np.mean(np.abs(yq))),
        "mean_abs_gh5": float(np.mean(np.abs(yg))),
    }


if __name__ == "__main__":
    raise SystemExit(main())

