#!/usr/bin/env python3
"""Q3 Fock-probe kernel reconstruction. No propagator.

Requires the eight production series. Reconstructs K00, K22, K02, K20
and compares v_n = K22-K00 against v_c = K02 (and v_barc = K20).
Does not rewrite fourangle_fit.json. Does not start Q2.
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import q8_m0_qinput_1600nm_protocolA_qed_postprocess_20261102 as pp
import q8_m0_qinput_1600nm_protocolA_qed_q3fock_20261103 as runner
from qinput_spectrum_reanalysis_20261028 import packed, spectrum

ROOT = runner.ROOT
OUT = runner.OUT
FIT = pp.FIT_PATH
FROZEN_SHA = "17a8e37ec3e0ab54096ff4edd1440c5a3f0412b09450c2fd8960cbc00d8e69db"
OMEGA0 = pp.OMEGA0
PERIOD = pp.PERIOD
WINDOWS = pp.WINDOWS
CHANNELS = (
    ("h2_odd_x", 2, "odd", "x"),
    ("h2_odd_y", 2, "odd", "y"),
    ("h3_odd_x", 3, "odd", "x"),
    ("h3_odd_y", 3, "odd", "y"),
)
ABS_FLOOR = 1e-12
REL_FLOOR = 1e-4
EXPLORE_DEG = 10.0
PAIRS = {
    "fock0": ("p_0", "m_0"),
    "fock2": ("p_2", "m_2"),
    "plus": ("p_plus", "m_plus"),
    "iplus": ("p_i", "m_i"),
}


def dump_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def load_series(tag: str):
    with np.load(OUT / f"series_{tag}.npz", allow_pickle=False) as z:
        t = np.asarray(z["t"], float)
        j = np.column_stack([np.asarray(z["Jx"]), np.asarray(z["Jy"])]).astype(complex, copy=False)
    return t, j


def pair_currents(ptag: str, mtag: str):
    tp, jp = load_series(ptag)
    tm, jm = load_series(mtag)
    if tp.shape != tm.shape or not np.allclose(tp, tm, atol=0.0, rtol=0.0):
        raise RuntimeError(f"time grid mismatch {ptag}/{mtag}")
    return tp, jp, jm


def z_channel(t, jp, jm, window: str, q: int, parity: str, axis: str, method="linear_exact"):
    je = 0.5 * (jp + jm)
    jo = 0.5 * (jp - jm)
    j = jo if parity == "odd" else je
    a, b = WINDOWS[window]
    spec = spectrum(t, j, a * PERIOD, b * PERIOD, np.arange(1, 7, dtype=float),
                    method=method, omega=OMEGA0)
    iq = int(q) - 1
    ia = 0 if axis == "x" else 1
    return complex(spec[iq, ia])


def reconstruct(j00, j22, jplus, ji):
    s = 2.0 * jplus - j00 - j22
    d = 2.0 * ji - j00 - j22
    k00 = j00
    k22 = j22
    k02 = 0.5 * (s - 1j * d)
    k20 = 0.5 * (s + 1j * d)
    jplus_pred = 0.5 * (k00 + k22 + k02 + k20)
    ji_pred = 0.5 * (k00 + k22) + 0.5j * (k02 - k20)
    return {
        "K00": k00, "K22": k22, "K02": k02, "K20": k20,
        "S": s, "D": d,
        "Jplus_pred": jplus_pred, "Ji_pred": ji_pred,
        "Jplus_resid": jplus - jplus_pred,
        "Ji_resid": ji - ji_pred,
    }


def vec4(chs: dict, key: str) -> np.ndarray:
    return np.array([chs[name][key] for name, *_ in CHANNELS], complex)


def herm_inner(u, v) -> complex:
    return complex(np.vdot(u, v))


def nrm(u) -> float:
    return float(np.linalg.norm(u))


def angle_record(vn, vc, scale: float) -> dict:
    nn, nc = nrm(vn), nrm(vc)
    floor = max(ABS_FLOOR, REL_FLOOR * max(scale, 1e-30))
    if nn <= floor or nc <= floor:
        return {
            "decidable": False,
            "verdict": "direction_undecidable",
            "why": "norm near numerical floor",
            "norm_vn": nn, "norm_vc": nc, "floor": floor,
            "cos": None, "phi_deg": None,
            "explore_ge_10deg": None,
        }
    cos = abs(herm_inner(vn, vc)) / (nn * nc)
    cos = float(np.clip(cos, 0.0, 1.0))
    phi = float(np.degrees(np.arccos(cos)))
    return {
        "decidable": True,
        "verdict": "exploratory_rotated" if phi >= EXPLORE_DEG else "exploratory_near_collinear",
        "norm_vn": nn, "norm_vc": nc, "floor": floor,
        "cos": cos, "phi_deg": phi,
        "explore_ge_10deg": bool(phi >= EXPLORE_DEG),
    }


def scale_channels(k00, k22, k02, k20) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    out = []
    for i in range(4):
        m = max(abs(k00[i]), abs(k22[i]), abs(k02[i]), abs(k20[i]), 1e-30)
        out.append((k00[i] / m, k22[i] / m, k02[i] / m, k20[i] / m))
    arr = np.array(out, complex)
    return arr[:, 0], arr[:, 1], arr[:, 2], arr[:, 3]


def pack_vec(v) -> list:
    return [packed(complex(x)) for x in v]


def analyze_window(window: str, series: dict) -> dict:
    chs = {}
    for cname, q, parity, axis in CHANNELS:
        j00 = z_channel(*series["fock0"], window, q, parity, axis)
        j22 = z_channel(*series["fock2"], window, q, parity, axis)
        jplus = z_channel(*series["plus"], window, q, parity, axis)
        ji = z_channel(*series["iplus"], window, q, parity, axis)
        rec = reconstruct(j00, j22, jplus, ji)
        rec.update({"J00": j00, "J22": j22, "Jplus": jplus, "Ji": ji})
        chs[cname] = rec
    k00 = vec4(chs, "K00")
    k22 = vec4(chs, "K22")
    k02 = vec4(chs, "K02")
    k20 = vec4(chs, "K20")
    vn = k22 - k00
    vc = k02
    vbar = k20
    scale = max(float(np.max(np.abs(np.stack([k00, k22, k02, k20])))), 1e-30)
    k00s, k22s, k02s, k20s = scale_channels(k00, k22, k02, k20)
    raw = {
        "vn_vs_vc": angle_record(vn, vc, scale),
        "vn_vs_vbarc": angle_record(vn, vbar, scale),
        "vc_vs_vbarc": angle_record(vc, vbar, scale),
    }
    scaled = {
        "vn_vs_vc": angle_record(k22s - k00s, k02s, 1.0),
        "vn_vs_vbarc": angle_record(k22s - k00s, k20s, 1.0),
    }
    h2 = {
        "vn_vs_vc": angle_record(vn[:2], vc[:2], scale),
        "vn_vs_vbarc": angle_record(vn[:2], vbar[:2], scale),
    }
    h3 = {
        "vn_vs_vc": angle_record(vn[2:], vc[2:], scale),
        "vn_vs_vbarc": angle_record(vn[2:], vbar[2:], scale),
    }
    resid = {
        cname: {
            "Jplus_resid": packed(chs[cname]["Jplus_resid"]),
            "Ji_resid": packed(chs[cname]["Ji_resid"]),
            "abs_Jplus_resid": float(abs(chs[cname]["Jplus_resid"])),
            "abs_Ji_resid": float(abs(chs[cname]["Ji_resid"])),
        }
        for cname, *_ in CHANNELS
    }
    packed_ch = {}
    for cname, *_ in CHANNELS:
        packed_ch[cname] = {k: packed(chs[cname][k]) for k in
                            ("J00", "J22", "Jplus", "Ji", "K00", "K22", "K02", "K20", "S", "D")}
    return {
        "lo_over_T": WINDOWS[window][0],
        "hi_over_T": WINDOWS[window][1],
        "channels": packed_ch,
        "vectors": {
            "K00": pack_vec(k00), "K22": pack_vec(k22),
            "K02": pack_vec(k02), "K20": pack_vec(k20),
            "vn": pack_vec(vn), "vc": pack_vec(vc), "vbarc": pack_vec(vbar),
            "norm_K00": nrm(k00), "norm_K22": nrm(k22),
            "norm_vn": nrm(vn), "norm_vc": nrm(vc), "norm_vbarc": nrm(vbar),
            "max_abs_K": scale,
        },
        "reconstruction_resid": resid,
        "angles": {
            "raw_C4": raw,
            "channel_scaled_C4": scaled,
            "H2_only": h2,
            "H3_only": h3,
        },
    }


def window_verdict(win: dict) -> str:
    row = win["angles"]["raw_C4"]["vn_vs_vc"]
    if not row["decidable"]:
        return "undecidable"
    return "ge_10" if row["explore_ge_10deg"] else "lt_10"


def combined_verdict(windows: dict) -> dict:
    labs = ("front", "main", "back")
    v = {w: window_verdict(windows[w]) for w in labs}
    main = v["main"]
    same = all(v[w] == main for w in labs)
    if main == "undecidable" or not same:
        action = "record_and_stop"
        label = "direction_undecidable_or_window_dependent"
    elif main == "lt_10":
        action = "close_scalar_reweight"
        label = "exploratory_near_collinear_all_windows"
    else:
        action = "record_needs_new_authorization"
        label = "exploratory_rotated_all_windows"
    return {
        "per_window": v,
        "windows_agree": same,
        "label": label,
        "action": action,
        "note": "10 deg is exploratory only; not a quantum-advantage bound. 2x2 subspace only.",
    }


def assert_frozen() -> str:
    got = pp.sha256_file(FIT)
    if got != FROZEN_SHA:
        raise RuntimeError(f"frozen fit SHA changed: {got}")
    return got


def cz(z) -> complex:
    if isinstance(z, dict):
        return complex(float(z["re"]), float(z["im"]))
    return complex(z)


def fmt_z(z, scale: float = 1.0, digits: int = 4) -> str:
    w = cz(z) / float(scale)
    return f"{w.real:+.{digits}f}{w.imag:+.{digits}f}i"


def fmt_phi(row: dict) -> str:
    if not row.get("decidable"):
        return "不可判定"
    return f"{float(row['phi_deg']):.3f}°"


def fmt_verdict(row: dict) -> str:
    if not row.get("decidable"):
        return "direction_undecidable"
    return str(row.get("verdict"))


def collect_jobs() -> list:
    rows = []
    for tag in runner.STAGE1_TAGS:
        rec = runner.load_job(tag) or {}
        defn = rec.get("definition") or {}
        ledger = rec.get("ledger") or {}
        rows.append({
            "tag": tag,
            "status": rec.get("status"),
            "probe": rec.get("probe") or defn.get("probe"),
            "probe_label": rec.get("probe_label") or defn.get("probe_label"),
            "neel": rec.get("neel"),
            "steps": ledger.get("steps_completed"),
            "max_top_fock": rec.get("max_top_fock"),
            "max_top_two": rec.get("max_top_two"),
            "max_n_ph": rec.get("max_n_ph"),
            "end_n_ph": rec.get("end_n_ph"),
            "max_abs_norm_minus_1": rec.get("max_abs_norm_minus_1"),
            "wall_seconds": rec.get("wall_seconds"),
            "fock_alarm_0p05": rec.get("fock_alarm_0p05"),
            "not_DS": rec.get("not_DS"),
            "operator_order": defn.get("operator_order"),
            "omega0": rec.get("omega0"),
            "g": rec.get("g"),
            "n_max": rec.get("n_max"),
            "used_pf_OMEGA0": rec.get("used_pf_OMEGA0"),
        })
    return rows


def make_figure(payload: dict, path: Path) -> None:
    import matplotlib.pyplot as plt

    labs = ("front", "main", "back")
    kinds = (
        ("raw_C4", r"raw $\mathbb{C}^4$"),
        ("channel_scaled_C4", "channel-scaled"),
        ("H2_only", "H2 only"),
        ("H3_only", "H3 only"),
    )
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.0), dpi=140)
    x = np.arange(len(labs))
    width = 0.18
    ax = axes[0]
    for i, (key, lab) in enumerate(kinds):
        ys = []
        for w in labs:
            row = payload["windows"][w]["angles"][key]["vn_vs_vc"]
            ys.append(float(row["phi_deg"]) if row.get("decidable") else np.nan)
        ax.bar(x + (i - 1.5) * width, ys, width, label=lab)
    ax.axhline(EXPLORE_DEG, color="0.35", ls="--", lw=1.0, label=r"$10^\circ$ exploratory")
    ax.set_xticks(x)
    ax.set_xticklabels(labs)
    ax.set_ylabel(r"$\phi(\mathbf{v}_n,\mathbf{v}_c)$ (deg)")
    ax.set_title(r"occupation vs coherence direction")
    ax.legend(fontsize=8, loc="best")
    ax.set_ylim(0.0, max(12.0, ax.get_ylim()[1]))
    ax = axes[1]
    names = [r"$\|K_{00}\|$", r"$\|\mathbf{v}_n\|$", r"$\|\mathbf{v}_c\|$", r"$\|\mathbf{v}_{\bar c}\|$"]
    keys = ("norm_K00", "norm_vn", "norm_vc", "norm_vbarc")
    for i, w in enumerate(labs):
        ys = [float(payload["windows"][w]["vectors"][k]) for k in keys]
        ax.bar(np.arange(4) + (i - 1) * 0.25, ys, 0.22, label=w)
    ax.set_xticks(np.arange(4))
    ax.set_xticklabels(names)
    ax.set_yscale("log")
    ax.set_ylabel("vector norm")
    ax.set_title("kernel vector sizes")
    ax.legend(fontsize=8)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)


def write_report(payload: dict, path: Path) -> None:
    main = payload["windows"]["main"]
    comb = payload["combined"]
    raw = main["angles"]["raw_C4"]["vn_vs_vc"]
    ch_names = payload["channel_order"]
    scale5 = 1.0e-5
    jobs = payload.get("jobs") or []
    leak = float(payload["constant_leak_abs_max"])
    fig_rel = payload.get("figure_rel")

    def ktab(win_name: str, key: str) -> str:
        chs = payload["windows"][win_name]["channels"]
        return " / ".join(fmt_z(chs[n][key], scale5) for n in ch_names)

    def ang_row(win_name: str, kind: str, pair: str) -> dict:
        return payload["windows"][win_name]["angles"][kind][pair]

    lines = [
        "# Q3 最小 Fock 探针核重构（2026-11-03）",
        "",
        "状态必须写成：**8/8 精确 QED 过闸后，已重构** "
        "\\(\\{K_{00},K_{22},K_{02},K_{20}\\}\\)；"
        "主比较是 \\(\\mathbf v_n=K_{22}-K_{00}\\) 对 \\(\\mathbf v_c=K_{02}\\)，"
        "不是 \\(K_{00}\\) 对 \\(K_{02}\\)。"
        "本文件只做后处理，不改冻结 A/B/C，不开 Q2 / 正 P / 纯 BSV / CrI₃。",
        "",
        f"- 冻结拟合 SHA 未改：`{payload['fit_sha256']}`",
        "- 探针：实验室 Fock 纯态 "
        "\\(|0\\rangle,\\ |2\\rangle,\\ (|0\\rangle+|2\\rangle)/\\sqrt{2},\\ (|0\\rangle+i|2\\rangle)/\\sqrt{2}\\)",
        "- 模型：M0 单 \\(k/-k\\)，\\(\\omega=0.775\\,\\mathrm{eV}\\)，"
        "\\(g=0.005\\,\\mathrm{\\AA}^{-1}\\) 不缩放，\\(n_{\\max}=8\\)，1+4+1，`linear_exact`",
        f"- 常数信号假 H2 泄漏 \\(\\lvert z\\rvert_{{\\max}}={leak:.3e}\\)",
        "- 10° 只是探索性阈值，不是量子优势界限。",
        "- 四个探针只覆盖 \\(\\{|0\\rangle,|2\\rangle\\}\\) 子空间。",
        "",
        "## 0. 八轴闸门",
        "",
        "| 轴 | 探针 | Néel | 步数 | \\(\\max P_8\\) | \\(\\max(P_7+P_8)\\) | "
        "\\(\\max\\langle n\\rangle\\) | 范数误差 | 报警 |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for rec in jobs:
        lines.append(
            "| `{tag}` | {lab} | {neel:+d} | {steps} | {p8:.3e} | {p78:.3e} | "
            "{nph:.6f} | {nrm:.3e} | {alarm} |".format(
                tag=rec["tag"],
                lab=rec.get("probe_label") or rec.get("probe"),
                neel=int(rec["neel"]),
                steps=int(rec["steps"]),
                p8=float(rec["max_top_fock"]),
                p78=float(rec["max_top_two"]),
                nph=float(rec["max_n_ph"]),
                nrm=float(rec["max_abs_norm_minus_1"]),
                alarm="yes" if rec.get("fock_alarm_0p05") else "no",
            )
        )
    lines += [
        "",
        "输入确认：`operator_order` 全部是 Fock 探针，不是 \\(D(\\alpha)S(\\xi)|0\\rangle\\)。"
        "\\(|0\\rangle\\) 轴的 \\(\\max P_8\\) 在数值地板；\\(|2\\rangle\\) 轴平均光子数钉在 2。"
        "本轮没有用 fockaudit 的 `n8_p_q` 代替任何 Q3 轴。",
        "",
        "## 1. 重构公式（外审后的判据）",
        "",
        "Néel-odd 线性读出：",
        "",
        "\\[",
        "K_{00}=J_{00},\\qquad K_{22}=J_{22},",
        "\\]",
        "\\[",
        "S=2J_{+}-J_{00}-J_{22},\\qquad D=2J_{i}-J_{00}-J_{22},",
        "\\]",
        "\\[",
        "K_{02}=(S-iD)/2,\\qquad K_{20}=(S+iD)/2.",
        "\\]",
        "",
        "子空间内任意输入：",
        "",
        "\\[",
        r"\mathcal J =K_{00}+\rho_{22}(K_{22}-K_{00})+\rho_{02}K_{20}+\rho_{20}K_{02}.",
        "\\]",
        "",
        "因此主比较必须是光子数变化方向对相干方向：",
        "",
        "\\[",
        r"\mathbf v_n=\mathbf K_{22}-\mathbf K_{00},\qquad \mathbf v_c=\mathbf K_{02},\qquad \mathbf v_{\bar c}=\mathbf K_{20}.",
        "\\]",
        "",
        "通道顺序固定为 \\((J_{2x}^{o},J_{2y}^{o},J_{3x}^{o},J_{3y}^{o})\\)。"
        "正频核一般非 Hermitian，所以 \\(K_{02}\\) 与 \\(K_{20}\\) 都保留。"
        "拟合 \\(A,B,C\\) **不是** 这些矩阵元。",
        "",
        "## 2. 主窗四个复矩阵元",
        "",
        f"单位：下表已除以 \\(10^{{-5}}\\)。通道顺序：`{' , '.join(ch_names)}`。",
        "",
        "| 矩阵元 | \\(J_{2x}^{o}\\) / \\(J_{2y}^{o}\\) / \\(J_{3x}^{o}\\) / \\(J_{3y}^{o}\\) |",
        "|---|---|",
        f"| \\(K_{{00}}\\) | {ktab('main', 'K00')} |",
        f"| \\(K_{{22}}\\) | {ktab('main', 'K22')} |",
        f"| \\(K_{{02}}\\) | {ktab('main', 'K02')} |",
        f"| \\(K_{{20}}\\) | {ktab('main', 'K20')} |",
        "",
        "对应向量范数：",
        "",
        f"- \\(\\|K_{{00}}\\|={main['vectors']['norm_K00']:.6e}\\)",
        f"- \\(\\|K_{{22}}\\|={main['vectors']['norm_K22']:.6e}\\)",
        f"- \\(\\|\\mathbf v_n\\|={main['vectors']['norm_vn']:.6e}\\)",
        f"- \\(\\|\\mathbf v_c\\|={main['vectors']['norm_vc']:.6e}\\)",
        f"- \\(\\|\\mathbf v_{{\\bar c}}\\|={main['vectors']['norm_vbarc']:.6e}\\)",
        f"- 本窗 \\(\\max\\lvert K\\rvert={main['vectors']['max_abs_K']:.6e}\\)，"
        f"地板 \\(\\max(10^{{-12}},10^{{-4}}\\max\\lvert K\\rvert)\\)",
        "",
        "## 3. 主窗方向：vn vs vc 与 K20",
        "",
        "| 归一化 | phi(vn, vc) | 判定 | phi(vn, K20) | 判定 |",
        "|---|---|---|---|---|",
    ]
    for kind, lab in (
        ("raw_C4", "raw C^4"),
        ("channel_scaled_C4", "按通道 max|K| 缩放"),
        ("H2_only", "仅 H2"),
        ("H3_only", "仅 H3"),
    ):
        a = ang_row("main", kind, "vn_vs_vc")
        b = ang_row("main", kind, "vn_vs_vbarc")
        lines.append(
            f"| {lab} | {fmt_phi(a)} | `{fmt_verdict(a)}` | {fmt_phi(b)} | `{fmt_verdict(b)}` |"
        )
    raw_bar = main["angles"]["raw_C4"]["vn_vs_vbarc"]
    raw_cc = main["angles"]["raw_C4"]["vc_vs_vbarc"]
    lines += [
        "",
        f"主窗 raw C^4：decidable=`{raw['decidable']}`，"
        f"phi(vn,vc)={fmt_phi(raw)}，verdict=`{raw.get('verdict')}`；"
        f"phi(vn,K20)={fmt_phi(raw_bar)}；"
        f"phi(K02,K20)={fmt_phi(raw_cc)}。",
        "",
        "解释规则：范数近地板则写**方向不可判定**，不得标共线。"
        "10° 只用于预先写下的探索性分叉，不是量子优势。",
        "",
        "## 4. 三窗方向是否一致",
        "",
        "| 窗 | 区间 | raw phi(vn,vc) | 判定 | 缩放后 | 仅 H2 | 仅 H3 |",
        "|---|---|---|---|---|---|---|",
    ]
    for w, interval in (("front", "[T0,3T0]"), ("main", "[2T0,4T0]"), ("back", "[3T0,5T0]")):
        r = ang_row(w, "raw_C4", "vn_vs_vc")
        s = ang_row(w, "channel_scaled_C4", "vn_vs_vc")
        h2 = ang_row(w, "H2_only", "vn_vs_vc")
        h3 = ang_row(w, "H3_only", "vn_vs_vc")
        lines.append(
            f"| {w} | {interval} | {fmt_phi(r)} | `{fmt_verdict(r)}` | "
            f"{fmt_phi(s)} | {fmt_phi(h2)} | {fmt_phi(h3)} |"
        )
    lines += [
        "",
        f"- front/main/back 主判据：`{comb['per_window']}`",
        f"- windows_agree: `{comb['windows_agree']}`",
        f"- label: `{comb['label']}`",
        f"- action: `{comb['action']}`",
        "",
        "三窗共用同一套八条轨迹，而且窗口重叠，所以这不是三次独立实验。"
        "它们只回答：方向判据有没有被某一个矩形窗偶然选出来。",
        "",
        "## 5. 重构残差",
        "",
        "由四个探针线性重构 K 后，应精确回到 J+ 与 Ji。"
        "下表是主窗残差绝对值，只用于数值自洽，不是拟合优度。",
        "",
        "| 通道 | |Jplus resid| | |Ji resid| |",
        "|---|---|---|",
    ]
    resid = main["reconstruction_resid"]
    for name in ch_names:
        lines.append(
            f"| `{name}` | {resid[name]['abs_Jplus_resid']:.3e} | {resid[name]['abs_Ji_resid']:.3e} |"
        )
    lines += [
        "",
        "## 6. 三层结论",
        "",
        "**已经支持**",
        "",
        "- 四个合法纯态 × 两个 Néel 磁畴足以重构所选 2×2 子空间的四个复矩阵元。",
        "- 正频核不必 Hermitian：K02 与 K20 作为独立复数保留。",
        "- 主比较已按外审改为光子数响应 K22−K00 对相干响应 K02。",
        "",
        "**有证据，但只到这个子空间**",
        "",
        f"- 主窗 raw 方向：`{raw.get('verdict')}`，phi={fmt_phi(raw)}。",
        f"- 三窗主判据是否一致：`{comb['windows_agree']}`，动作 `{comb['action']}`。",
        "- 若 H2 与 H3 的夹角明显不同，这与 Q1 中 H2-odd 的 rho≈R、H3-odd 偏离 R 同方向："
        "不同谐波阶不是同一组标量权重。这仍不是量子优势。",
        "",
        "**尚未证明**",
        "",
        "- 这个 2×2 子空间并不自动主导原始位移压缩态。",
        "- 方向旋转或不旋转，都不能代替正 P 对照。",
        "- 没有 nmax=8→10 或 dt/2 的观测量收敛。",
        "- 没有输出 g^(2)、纯 BSV、真实 CrI3。",
        "",
        "## 7. 本轮明确停在哪里",
        "",
        "Q3 最小核重构完成。**不要开 Q2，不要增加探针。** "
        "不升 nmax，不减 dt，不调 g，不跑纯 BSV，不做正 P，"
        "不碰 M52/full112，不 git commit/push。",
        "",
        "无方向旋转不是失败：它意味着在这个子空间里，相干通道看起来像光子数通道的复标量重加权。"
        "有方向旋转也只是进入下一步的候选，仍需新授权。",
        "",
    ]
    if fig_rel:
        lines += [
            f"![Q3 kernel angles]({fig_rel})",
            "",
            "图注：左图是四种归一化下 phi(vn,vc)；虚线是探索性 10°。"
            "右图是核向量范数。三窗共用同一套轨迹。",
            "",
        ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    sha = assert_frozen()
    st = runner.analyze_status()
    if not st.get("stage1_complete"):
        raise RuntimeError(f"q3fock stage1 incomplete: {st.get('n_ok_no_alarm')}/8")
    if st.get("alarm_tags"):
        raise RuntimeError(f"fock alarm tags {st.get('alarm_tags')}")
    series = {}
    for probe, (ptag, mtag) in PAIRS.items():
        t, jp, jm = pair_currents(ptag, mtag)
        series[probe] = (t, jp, jm)
    leak = spectrum(series["fock0"][0], np.ones((series["fock0"][0].size, 2)),
                    2 * PERIOD, 4 * PERIOD, np.array([2.0]),
                    method="linear_exact", omega=OMEGA0)
    windows = {w: analyze_window(w, series) for w in ("front", "main", "back")}
    payload = {
        "schema": "1600nm_protocolA_qed_q3fock_kernel_v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "fit_sha256": sha,
        "did_not_rewrite_fourangle_fit": True,
        "did_not_propagate": True,
        "did_not_start_Q2": True,
        "primary_comparison": "vn=K22-K00 vs vc=K02; also report K20",
        "not_K00_vs_K02": True,
        "explore_deg": EXPLORE_DEG,
        "abs_floor": ABS_FLOOR,
        "rel_floor": REL_FLOOR,
        "constant_leak_abs_max": float(np.max(np.abs(leak))),
        "channel_order": [c[0] for c in CHANNELS],
        "windows": windows,
        "combined": combined_verdict(windows),
        "not_quantum_advantage": True,
        "subspace_is_2x2_only": True,
        "jobs": collect_jobs(),
    }
    fig_path = OUT / "analysis_q3fock_20261103" / "figures" / "fig_q3fock_angles.png"
    make_figure(payload, fig_path)
    payload["figure_rel"] = str(fig_path).replace("\\", "/")
    out_json = OUT / "analysis_q3fock_20261103" / "kernel.json"
    dump_json(out_json, payload)
    md = ROOT / "docs/04_BSV量子光/线Q_Q3最小Fock探针_核重构_20261103.md"
    write_report(payload, md)
    print("Q3FOCK_ANALYZE_OK", sha)
    print("combined", json.dumps(payload["combined"], ensure_ascii=False))
    raw = windows["main"]["angles"]["raw_C4"]["vn_vs_vc"]
    print("main_raw_vn_vc", raw)
    print("figure", payload.get("figure_rel"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
