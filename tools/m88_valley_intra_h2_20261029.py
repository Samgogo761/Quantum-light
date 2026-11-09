"""Decompose existing 1600 nm coherent H2-odd currents.

No new SBE. Reuse B1 Fourier protocol. Explain the M88 2x2->4x4
phase jump using valley (K/K') and intra/inter files already on disk.
"""
from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import b1_1600nm_model_compare_20261029 as b1

OUT = ROOT / "results/q_line/m88_valley_intra_20261029"
DOCS = ROOT / "docs/04_BSV量子光"
FIG = OUT / "figures"
TWOPI = b1.TWOPI

PAIRS = {
    "m52_2x2": {
        "plus": ROOT / "results/q_line/m52_1600nm_fewk_coherent_20261028/plusN",
        "minus": ROOT / "results/q_line/m52_1600nm_fewk_coherent_20261028/minusN",
        "model": "m52",
        "kgrid": "2x2",
    },
    "m52_4x4": {
        "plus": ROOT / "results/q_line/m52_1600nm_fewk_coherent_20261028/k4_plusN",
        "minus": ROOT / "results/q_line/m52_1600nm_fewk_coherent_20261028/k4_minusN",
        "model": "m52",
        "kgrid": "4x4",
    },
    "m88_2x2": {
        "plus": ROOT / "results/q_line/b1_1600nm_model_compare_20261029/m88_plusN",
        "minus": ROOT / "results/q_line/b1_1600nm_model_compare_20261029/m88_minusN",
        "model": "m88",
        "kgrid": "2x2",
    },
    "m88_4x4": {
        "plus": ROOT / "results/q_line/m88_1600nm_4x4_20261029/m88_plusN",
        "minus": ROOT / "results/q_line/m88_1600nm_4x4_20261029/m88_minusN",
        "model": "m88",
        "kgrid": "4x4",
    },
    "full112_2x2": {
        "plus": ROOT / "results/q_line/b1_1600nm_model_compare_20261029/full112_plusN",
        "minus": ROOT / "results/q_line/b1_1600nm_model_compare_20261029/full112_minusN",
        "model": "full112",
        "kgrid": "2x2",
    },
}


def dump(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    tmp.replace(path)


def progress(msg: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    line = f"{time.strftime('%H:%M:%S')} {msg}"
    with (OUT / "progress.txt").open("a", encoding="utf-8") as fh:
        fh.write(line + "\n")
    print(line, flush=True)


def read_cols(path: Path, cols: dict[str, int]):
    rows = []
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        p = line.split()
        need = max(cols.values()) + 1
        if len(p) < need:
            continue
        rec = {"t": float(p[1])}
        for name, idx in cols.items():
            rec[name] = float(p[idx])
        rows.append(rec)
    t = np.array([r["t"] for r in rows], dtype=float)
    data = {name: np.array([r[name] for r in rows], dtype=float) for name in cols}
    return t, data


def odd_h2(t, jx_p, jy_p, jx_m, jy_m, omega0: float) -> dict:
    axp, ayp = b1.exact_freq_integral(t, jx_p, jy_p, 2.0 * omega0, 0.0)
    axm, aym = b1.exact_freq_integral(t, jx_m, jy_m, 2.0 * omega0, 0.0)
    oddx = 0.5 * (axp - axm)
    oddy = 0.5 * (ayp - aym)
    return {
        "jx": b1.cinfo(oddx),
        "jy": b1.cinfo(oddy),
        "amp2d": float(math.hypot(abs(oddx), abs(oddy))),
        "jy_over_jx": float(abs(oddy) / abs(oddx)) if abs(oddx) > 0 else None,
        "jx_complex": [float(oddx.real), float(oddx.imag)],
        "jy_complex": [float(oddy.real), float(oddy.imag)],
    }


def wrap_deg(a: float) -> float:
    return (a + 180.0) % 360.0 - 180.0


def angle_between(z1, z2) -> float | None:
    if abs(z1) == 0 or abs(z2) == 0:
        return None
    return wrap_deg(math.degrees(np.angle(z2) - np.angle(z1)))


def cancel_ratio(a: dict, b: dict) -> float | None:
    za = complex(*a["jx_complex"])
    zb = complex(*b["jx_complex"])
    denom = abs(za) + abs(zb)
    if denom <= 0:
        return None
    return float(abs(za + zb) / denom)


def analyze_one(name: str, spec: dict) -> dict:
    plus = spec["plus"]
    minus = spec["minus"]
    t_p, jx_p, jy_p = b1.read_jt(plus / "Jt.dat")
    t_m, jx_m, jy_m = b1.read_jt(minus / "Jt.dat")
    if len(t_p) != len(t_m) or np.max(np.abs(t_p - t_m)) > 1e-8:
        raise RuntimeError(f"{name} Jt time grids differ")
    t_cycle_fs = 1600.0e-9 / 2.99792458e8 * 1e15
    omega0 = TWOPI / t_cycle_fs * b1.AU_TO_FS
    rec = {
        "name": name,
        "model": spec["model"],
        "kgrid": spec["kgrid"],
        "nt": int(len(t_p)),
        "dt_fs": float(t_p[1] - t_p[0]),
        "channels": {"total": odd_h2(t_p, jx_p, jy_p, jx_m, jy_m, omega0)},
        "reconstruction": {},
    }
    t_vp, val_p = read_cols(
        plus / "Jt_valley.dat",
        {"jx_k": 2, "jy_k": 3, "jx_kp": 4, "jy_kp": 5},
    )
    t_vm, val_m = read_cols(
        minus / "Jt_valley.dat",
        {"jx_k": 2, "jy_k": 3, "jx_kp": 4, "jy_kp": 5},
    )
    if len(t_vp) != len(t_p) or np.max(np.abs(t_vp - t_p)) > 1e-8:
        raise RuntimeError(f"{name} valley time grid mismatch")
    rec["channels"]["K"] = odd_h2(t_vp, val_p["jx_k"], val_p["jy_k"], val_m["jx_k"], val_m["jy_k"], omega0)
    rec["channels"]["Kp"] = odd_h2(t_vp, val_p["jx_kp"], val_p["jy_kp"], val_m["jx_kp"], val_m["jy_kp"], omega0)
    rec["channels"]["KplusKp"] = odd_h2(
        t_vp,
        val_p["jx_k"] + val_p["jx_kp"],
        val_p["jy_k"] + val_p["jy_kp"],
        val_m["jx_k"] + val_m["jx_kp"],
        val_m["jy_k"] + val_m["jy_kp"],
        omega0,
    )
    t_dp, dec_p = read_cols(
        plus / "Jt_decomposed.dat",
        {"jx_intra": 2, "jy_intra": 3, "jx_inter": 4, "jy_inter": 5},
    )
    t_dm, dec_m = read_cols(
        minus / "Jt_decomposed.dat",
        {"jx_intra": 2, "jy_intra": 3, "jx_inter": 4, "jy_inter": 5},
    )
    if len(t_dp) != len(t_p) or np.max(np.abs(t_dp - t_p)) > 1e-8:
        raise RuntimeError(f"{name} decomposed time grid mismatch")
    rec["channels"]["intra"] = odd_h2(
        t_dp, dec_p["jx_intra"], dec_p["jy_intra"], dec_m["jx_intra"], dec_m["jy_intra"], omega0
    )
    rec["channels"]["inter"] = odd_h2(
        t_dp, dec_p["jx_inter"], dec_p["jy_inter"], dec_m["jx_inter"], dec_m["jy_inter"], omega0
    )
    rec["channels"]["intra_plus_inter"] = odd_h2(
        t_dp,
        dec_p["jx_intra"] + dec_p["jx_inter"],
        dec_p["jy_intra"] + dec_p["jy_inter"],
        dec_m["jx_intra"] + dec_m["jx_inter"],
        dec_m["jy_intra"] + dec_m["jy_inter"],
        omega0,
    )
    tot = rec["channels"]["total"]
    rec["reconstruction"]["valley_over_total_amp"] = rec["channels"]["KplusKp"]["amp2d"] / tot["amp2d"]
    rec["reconstruction"]["intra_inter_over_total_amp"] = rec["channels"]["intra_plus_inter"]["amp2d"] / tot["amp2d"]
    rec["reconstruction"]["valley_jx_phase_minus_total_deg"] = wrap_deg(
        rec["channels"]["KplusKp"]["jx"]["phase_deg"] - tot["jx"]["phase_deg"]
    )
    rec["K_Kp_jx_angle_deg"] = angle_between(
        complex(*rec["channels"]["K"]["jx_complex"]),
        complex(*rec["channels"]["Kp"]["jx_complex"]),
    )
    rec["K_Kp_cancel_jx"] = cancel_ratio(rec["channels"]["K"], rec["channels"]["Kp"])
    rec["intra_inter_jx_angle_deg"] = angle_between(
        complex(*rec["channels"]["intra"]["jx_complex"]),
        complex(*rec["channels"]["inter"]["jx_complex"]),
    )
    rec["intra_inter_cancel_jx"] = cancel_ratio(rec["channels"]["intra"], rec["channels"]["inter"])
    rec["K_share"] = rec["channels"]["K"]["amp2d"] / tot["amp2d"]
    rec["Kp_share"] = rec["channels"]["Kp"]["amp2d"] / tot["amp2d"]
    rec["intra_share"] = rec["channels"]["intra"]["amp2d"] / tot["amp2d"]
    rec["inter_share"] = rec["channels"]["inter"]["amp2d"] / tot["amp2d"]
    return rec


def compare_grids(a: dict, b: dict) -> dict:
    def dphi(ch: str) -> float | None:
        pa = a["channels"][ch]["jx"]["phase_deg"]
        pb = b["channels"][ch]["jx"]["phase_deg"]
        if pa is None or pb is None:
            return None
        return wrap_deg(pb - pa)

    return {
        "amp_drop_total": a["channels"]["total"]["amp2d"] / max(b["channels"]["total"]["amp2d"], 1e-30),
        "dphi_total_jx": dphi("total"),
        "dphi_K_jx": dphi("K"),
        "dphi_Kp_jx": dphi("Kp"),
        "dphi_intra_jx": dphi("intra"),
        "dphi_inter_jx": dphi("inter"),
        "K_Kp_angle_2x2": a["K_Kp_jx_angle_deg"],
        "K_Kp_angle_4x4": b["K_Kp_jx_angle_deg"],
        "K_Kp_cancel_2x2": a["K_Kp_cancel_jx"],
        "K_Kp_cancel_4x4": b["K_Kp_cancel_jx"],
        "intra_share_2x2": a["intra_share"],
        "intra_share_4x4": b["intra_share"],
        "inter_share_2x2": a["inter_share"],
        "inter_share_4x4": b["inter_share"],
    }


def make_figure(payload: dict) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    FIG.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 4.2), constrained_layout=True)
    colors = {"K": "#1f77b4", "Kp": "#ff7f0e", "total": "#111111", "intra": "#2ca02c", "inter": "#d62728"}
    for ax, key, title in (
        (axes[0], "m88_2x2", "M88 2x2 H2-odd Jx"),
        (axes[1], "m88_4x4", "M88 4x4 H2-odd Jx"),
    ):
        rec = payload["jobs"][key]
        ax.axhline(0, color="0.8", lw=0.6)
        ax.axvline(0, color="0.8", lw=0.6)
        for ch in ("K", "Kp", "intra", "inter", "total"):
            z = complex(*rec["channels"][ch]["jx_complex"])
            ax.annotate(
                "",
                xy=(z.real, z.imag),
                xytext=(0, 0),
                arrowprops={"arrowstyle": "->", "color": colors[ch], "lw": 1.6},
            )
            ax.plot([z.real], [z.imag], "o", color=colors[ch], ms=4, label=ch)
        ax.set_title(title)
        ax.set_xlabel("Re Jx odd")
        ax.set_ylabel("Im Jx odd")
        ax.set_aspect("equal", adjustable="datalim")
        ax.legend(fontsize=8, loc="best")
    path = FIG / "m88_h2_odd_phasors.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def write_reports(payload: dict) -> None:
    m88_2 = payload["jobs"]["m88_2x2"]
    m88_4 = payload["jobs"]["m88_4x4"]
    cmp88 = payload["compare"]["m88"]
    cmp52 = payload["compare"].get("m52")
    f112 = payload["jobs"]["full112_2x2"]
    md = DOCS / "线Q_M88谷间分解_一页_20261029.md"
    lines = [
        "# 线 Q：M88 谷间/带内分解（2026-10-29）",
        "",
        "无新 SBE。同一 1600 nm、25 fs、10¹⁰ W/cm²、Hann 全窗 H2 奇电流。",
        "目的：解释 2×2→4×4 的 134° 相位跳变来自 K/K′ 相消还是 intra/inter。不上 8×8。",
        "",
        "| 任务 | 总振幅 | Jx 相位 | K∠Kp | 谷相消 | intra 份额 | inter 份额 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for key in ("m88_2x2", "m88_4x4", "m52_2x2", "m52_4x4", "full112_2x2"):
        rec = payload["jobs"][key]
        tot = rec["channels"]["total"]
        lines.append(
            f"| {key} | {tot['amp2d']:.3e} | {tot['jx']['phase_deg']:.1f}° | "
            f"{rec['K_Kp_jx_angle_deg']:.1f}° | {rec['K_Kp_cancel_jx']:.3f} | "
            f"{rec['intra_share']:.2f} | {rec['inter_share']:.2f} |"
        )
    lines += [
        "",
        "## M88 判定",
        "",
        f"- 总振幅 2×2/4×4 = **{cmp88['amp_drop_total']:.2f}**；总 Jx Δφ = **{cmp88['dphi_total_jx']:.1f}°**。",
        f"- K 自身 Δφ = {cmp88['dphi_K_jx']:.1f}°，K′ 自身 Δφ = {cmp88['dphi_Kp_jx']:.1f}°。",
        f"- 谷夹角 2×2 **{cmp88['K_Kp_angle_2x2']:.1f}°** → 4×4 **{cmp88['K_Kp_angle_4x4']:.1f}°**；",
        f"  谷相消比（1=无相消）2×2 **{cmp88['K_Kp_cancel_2x2']:.3f}** → 4×4 **{cmp88['K_Kp_cancel_4x4']:.3f}**。",
        f"- intra 份额 {cmp88['intra_share_2x2']:.2f} → {cmp88['intra_share_4x4']:.2f}；"
        f"inter {cmp88['inter_share_2x2']:.2f} → {cmp88['inter_share_4x4']:.2f}。",
        f"- intra Δφ = {cmp88['dphi_intra_jx']:.1f}°，inter Δφ = {cmp88['dphi_inter_jx']:.1f}°。",
        f"- 谷重构/总振幅：2×2 {m88_2['reconstruction']['valley_over_total_amp']:.4f}，"
        f"4×4 {m88_4['reconstruction']['valley_over_total_amp']:.4f}。",
        f"- full112 2×2：振幅 {f112['channels']['total']['amp2d']:.3e}，"
        f"相位 {f112['channels']['total']['jx']['phase_deg']:.1f}°，"
        f"K∠Kp {f112['K_Kp_jx_angle_deg']:.1f}°。",
    ]
    if cmp52:
        lines += [
            "",
            "## M52 对照（不加密）",
            "",
            f"- 振幅降 {cmp52['amp_drop_total']:.2f}；总 Δφ {cmp52['dphi_total_jx']:.1f}°。",
            f"- 谷夹角 {cmp52['K_Kp_angle_2x2']:.1f}° → {cmp52['K_Kp_angle_4x4']:.1f}°；"
            f"相消比 {cmp52['K_Kp_cancel_2x2']:.3f} → {cmp52['K_Kp_cancel_4x4']:.3f}。",
        ]
    k_angle_jump = abs(cmp88["K_Kp_angle_4x4"]) > 90.0 and abs(cmp88["K_Kp_angle_2x2"]) < 90.0
    intra_stable = abs(cmp88["dphi_intra_jx"]) < 40.0
    if k_angle_jump and cmp88["K_Kp_cancel_4x4"] < cmp88["K_Kp_cancel_2x2"] - 0.15:
        verdict = "phase_jump_is_valley_cancellation"
        lines += [
            "",
            f"判定串：`{verdict}`。4×4 把 K 与 K′ 推到接近反相，净相位被谷间相消改写。",
            "这不是 8×8 的理由，而是停止密网格、改看代表 k 的理由。",
        ]
    elif intra_stable and abs(cmp88["dphi_inter_jx"]) > 60.0:
        verdict = "phase_jump_is_interband_mixing"
        lines += [
            "",
            f"判定串：`{verdict}`。带内相位较稳，带间通道被网格改写。",
        ]
    else:
        verdict = "phase_jump_mixed_fewk_sampling"
        lines += [
            "",
            f"判定串：`{verdict}`。相位跳变不能单归因于一种分解；2×2/4×4 都不是材料定量。",
        ]
    payload["verdict"] = verdict
    lines += [
        "",
        f"图：`{payload['figure']}`。",
        f"归档：`{OUT.as_posix()}`。",
        "",
        "不问量子光。不把相干光磁性 H2 当论文终点。",
    ]
    md.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    if OUT.exists():
        (OUT / "progress.txt").write_text("", encoding="utf-8")
    payload = {"jobs": {}, "compare": {}, "no_new_sbe": True, "not_quantum_light": True}
    for name, spec in PAIRS.items():
        progress(f"analyze {name}")
        payload["jobs"][name] = analyze_one(name, spec)
    payload["compare"]["m88"] = compare_grids(payload["jobs"]["m88_2x2"], payload["jobs"]["m88_4x4"])
    payload["compare"]["m52"] = compare_grids(payload["jobs"]["m52_2x2"], payload["jobs"]["m52_4x4"])
    fig = make_figure(payload)
    payload["figure"] = str(fig)
    write_reports(payload)
    dump(OUT / "valley_intra.json", payload)
    progress(f"done verdict={payload.get('verdict')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
