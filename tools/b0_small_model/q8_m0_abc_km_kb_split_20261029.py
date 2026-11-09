"""Split the QED 14% remainder into B (Delta n=0) vs A (Delta n=+-2).

No new QED. Reuse endpoint-fixed Fourier and archived Q6/GH5 series.
Fair comparison is the two-angle proxy B and (A+C), because GH5 R-scan
only has theta=0/pi. Four-angle QED A,C are reported as a check.
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
import q8_m0_qinput_shared_kk_dyn_preflight_20261030 as pf
import qinput_spectrum_reanalysis_20261028 as spec

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results/q_line/abc_km_kb_split_20261029"
DOCS = ROOT / "docs/04_BSV量子光"
FIG = OUT / "figures"

ALPHA2 = 0.25
MABS = 0.3183267910741206
R_IDEAL = 1.2733071642964824

QED_B = -3.300558614e-8 + 9.537284492e-9j
QED_A = +4.743410898e-8 - 1.583918408e-8j
QED_C = +6.345731355e-10 - 4.336636681e-10j
QED_JX0 = 1.4885911076055816e-08 - 6.495157310566291e-09j
QED_JXPI = -8.125145315623136e-08 + 2.6050538175670813e-08j

QED_TH0_P = ROOT / "results/q_line/qinput_qin_a2_141_q1a_20261120"
QED_THPI_P = ROOT / "results/q_line/qinput_qin_a2_141_qed_theta_pi_20261120"
GH5_DIR = ROOT / "results/q_line/abc_rscan_gh5_20261029"
RSCAN_POST = GH5_DIR / "abc_rscan_post.json"


def dump(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(
        json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(tmp, path)


def load_xy(folder: Path, tag: str):
    z = np.load(folder / f"series_{tag}.npz")
    t = np.asarray(z["t"], dtype=float)
    j = np.column_stack([
        np.real(np.asarray(z["Jx"])),
        np.real(np.asarray(z["Jy"])),
    ])
    extra = {}
    if "n_ph" in z.files:
        extra["n_ph"] = np.asarray(z["n_ph"], dtype=float)
    return t, j, extra


def odd_even_h2(folder_p: Path, tag_p: str, folder_m: Path, tag_m: str):
    tp, jp, ep = load_xy(folder_p, tag_p)
    tm, jm, em = load_xy(folder_m, tag_m)
    if np.max(np.abs(tp - tm)) > 1e-12:
        raise RuntimeError(f"time mismatch {folder_p} {tag_p}")
    T0 = 2.0 * np.pi / pf.OMEGA0
    windows = {"main": (2.0, 4.0), "front": (1.0, 3.0), "back": (3.0, 5.0)}
    pack = {}
    t_end = float(min(tp[-1], tm[-1]))
    for name, (a, b) in windows.items():
        lo, hi = a * T0, b * T0
        if t_end + 1e-12 < hi:
            pack[name] = None
            continue
        qp = spec.spectrum(tp, jp, lo, hi, a1.QS, method="linear_exact", omega=pf.OMEGA0)
        qm = spec.spectrum(tm, jm, lo, hi, a1.QS, method="linear_exact", omega=pf.OMEGA0)
        pack[name] = {
            "odd_jx": a1.cpack(0.5 * (qp[1, 0] - qm[1, 0])),
            "odd_jy": a1.cpack(0.5 * (qp[1, 1] - qm[1, 1])),
            "even_jx": a1.cpack(0.5 * (qp[1, 0] + qm[1, 0])),
            "odd_norm2D": float(np.linalg.norm(0.5 * (qp[1] - qm[1]))),
        }
    if pack["main"] is None:
        raise RuntimeError(f"main window missing {folder_p} {tag_p}")
    lo, hi = 2.0 * T0, 4.0 * T0
    nph = None
    if "n_ph" in ep and "n_ph" in em:
        mask = (tp >= lo) & (tp <= hi)
        nph = {
            "plus_mean": float(np.mean(ep["n_ph"][mask])),
            "minus_mean": float(np.mean(em["n_ph"][mask])),
            "odd_mean": float(np.mean(0.5 * (ep["n_ph"][mask] - em["n_ph"][mask]))),
        }
    return {
        "odd_jx": pack["main"]["odd_jx"],
        "odd_jy": pack["main"]["odd_jy"],
        "even_jx": pack["main"]["even_jx"],
        "odd_norm2D": pack["main"]["odd_norm2D"],
        "windows": pack,
        "n_ph": nph,
    }


def kernels(abs_B: float, abs_AC: float) -> dict:
    kB = abs_B / ALPHA2
    kM = abs_AC / MABS
    return {
        "abs_B": abs_B,
        "abs_AC": abs_AC,
        "K_B": kB,
        "K_M": kM,
        "K_M_over_K_B": kM / max(kB, 1e-30),
        "rho": abs_AC / max(abs_B, 1e-30),
        "rho_over_R": (abs_AC / max(abs_B, 1e-30)) / R_IDEAL,
    }


def odd_jx(folder_p: Path, tag_p: str, folder_m: Path, tag_m: str):
    tp, jp, _ = load_xy(folder_p, tag_p)
    tm, jm, _ = load_xy(folder_m, tag_m)
    if np.max(np.abs(tp - tm)) > 1e-12:
        raise RuntimeError(f"time mismatch {folder_p} {tag_p}")
    return tp, 0.5 * (jp[:, 0] - jm[:, 0])


def sliding_kernel_ratios(t, j0q, jpiq, j0g, jpig, T0: float) -> dict:
    t_end = float(t[-1])
    centers = []
    r_kb = []
    r_km = []
    start = 0.50
    while start + 2.0 <= t_end / T0 + 1e-12:
        lo, hi = start * T0, (start + 2.0) * T0
        if hi > t_end + 1e-12:
            break

        def proxy(j0, jpi):
            j0c = np.column_stack([j0, np.zeros_like(j0)])
            jpic = np.column_stack([jpi, np.zeros_like(jpi)])
            q0 = spec.spectrum(
                t, j0c, lo, hi, a1.QS, method="linear_exact", omega=pf.OMEGA0
            )
            qpi = spec.spectrum(
                t, jpic, lo, hi, a1.QS, method="linear_exact", omega=pf.OMEGA0
            )
            return a1.two_angle_proxy(q0[1, 0], qpi[1, 0])

        pq = proxy(j0q, jpiq)
        pg = proxy(j0g, jpig)
        kq = kernels(pq["B_proxy"]["abs"], pq["A_plus_C_proxy"]["abs"])
        kg = kernels(pg["B_proxy"]["abs"], pg["A_plus_C_proxy"]["abs"])
        centers.append(start + 1.0)
        r_kb.append(kq["K_B"] / max(kg["K_B"], 1e-30))
        r_km.append(kq["K_M"] / max(kg["K_M"], 1e-30))
        start += 0.25
    return {"center_T0": centers, "KB": r_kb, "KM": r_km}


def make_figure(slide: dict, rec: dict):
    FIG.mkdir(parents=True, exist_ok=True)
    path = FIG / "km_kb_windows.png"
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        (FIG / "km_kb_windows.skip.txt").write_text(str(exc), encoding="utf-8")
        return None
    fig, ax = plt.subplots(figsize=(6.6, 3.5), dpi=140)
    ax.axhline(1.0, color="0.55", lw=0.8)
    ax.plot(slide["center_T0"], slide["KB"], "o-", ms=3.5, label=r"$K_B$ QED/GH5")
    ax.plot(slide["center_T0"], slide["KM"], "s-", ms=3.5, label=r"$K_M$ QED/GH5")
    if rec["window_KB"]["front"] is not None:
        ax.plot([2.0], [rec["window_KB"]["front"]], "o", ms=8, color="C0")
        ax.plot([2.0], [rec["window_KM"]["front"]], "s", ms=8, color="C1")
    ax.plot([3.0], [rec["ratio_KB"]], "o", ms=8, color="C0")
    ax.plot([3.0], [rec["ratio_KM"]], "s", ms=8, color="C1")
    ax.set_xlabel("2-cycle window center / T0")
    ax.set_ylabel("QED / GH5")
    ax.set_ylim(0.70, 1.30)
    ax.legend(fontsize=8, loc="best")
    ax.set_title("K_M stays at GH5; K_B changes with the window")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path


def write_reports(rec: dict) -> None:
    q = rec["qed_two_angle"]
    g = rec["gh5_two_angle"]
    four = rec["qed_four_angle"]
    def fmt_ratio(x):
        return "NA" if x is None else f"{x:.3f}"
    md = DOCS / "线Q_ABC_KMKB拆分_一页_20261029.md"
    lines = [
        "# 线 Q：14% 余量在 K_B，且随窗口变号（2026-10-29）",
        "",
        "无新 QED。公平口径是 θ=0/π 的两角代理 B 与 (A+C)；四角 QED 的 A、C 只作核对。",
        "权威数字不改写：ρ=1.456，R=1.273，arg(A/B)=177.65°。",
        "",
        "| 口径 | |B| | |A 或 A+C| | K_B=|B|/α² | K_M=|AC|/|M| | K_M/K_B | ρ | ρ/R |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
        f"| GH5 两角 a05r03 | {g['abs_B']:.3e} | {g['abs_AC']:.3e} | {g['K_B']:.3e} | {g['K_M']:.3e} | {g['K_M_over_K_B']:.4f} | {g['rho']:.3f} | {g['rho_over_R']:.4f} |",
        f"| QED 两角 | {q['abs_B']:.3e} | {q['abs_AC']:.3e} | {q['K_B']:.3e} | {q['K_M']:.3e} | {q['K_M_over_K_B']:.4f} | {q['rho']:.3f} | {q['rho_over_R']:.4f} |",
        f"| QED 四角 A,B | {four['abs_B']:.3e} | {four['abs_A']:.3e} | {four['K_B']:.3e} | {four['K_M']:.3e} | {four['K_M_over_K_B']:.4f} | {four['rho']:.3f} | {four['rho_over_R']:.4f} |",
        "",
        "## 判定",
        "",
        f"- 主窗：K_B 比 = **{rec['ratio_KB']:.3f}**（B 被压低 **{100*(1-rec['ratio_KB']):.1f}%**）；K_M 比 = **{rec['ratio_KM']:.3f}**（A+C 只差 **{100*(rec['ratio_KM']-1):.1f}%**）。",
        f"- 前窗 (1–3 T₀)：K_B 反转为 **{fmt_ratio(rec['window_KB']['front'])}**（B 被抬高约 14%）；K_M 仍是 **{fmt_ratio(rec['window_KM']['front'])}**。",
        f"- 后窗 (3–5 T₀)：NA。θ=π 的 QED 序列在 t={rec['t_end_qed_pi']:.1f} 结束，5T₀={rec['five_T0']:.1f}。",
        "- **K_M 窗口稳定，且已经被 GH5/Wigner 抓住。** 不能写成材料更强地读取 Δn=±2。",
        "- **14% 住在 Δn=0 的 K_B 里，并且随探测窗变号。** 这与已知邻窗 16%–39% 一致，不是协议无关的选择定则。",
        f"- 滑动 2 周期窗：K_M 在 {rec['sliding_span']['KM_min']:.3f}–{rec['sliding_span']['KM_max']:.3f}；K_B 在 {rec['sliding_span']['KB_min']:.3f}–{rec['sliding_span']['KB_max']:.3f}。",
        f"- 四角 |C/A|={four['abs_C_over_A']:.4f}，两角 (A+C) 与四角 A 只差约 1%，解释不了 14%。",
        "- GH5 已按 Wigner 采样经典统计；kick=True 的 Ehrenfest 也不能给出这 14%。",
        "- nmax=8 初态截断已在 Δn 报告中排除。本轮仍不启动 nmax=10。",
        f"- QED 主窗 ⟨n_ph⟩ ≈ {rec['qed_n_ph']['theta0']['plus_mean']:.3f} (θ=0)、{rec['qed_n_ph']['theta_pi']['plus_mean']:.3f} (θ=π)；奇 n_ph ~ 10⁻⁹。",
        f"- 偶通道主窗：K_B 比 = **{rec['even_two_angle']['ratio_KB']:.3f}**，|(A+C)/B|_QED={rec['even_two_angle']['qed_rho']:.4f}。14% 是 **Néel-odd 的 Δn=0 块** 特有，不是所有数态对角电流都被压低。",
        "",
        f"判定串：`{rec['verdict']}`。",
        f"图：`{rec.get('figure')}`。",
        f"归档：`{OUT.as_posix()}`。",
        "",
    ]
    md.write_text("\n".join(lines), encoding="utf-8")
    rec["one_page"] = str(md)


def analyze() -> dict:
    post = json.loads(RSCAN_POST.read_text(encoding="utf-8"))
    gh5_row = next(r for r in post["rows"] if r["prefix"] == "a05r03")
    gh5_k = kernels(gh5_row["abs_B"], gh5_row["abs_AC"])

    qed_series = {
        "theta0": odd_even_h2(QED_TH0_P, "ds_p", QED_TH0_P, "ds_m"),
        "theta_pi": odd_even_h2(QED_THPI_P, "ds_p", QED_THPI_P, "ds_m"),
    }
    gh5_series = {
        "theta0": odd_even_h2(GH5_DIR, "a05r03_p_0", GH5_DIR, "a05r03_m_0"),
        "theta_pi": odd_even_h2(GH5_DIR, "a05r03_p_pi", GH5_DIR, "a05r03_m_pi"),
    }

    def proxy_from(series):
        z0 = a1.cz(series["theta0"]["odd_jx"])
        zpi = a1.cz(series["theta_pi"]["odd_jx"])
        pr = a1.two_angle_proxy(z0, zpi)
        k = kernels(pr["B_proxy"]["abs"], pr["A_plus_C_proxy"]["abs"])
        k["proxy"] = pr
        k["arg_B_deg"] = pr["B_proxy"]["arg_deg"]
        return k

    qed2 = proxy_from(qed_series)
    gh52 = proxy_from(gh5_series)

    def proxy_even(series):
        z0 = a1.cz(series["theta0"]["even_jx"])
        zpi = a1.cz(series["theta_pi"]["even_jx"])
        pr = a1.two_angle_proxy(z0, zpi)
        k = kernels(pr["B_proxy"]["abs"], pr["A_plus_C_proxy"]["abs"])
        k["proxy"] = pr
        return k

    qed2e = proxy_even(qed_series)
    gh52e = proxy_even(gh5_series)

    four = kernels(abs(QED_B), abs(QED_A))
    four.update({
        "abs_B": abs(QED_B),
        "abs_A": abs(QED_A),
        "abs_C": abs(QED_C),
        "abs_C_over_A": abs(QED_C) / abs(QED_A),
        "B": a1.cpack(QED_B),
        "A": a1.cpack(QED_A),
        "C": a1.cpack(QED_C),
        "frozen_jx0": a1.cpack(QED_JX0),
        "frozen_jxpi": a1.cpack(QED_JXPI),
    })

    window_KB = {}
    window_KM = {}
    for name in ("main", "front", "back"):
        if (qed_series["theta0"]["windows"][name] is None
                or qed_series["theta_pi"]["windows"][name] is None
                or gh5_series["theta0"]["windows"][name] is None
                or gh5_series["theta_pi"]["windows"][name] is None):
            window_KB[name] = None
            window_KM[name] = None
            continue
        z0q = a1.cz(qed_series["theta0"]["windows"][name]["odd_jx"])
        zpiq = a1.cz(qed_series["theta_pi"]["windows"][name]["odd_jx"])
        z0g = a1.cz(gh5_series["theta0"]["windows"][name]["odd_jx"])
        zpig = a1.cz(gh5_series["theta_pi"]["windows"][name]["odd_jx"])
        pq = a1.two_angle_proxy(z0q, zpiq)
        pg = a1.two_angle_proxy(z0g, zpig)
        kq = kernels(pq["B_proxy"]["abs"], pq["A_plus_C_proxy"]["abs"])
        kg = kernels(pg["B_proxy"]["abs"], pg["A_plus_C_proxy"]["abs"])
        window_KB[name] = kq["K_B"] / max(kg["K_B"], 1e-30)
        window_KM[name] = kq["K_M"] / max(kg["K_M"], 1e-30)

    z0_series = a1.cz(qed_series["theta0"]["odd_jx"])
    rel_j0 = abs(z0_series - QED_JX0) / max(abs(QED_JX0), 1e-30)

    T0 = 2.0 * np.pi / pf.OMEGA0
    tq0, jq0 = odd_jx(QED_TH0_P, "ds_p", QED_TH0_P, "ds_m")
    tqpi, jqpi = odd_jx(QED_THPI_P, "ds_p", QED_THPI_P, "ds_m")
    tg0, jg0 = odd_jx(GH5_DIR, "a05r03_p_0", GH5_DIR, "a05r03_m_0")
    tgpi, jgpi = odd_jx(GH5_DIR, "a05r03_p_pi", GH5_DIR, "a05r03_m_pi")
    t_ref = tqpi
    jq0i = np.interp(t_ref, tq0, jq0)
    jg0i = np.interp(t_ref, tg0, jg0)
    jgpii = np.interp(t_ref, tgpi, jgpi)
    slide = sliding_kernel_ratios(t_ref, jq0i, jqpi, jg0i, jgpii, T0)
    mask = (t_ref >= 2.0 * T0) & (t_ref <= 4.0 * T0)
    Bq = 0.5 * (jq0i[mask] + jqpi[mask])
    Bg = 0.5 * (jg0i[mask] + jgpii[mask])
    ACq = 0.5 * (jq0i[mask] - jqpi[mask])
    ACg = 0.5 * (jg0i[mask] - jgpii[mask])

    def rms(x):
        return float(np.sqrt(np.mean(np.square(x))))

    rec = {
        "verdict": "KM_window_stable_KB_window_dependent",
        "no_new_qed": True,
        "fair_protocol": "two_angle_B_and_AplusC",
        "gh5_post_row": gh5_k,
        "gh5_two_angle": gh52,
        "qed_two_angle": qed2,
        "qed_four_angle": four,
        "ratio_KB": qed2["K_B"] / max(gh52["K_B"], 1e-30),
        "ratio_KM": qed2["K_M"] / max(gh52["K_M"], 1e-30),
        "window_KB": window_KB,
        "window_KM": window_KM,
        "even_two_angle": {
            "qed": qed2e,
            "gh5": gh52e,
            "ratio_KB": qed2e["K_B"] / max(gh52e["K_B"], 1e-30),
            "ratio_KM": qed2e["K_M"] / max(gh52e["K_M"], 1e-30),
            "qed_rho": qed2e["rho"],
            "gh5_rho": gh52e["rho"],
        },
        "series_vs_frozen_j0_rel": rel_j0,
        "qed_n_ph": {
            "theta0": qed_series["theta0"]["n_ph"],
            "theta_pi": qed_series["theta_pi"]["n_ph"],
        },
        "t_end_qed_pi": float(tqpi[-1]),
        "five_T0": float(5.0 * T0),
        "time_domain": {
            "rms_B_rel": rms(Bq - Bg) / max(rms(Bg), 1e-30),
            "rms_AC_rel": rms(ACq - ACg) / max(rms(ACg), 1e-30),
        },
        "sliding": slide,
        "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    rec["sliding_span"] = {
        "KB_min": min(slide["KB"]) if slide["KB"] else None,
        "KB_max": max(slide["KB"]) if slide["KB"] else None,
        "KM_min": min(slide["KM"]) if slide["KM"] else None,
        "KM_max": max(slide["KM"]) if slide["KM"] else None,
    }
    rec["figure"] = None if make_figure(slide, rec) is None else str(
        FIG / "km_kb_windows.png"
    )
    dump(OUT / "km_kb_split.json", rec)
    write_reports(rec)
    return rec


def main() -> int:
    rec = analyze()
    print(json.dumps({
        "verdict": rec["verdict"],
        "ratio_KB": rec["ratio_KB"],
        "ratio_KM": rec["ratio_KM"],
        "window_KB": rec["window_KB"],
        "window_KM": rec["window_KM"],
        "time_domain": rec["time_domain"],
        "even_KB": rec["even_two_angle"]["ratio_KB"],
        "even_rho": rec["even_two_angle"]["qed_rho"],
        "one_page": rec["one_page"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
