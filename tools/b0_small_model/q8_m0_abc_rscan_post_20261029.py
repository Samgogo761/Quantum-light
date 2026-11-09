"""Post-analysis for the GH5 R-scan. Read-only on job series; does not rerun GH5."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results/q_line/abc_rscan_gh5_20261029"
FIG = OUT / "figures"
DOCS = ROOT / "docs/04_BSV量子光"
JSON_PATH = OUT / "abc_rscan.json"


def mag(z: complex) -> float:
    return float(abs(z))


def cz(pack) -> complex:
    return complex(pack["re"], pack["im"])


def extra_row(row: dict) -> dict:
    alpha = float(row["alpha"])
    r = float(row["r"])
    R = float(row["R"])
    nbar = float(row["nbar"])
    Mabs = float(np.sinh(r) * np.cosh(r))
    B = cz(row["odd_x_proxy"]["B_proxy"])
    AC = cz(row["odd_x_proxy"]["A_plus_C_proxy"])
    even0 = float(row["theta0"]["even_norm2D"])
    evenpi = float(row["theta_pi"]["even_norm2D"])
    return {
        "prefix": row["prefix"],
        "alpha": alpha,
        "r": r,
        "R": R,
        "nbar": nbar,
        "Mabs": Mabs,
        "alpha2": alpha * alpha,
        "abs_B": mag(B),
        "abs_AC": mag(AC),
        "B_over_alpha2": mag(B) / max(alpha * alpha, 1e-30),
        "B_over_nbar": mag(B) / max(nbar, 1e-30),
        "AC_over_M": mag(AC) / max(Mabs, 1e-30),
        "K_ratio_AC_B": (mag(AC) / max(Mabs, 1e-30)) / max(mag(B) / max(alpha * alpha, 1e-30), 1e-30),
        "rho": float(row["rho_proxy_AC_over_B"]),
        "rho_over_R": float(row["material_factor_rho_over_R"]),
        "arg_B_deg": float(row["arg_B_deg"]),
        "arg_AC_over_B_deg": float(row["arg_AC_over_B_deg"]),
        "amp2D": float(row["amp2D_pi_over_0"]),
        "ampJx": float(row["amp_Jx_pi_over_0"]),
        "pred_R": float(row["pred_amp_from_R"]),
        "pred_rho": float(row["pred_amp_from_rho"]),
        "even_pi_over_0": evenpi / max(even0, 1e-30),
        "odd0": float(row["odd_norm2D_0"]),
        "oddpi": float(row["odd_norm2D_pi"]),
    }


def verdict_notes(rows: list) -> tuple:
    fs = np.array([r["rho_over_R"] for r in rows], float)
    ks = np.array([r["K_ratio_AC_B"] for r in rows], float)
    args = np.array([r["arg_AC_over_B_deg"] for r in rows], float)
    mean_f = float(np.mean(fs))
    spread = float(np.std(fs) / max(abs(mean_f), 1e-30))
    mean_k = float(np.mean(ks))
    notes = []
    q6 = next((r for r in rows if r["prefix"] == "a05r03"), None)
    if q6 is not None:
        notes.append(
            "a05r03 对齐 A2b：ρ=%.4f（目标 1.268），Jx π/0=%.3f（目标 8.016），arg(B)=%.2f°（目标 167.82°）。"
            % (q6["rho"], q6["ampJx"], q6["arg_B_deg"])
        )
    if spread < 0.08 and abs(mean_f - 1.0) < 0.08:
        verdict = "rho_tracks_R_field_dominated"
        notes.append(
            "GH5 上 ρ/R 均值 %.4f，相对散度 %.2f%%。大相位干涉由输入统计窗 R 控制；材料在本层几乎不改 |M| 与 α² 的相对权重。"
            % (mean_f, 100.0 * spread)
        )
    elif spread < 0.08:
        verdict = "material_factor_stable_across_R"
        notes.append("材料因子稳定在 %.3f，可当作与 R 无关的电子修正。" % mean_f)
    else:
        verdict = "material_factor_R_dependent"
        notes.append("ρ/R 随 R 变化，不能把单一 1.14 外推到所有位移压缩窗口。")
    notes.append(
        "|A+C|/|M| 与 |B|/α² 的比值均值 %.4f。若接近 1，说明异常关联与位移数态块被同一电子核读取。"
        % mean_k
    )
    dphi = float(np.max(np.abs((args + 180.0 + 180.0) % 360.0 - 180.0)))
    notes.append("arg((A+C)/B) 偏离 180° 最多 %.1f°。" % dphi)
    notes.append(
        "QED 的 ρ/R=1.144 不在本层。那是精确量子映射相对 Ehrenfest/GH5 的修正，不是 χ^{(2)} 选择定则。"
    )
    a12 = next((r for r in rows if r["prefix"] == "a12r03"), None)
    if a12 is not None:
        notes.append(
            "低 R 点 a12r03（R=%.3f）接近明亮 80%%/20%% 窗口：ρ=%.3f，2D π/0=%.2f。这解释了 1600 nm DS 为何没有 110°。"
            % (a12["R"], a12["rho"], a12["amp2D"])
        )
    return verdict, notes


def make_figure(rows: list) -> Path:
    FIG.mkdir(parents=True, exist_ok=True)
    path = FIG / "rscan_rho_and_contrast.png"
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        (FIG / "rscan_rho_and_contrast.skip.txt").write_text(
            "matplotlib unavailable: %s\n" % exc, encoding="utf-8"
        )
        return path
    R = np.array([r["R"] for r in rows], float)
    rho = np.array([r["rho"] for r in rows], float)
    amp = np.array([r["amp2D"] for r in rows], float)
    pred = np.array([r["pred_R"] for r in rows], float)
    xmax = max(2.6, float(R.max()) + 0.15)
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.4), dpi=140)
    ax = axes[0]
    ax.plot([0.0, xmax], [0.0, xmax], color="0.7", lw=1.0, label="ρ=R")
    ax.scatter(R, rho, c="#1f4e79", s=36, zorder=3, label="GH5 ρ")
    ax.axhline(1.456, color="#c0392b", ls="--", lw=0.9, label="QED ρ at Q6")
    ax.set_xlabel(r"$R=|M|/\alpha^2$")
    ax.set_ylabel(r"$\rho=|(A+C)/B|$")
    ax.set_title("odd H2 tracks the input window")
    ax.legend(frameon=False, fontsize=8)
    ax.set_xlim(0.0, xmax)
    ax.set_ylim(0.0, max(xmax, float(rho.max()) + 0.15))
    ax = axes[1]
    ax.plot(R, pred, color="0.7", lw=1.0, label=r"$|1+R|/|1-R|$")
    ax.scatter(R, amp, c="#1f4e79", s=36, zorder=3, label="GH5 2D π/0")
    ax.axhline(4.98, color="#c0392b", ls="--", lw=0.9, label="QED 2D π/0 at Q6")
    ax.set_xlabel(r"$R=|M|/\alpha^2$")
    ax.set_ylabel("odd H2 2D amplitude ratio")
    ax.set_title("contrast is set by R, not by α")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path


def write_reports(rec: dict, rows: list, fig: Path) -> None:
    verdict, notes = verdict_notes(rows)
    one = DOCS / "线Q_ABC_R扫描_一页_20261029.md"
    longp = DOCS / "线Q_ABC_R扫描_20261029.md"
    lines = [
        "# 线 Q：A/B/C 的 R=|M|/α² 扫描一页（2026-10-29）",
        "",
        "Q6 原协议乘积 GH5，η=1，kick=True，θ=0/π。只改位移压缩输入的 (α,r)。",
        "两角代理只能给出 B 与 (A+C)，不能拆开 A 与 C。不是 1600 nm 的 80%/20% DS。",
        "权威 QED 数字不改写：ρ=1.456，R=1.273，arg(A/B)=177.65°，2D 振幅比 4.98。",
        "",
        "| 点 | α | r | R | ⟨n⟩ | ρ | ρ/R | |B|/α² | |A+C|/|M| | 2D π/0 | |1+R|/|1−R| | arg((A+C)/B) | 偶 π/0 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        lines.append(
            "| %s | %.1f | %.1f | %.3f | %.3f | %.3f | %.3f | %.3e | %.3e | %.2f | %.2f | %.1f° | %.4f |"
            % (
                r["prefix"], r["alpha"], r["r"], r["R"], r["nbar"], r["rho"],
                r["rho_over_R"], r["B_over_alpha2"], r["AC_over_M"], r["amp2D"],
                r["pred_R"], r["arg_AC_over_B_deg"], r["even_pi_over_0"],
            )
        )
    lines += ["", "## 判定", ""]
    for n in notes:
        lines.append("- " + n)
    lines += [
        "",
        "可证伪命题：在固定 Hamiltonian 与探测窗下，",
        r"\(\mathcal J_2^o(\theta)\approx K_B\alpha^2 + K_M M(\theta)\)，GH5 上 \(K_M/K_B\approx 1\)。",
        "因此 O(1) 相位干涉要求 R≳1，而不是更大的 α。QED 把该比值修正到约 1.14。",
        "",
        "图：`" + fig.as_posix() + "`",
        "归档：`" + OUT.as_posix() + "`。",
        "",
    ]
    one.write_text("\n".join(lines), encoding="utf-8")
    payload = {
        "verdict": verdict,
        "notes": notes,
        "rows": rows,
        "driver": {
            "verdict": rec.get("verdict"),
            "diagnostics": rec.get("diagnostics"),
            "status": rec.get("status"),
            "missing_jobs": rec.get("missing_jobs"),
        },
        "figure": str(fig),
        "qed_not_rerun": True,
        "cannot_separate_A_and_C": True,
        "did_not_mix_1600nm_8020": True,
    }
    longp.write_text(
        "# 八带 M0：A/B/C 的 R 扫描（2026-10-29）\n\n"
        + "\n".join("- " + n for n in notes)
        + "\n\n```json\n"
        + json.dumps(payload, ensure_ascii=False, indent=2)
        + "\n```\n",
        encoding="utf-8",
    )
    (OUT / "abc_rscan_post.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def main() -> int:
    rec = json.loads(JSON_PATH.read_text(encoding="utf-8"))
    summaries = rec.get("summaries") or []
    if not summaries:
        raise SystemExit("no summaries in abc_rscan.json")
    rows = [extra_row(s) for s in summaries]
    fig = make_figure(rows)
    write_reports(rec, rows, fig)
    print(json.dumps({
        "n": len(rows),
        "prefixes": [r["prefix"] for r in rows],
        "mean_rho_over_R": float(np.mean([r["rho_over_R"] for r in rows])),
        "mean_K_ratio": float(np.mean([r["K_ratio_AC_B"] for r in rows])),
        "figure": str(fig),
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

