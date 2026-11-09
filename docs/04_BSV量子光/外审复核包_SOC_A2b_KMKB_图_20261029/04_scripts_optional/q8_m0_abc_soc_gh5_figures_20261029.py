"""Plot GH5/kernel SOC scan. Read-only on A1/A2 archives."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
GH5 = ROOT / "results/q_line/abc_soc_gh5_scan_20261029"
KER = ROOT / "results/q_line/abc_soc_kernel_scan_20261029"
FIG = GH5 / "figures"


def near_pi_deg(x: float) -> float:
    a = float(x) % 360.0
    if a < 0:
        a += 360.0
    return a


def main() -> int:
    FIG.mkdir(parents=True, exist_ok=True)
    rec = json.loads((GH5 / "soc_gh5_scan.json").read_text(encoding="utf-8"))
    ker = json.loads((KER / "soc_kernel_scan.json").read_text(encoding="utf-8"))

    order = ["soc025", "soc050", "soc075", "soc100_frozen"]
    rows = [rec["rows"][k] for k in order]
    frac = np.array([r["lam_so_frac"] for r in rows], dtype=float)
    arg_acb = np.array([near_pi_deg(r["arg_AC_over_B_deg"]) for r in rows])
    kmkb = np.array([r["K_M_over_K_B"] for r in rows], dtype=float)
    acb = np.array([r["abs_AC_over_B"] for r in rows], dtype=float)
    odd0 = np.array([r["odd0"] for r in rows], dtype=float)
    oddpi = np.array([r["odd_pi"] for r in rows], dtype=float)
    amp = np.array([r["amp_ratio_pi_over_0"] for r in rows], dtype=float)
    pi_dev = np.array([r["pi_dev_deg"] for r in rows], dtype=float)
    arg_b = np.array([near_pi_deg(r["B_proxy"]["arg_deg"]) for r in rows])

    ad = [row for row in ker["adiabatic_soc"] if not row.get("gated_off")]
    ad_frac = np.array([row["lam_so_frac"] for row in ad], dtype=float)
    ad_abs = np.array([row["odd_x"]["abs"] for row in ad], dtype=float)
    ad_arg = np.array([row["odd_x"]["arg_deg"] for row in ad], dtype=float)
    ad_at_1 = float(ad_abs[np.argmin(np.abs(ad_frac - 1.0))])

    ft = ker["finite_time"]
    ft_order = [name for name in ("soc025", "soc050", "soc075", "soc100") if name in ft]
    ft_frac = np.array([float(ft[n]["lam_so_frac"]) for n in ft_order])
    ft_arg = np.array([near_pi_deg(ft[n]["finite_time_arg_deg"]) for n in ft_order])
    ft_odd = np.array([float(ft[n]["odd_norm2D"]) for n in ft_order])

    plt.rcParams.update({
        "font.size": 11,
        "axes.linewidth": 1.0,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.bbox": "tight",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })

    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    ax.axhline(180.0, color="0.6", ls="--", lw=1.0, label="π")
    ax.plot(frac, arg_acb, "o-", color="#1f4e79", lw=1.6, ms=7, label=r"GH5 $\arg((A+C)/B)$")
    ax.set_xlabel(r"$\lambda_{\mathrm{SO}}/\lambda_{\mathrm{SO}}^{(0)}$")
    ax.set_ylabel(r"$\arg((A+C)/B)$ (deg)")
    ax.set_ylim(170, 185)
    ax.set_xlim(0.15, 1.10)
    ax.legend(frameon=False)
    ax.set_title("SOC does not pull the two-angle phase off π")
    fig.savefig(FIG / "arg_ACB_vs_soc.svg")
    fig.savefig(FIG / "arg_ACB_vs_soc.png", dpi=160)
    fig.savefig(FIG / "arg_ACB_vs_soc.pdf")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    ax.axhline(1.0, color="0.6", ls="--", lw=1.0, label=r"$K_M/K_B=1$")
    ax.plot(frac, kmkb, "s-", color="#9c3d0e", lw=1.6, ms=7, label=r"GH5 $K_M/K_B$")
    ax.plot(frac, acb / rec["R"], "o--", color="#2e7d4f", lw=1.2, ms=6,
            label=r"$|A+C|/|B|$ / $R$")
    ax.set_xlabel(r"$\lambda_{\mathrm{SO}}/\lambda_{\mathrm{SO}}^{(0)}$")
    ax.set_ylabel("dimensionless ratio")
    ax.set_ylim(0.92, 1.08)
    ax.set_xlim(0.15, 1.10)
    ax.legend(frameon=False)
    ax.set_title("Same kernel reads α² and M")
    fig.savefig(FIG / "km_over_kb_vs_soc.svg")
    fig.savefig(FIG / "km_over_kb_vs_soc.png", dpi=160)
    fig.savefig(FIG / "km_over_kb_vs_soc.pdf")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    ax.plot(ad_frac, ad_abs, "-", color="#1f4e79", lw=1.5, label="adiabatic |K_odd|")
    ax.set_xlabel(r"$\lambda_{\mathrm{SO}}/\lambda_{\mathrm{SO}}^{(0)}$")
    ax.set_ylabel(r"$|\chi^{(2)}_{x,\mathrm{odd}}|$ (model units)")
    ax2 = ax.twinx()
    ax2.plot(frac, oddpi, "o", color="#9c3d0e", ms=7, label="GH5 odd π")
    ax2.plot(frac, odd0, "s", color="#2e7d4f", ms=7, label="GH5 odd 0")
    ax2.set_ylabel("GH5 odd 2D amplitude")
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, frameon=False, loc="upper left")
    ax.set_title("SOC scales the odd channel, does not rotate it")
    fig.savefig(FIG / "odd_amp_vs_soc.svg")
    fig.savefig(FIG / "odd_amp_vs_soc.png", dpi=160)
    fig.savefig(FIG / "odd_amp_vs_soc.pdf")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    ax.plot(ad_frac, ad_arg, "-", color="#1f4e79", lw=1.6, label="adiabatic arg(K)")
    ax.axhline(-117.2, color="0.6", ls="--", lw=1.0)
    ax.set_xlabel(r"$\lambda_{\mathrm{SO}}/\lambda_{\mathrm{SO}}^{(0)}$")
    ax.set_ylabel("adiabatic arg(K_odd) (deg)")
    ax.set_ylim(-118.0, -116.5)
    ax.legend(frameon=False)
    ax.set_title("Adiabatic kernel phase pinned (span 0.28°)")
    fig.savefig(FIG / "adiabatic_arg_vs_soc.svg")
    fig.savefig(FIG / "adiabatic_arg_vs_soc.png", dpi=160)
    fig.savefig(FIG / "adiabatic_arg_vs_soc.pdf")
    plt.close(fig)

    table = {
        "verdict": rec["verdict"],
        "arg_span_deg": rec["arg_span_deg"],
        "max_pi_dev_deg": rec["max_pi_dev_deg"],
        "KM_over_KB_span": rec["KM_over_KB_span"],
        "R": rec["R"],
        "amp": amp.tolist(),
        "pi_dev": pi_dev.tolist(),
        "arg_b_near_pi": arg_b.tolist(),
        "ft_frac": ft_frac.tolist(),
        "ft_arg_near_pi": ft_arg.tolist(),
        "ft_odd": ft_odd.tolist(),
        "ad_arg_min": float(np.min(ad_arg)),
        "ad_arg_max": float(np.max(ad_arg)),
        "ad_abs_at_R0": ad_at_1,
        "rows": {
            k: {
                "lam_so_frac": rec["rows"][k]["lam_so_frac"],
                "arg_AC_over_B_deg": rec["rows"][k]["arg_AC_over_B_deg"],
                "pi_dev_deg": rec["rows"][k]["pi_dev_deg"],
                "abs_AC_over_B": rec["rows"][k]["abs_AC_over_B"],
                "K_M_over_K_B": rec["rows"][k]["K_M_over_K_B"],
                "amp_ratio_pi_over_0": rec["rows"][k]["amp_ratio_pi_over_0"],
                "arg_B_deg": rec["rows"][k]["B_proxy"]["arg_deg"],
                "odd0": rec["rows"][k]["odd0"],
                "odd_pi": rec["rows"][k]["odd_pi"],
            }
            for k in order
        },
        "kernel": {
            "verdict": ker["verdict"],
            "adiabatic_soc_arg_span_deg": ker["adiabatic_soc_arg_span_deg"],
            "finite_time_arg_span_deg": ker["finite_time_arg_span_deg"],
        },
    }
    (GH5 / "soc_gh5_figure_table.json").write_text(
        json.dumps(table, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        "verdict": rec["verdict"],
        "arg_span_deg": rec["arg_span_deg"],
        "max_pi_dev_deg": rec["max_pi_dev_deg"],
        "out": str(FIG),
        "n_ad": int(len(ad)),
        "ad_arg_min": float(np.min(ad_arg)),
        "ad_arg_max": float(np.max(ad_arg)),
        "ad_abs_at_1": ad_at_1,
        "amp": amp.tolist(),
        "pi_dev": pi_dev.tolist(),
        "arg_b": arg_b.tolist(),
        "ft_arg": ft_arg.tolist(),
        "ft_odd": ft_odd.tolist(),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
