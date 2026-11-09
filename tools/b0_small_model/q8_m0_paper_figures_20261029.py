"""Paper figures from frozen Q-line JSON. No new QED or SBE."""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results/q_line/paper_figures_20261029"
FIG = OUT / "figures"

MIC = ROOT / "results/q_line/abc_microscopic_20261029/abc_microscopic.json"
RSCAN = ROOT / "results/q_line/abc_rscan_gh5_20261029/abc_rscan_post.json"
KB = ROOT / "results/q_line/abc_kb_window_20261029/kb_window.json"
VAL = ROOT / "results/q_line/m88_valley_intra_20261029/valley_intra.json"
A1 = ROOT / "results/q_line/abc_a1_gate_gh5_20261029/a1_gate.json"
A2 = ROOT / "results/q_line/abc_a2_finite_time_kernel_20261029/a2_finite_time.json"
A2B = ROOT / "results/q_line/abc_a2b_ehrenfest_kick_20261029/a2b_ehrenfest.json"

COL = {
    "blue": "#0072B2",
    "orange": "#D55E00",
    "green": "#009E73",
    "pink": "#CC79A7",
    "gold": "#E69F00",
    "sky": "#56B4E9",
    "grey": "#444444",
}


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def dump(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def setup_mpl():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.labelsize": 10,
            "axes.titlesize": 10.5,
            "legend.fontsize": 7.5,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "axes.linewidth": 0.8,
            "figure.dpi": 140,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "mathtext.fontset": "dejavusans",
        }
    )
    return plt


def save(fig, name: str) -> Path:
    FIG.mkdir(parents=True, exist_ok=True)
    png = FIG / f"{name}.png"
    pdf = FIG / f"{name}.pdf"
    fig.savefig(png, facecolor="white")
    fig.savefig(pdf, facecolor="white")
    return png


def wrap_deg(x: float) -> float:
    return float((x + 180.0) % 360.0 - 180.0)


def to_360(x: float) -> float:
    """Map a degree to (0, 360) so near-pi values stay on one side."""
    x = float(x)
    if x < 0.0:
        x += 360.0
    return x


def fig1_ellipses(plt, moments):
    alpha = float(moments["alpha"]["abs"])
    r = float(moments["r"])
    R = float(moments["M_over_alpha2"])
    cx, cy = 2.0 * alpha, 0.0
    t = np.linspace(0.0, 2.0 * np.pi, 400)
    fig, ax = plt.subplots(figsize=(4.4, 4.0))
    vac = np.exp(1j * t)
    ax.plot(cx + vac.real, cy + vac.imag, color="#bbbbbb", lw=0.9, ls="--", label="vacuum 1$\\sigma$")
    colors = [COL["blue"], COL["green"], COL["orange"], COL["pink"]]
    thetas = [0.0, 0.5 * np.pi, np.pi, 1.5 * np.pi]
    labels = [r"$\theta=0$ (X sq.)", r"$\theta=\pi/2$", r"$\theta=\pi$ (P sq.)", r"$\theta=3\pi/2$"]
    for th, c, lab in zip(thetas, colors, labels):
        phi = 0.5 * th
        rot = np.array([[np.cos(phi), -np.sin(phi)], [np.sin(phi), np.cos(phi)]])
        pts = rot @ np.vstack([np.exp(-r) * np.cos(t), np.exp(+r) * np.sin(t)])
        ax.plot(cx + pts[0], cy + pts[1], color=c, lw=1.6, label=lab)
    ax.plot([0], [0], "k+", ms=7, mew=1.2)
    ax.plot([cx], [cy], "ko", ms=5, label=rf"$\langle X\rangle=2\alpha={2*alpha:.1f}$")
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel(r"$X=a+a^\dagger$")
    ax.set_ylabel(r"$P=(a-a^\dagger)/i$")
    ax.set_title(rf"Displaced squeezed input, $R=|M|/\alpha^2={R:.3f}$")
    ax.legend(loc="upper right", frameon=False)
    ax.set_xlim(-1.6, 3.8)
    ax.set_ylim(-2.4, 2.4)
    ax.axhline(0, color="#dddddd", lw=0.6)
    ax.axvline(0, color="#dddddd", lw=0.6)
    fig.text(
        0.02,
        -0.02,
        r"$D(0.5)S(0.3e^{i\theta})|0\rangle$; mean displacement fixed, ellipse rotates.",
        fontsize=7.5,
        color="#444444",
    )
    return save(fig, "fig1_ds_ellipse")


def fig2_plane(plt, bac, geom):
    B = bac["B"]["re"] + 1j * bac["B"]["im"]
    A = bac["A"]["re"] + 1j * bac["A"]["im"]
    C = bac["C"]["re"] + 1j * bac["C"]["im"]
    scale = 1e8
    th = np.linspace(0.0, 2.0 * np.pi, 400)
    curve = B + A * np.exp(1j * th) + C * np.exp(-1j * th)
    pts = {
        "0": geom["theta0"],
        r"\pi/2": geom["theta_pi2"],
        r"\pi": geom["theta_pi"],
        r"3\pi/2": geom["theta_3pi2"],
    }
    fig, ax = plt.subplots(figsize=(4.6, 4.2))
    ax.plot(curve.real * scale, curve.imag * scale, color=COL["grey"], lw=1.1, label=r"$B+Ae^{i\theta}+Ce^{-i\theta}$")
    colors = [COL["blue"], COL["green"], COL["orange"], COL["pink"]]
    for (lab, p), c in zip(pts.items(), colors):
        ax.scatter([p["re"] * scale], [p["im"] * scale], s=36, color=c, zorder=5)
        off = {
            "0": (10, -12),
            r"\pi/2": (-46, 8),
            r"\pi": (8, 10),
            r"3\pi/2": (10, -14),
        }[lab]
        ax.annotate(rf"$\theta={lab}$", (p["re"] * scale, p["im"] * scale), textcoords="offset points", xytext=off, fontsize=8)
    ax.scatter([B.real * scale], [B.imag * scale], marker="x", s=45, color="k", zorder=6, label="B")
    ax.axhline(0, color="#dddddd", lw=0.6)
    ax.axvline(0, color="#dddddd", lw=0.6)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel(r"$\mathrm{Re}\,J_x^{o}\ (\times 10^{-8})$")
    ax.set_ylabel(r"$\mathrm{Im}\,J_x^{o}\ (\times 10^{-8})$")
    rho = geom["rho_absA_over_absB"]
    argab = geom["arg_A_over_B_deg"]
    ax.set_title(rf"Q6 QED $\eta=1$: $\rho=|A|/|B|={rho:.3f}$, $\arg(A/B)={argab:.2f}^\circ$")
    ax.legend(loc="lower right", frameon=False)
    fig.text(
        0.02,
        -0.02,
        r"Main-window odd $J_x$; $|C/A|\approx 1.54\%$. Frozen 2D $\theta=\pi/\theta=0$ ratio is 4.98.",
        fontsize=7.5,
        color="#444444",
    )
    return save(fig, "fig2_qed_complex_plane")


def fig3_rscan(plt, rscan):
    rows = rscan["rows"]
    R = np.array([r["R"] for r in rows], dtype=float)
    rho = np.array([r["rho"] for r in rows], dtype=float)
    amp = np.array([r["amp2D"] for r in rows], dtype=float)
    xx = np.linspace(0.05, 2.55, 80)
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.15))
    ax = axes[0]
    ax.plot(xx, xx, color="#aaaaaa", lw=1.0, ls="--", label=r"$\rho=R$")
    ax.scatter(R, rho, s=38, color=COL["blue"], zorder=3)
    for r in rows:
        ax.annotate(r["prefix"], (r["R"], r["rho"]), textcoords="offset points", xytext=(4, 3), fontsize=6.5)
    ax.set_xlabel(r"$R=|M|/\alpha^2$")
    ax.set_ylabel(r"GH5 $\rho=|A+C|/|B|$")
    ax.set_title(r"$\rho$ tracks $R$ (mean $\rho/R=0.9976$)")
    ax.legend(frameon=False)
    ax.set_xlim(0, 2.6)
    ax.set_ylim(0, 2.6)
    ax = axes[1]
    ax.plot(xx, np.abs(1 + xx) / np.abs(1 - xx), color="#aaaaaa", lw=1.0, ls="--", label=r"$|1+R|/|1-R|$")
    ax.scatter(R, amp, s=38, color=COL["orange"], zorder=3, label=r"GH5 2D $\theta=\pi/\theta=0$")
    i = [k for k, r in enumerate(rows) if r["prefix"] == "a12r03"][0]
    ax.scatter([R[i]], [amp[i]], s=70, facecolors="none", edgecolors=COL["green"], lw=1.6, zorder=4)
    ax.annotate("a12r03\n(bright-like R)", (R[i], amp[i]), textcoords="offset points", xytext=(8, -18), fontsize=7, color=COL["green"])
    ax.set_xlabel(r"$R=|M|/\alpha^2$")
    ax.set_ylabel("amplitude ratio")
    ax.set_title(r"$R=O(1)$: interference is strong; near $R\approx 1$ it cancels")
    ax.legend(frameon=False, loc="upper right")
    ax.set_xlim(0, 2.6)
    ax.set_ylim(0, 14)
    fig.tight_layout()
    return save(fig, "fig3_rscan_rho")


def fig4_gating(plt, a1):
    d0 = a1["summaries"]["d0"]
    s0 = a1["summaries"]["s0"]
    fr = a1["frozen_gh5"]
    odd_p = [
        fr["theta0"]["odd_norm2D"],
        fr["theta_pi"]["odd_norm2D"],
        max(d0["odd_max_norm2D"], 1e-30),
        s0["odd_max_norm2D"],
    ]
    even_p = [
        fr["theta0"]["even_norm2D"],
        fr["theta_pi"]["even_norm2D"],
        d0["even_max_norm2D"],
        s0["even_max_norm2D"],
    ]
    labels = [r"R0 $\theta=0$", r"R0 $\theta=\pi$", r"$\Delta=0$", r"SOC$=0$"]
    x = np.arange(len(labels))
    w = 0.36
    fig, ax = plt.subplots(figsize=(5.2, 3.3))
    ax.bar(x - w / 2, odd_p, w, color=COL["orange"], label="Neel-odd H2")
    ax.bar(x + w / 2, even_p, w, color=COL["blue"], label="even H2")
    ax.set_yscale("log")
    ax.set_xticks(x, labels)
    ax.set_ylabel("2D H2 amplitude")
    ax.set_title(r"Gating: $\Delta=0$ and SOC$=0$ kill the odd channel")
    ax.legend(frameon=False)
    ax.set_ylim(1e-16, 5e-3)
    fig.text(
        0.02,
        -0.04,
        rf"SOC=0 gap {s0['min_gap']:.3f} eV (not a unique-mechanism certificate). "
        r"$\Delta=0$ odd vanishing is definitional.",
        fontsize=7.5,
        color="#444444",
    )
    return save(fig, "fig4_gating")


def fig5_phase(plt, a2, a2b, bac):
    adi = float(a2["cases"]["eta1.0"]["adiabatic_arg_deg"])
    fin = float(a2["cases"]["eta1.0"]["finite_time_arg_deg"])
    off = float(a2b["summaries"]["e10_off"]["arg_B_deg"])
    on = float(a2b["summaries"]["e10_on"]["arg_B_deg"])
    qed = float(bac["B"]["phase_deg"])
    names = ["adiabatic", "finite-time", "Ehrenfest off", "Ehrenfest on", "QED"]
    vals = [adi, fin, off, on, qed]
    colors = [COL["grey"], COL["sky"], COL["green"], COL["gold"], COL["orange"]]
    fig, axes = plt.subplots(1, 2, figsize=(7.3, 3.35))
    ax = axes[0]
    th = np.linspace(0, 2 * np.pi, 400)
    ax.plot(np.cos(th), np.sin(th), color="#e0e0e0", lw=0.8)
    ax.axhline(0, color="#eeeeee", lw=0.6)
    ax.axvline(0, color="#eeeeee", lw=0.6)
    for name, ang, c in zip(names, vals, colors):
        rad = np.deg2rad(ang)
        ax.annotate("", xy=(np.cos(rad), np.sin(rad)), xytext=(0, 0), arrowprops=dict(arrowstyle="-|>", color=c, lw=1.6))
        ax.scatter([np.cos(rad)], [np.sin(rad)], color=c, s=18, zorder=5, label=rf"{name} {ang:.1f}$^\circ$")
    ax.set_aspect("equal")
    ax.set_xlim(-1.35, 1.35)
    ax.set_ylim(-1.35, 1.35)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(r"$\arg(B)$ on the unit circle")
    ax.legend(frameon=False, loc="lower left", fontsize=6.8)
    ax = axes[1]
    unwrapped = [adi, fin, off, on - 360.0 if on > 0 else on, qed - 360.0 if qed > 0 else qed]
    ax.plot(np.arange(len(names)), unwrapped, color=COL["grey"], lw=1.0)
    ax.scatter(np.arange(len(names)), unwrapped, c=colors, s=42, zorder=3)
    ax.set_xticks(np.arange(len(names)), ["adiab.", "finite", "kick off", "kick on", "QED"], rotation=20)
    ax.set_ylabel(r"unwrapped $\arg(B)$ (deg)")
    ax.set_title(r"finite-time 73.7% of 79$^\circ$; kick most of remainder")
    fig.tight_layout()
    return save(fig, "fig5_phase_stack")


def fig6_kb(plt, kb):
    slide = kb["sliding"]
    cT = np.array([s["center_T0"] for s in slide], dtype=float)
    kB = np.array([s["ratio"]["qed_over_gh5"]["KB"] for s in slide], dtype=float)
    kM = np.array([s["ratio"]["qed_over_gh5"]["KM"] for s in slide], dtype=float)
    fig, ax = plt.subplots(figsize=(5.4, 3.3))
    ax.axvspan(1.25, 2.25, color="#f6e8c3", alpha=0.7, label="front / switch")
    ax.axvspan(3.0, 4.0, color="#d5e8d4", alpha=0.7, label="plateau interior")
    ax.axhline(1.0, color="#aaaaaa", lw=0.9, ls="--")
    ax.plot(cT, kM, "-o", color=COL["blue"], ms=4.5, label=r"$K_M$ (QED/GH5)")
    ax.plot(cT, kB, "-s", color=COL["orange"], ms=4.5, label=r"$K_B$ (QED/GH5)")
    ax.set_xlabel(r"window centre / $T_0$")
    ax.set_ylabel("QED / GH5 kernel ratio")
    plat = kb["plateau_interior"]["KB_mean"]
    ax.set_title(rf"Odd $K_B$ plateau {plat:.3f} (4 overlapping windows); $K_M\approx 1$")
    ax.legend(frameon=False, loc="upper right")
    ax.set_xlim(1.35, 3.9)
    ax.set_ylim(0.75, 1.28)
    fig.text(
        0.02,
        -0.04,
        "Front-window reversal is a switch-related transient, not a second selection rule.",
        fontsize=7.5,
        color="#444444",
    )
    return save(fig, "fig6_kb_window")


def fig7_material(plt, val):
    order = ["m52_2x2", "m52_4x4", "m88_2x2", "m88_4x4", "full112_2x2"]
    amps = [val["jobs"][k]["channels"]["total"]["amp2d"] for k in order]
    phis = [val["jobs"][k]["channels"]["total"]["jx"]["phase_deg"] for k in order]
    labels = ["M52 2x2", "M52 4x4", "M88 2x2", "M88 4x4", "full112 2x2"]
    colors = [COL["sky"], COL["blue"], COL["gold"], COL["orange"], COL["green"]]
    fig, axes = plt.subplots(1, 2, figsize=(7.3, 3.25))
    ax = axes[0]
    ax.bar(np.arange(len(order)), amps, color=colors)
    ax.set_xticks(np.arange(len(order)), labels, rotation=25, ha="right")
    ax.set_ylabel("Neel-odd H2 amplitude")
    ax.set_title(r"1600 nm coherent: magnetic H2 exists (few-$k$ only)")
    ax = axes[1]
    ax.scatter(np.arange(len(order)), phis, c=colors, s=55, zorder=3)
    ax.plot(np.arange(len(order)), phis, color="#cccccc", lw=0.8)
    ax.set_xticks(np.arange(len(order)), labels, rotation=25, ha="right")
    ax.set_ylabel(r"odd $J_x$ phase (deg)")
    ax.set_title(r"M88 phase jump is few-$k$ sampling, not a material law")
    ax.annotate(
        "not K/K' cancel",
        xy=(3, phis[3]),
        xytext=(2.05, 40),
        fontsize=7,
        arrowprops=dict(arrowstyle="->", color=COL["orange"], lw=0.8),
        color=COL["orange"],
    )
    fig.tight_layout()
    fig.text(
        0.02,
        -0.06,
        "Existence only. Do not densify M52. 2x2/4x4 phases are not quantitative.",
        fontsize=7.5,
        color="#444444",
    )
    return save(fig, "fig7_material_fewk")


def fig8_eta(plt, mic):
    xs = [0.5, 1.0, 1.5]
    scan = {item["eta"]: item for item in mic["lowest_order_test"]["eta_scan"]}
    adi = [scan[e]["kernel_odd_x"]["phase_deg"] for e in xs]
    gh5 = [mic["bac"][f"GH5_eta{e:.1f}_odd_x"]["B"]["phase_deg"] for e in xs]
    rows = [
        ("GH5 eta=0.5", COL["sky"], mic["bac"]["GH5_eta0.5_odd_x"]),
        ("GH5 eta=1.0", COL["blue"], mic["bac"]["GH5_eta1.0_odd_x"]),
        ("GH5 eta=1.5", COL["green"], mic["bac"]["GH5_eta1.5_odd_x"]),
        ("QED eta=1.0", COL["orange"], mic["bac"]["QED_eta1.0_odd_x"]),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(7.3, 3.2))
    ax = axes[0]
    ax.plot(xs, adi, "-o", color=COL["grey"], label=r"adiabatic $\chi^{(2)}$")
    ax.plot(xs, gh5, "-s", color=COL["blue"], label="GH5 arg(B)")
    ax.scatter([1.0], [mic["bac"]["QED_eta1.0_odd_x"]["B"]["phase_deg"]], color=COL["orange"], s=45, zorder=4, label="QED eta=1")
    ax.set_xlabel(r"exchange scale $\eta$")
    ax.set_ylabel("phase (deg)")
    ax.set_title(r"GH5 $\arg(B)$ vs adiabatic; QED only at $\eta=1$")
    ax.legend(frameon=False, fontsize=7)
    ax.set_xticks(xs)
    ax = axes[1]
    args360 = [to_360(r[2]["geometry"]["arg_A_over_B_deg"]) for r in rows]
    ax.bar(np.arange(len(rows)), args360, color=[r[1] for r in rows])
    ax.set_xticks(np.arange(len(rows)), [r[0] for r in rows], rotation=20, ha="right")
    ax.set_ylabel(r"$\arg(A/B)$ (deg)")
    ax.set_title(r"Near-$\pi$ $\arg(A/B)$: GH5 $\eta$ scan vs one QED point")
    ax.axhline(180.0, color="#aaaaaa", lw=0.9, ls="--")
    ax.set_ylim(0.0, 360.0)
    fig.tight_layout()
    return save(fig, "fig8_eta_phase")


def main() -> int:
    plt = setup_mpl()
    mic = load(MIC)
    rscan = load(RSCAN)
    kb = load(KB)
    val = load(VAL)
    a1 = load(A1)
    a2 = load(A2)
    a2b = load(A2B)
    bac = mic["bac"]["QED_eta1.0_odd_x"]
    geom = mic["geometry_qed_eta1_odd_x"]
    paths = {
        "fig1": str(fig1_ellipses(plt, mic["input_moments_q6"])),
        "fig2": str(fig2_plane(plt, bac, geom)),
        "fig3": str(fig3_rscan(plt, rscan)),
        "fig4": str(fig4_gating(plt, a1)),
        "fig5": str(fig5_phase(plt, a2, a2b, bac)),
        "fig6": str(fig6_kb(plt, kb)),
        "fig7": str(fig7_material(plt, val)),
        "fig8": str(fig8_eta(plt, mic)),
    }
    plt.close("all")
    man = {
        "no_new_qed": True,
        "no_new_sbe": True,
        "figures": paths,
        "frozen": {
            "pi_over_0_odd_2d": mic["frozen_qed"]["pi_over_0_odd_2d"],
            "covflip_phase_x_deg": mic["frozen_qed"]["covflip_phase_x_deg"],
            "R": mic["input_moments_q6"]["M_over_alpha2"],
            "rho": geom["rho_absA_over_absB"],
            "arg_A_over_B_deg": geom["arg_A_over_B_deg"],
        },
        "kb_verdict": kb["verdict"],
        "rscan_verdict": rscan["verdict"],
        "valley_verdict": val.get("verdict"),
    }
    dump(OUT / "paper_figures_manifest.json", man)
    print(json.dumps(paths, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
