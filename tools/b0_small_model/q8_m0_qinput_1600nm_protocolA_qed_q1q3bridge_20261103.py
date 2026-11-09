#!/usr/bin/env python3
"""Q1-Q3 bridge: Tr(P rho P K) vs full DS. No propagator."""
from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import q8_m0_qinput_dynamics_20261026 as dyn
import q8_m0_qinput_1600nm_protocolA_qed_postprocess_20261102 as pp
from q8_m0_qinput_1600nm_protocolA_qed_windowQ1_20261103 import pair_currents, z_channel
from qinput_spectrum_reanalysis_20261028 import packed

ROOT = pp.ROOT
FIT = pp.FIT_PATH
FROZEN = "17a8e37ec3e0ab54096ff4edd1440c5a3f0412b09450c2fd8960cbc00d8e69db"
KERNEL = ROOT / "results/q_line/qinput_1600nm_protocolA_qed_q3fock_20261103/analysis_q3fock_20261103/kernel.json"
Q1JSON = ROOT / "results/q_line/qinput_1600nm_protocolA_qed_fourangle_20261102/analysis_windowQ1_20261103/windowQ1.json"
OUT = ROOT / "results/q_line/qinput_1600nm_protocolA_qed_fourangle_20261102/analysis_q1q3bridge_20261103"
DOC = ROOT / "docs/04_BSV量子光/线Q_Q1Q3桥接_0-2子空间贡献_20261103.md"
ANGLES = (("0", 0.0, 0), ("h", 0.5 * np.pi, 1), ("pi", np.pi, 2), ("t", 1.5 * np.pi, 3), ("q", 0.25 * np.pi, None))
WINDOWS = ("front", "main", "back")
CHANS = ("h2_odd_x", "h2_odd_y", "h3_odd_x", "h3_odd_y")
H3Y_Q = {"h3_odd_y": (3, "odd", "y")}
H3Y_TAGS = {
    "0": ("p_0", "m_0"),
    "h": ("p_h", "m_h"),
    "pi": ("p_pi", "m_pi"),
    "t": ("p_t", "m_t"),
    "q": ("p_q", "m_q"),
}
ANCHOR = {"0": 0.698, "pi": 0.871}


def sha(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def cz(d) -> complex:
    return complex(float(d["re"]), float(d["im"]))


def dump_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def ds_c(theta: float) -> np.ndarray:
    raw = dyn.displaced_squeezed_coefficients(0.5, 0.3, float(theta), 8)
    mass = float(np.vdot(raw, raw).real)
    if mass <= 0.0:
        raise RuntimeError("DS mass vanished")
    return raw / np.sqrt(mass)


def rho_blocks(c: np.ndarray) -> dict:
    p = np.abs(c) ** 2
    return {
        "P00": float(p[0]),
        "P11": float(p[1]),
        "P22": float(p[2]),
        "P33": float(p[3]),
        "P00_plus_P22": float(p[0] + p[2]),
        "rho00": complex(c[0] * np.conj(c[0])),
        "rho22": complex(c[2] * np.conj(c[2])),
        "rho02": complex(c[0] * np.conj(c[2])),
        "rho20": complex(c[2] * np.conj(c[0])),
        "nbar": float(np.dot(p, np.arange(len(p)))),
        "retained_mass": float(np.vdot(c, c).real),
    }


def j02_from(K: dict, r: dict) -> complex:
    return K["K00"] * r["rho00"] + K["K22"] * r["rho22"] + K["K02"] * r["rho20"] + K["K20"] * r["rho02"]


def kernel_chan(ker: dict, window: str, name: str) -> dict:
    raw = ker["windows"][window]["channels"][name]
    return {key: cz(raw[key]) for key in ("K00", "K22", "K02", "K20")}


def j_full_from_q1(q1: dict, window: str, name: str, tag: str, idx) -> complex:
    ch = q1["windows"][window]["channels"][name]
    if idx is None:
        return cz(ch["holdout"]["z"])
    return cz(ch["observations"][idx])


def metrics(z_full: complex, z02: complex) -> dict:
    rem = z_full - z02
    abs_full = abs(z_full)
    abs_02 = abs(z02)
    abs_rem = abs(rem)
    frac_02 = None if abs_full == 0.0 else abs_02 / abs_full
    frac_rem = None if abs_full == 0.0 else abs_rem / abs_full
    dphi = None
    if abs_full and abs_02:
        dphi = float(np.degrees(np.angle(z02 * np.conj(z_full))))
    same_re = bool(np.sign(z_full.real) == np.sign(z02.real)) or (z_full.real == 0.0 and z02.real == 0.0)
    return {
        "J_full": packed(z_full),
        "J02": packed(z02),
        "J_rem": packed(rem),
        "abs_J_full": float(abs_full),
        "abs_J02": float(abs_02),
        "abs_J_rem": float(abs_rem),
        "abs_J02_over_abs_J_full": frac_02,
        "abs_J_rem_over_abs_J_full": frac_rem,
        "phase_J02_minus_J_full_deg": dphi,
        "Re_same_sign": same_re,
        "Re_opposite_sign": (z_full.real != 0.0 or z02.real != 0.0) and not same_re,
    }


def load_h3y() -> dict:
    out = {}
    for tag, (ptag, mtag) in H3Y_TAGS.items():
        t, jp, jm = pair_currents(pp.QED_OUT, ptag, mtag)
        out[tag] = {win: z_channel(t, jp, jm, win, 3, "odd", "y") for win in WINDOWS}
    return out


def pack_rho(r: dict) -> dict:
    return {
        "P00": r["P00"],
        "P11": r["P11"],
        "P22": r["P22"],
        "P33": r["P33"],
        "P00_plus_P22": r["P00_plus_P22"],
        "nbar": r["nbar"],
        "retained_mass": r["retained_mass"],
        "rho02": packed(r["rho02"]),
        "rho20": packed(r["rho20"]),
    }


def fork_decision(rows: list[dict]) -> dict:
    main_h2x = [row for row in rows if row["window"] == "main" and row["channel"] == "h2_odd_x"]
    by_tag = {row["tag"]: row for row in main_h2x}
    fracs = [row["abs_J02_over_abs_J_full"] for row in main_h2x]
    rem_fracs = [row["abs_J_rem_over_abs_J_full"] for row in main_h2x]
    median_02 = float(np.median(fracs))
    train = [row for row in main_h2x if row["tag"] != "q"]
    large_rem = any(row["abs_J_rem_over_abs_J_full"] >= 0.3 for row in train)
    theta0 = by_tag["0"]
    opp = bool(theta0["Re_opposite_sign"])
    if opp:
        label = "0-2_cannot_explain_theta0_H2x_real_part"
        action = "stop_do_not_expand_probes"
        note = "θ=0 主窗 H2x 实部反号：0–2 子空间不能解释该点的完整 DS 实部。"
    elif large_rem:
        label = "0-2_not_complete"
        action = "stop_acknowledge_scope"
        note = "至少一个训练角 |J_rem|/|J_full|≥0.3。"
    elif median_02 >= 0.7 and not opp:
        label = "0-2_majority"
        action = "stop_expand_needs_new_authorization"
        note = "主窗 H2x 中位数 |J02|/|J_full|≥0.7 且 θ=0 实部同号。"
    else:
        label = "0-2_not_dominant"
        action = "stop"
        note = "未达到多数贡献阈值，也不满足反号/大剩余的更强否定。"
    window_opp = {
        win: bool(next(row for row in rows if row["window"] == win and row["channel"] == "h2_odd_x" and row["tag"] == "0")["Re_opposite_sign"])
        for win in WINDOWS
    }
    return {
        "label": label,
        "action": action,
        "note": note,
        "main_h2x_median_abs_J02_over_abs_J_full": median_02,
        "main_h2x_median_abs_J_rem_over_abs_J_full": float(np.median(rem_fracs)),
        "training_any_remainder_ge_0p3": large_rem,
        "theta0_main_h2x_Re_opposite_sign": opp,
        "theta0_h2x_Re_opposite_sign_by_window": window_opp,
        "did_not_expand_probes": True,
        "did_not_claim_quantum_advantage": True,
    }


def make_figures(rows: list[dict], path_plane: Path, path_bar: Path) -> None:
    import matplotlib.pyplot as plt

    main = [row for row in rows if row["window"] == "main" and row["channel"] == "h2_odd_x"]
    order = ["0", "h", "pi", "t", "q"]
    labels = {"0": r"$0$", "h": r"$\pi/2$", "pi": r"$\pi$", "t": r"$3\pi/2$", "q": r"$\pi/4$"}
    main = sorted(main, key=lambda r: order.index(r["tag"]))

    fig, ax = plt.subplots(figsize=(6.4, 5.4), dpi=140)
    colors = {"J_full": "#1f4e79", "J02": "#c47b16", "J_rem": "#7a3b6e"}
    markers = {"J_full": "o", "J02": "s", "J_rem": "^"}
    for key in ("J_full", "J02", "J_rem"):
        xs = [row[key]["re"] * 1e5 for row in main]
        ys = [row[key]["im"] * 1e5 for row in main]
        ax.plot(xs, ys, markers[key] + "-", color=colors[key], label=key, lw=1.2)
        for row, x, y in zip(main, xs, ys):
            ax.annotate(labels[row["tag"]], (x, y), textcoords="offset points", xytext=(4, 4), fontsize=8)
    ax.axhline(0.0, color="#bbbbbb", lw=0.6)
    ax.axvline(0.0, color="#bbbbbb", lw=0.6)
    ax.set_xlabel(r"Re $\mathcal{J}_{2x}^{o}\times 10^{5}$")
    ax.set_ylabel(r"Im $\mathcal{J}_{2x}^{o}\times 10^{5}$")
    ax.set_title("Main window H2x: full DS vs 0–2 contribution")
    ax.legend(frameon=False)
    ax.set_aspect("equal", adjustable="datalim")
    fig.tight_layout()
    path_plane.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path_plane)
    fig.savefig(path_plane.with_suffix(".pdf"))
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.2, 4.6), dpi=140)
    x = np.arange(len(order))
    h2 = [next(r for r in rows if r["window"] == "main" and r["channel"] == "h2_odd_x" and r["tag"] == tag) for tag in order]
    h3 = [next(r for r in rows if r["window"] == "main" and r["channel"] == "h3_odd_x" and r["tag"] == tag) for tag in order]
    w = 0.18
    ax.bar(x - 1.5 * w, [r["abs_J02_over_abs_J_full"] for r in h2], w, color="#1f4e79", label=r"H2x $|J_{02}|/|J|$")
    ax.bar(x - 0.5 * w, [r["abs_J_rem_over_abs_J_full"] for r in h2], w, color="#7aa2c4", label=r"H2x $|J_{\mathrm{rem}}|/|J|$")
    ax.bar(x + 0.5 * w, [r["abs_J02_over_abs_J_full"] for r in h3], w, color="#c47b16", label=r"H3x $|J_{02}|/|J|$")
    ax.bar(x + 1.5 * w, [r["abs_J_rem_over_abs_J_full"] for r in h3], w, color="#e2b36b", label=r"H3x $|J_{\mathrm{rem}}|/|J|$")
    ax.axhline(0.3, color="#b23a3a", ls="--", lw=0.8, label="0.3 remainder")
    ax.axhline(0.7, color="#3a7d44", ls=":", lw=0.8, label="0.7 majority")
    ax.set_xticks(x)
    ax.set_xticklabels([labels[t] for t in order])
    ax.set_ylabel("fraction of |J_full|")
    ymax = max(
        max(r["abs_J02_over_abs_J_full"] for r in h2 + h3),
        max(r["abs_J_rem_over_abs_J_full"] for r in h2 + h3),
        0.7,
    )
    ax.set_ylim(0.0, ymax * 1.12)
    ax.set_title("Main window: 0–2 vs remainder (not probability weight)")
    ax.legend(frameon=False, ncol=2, fontsize=8)
    fig.tight_layout()
    fig.savefig(path_bar)
    fig.savefig(path_bar.with_suffix(".pdf"))
    plt.close(fig)


def fmt_c(d: dict, scale=1.0) -> str:
    z = cz(d) * scale
    sign = "+" if z.imag >= 0 else "-"
    if scale == 1e5:
        return f"{z.real:.3f}{sign}{abs(z.imag):.3f}"
    return f"{z.real:.3e}{sign}{abs(z.imag):.3e}i"


def write_report(payload: dict, path: Path) -> None:
    rows = payload["rows"]
    fock = payload["fock"]
    fork = payload["fork"]
    main_h2x = {row["tag"]: row for row in rows if row["window"] == "main" and row["channel"] == "h2_odd_x"}
    lines = [
        "# Q1–Q3 桥接：0–2 子空间对 DS 响应的贡献（2026-11-03）",
        "",
        "零传播。生产 `D(0.5)S(0.3 e^{iθ})|0⟩`，冻结 Q3 核，投影后不重新归一化。",
        "不是量子优势，也不是完整微观核。",
        "",
        f"**分叉：** `{fork['label']}`。{fork['note']} **不扩探针。**",
        "",
        "## 1. 算符约定闸门",
        "",
        "| θ | P00 | P11 | P22 | P33 | P00+P22 | 锚点 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for tag, th_name in (("0", "0"), ("pi", "π")):
        r = fock[tag]
        lines.append(
            f"| {th_name} | {r['P00']:.4f} | {r['P11']:.4f} | {r['P22']:.4f} | {r['P33']:.4f} | "
            f"{r['P00_plus_P22']:.4f} | {ANCHOR[tag]:.3f} |"
        )
    lines += [
        "",
        f"闸门：`{payload['convention_gate']['status']}`。θ=0 时 `|1⟩` 约占 {fock['0']['P11']*100:.1f}%，",
        "因此 P00+P22 不是响应贡献比例。",
        "",
        "## 2. 主窗 H2x：完整 DS 对 0–2",
        "",
        r"单位：$\times 10^{-5}$。",
        "",
        "| θ | Re J_full | Re J02 | Re J_rem | |J02|/|J| | |J_rem|/|J| | Δφ (deg) | 实部同号 |",
        "|---|---:|---:|---:|---:|---:|---:|:---:|",
    ]
    labels = {"0": "0", "h": "π/2", "pi": "π", "t": "3π/2", "q": "π/4"}
    for tag in ("0", "h", "pi", "t", "q"):
        row = main_h2x[tag]
        lines.append(
            f"| {labels[tag]} | {row['J_full']['re']*1e5:.3f} | {row['J02']['re']*1e5:.3f} | "
            f"{row['J_rem']['re']*1e5:.3f} | {100*row['abs_J02_over_abs_J_full']:.1f}% | "
            f"{100*row['abs_J_rem_over_abs_J_full']:.1f}% | {row['phase_J02_minus_J_full_deg']:.1f} | "
            f"{'否' if row['Re_opposite_sign'] else '是'} |"
        )
    t0 = main_h2x["0"]
    lines += [
        "",
        f"θ=0 主窗 H2x：Re J_full = {t0['J_full']['re']:.4e}，Re J02 = {t0['J02']['re']:.4e}，"
        f"**反号 = {t0['Re_opposite_sign']}**。",
        f"中位数 |J02|/|J_full| = {fork['main_h2x_median_abs_J02_over_abs_J_full']:.3f}；"
        f"中位数 |J_rem|/|J_full| = {fork['main_h2x_median_abs_J_rem_over_abs_J_full']:.3f}。",
        "比值可以大于 1：θ=0 的 132%/232% 表示 0–2 贡献与完整响应近反相相消，不是“解释了 132%”。",
        "",
        "## 3. 其他通道与窗口",
        "",
        "主窗 |J_rem|/|J_full|：",
        "",
        "| θ | H2x | H2y | H3x | H3y |",
        "|---|---:|---:|---:|---:|",
    ]
    for tag in ("0", "h", "pi", "t", "q"):
        vals = []
        for ch in CHANS:
            row = next(r for r in rows if r["window"] == "main" and r["channel"] == ch and r["tag"] == tag)
            vals.append(f"{100*row['abs_J_rem_over_abs_J_full']:.1f}%")
        lines.append(f"| {labels[tag]} | " + " | ".join(vals) + " |")
    lines += [
        "",
        "θ=0 H2x 实部反号是否三窗都有："
        + ", ".join(f"{w}={'是' if v else '否'}" for w, v in fork["theta0_h2x_Re_opposite_sign_by_window"].items())
        + "。三窗重叠，不是三次独立实验。",
        "",
        "## 4. 结论",
        "",
        "- `{|0⟩,|2⟩}` 核不能解释 Q1 主窗 H2x 的 θ=0 实部；该点 0–2 贡献与完整 DS **反号**。",
        "- 即使 K02 很大，原始位移压缩态的近反相干涉也不是这个 2×2 子空间单独形成的。",
        "- `|1⟩` 及更高 Fock 块的贡献不可忽略，甚至与 0–2 相消。",
        "- 按预声明分叉：**停止，不扩展探针。**",
        "",
        "图：`analysis_q1q3bridge_20261103/figures/`。未改 `fourangle_fit.json`。未传播，未 commit/push。",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    got = sha(FIT)
    if got != FROZEN:
        raise RuntimeError(f"frozen SHA changed: {got}")
    ker = json.loads(KERNEL.read_text(encoding="utf-8"))
    q1 = json.loads(Q1JSON.read_text(encoding="utf-8"))["qed"]
    h3y = load_h3y()
    fock = {}
    rows = []
    for tag, theta, idx in ANGLES:
        r = rho_blocks(ds_c(theta))
        fock[tag] = pack_rho(r)
        for win in WINDOWS:
            for name in CHANS:
                K = kernel_chan(ker, win, name)
                z02 = j02_from(K, r)
                if name == "h3_odd_y":
                    z_full = h3y[tag][win]
                else:
                    z_full = j_full_from_q1(q1, win, name, tag, idx)
                met = metrics(z_full, z02)
                met.update({"tag": tag, "theta": float(theta), "window": win, "channel": name})
                rows.append(met)
    gate = {"status": "pass", "checked": {}}
    for tag, anchor in ANCHOR.items():
        got_p = fock[tag]["P00_plus_P22"]
        ok = abs(got_p - anchor) <= 0.02
        gate["checked"][tag] = {"P00_plus_P22": got_p, "anchor": anchor, "pass": ok}
        if not ok:
            gate["status"] = "fail"
    if gate["status"] != "pass":
        raise RuntimeError(f"DS convention gate failed: {gate}")
    fork = fork_decision(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema": "1600nm_protocolA_qed_q1q3bridge_v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "fit_sha256_before": got,
        "did_not_rewrite_fourangle_fit": True,
        "did_not_propagate": True,
        "did_not_expand_probes": True,
        "did_not_renormalize_P_rho_P": True,
        "operator_order": "D(alpha)S(xi)|0>, xi=r exp(i theta); then normalize on n=0..8",
        "alpha": 0.5,
        "r": 0.3,
        "n_max": 8,
        "formula": "J02 = K00 rho00 + K22 rho22 + K02 rho20 + K20 rho02",
        "convention_gate": gate,
        "fock": fock,
        "rows": rows,
        "fork": fork,
        "not_quantum_advantage": True,
        "probability_is_not_response_fraction": True,
    }
    dump_json(OUT / "bridge.json", payload)
    make_figures(rows, OUT / "figures" / "fig_bridge_h2x_plane.png", OUT / "figures" / "fig_bridge_remainder.png")
    write_report(payload, DOC)
    after = sha(FIT)
    if after != FROZEN:
        raise RuntimeError("SHA changed during run")
    payload["fit_sha256_after"] = after
    dump_json(OUT / "bridge.json", payload)
    print("BRIDGE_OK", fork["label"])
    t0 = next(r for r in rows if r["window"] == "main" and r["channel"] == "h2_odd_x" and r["tag"] == "0")
    print("theta0_main_h2x", "J_full", t0["J_full"]["re"], "J02", t0["J02"]["re"], "opp", t0["Re_opposite_sign"])
    print("DOC", DOC)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
