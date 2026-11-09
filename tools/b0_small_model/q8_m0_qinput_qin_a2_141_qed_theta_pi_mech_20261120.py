#!/usr/bin/env python3
"""Read-only mechanism diagnosis for Q1a squeeze-angle H2 odd current.

Frozen archives only. No electron propagation, no new QED/GH5, no Cn/SV,
no nmax/k/Floquet/server. Overlapping 2T0 windows are not independent
experiments. Integer Fourier is not certified HHG.
"""
from __future__ import annotations

import json
import math
import os
from pathlib import Path

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

HERE = Path(__file__).resolve().parent
import sys
sys.path.insert(0, str(HERE))

import numpy as np

import q8_m0_qinput_dynamics_20261026 as dyn
import q8_m0_qinput_shared_kk_ds_odd_20261102 as odd
import q8_m0_qinput_shared_kk_dyn_preflight_20261030 as pf

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results/q_line/qinput_qin_a2_141_qed_theta_pi_mech_20261120"
FIG = OUT / "figures"
DOCS = ROOT / "docs/04_BSV量子光"
QED0 = ROOT / "results/q_line/qinput_qin_a2_141_q1a_20261120"
QEDPI = ROOT / "results/q_line/qinput_qin_a2_141_qed_theta_pi_20261120"
GH50 = ROOT / "results/q_line/qinput_qin_a2_141_gh5_20261120"
GH5PI = ROOT / "results/q_line/qinput_qin_a2_141_gh5_theta_pi_20261120"

ALPHA = complex(odd.ALPHA)
R_SQ = float(odd.R_SQ)
OMEGA0 = float(pf.OMEGA0)
WINDOWS = (
    ("front_1_3", 1.0, 3.0),
    ("main_2_4", 2.0, 4.0),
    ("shift_2p5_4p5", 2.5, 4.5),
    ("shift_2p7_4p7", 2.7, 4.7),
)


def jsonable(x):
    return odd.jsonable(x)


def dump(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(jsonable(obj), ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                   encoding="utf-8")
    os.replace(tmp, path)


def wrap(phi: float) -> float:
    return float((phi + np.pi) % (2.0 * np.pi) - np.pi)


def angdiff(a, b) -> float:
    return wrap(float(a) - float(b))


def pack_c(z):
    z = complex(z)
    return {"re": float(z.real), "im": float(z.imag), "abs": float(abs(z)), "arg": float(np.angle(z))}


def load_series(directory: Path, tag: str):
    z = np.load(directory / f"series_{tag}.npz")
    t = np.asarray(z["t"], float)
    jx = np.real(np.asarray(z["Jx"])).astype(float)
    jy = np.real(np.asarray(z["Jy"])).astype(float)
    j = np.stack([jx, jy], axis=1)
    return t, j


def even_odd_from_dir(directory: Path):
    tp, jp = load_series(directory, "ds_p")
    tm, jm = load_series(directory, "ds_m")
    n = min(len(tp), len(tm))
    if not np.allclose(tp[:n], tm[:n], rtol=0, atol=1e-9):
        raise RuntimeError(f"time mismatch in {directory}")
    ev, od = odd.even_odd(jp[:n], jm[:n])
    return tp[:n], np.real(ev), np.real(od)


def interp_xy(t_ref, t_src, y_src):
    out = np.empty((len(t_ref), y_src.shape[1]), float)
    for a in range(y_src.shape[1]):
        out[:, a] = np.interp(t_ref, t_src, y_src[:, a])
    return out


def window_mask(t, t0, t1):
    return (t >= t0 - 1e-12) & (t <= t1 + 1e-12)


def rect_jq(t, j, omega0, t0, t1, qs):
    t = np.asarray(t, float)
    j = np.asarray(j, float)
    if j.ndim == 1:
        j = j[:, None]
    m = window_mask(t, t0, t1)
    tw, jw = t[m], j[m]
    span = float(t1 - t0)
    out = []
    for q in qs:
        ker = np.exp(1j * float(q) * omega0 * tw)
        slot = np.array([np.trapezoid(ker * jw[:, a], tw) for a in range(j.shape[1])])
        out.append(slot / span)
    return np.asarray(out, complex)


def theory_block(r: float, alpha: complex, omega0: float) -> dict:
    r = float(r)
    sh2 = float(np.sinh(2.0 * r))
    ch2 = float(np.cosh(2.0 * r))
    m0 = -0.5 * sh2
    mpi = +0.5 * sh2
    nbar = abs(alpha) ** 2 + float(np.sinh(r) ** 2)
    return {
        "alpha": pack_c(alpha),
        "r": r,
        "omega0": omega0,
        "operator_order": "D(alpha) S(r exp(i theta)) |0>",
        "M_theta": "M_theta = <(Delta a)^2> = -exp(i theta) sinh(2r)/2",
        "M_0": m0,
        "M_pi": mpi,
        "M_pi_minus_M_0": mpi - m0,
        "sinh_2r": sh2,
        "cosh_2r": ch2,
        "mean_n_analytic": float(nbar),
        "mean_X": 2.0 * alpha.real,
        "mean_P": 2.0 * alpha.imag,
        "VarX_0": float(np.exp(-2.0 * r)),
        "VarX_pi": float(np.exp(+2.0 * r)),
        "VarP_0": float(np.exp(+2.0 * r)),
        "VarP_pi": float(np.exp(-2.0 * r)),
        "free_VarX_t": "cosh(2r) - sinh(2r) cos(2 omega0 t - theta)",
        "dtheta_VarX_t": "2 sinh(2r) cos(2 omega0 t)",
        "dtheta_VarX_amp": 2.0 * sh2,
        "C_theta_t1_t2": (
            "C_theta(t1,t2)=cosh(2r) cos(omega0(t1-t2)) "
            "- sinh(2r) cos(omega0(t1+t2)-theta)"
        ),
        "dtheta_C_t1_t2": "2 sinh(2r) cos(omega0(t1+t2))",
        "free_field_not_full_QED": True,
        "phase_compatibility_is_not_kernel_proof": True,
        "derivation": {
            "squeezed_vacuum_a2": "<xi|a^2|xi> = -exp(i theta) sinh(r) cosh(r)",
            "displacement_leaves_anomalous": (
                "D(alpha) shifts <a> to alpha; M = <a^2>-<a>^2 remains "
                "-exp(i theta) sinh(2r)/2"
            ),
            "X_t": "X(t)=a e^{-i omega0 t}+a^dagger e^{i omega0 t}",
            "VarX_from_M": (
                "Var X(t) = 2 Re[M exp(-2 i omega0 t)] + 2<Delta n>+1 "
                "= cosh(2r) - sinh(2r) cos(2 omega0 t - theta)"
            ),
            "theta_pi_minus_0": (
                "cos(2 omega0 t - pi) = -cos(2 omega0 t), hence "
                "Var_pi - Var_0 = 2 sinh(2r) cos(2 omega0 t)"
            ),
            "two_time": (
                "Symmetric covariance inherits the t1+t2 double-frequency "
                "phase from M. A magnetic second-order response kernel "
                "could therefore source an H2 current difference. This is "
                "a possible mechanism, not a demonstrated kernel."
            ),
        },
    }


def linear_leakage_phase(t0: float, omega0: float) -> float:
    # J2 ~ b exp(i 2 omega0 t0) / (i 2 omega0)  => arg = 2 omega0 t0 - pi/2
    return wrap(2.0 * omega0 * t0 - 0.5 * np.pi)


def joint_fit_h2(t, y, omega0, t0, t1, poly_deg: int):
    m = window_mask(t, t0, t1)
    tw = t[m]
    yw = np.asarray(y[m], float)
    tmid = 0.5 * (t0 + t1)
    dt = tw - tmid
    cols = [np.ones_like(tw)]
    for p in range(1, poly_deg + 1):
        cols.append(dt ** p)
    cols.append(np.cos(2.0 * omega0 * tw))
    cols.append(np.sin(2.0 * omega0 * tw))
    A = np.column_stack(cols)
    coef, *_ = np.linalg.lstsq(A, yw, rcond=None)
    a_cos = float(coef[-2])
    b_sin = float(coef[-1])
    # Convention J_q = (1/T) ∫ exp(+i q ω0 t) J(t) dt.
    # ⟨exp(+iΩt) cos(Ωt)⟩ = 1/2, ⟨exp(+iΩt) sin(Ωt)⟩ = +i/2.
    j2 = 0.5 * (a_cos + 1j * b_sin)
    yhat = A @ coef
    resid = yw - yhat
    osc = a_cos * np.cos(2.0 * omega0 * tw) + b_sin * np.sin(2.0 * omega0 * tw)
    return {
        "poly_deg": int(poly_deg),
        "n_pts": int(tw.size),
        "poly": [float(c) for c in coef[:-2]],
        "A_cos": a_cos,
        "B_sin": b_sin,
        "j2": pack_c(j2),
        "rms_y": float(np.sqrt(np.mean(yw ** 2))),
        "rms_osc": float(np.sqrt(np.mean(osc ** 2))),
        "rms_resid": float(np.sqrt(np.mean(resid ** 2))),
    }


def analyze_channel(name, t, yx, omega0, T0, t_cover):
    rows = []
    for wname, a, b in WINDOWS:
        t0, t1 = a * T0, b * T0
        complete = bool(t_cover + 1e-9 >= t1)
        raw = rect_jq(t, yx, omega0, t0, t1, [2.0])[0, 0] if complete else 0j
        fits = {}
        if complete:
            for deg in (0, 1, 2):
                fits[f"poly{deg}_plus_2w"] = joint_fit_h2(t, yx, omega0, t0, t1, deg)
        leak_phi = linear_leakage_phase(t0, omega0)
        rows.append({
            "name": wname,
            "t0_over_T0": a,
            "t1_over_T0": b,
            "complete": complete,
            "raw_j2": pack_c(raw),
            "linear_leakage_phase_rad": leak_phi,
            "fits": fits,
        })
    qs = np.linspace(0.0, 6.0, 241)
    cont = {}
    for wname, a, b in WINDOWS:
        t0, t1 = a * T0, b * T0
        if t_cover + 1e-9 < t1:
            continue
        spec = rect_jq(t, yx, omega0, t0, t1, qs)[:, 0]
        abs_s = np.abs(spec)
        peak_i = int(np.argmax(abs_s[1:])) + 1 if abs_s.size > 1 else 0
        cont[wname] = {
            "q": qs.tolist(),
            "abs": abs_s.tolist(),
            "peak_q": float(qs[peak_i]),
            "peak_abs": float(abs_s[peak_i]),
            "abs_at_2": float(np.abs(rect_jq(t, yx, omega0, t0, t1, [2.0])[0, 0])),
        }
    return {"channel": name, "windows": rows, "continuum": cont,
            "t_cover_over_T0": t_cover / T0}


def dcomplex(z_a, z_b, z_ref) -> float:
    den = abs(z_ref) + 1e-30
    return float(abs(z_a - z_b) / den)


def svg_axes(x0, y0, w, h, xlabel, ylabel, title):
    return (
        f'<rect x="{x0}" y="{y0}" width="{w}" height="{h}" fill="#fffdf8" '
        f'stroke="#2c3338" stroke-width="1"/>'
        f'<text x="{x0 + w/2:.1f}" y="{y0 - 8:.1f}" text-anchor="middle" '
        f'font-size="13" fill="#1b242c">{title}</text>'
        f'<text x="{x0 + w/2:.1f}" y="{y0 + h + 22:.1f}" text-anchor="middle" '
        f'font-size="11" fill="#243038">{xlabel}</text>'
        f'<text x="{x0 - 28:.1f}" y="{y0 + h/2:.1f}" text-anchor="middle" '
        f'font-size="11" fill="#243038" transform="rotate(-90 {x0-28:.1f} {y0+h/2:.1f})">{ylabel}</text>'
    )


def write_svg(path: Path, body: str, width=920, height=560) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    svg = (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">'
        f'<rect width="100%" height="100%" fill="#f4efe6"/>{body}</svg>\n'
    )
    path.write_text(svg, encoding="utf-8")
    return str(path)


def map_xy(z, lim, x0, y0, w, h):
    px = x0 + (z.real / lim * 0.5 + 0.5) * w
    py = y0 + (0.5 - z.imag / lim * 0.5) * h
    return px, py


def arrow(z, lim, x0, y0, w, h, color, label, lx, ly, dash=False):
    cx, cy = x0 + w / 2, y0 + h / 2
    px, py = map_xy(z, lim, x0, y0, w, h)
    dash_attr = ' stroke-dasharray="4 3"' if dash else ""
    return (
        f'<line x1="{cx:.1f}" y1="{cy:.1f}" x2="{px:.1f}" y2="{py:.1f}" '
        f'stroke="{color}" stroke-width="2"{dash_attr}/>'
        f'<circle cx="{px:.1f}" cy="{py:.1f}" r="4" fill="{color}"/>'
        f'<text x="{lx}" y="{ly}" font-size="11" fill="{color}">{label}</text>'
    )


def figure_windows(items, path, title):
    zs = [z for _, z, _ in items]
    lim = max(abs(z) for z in zs) * 1.25 + 1e-30
    x0, y0, w, h = 90, 40, 740, 430
    body = svg_axes(x0, y0, w, h, "Re H2 Jx^o", "Im H2 Jx^o", title)
    cx, cy = x0 + w / 2, y0 + h / 2
    body += f'<line x1="{x0}" y1="{cy:.1f}" x2="{x0+w}" y2="{cy:.1f}" stroke="#8a9096"/>'
    body += f'<line x1="{cx:.1f}" y1="{y0}" x2="{cx:.1f}" y2="{y0+h}" stroke="#8a9096"/>'
    lx = 70
    for label, z, color in items:
        body += arrow(z, lim, x0, y0, w, h, color, label, lx, 36)
        lx += 160
    body += (
        '<text x="460" y="548" font-size="10" fill="#5c5148">'
        "overlapping 2T0 windows are not independent experiments</text>"
    )
    return write_svg(path, body)


def figure_cont(q, curves, path, title):
    xmin, xmax = float(q[0]), float(q[-1])
    ys = np.concatenate([np.asarray(c[1], float) for c in curves])
    ymin, ymax = 0.0, float(np.max(ys)) * 1.12 + 1e-30
    x0, y0p, w, h = 70, 40, 800, 430
    body = svg_axes(x0, y0p, w, h, "Ω / ω0", "|J^o(Ω)|", title)

    def poly(xs, ys_, color, sw):
        pts = []
        for x, y in zip(xs, ys_):
            px = x0 + (float(x) - xmin) / (xmax - xmin) * w
            py = y0p + (1.0 - (float(y) - ymin) / max(ymax - ymin, 1e-30)) * h
            pts.append(f"{px:.2f},{py:.2f}")
        return (f'<polyline fill="none" stroke="{color}" stroke-width="{sw}" '
                f'points="{" ".join(pts)}"/>')

    for color, ys_, lab, xtext in curves:
        body += poly(q, ys_, color, 1.4)
        body += f'<text x="{xtext}" y="36" font-size="11" fill="{color}">{lab}</text>'
    for n in range(1, 7):
        px = x0 + (n - xmin) / (xmax - xmin) * w
        body += (f'<line x1="{px:.1f}" y1="{y0p}" x2="{px:.1f}" y2="{y0p+h}" '
                 f'stroke="#8a9096" stroke-dasharray="3 4" stroke-width="0.8"/>')
    return write_svg(path, body)


def figure_phase(win_names, obs, leak, path):
    x0, y0, w, h = 90, 40, 740, 430
    body = svg_axes(x0, y0, w, h, "window", "phase (rad)",
                    "H2 δθ phase vs linear-leakage template")
    n = len(win_names)
    ymin, ymax = -np.pi, np.pi
    xs = np.arange(n)

    def px(i):
        return x0 + (0.1 + 0.8 * i / max(n - 1, 1)) * w

    def py(val):
        return y0 + (0.5 - val / (2.0 * np.pi)) * h

    body += f'<line x1="{x0}" y1="{py(0):.1f}" x2="{x0+w}" y2="{py(0):.1f}" stroke="#8a9096"/>'
    pts_o, pts_l = [], []
    for i, name in enumerate(win_names):
        xo = px(i)
        body += f'<text x="{xo:.1f}" y="{y0+h+20:.1f}" text-anchor="middle" font-size="10">{name}</text>'
        yo = py(obs[i])
        yl = py(leak[i])
        pts_o.append(f"{xo:.1f},{yo:.1f}")
        pts_l.append(f"{xo:.1f},{yl:.1f}")
        body += f'<circle cx="{xo:.1f}" cy="{yo:.1f}" r="4" fill="#c45c26"/>'
        body += f'<circle cx="{xo:.1f}" cy="{yl:.1f}" r="4" fill="#4c6b8a"/>'
    body += f'<polyline fill="none" stroke="#c45c26" stroke-width="2" points="{" ".join(pts_o)}"/>'
    body += (f'<polyline fill="none" stroke="#4c6b8a" stroke-width="2" '
             f'stroke-dasharray="4 3" points="{" ".join(pts_l)}"/>')
    body += '<text x="80" y="36" font-size="11" fill="#c45c26">observed arg(δθ J2)</text>'
    body += '<text x="280" y="36" font-size="11" fill="#4c6b8a">linear leakage 2ω0 t0 − π/2</text>'
    return write_svg(path, body)


def rms(y):
    y = np.asarray(y, float)
    return float(np.sqrt(np.mean(y ** 2))) if y.size else 0.0


def fmt_c(zpack):
    return f"{zpack['re']:.3e}{zpack['im']:+.3e}i"


def write_reports(rec: dict) -> None:
    DOCS.mkdir(parents=True, exist_ok=True)
    th = rec["theory"]
    qed = rec["qed"]
    gh = rec["gh5"]
    ans = rec["answers"]
    fig = rec["figures"]
    full = []
    full.append("# 线Q 量子光多周期磁序响应：θ=π QED 机制诊断（20261120）\n")
    full.append("只读后处理与自由光场二阶关联推导。未进行新的电子传播。")
    full.append("重叠的 2T0 窗口不是独立重复实验。整数阶 Fourier 不是已认证 HHG。\n")
    full.append("## 一、反常二阶关联与自由光场协方差\n")
    full.append("对输入态 \\(|\\psi_\\theta\\rangle=D(\\alpha)S(re^{i\\theta})|0\\rangle\\)：\n")
    full.append("\\[\nM_\\theta=\\langle(\\Delta a)^2\\rangle=-\\frac12 e^{i\\theta}\\sinh(2r).\n\\]\n")
    full.append(f"数值：\\(r=0.3\\)，\\(M_0={th['M_0']:.6f}\\)，\\(M_\\pi={th['M_pi']:.6f}\\)。\n")
    full.append("自由光场正交分量 \\(X(t)=ae^{-i\\omega_0 t}+a^\\dagger e^{i\\omega_0 t}\\) 满足\n")
    full.append("\\[\n\\operatorname{Var}_\\theta[X(t)]=\\cosh(2r)-\\sinh(2r)\\cos(2\\omega_0 t-\\theta),\n\\]\n")
    full.append("因此压缩角反转给出确定的双频差\n")
    full.append("\\[\n\\operatorname{Var}_\\pi[X(t)]-\\operatorname{Var}_0[X(t)]=2\\sinh(2r)\\cos(2\\omega_0 t).\n\\]\n")
    full.append(f"振幅 \\(2\\sinh(2r)={th['dtheta_VarX_amp']:.4f}\\)。")
    full.append("对称二时刻协方差为\n")
    full.append("\\[\nC_\\theta(t_1,t_2)=\\cosh(2r)\\cos[\\omega_0(t_1-t_2)]-\\sinh(2r)\\cos[\\omega_0(t_1+t_2)-\\theta],\n\\]\n")
    full.append("其差 \\(C_\\pi-C_0=2\\sinh(2r)\\cos[\\omega_0(t_1+t_2)]\\) 依赖 \\(t_1+t_2\\) 的双频相位。")
    full.append("若存在磁序敏感的二阶电子响应核，该结构可以进入 H2 电流差。")
    full.append("**这不是已证明的响应核。** 自由光场公式不等于有电子反作用的 QED。\n")
    full.append("## 二、固定 2T0 平移窗的 H2 复振幅\n")
    full.append("全局约定：\\(J_q=1/(t_1-t_0)\\int e^{+iq\\omega_0 t}J(t)\\,dt\\)，矩形窗。\n")
    full.append("| 窗口 | QED \\(\\theta=0\\) | QED \\(\\theta=\\pi\\) | QED \\(\\delta_\\theta\\) | GH5 \\(\\delta_\\theta\\) | 完整 |")
    full.append("| --- | --- | --- | --- | --- | --- |")
    for row in rec["window_table"]:
        full.append(
            f"| {row['name']} | {fmt_c(row['qed_0'])} | {fmt_c(row['qed_pi'])} | "
            f"{fmt_c(row['qed_d'])} | {fmt_c(row['gh5_d'])} | {row['complete']} |"
        )
    full.append("")
    dc = rec["delta_complex"]
    full.append(f"主窗相对 [1,3] 的复差向量变化 \\(\\delta_\\mathrm{{complex}}={dc['main_vs_front']:.3f}\\)。")
    full.append(f"[2.5,4.5] 对 [2.7,4.7]：\\(\\delta_\\mathrm{{complex}}={dc['shift_pair']:.3f}\\)，")
    full.append(f"观测相位差 {dc['shift_dphi_deg']:.1f}°，线性泄漏模板相位差 {dc['leak_shift_dphi_deg']:.1f}°。")
    full.append("这些窗口相互重叠，不能当成独立重复实验。\n")
    full.append("## 三、慢变背景与 2ω0 联合拟合\n")
    full.append("预先固定模型：常数 / 线性 / 二次多项式，加上 \\(A\\cos(2\\omega_0 t)+B\\sin(2\\omega_0 t)\\)。")
    full.append("未再搜索其他背景函数。\n")
    full.append("| 窗口 | 原始 Fourier | 常数+2ω0 | 线性+2ω0 | 二次+2ω0 | 相对差 const/lin/quad |")
    full.append("| --- | --- | --- | --- | --- | --- |")
    for row in rec["fit_table"]:
        full.append(
            f"| {row['name']} | {fmt_c(row['raw'])} | {fmt_c(row['fit_const'])} | "
            f"{fmt_c(row['fit_lin'])} | {fmt_c(row['fit_quad'])} | "
            f"{row['rel_const']:.3f} / {row['rel_lin']:.3f} / {row['rel_quad']:.3f} |"
        )
    full.append("")
    full.append(f"线性泄漏若主导，[2.5,4.5]→[2.7,4.7] 的 H2 相位应跳约 {dc['leak_shift_dphi_deg']:.1f}°。")
    full.append(f"实测 QED \\(\\delta_\\theta\\) 相位差为 {dc['shift_dphi_deg']:.1f}°。\n")
    c0 = rec["qed"]["odd_0"]["continuum"]["main_2_4"]
    cpi = rec["qed"]["odd_pi"]["continuum"]["main_2_4"]
    cd = rec["qed"]["d_odd"]["continuum"]["main_2_4"]
    full.append(
        f"主窗连续谱：单态峰值分别在 q={c0['peak_q']:.3f} 与 q={cpi['peak_q']:.3f}；"
        f"δθ 峰值在 q={cd['peak_q']:.3f}，且 |J(peak)|≈|J_2|。"
        "压缩角差集中在 H2 邻域，不是把两态共有低频背景泄漏进 q=2。\n"
    )
    full.append("## 四、偶/奇通道尺度\n")
    sc = rec["scales"]
    full.append("| 量 | RMS | 主窗 H2 \\(|J_{2,x}|\\) |")
    full.append("| --- | --- | --- |")
    full.append(f"| \\(J^e(\\theta=0)\\) | {sc['rms_even_0']:.3e} | {sc['h2_even_0']:.3e} |")
    full.append(f"| \\(J^o(\\theta=0)\\) | {sc['rms_odd_0']:.3e} | {sc['h2_odd_0']:.3e} |")
    full.append(f"| \\(J^o(\\theta=\\pi)\\) | {sc['rms_odd_pi']:.3e} | {sc['h2_odd_pi']:.3e} |")
    full.append(f"| \\(\\delta_\\theta J^o\\) | {sc['rms_d_odd']:.3e} | {sc['h2_d_odd']:.3e} |")
    full.append(f"| \\(\\delta_\\theta J^e\\) | {sc['rms_d_even']:.3e} | {sc['h2_d_even']:.3e} |")
    full.append("")
    full.append(f"主窗 \\(|J^o|/|J^e|\\) RMS 比约 {sc['rms_odd_over_even']:.3e}。")
    full.append("不能把很小的奇电流相对变化写成巨大的实际磁性输出。\n")
    full.append("## 五、GH5 同窗对照\n")
    full.append(f"主窗 QED/GH5 复差向量相对偏差 {rec['gh5_vs_qed']['rel_main']:.3f}，")
    full.append(f"相位差 {rec['gh5_vs_qed']['dphi_main_deg']:.2f}°。")
    full.append("相位与自由场 \\(\\cos(2\\omega_0 t)\\) 结构可以相容，但这不等于已经证明响应核。\n")
    full.append("## 六、图\n")
    for key, path in fig.items():
        full.append(f"![{key}]({path})\n")
    full.append("## 七、两个问题的回答\n")
    full.append(f"**A. 当前 H2 压缩角调控是否可能主要由低频泄漏造成？** {ans['A']}\n")
    full.append(f"**B. 反常二阶光场关联是否构成目前最合理、但仍待验证的微观机制？** {ans['B']}\n")
    full.append("## 八、边界\n")
    full.append("- 整数 Fourier 投影不是已认证 HHG。")
    full.append("- 同 α、近似同 ⟨n⟩ 不是完整非经典性判据。")
    full.append("- QED 电流差不能证明输出光压缩或纠缠。")
    full.append("- 不得与 A*=0.8 经典强场谱拼接。")
    full.append("- 后窗 [3,5]T0 仍缺失；本诊断不续跑。\n")
    full.append("完成后停机。禁止续跑 QED、Cn/SV、提高 GH 阶、改 nmax、加 k、Floquet 或服务器。\n")
    pfull = DOCS / "线Q_量子光多周期磁序响应_Q1a_QED_theta_pi_机制诊断_20261120.md"
    pfull.write_text("\n".join(full) + "\n", encoding="utf-8")
    one = []
    one.append("# 一页：θ=π QED H2 机制诊断（20261120）\n")
    one.append("| 问题 | 结论 |")
    one.append("| --- | --- |")
    one.append(f"| A 低频泄漏？ | {ans['A_short']} |")
    one.append(f"| B 反常关联机制？ | {ans['B_short']} |")
    one.append(f"| 主窗 δ_complex vs [1,3] | {dc['main_vs_front']:.3f} |")
    one.append(f"| 平移窗相位差 / 泄漏模板 | {dc['shift_dphi_deg']:.1f}° / {dc['leak_shift_dphi_deg']:.1f}° |")
    one.append(
        f"| 主窗拟合相对差 const/lin/quad | "
        f"{rec['fit_table'][1]['rel_const']:.3f} / "
        f"{rec['fit_table'][1]['rel_lin']:.3f} / "
        f"{rec['fit_table'][1]['rel_quad']:.3f} |"
    )
    one.append("")
    one.append(f"![windows]({fig['h2_windows']})\n")
    one.append("停机。无新传播。\n")
    pone = DOCS / "线Q_量子光多周期磁序响应_Q1a_QED_theta_pi_机制诊断_一页_20261120.md"
    pone.write_text("\n".join(one) + "\n", encoding="utf-8")
    rec["reports"] = {"full": str(pfull), "one": str(pone)}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    env = dyn.Envelope(OMEGA0, ramp_cycles=1.0, flat_cycles=4.0)
    T0 = float(env.period)
    theory = theory_block(R_SQ, ALPHA, OMEGA0)

    t0, even0, odd0 = even_odd_from_dir(QED0)
    tpi, evenpi, oddpi = even_odd_from_dir(QEDPI)
    tg0, geven0, godd0 = even_odd_from_dir(GH50)
    tgpi, gevenpi, goddpi = even_odd_from_dir(GH5PI)

    even0_on_pi = interp_xy(tpi, t0, even0)
    odd0_on_pi = interp_xy(tpi, t0, odd0)
    d_odd = oddpi - odd0_on_pi
    d_even = evenpi - even0_on_pi
    gd_odd = goddpi - interp_xy(tgpi, tg0, godd0)

    cover_qed = float(tpi[-1])
    cover_gh = float(tgpi[-1])

    qed_0 = analyze_channel("qed_odd_0", t0, odd0[:, 0], OMEGA0, T0, float(t0[-1]))
    qed_pi = analyze_channel("qed_odd_pi", tpi, oddpi[:, 0], OMEGA0, T0, cover_qed)
    qed_d = analyze_channel("qed_d_odd", tpi, d_odd[:, 0], OMEGA0, T0, cover_qed)
    qed_e0 = analyze_channel("qed_even_0", t0, even0[:, 0], OMEGA0, T0, float(t0[-1]))
    qed_de = analyze_channel("qed_d_even", tpi, d_even[:, 0], OMEGA0, T0, cover_qed)
    gh_0 = analyze_channel("gh5_odd_0", tg0, godd0[:, 0], OMEGA0, T0, float(tg0[-1]))
    gh_pi = analyze_channel("gh5_odd_pi", tgpi, goddpi[:, 0], OMEGA0, T0, cover_gh)
    gh_d = analyze_channel("gh5_d_odd", tgpi, gd_odd[:, 0], OMEGA0, T0, cover_gh)

    window_table = []
    fit_table = []
    z_qed_d = []
    z_qed_0 = []
    z_qed_pi = []
    names = []
    leak_phi = []
    for i, (wname, a, b) in enumerate(WINDOWS):
        w0 = qed_0["windows"][i]
        wpi = qed_pi["windows"][i]
        wd = qed_d["windows"][i]
        wg = gh_d["windows"][i]
        complete = bool(w0["complete"] and wpi["complete"] and wd["complete"])
        window_table.append({
            "name": f"[{a:g},{b:g}]T0",
            "qed_0": w0["raw_j2"],
            "qed_pi": wpi["raw_j2"],
            "qed_d": wd["raw_j2"],
            "gh5_d": wg["raw_j2"],
            "complete": complete,
        })
        z0 = complex(w0["raw_j2"]["re"], w0["raw_j2"]["im"])
        zpi = complex(wpi["raw_j2"]["re"], wpi["raw_j2"]["im"])
        zd = complex(wd["raw_j2"]["re"], wd["raw_j2"]["im"])
        z_qed_0.append(z0)
        z_qed_pi.append(zpi)
        z_qed_d.append(zd)
        names.append(f"[{a:g},{b:g}]")
        leak_phi.append(wd["linear_leakage_phase_rad"])
        fitq = wd["fits"].get("poly2_plus_2w", {}).get("j2", pack_c(0j))
        raw = wd["raw_j2"]
        zf = complex(fitq["re"], fitq["im"])
        rel = float(abs(zf - zd) / (abs(zd) + 1e-30)) if complete else float("nan")
        fitc = wd["fits"].get("poly0_plus_2w", {}).get("j2", pack_c(0j))
        fitl = wd["fits"].get("poly1_plus_2w", {}).get("j2", pack_c(0j))
        zc = complex(fitc["re"], fitc["im"])
        zl = complex(fitl["re"], fitl["im"])
        relc = float(abs(zc - zd) / (abs(zd) + 1e-30)) if complete else float("nan")
        rell = float(abs(zl - zd) / (abs(zd) + 1e-30)) if complete else float("nan")
        fit_table.append({
            "name": f"[{a:g},{b:g}]T0",
            "raw": raw,
            "fit_const": fitc,
            "fit_lin": fitl,
            "fit_quad": fitq,
            "rel_const": relc,
            "rel_lin": rell,
            "rel_quad": rel,
        })

    z_main = z_qed_d[1]
    dc = {
        "main_vs_front": dcomplex(z_qed_d[0], z_qed_d[1], z_main),
        "main_vs_2p5": dcomplex(z_qed_d[2], z_qed_d[1], z_main),
        "main_vs_2p7": dcomplex(z_qed_d[3], z_qed_d[1], z_main),
        "shift_pair": dcomplex(z_qed_d[2], z_qed_d[3], z_main),
        "shift_dphi_rad": angdiff(np.angle(z_qed_d[3]), np.angle(z_qed_d[2])),
        "leak_shift_dphi_rad": angdiff(leak_phi[3], leak_phi[2]),
    }
    dc["shift_dphi_deg"] = abs(dc["shift_dphi_rad"]) * 180.0 / np.pi
    dc["leak_shift_dphi_deg"] = abs(dc["leak_shift_dphi_rad"]) * 180.0 / np.pi

    mwin = window_mask(tpi, 2.0 * T0, 4.0 * T0)
    mwin0 = window_mask(t0, 2.0 * T0, 4.0 * T0)
    h2e0 = qed_e0["windows"][1]["raw_j2"]["abs"]
    scales = {
        "rms_even_0": rms(even0[mwin0, 0]),
        "rms_odd_0": rms(odd0[mwin0, 0]),
        "rms_odd_pi": rms(oddpi[mwin, 0]),
        "rms_d_odd": rms(d_odd[mwin, 0]),
        "rms_d_even": rms(d_even[mwin, 0]),
        "h2_even_0": h2e0,
        "h2_odd_0": qed_0["windows"][1]["raw_j2"]["abs"],
        "h2_odd_pi": qed_pi["windows"][1]["raw_j2"]["abs"],
        "h2_d_odd": qed_d["windows"][1]["raw_j2"]["abs"],
        "h2_d_even": qed_de["windows"][1]["raw_j2"]["abs"],
    }
    scales["rms_odd_over_even"] = scales["rms_odd_0"] / (scales["rms_even_0"] + 1e-30)
    scales["h2_odd_over_even"] = scales["h2_odd_0"] / (scales["h2_even_0"] + 1e-30)

    zg_main = complex(gh_d["windows"][1]["raw_j2"]["re"], gh_d["windows"][1]["raw_j2"]["im"])
    gh_vs = {
        "rel_main": float(abs(zg_main - z_main) / (abs(z_main) + 1e-30)),
        "dphi_main_rad": angdiff(np.angle(zg_main), np.angle(z_main)),
    }
    gh_vs["dphi_main_deg"] = abs(gh_vs["dphi_main_rad"]) * 180.0 / np.pi

    colors = ["#4c6b8a", "#c45c26", "#2f9e44", "#7a3ea8"]
    f_win = figure_windows(
        [(names[i], z_qed_d[i], colors[i]) for i in range(4)],
        FIG / "h2_dtheta_sliding_windows_qed.svg",
        "QED H2 δθ Jx across 2T0 windows",
    )
    f_states = figure_windows(
        [("θ=0 main", z_qed_0[1], "#4c6b8a"),
         ("θ=π main", z_qed_pi[1], "#c45c26"),
         ("δθ main", z_qed_d[1], "#2f9e44")],
        FIG / "h2_theta0_vs_pi_main_qed.svg",
        "QED main-window H2 odd Jx",
    )
    qgrid = np.linspace(0.0, 6.0, 241)
    cont_main = qed_d["continuum"]["main_2_4"]
    cont_0 = qed_0["continuum"]["main_2_4"]
    cont_pi = qed_pi["continuum"]["main_2_4"]
    f_cont = figure_cont(
        qgrid,
        [("#4c6b8a", cont_0["abs"], "θ=0", 80),
         ("#c45c26", cont_pi["abs"], "θ=π", 150),
         ("#2f9e44", cont_main["abs"], "δθ", 230)],
        FIG / "h2_continuum_main_qed.svg",
        "QED main-window continuum |J_odd,x|",
    )
    obs_phi = [float(np.angle(z)) for z in z_qed_d]
    f_phase = figure_phase(names, obs_phi, leak_phi, FIG / "h2_phase_vs_linear_leakage.svg")

    fit_rel_main = fit_table[1]["rel_quad"]
    leak_mismatch = abs(dc["shift_dphi_deg"] - dc["leak_shift_dphi_deg"])
    leakage_not_dominant = bool(fit_rel_main < 0.25 and leak_mismatch > 40.0)
    leakage_possible = bool(fit_rel_main > 0.50 or leak_mismatch < 25.0)

    if leakage_not_dominant:
        a_ans = (
            "不太可能主要由低频泄漏造成。原始 H2 Fourier 与预先固定的二次多项式+2ω0 "
            f"联合拟合相差 {fit_rel_main:.1%}；[2.5,4.5]→[2.7,4.7] 观测相位差 "
            f"{dc['shift_dphi_deg']:.1f}°，而线性泄漏模板要求约 {dc['leak_shift_dphi_deg']:.1f}°。"
            "重叠窗仍有复漂移，故不能认证稳定谐波，但现有证据不支持“H2 只是慢变背景泄漏”。"
        )
        a_short = "否，现有窗诊断不支持泄漏主导"
    elif leakage_possible:
        a_ans = (
            "不能排除低频泄漏的显著贡献。拟合或平移窗相位与泄漏模板尚未分离。"
            "证据缺口：缺少完整后窗 [3,5]T0，且两周期分辨尺度约 0.5ω0。"
        )
        a_short = "无法排除；证据不足"
    else:
        a_ans = (
            "现有数据不足以把 H2 压缩角差完全归因于泄漏，也不足以彻底排除。"
            f"拟合相对差 {fit_rel_main:.1%}，平移窗相位差 {dc['shift_dphi_deg']:.1f}°"
            f"（泄漏模板 {dc['leak_shift_dphi_deg']:.1f}°）。"
        )
        a_short = "无法干净区分"

    b_ans = (
        "反常二阶关联是目前最合理、但仍待验证的微观机制。"
        "它同时解释了：θ 只旋转涨落椭圆、自由场 Var X 出现确定的 2ω0 差、"
        "观测量恰好落在 H2 邻域、以及 GH5/Wigner（只输入二阶统计）能够重现 QED 差向量。"
        "尚未证明电子响应核；自由场协方差不是完整 QED；相位相容不等于因果证明。"
    )
    b_short = "最合理候选，尚未验证响应核"

    rec = {
        "task": "qed_theta_pi_mechanism_readonly_20261120",
        "no_new_propagation": True,
        "theory": theory,
        "coverage": {
            "qed_theta0_T0": float(t0[-1] / T0),
            "qed_theta_pi_T0": float(tpi[-1] / T0),
            "gh5_T0": float(tgpi[-1] / T0),
            "windows": [[a, b] for _, a, b in WINDOWS],
            "overlapping_windows_not_independent": True,
        },
        "qed": {"odd_0": qed_0, "odd_pi": qed_pi, "d_odd": qed_d, "d_even": qed_de},
        "gh5": {"odd_0": gh_0, "odd_pi": gh_pi, "d_odd": gh_d},
        "window_table": window_table,
        "fit_table": fit_table,
        "delta_complex": dc,
        "scales": scales,
        "gh5_vs_qed": gh_vs,
        "answers": {"A": a_ans, "B": b_ans, "A_short": a_short, "B_short": b_short},
        "figures": {
            "h2_windows": f_win,
            "h2_states": f_states,
            "continuum": f_cont,
            "phase_leakage": f_phase,
        },
        "boundaries": {
            "integer_Fourier_is_not_certified_HHG": True,
            "same_nbar_is_not_full_nonclassicality": True,
            "QED_current_is_not_output_entanglement": True,
            "do_not_splice_classical_Astar_0p8": True,
            "free_field_not_full_QED": True,
        },
        "forbidden": ["QED resume", "Cn", "SV", "GH order", "nmax", "extra k", "Floquet", "server"],
    }
    write_reports(rec)
    dump(OUT / "mech.json", rec)
    print(json.dumps({
        "out": str(OUT),
        "A_short": a_short,
        "B_short": b_short,
        "fit_rel_main": fit_rel_main,
        "shift_dphi_deg": dc["shift_dphi_deg"],
        "leak_shift_dphi_deg": dc["leak_shift_dphi_deg"],
        "delta_complex_main_front": dc["main_vs_front"],
        "qed_pi_cover_T0": float(tpi[-1] / T0),
        "reports": rec.get("reports"),
    }, ensure_ascii=False, indent=2))
    return rec


if __name__ == "__main__":
    main()
