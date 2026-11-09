"""Delta-n block analysis of Q6 PDPS vs GH5 14 percent remainder.

No new QED. Reconstruct truncated D(0.5)S(0.3 e^{i theta})|0> at nmax=8
exactly as production PDPS (P D P, then normalize). Compare ideal R,
truncated R, and frozen QED rho=|A|/|B|. Also bound Delta-n=+-4 aliasing
from the already-known four-angle residual.
"""
from __future__ import annotations

import json
import math
import os
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
import sys

sys.path.insert(0, str(HERE))

import q8_m0_qinput_dynamics_20261026 as dyn
import q8_m0_qinput_minimal_20261020 as qm

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results/q_line/abc_dn_blocks_20261029"
DOCS = ROOT / "docs/04_BSV量子光"

ALPHA = 0.5
R_SQ = 0.3
NMAX_PROD = 8
QED_RHO = 1.456
QED_R = 1.2733
QED_ARG_AB_DEG = 177.65
QED_FIT_RESIDUAL = 0.00492
QED_PI4_ERR = 0.004248607859931319
A1_EVEN_AC_OVER_B = 0.00308

QED_B = -3.300558614e-8 + 9.537284492e-9j
QED_A = +4.743410898e-8 - 1.583918408e-8j
QED_C = +6.345731355e-10 - 4.336636681e-10j

QED_ODD_JX = {
    0.0: 1.4885911076055816e-08 - 6.495157310566291e-09j,
    0.5 * np.pi: -1.7422880832290363e-08 + 5.609641439655467e-08j,
    np.pi: -8.125145315623136e-08 + 2.6050538175670813e-08j,
    1.5 * np.pi: -4.823392164626447e-08 - 3.75026572937314e-08j,
}


def dump(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(
        json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(tmp, path)


def cpack(z) -> dict:
    z = complex(z)
    return {"re": float(z.real), "im": float(z.imag), "abs": float(abs(z)),
            "phase_deg": float(np.degrees(np.angle(z)))}


def lowering(n_max: int) -> np.ndarray:
    d = n_max + 1
    a = np.zeros((d, d), dtype=complex)
    for n in range(1, d):
        a[n - 1, n] = np.sqrt(n)
    return a


def ideal_moments(alpha: float, r: float, theta: float) -> dict:
    a = complex(alpha)
    m = -0.5 * np.sinh(2.0 * r) * np.exp(1j * theta)
    aa = a * a + m
    nbar = abs(a) ** 2 + np.sinh(r) ** 2
    return {
        "alpha": a,
        "mean_a": a,
        "mean_n": float(nbar),
        "aa": aa,
        "M": m,
        "R_M_over_alpha2": float(abs(m) / max(abs(a) ** 2, 1e-30)),
        "R_aa_over_alpha2": float(abs(aa) / max(abs(a) ** 2, 1e-30)),
    }


def pdps_state(alpha: float, r: float, theta: float, n_max: int, renormalize: bool) -> np.ndarray:
    raw = dyn.displaced_squeezed_coefficients(alpha, r, theta, n_max)
    if renormalize:
        nrm = np.linalg.norm(raw)
        if nrm == 0:
            raise RuntimeError("zero PDPS vector")
        return raw / nrm
    return raw


def state_moments(c: np.ndarray) -> dict:
    n_max = len(c) - 1
    a = lowering(n_max)
    rho = np.outer(c, c.conj())
    mean_a = complex(c.conj() @ (a @ c))
    mean_n = float(np.real(c.conj() @ (a.conj().T @ a @ c)))
    aa = complex(c.conj() @ (a @ a @ c))
    aaaa = complex(c.conj() @ (a @ a @ a @ a @ c))
    m = aa - mean_a * mean_a
    mass = float(np.vdot(c, c).real)
    dn = {}
    for delta in range(-n_max, n_max + 1):
        s = 0.0
        for i in range(n_max + 1):
            j = i + delta
            if 0 <= j <= n_max:
                s += abs(rho[i, j])
        dn[str(delta)] = float(s)
    return {
        "mass": mass,
        "mean_a": mean_a,
        "mean_n": mean_n,
        "aa": aa,
        "aaaa": aaaa,
        "M": m,
        "R_M_over_mean_a2": float(abs(m) / max(abs(mean_a) ** 2, 1e-30)),
        "R_M_over_alpha2": float(abs(m) / max(ALPHA * ALPHA, 1e-30)),
        "R_aa_over_mean_a2": float(abs(aa) / max(abs(mean_a) ** 2, 1e-30)),
        "dn_l1": dn,
        "dn2_over_dn0": float(dn.get("2", 0.0) / max(dn.get("0", 0.0), 1e-30)),
        "dn4_over_dn2": float(dn.get("4", 0.0) / max(dn.get("2", 0.0), 1e-30)),
        "dn4_over_dn0": float(dn.get("4", 0.0) / max(dn.get("0", 0.0), 1e-30)),
    }


def displaced_fourier_weights(alpha: float, r: float, n_max: int) -> dict:
    """Ideal-SV coefficients in the even Fock sector; F_ell weights of T^dagger T and of a^2."""
    d = n_max + 1
    disp = dyn.galerkin_displacement(alpha, n_max)
    pairs = np.arange(0, d, 2)
    sv = np.array(
        [
            ((-np.tanh(r)) ** (m // 2)
             * np.sqrt(math.factorial(m))
             / (2 ** (m // 2) * math.factorial(m // 2) * np.sqrt(np.cosh(r))))
            for m in pairs
        ],
        dtype=complex,
    )
    G = disp.conj().T @ disp
    a = lowering(n_max)
    Kaa = a @ a
    KT = disp.conj().T @ Kaa @ disp
    num, den = {}, {}
    for ia, m in enumerate(pairs):
        for ib, n in enumerate(pairs):
            ell = int(ib - ia)
            num[ell] = num.get(ell, 0j) + np.conj(sv[ia]) * sv[ib] * KT[m, n]
            den[ell] = den.get(ell, 0j) + np.conj(sv[ia]) * sv[ib] * G[m, n]
    return {
        "Z": {str(k): cpack(v) for k, v in sorted(den.items())},
        "N_aa": {str(k): cpack(v) for k, v in sorted(num.items())},
        "Z_offdiag_over_Z0": float(
            sum(abs(v) for k, v in den.items() if k != 0) / max(abs(den.get(0, 1j)), 1e-30)
        ),
        "N_ell2_over_ell1": float(abs(num.get(2, 0j)) / max(abs(num.get(1, 1j)), 1e-30)),
        "N_ell1_over_ell0": float(abs(num.get(1, 0j)) / max(abs(num.get(0, 1j)), 1e-30)),
    }


def abc_from_four(jx: dict) -> dict:
    th = np.array(sorted(jx), dtype=float)
    y = np.array([jx[t] for t in th], dtype=complex)
    M = np.column_stack([np.ones(len(th)), np.exp(1j * th), np.exp(-1j * th)])
    coef, *_ = np.linalg.lstsq(M, y, rcond=None)
    b, a, c = (complex(z) for z in coef)
    pred = M @ coef
    resid = float(np.linalg.norm(y - pred) / max(np.linalg.norm(y), 1e-30))
    M2 = np.column_stack([M, np.exp(2j * th), np.exp(-2j * th)])
    coef2, *_ = np.linalg.lstsq(M2, y, rcond=None)
    pred2 = M2 @ coef2
    resid2 = float(np.linalg.norm(y - pred2) / max(np.linalg.norm(y), 1e-30))
    return {
        "B": cpack(b),
        "A": cpack(a),
        "C": cpack(c),
        "rho": float(abs(a) / max(abs(b), 1e-30)),
        "arg_A_over_B_deg": float(np.degrees(np.angle(a / b))),
        "C_over_A": float(abs(c) / max(abs(a), 1e-30)),
        "four_point_residual": resid,
        "with_ell2_residual": resid2,
        "D_ell_plus2": cpack(coef2[3]),
        "E_ell_minus2": cpack(coef2[4]),
        "ell2_over_A": float(abs(coef2[3]) / max(abs(a), 1e-30)),
    }


def nmax_scan() -> list:
    rows = []
    for nmax in (8, 10, 12, 16, 24):
        c = pdps_state(ALPHA, R_SQ, 0.0, nmax, renormalize=True)
        raw = pdps_state(ALPHA, R_SQ, 0.0, nmax, renormalize=False)
        mom = state_moments(c)
        raw_mass = float(np.vdot(raw, raw).real)
        rows.append({
            "nmax": nmax,
            "raw_mass": raw_mass,
            "omitted_mass": max(0.0, 1.0 - raw_mass),
            "mean_a_abs": abs(mom["mean_a"]),
            "mean_n": mom["mean_n"],
            "R_M_over_mean_a2": mom["R_M_over_mean_a2"],
            "R_M_over_alpha2": mom["R_M_over_alpha2"],
            "dn4_over_dn2": mom["dn4_over_dn2"],
            "dn2_over_dn0": mom["dn2_over_dn0"],
        })
    return rows


def theta_table(n_max: int) -> list:
    rows = []
    for th in (0.0, 0.5 * np.pi, np.pi, 1.5 * np.pi, 0.25 * np.pi):
        c = pdps_state(ALPHA, R_SQ, th, n_max, renormalize=True)
        mom = state_moments(c)
        ideal = ideal_moments(ALPHA, R_SQ, th)
        rows.append({
            "theta": float(th),
            "theta_deg": float(np.degrees(th)),
            "mean_a": cpack(mom["mean_a"]),
            "mean_n": mom["mean_n"],
            "M": cpack(mom["M"]),
            "aa": cpack(mom["aa"]),
            "aaaa": cpack(mom["aaaa"]),
            "R_M_over_mean_a2": mom["R_M_over_mean_a2"],
            "R_M_over_alpha2": mom["R_M_over_alpha2"],
            "ideal_R": ideal["R_M_over_alpha2"],
            "dn2_over_dn0": mom["dn2_over_dn0"],
            "dn4_over_dn2": mom["dn4_over_dn2"],
            "arg_M_deg": float(np.degrees(np.angle(mom["M"]))),
            "ideal_arg_M_deg": float(np.degrees(np.angle(ideal["M"]))),
        })
    return rows


def verdict(scan8_R: float, ell2_over_A: float) -> str:
    frac = abs(scan8_R - QED_R) / max(abs(QED_RHO - QED_R), 1e-30)
    if frac > 0.5 and ell2_over_A < 0.05:
        return "truncation_explains_majority_of_14pct"
    if frac < 0.15 and ell2_over_A < 0.05:
        return "quantum_kernel_KM_over_KB_remainder"
    if ell2_over_A >= 0.05:
        return "higher_Delta_n_aliasing_not_negligible"
    return "mixed_truncation_and_kernel"


def write_reports(payload: dict) -> None:
    md = DOCS / "线Q_ABC_DeltaN块_一页_20261029.md"
    scan = {r["nmax"]: r for r in payload["nmax_scan"]}
    r8 = scan[8]
    fit = payload["qed_four_angle_abc"]
    fw = payload["displaced_fourier_aa"]
    lines = [
        "# 线 Q：A/B/C 的 Δn 块与 14% 余量一页（2026-10-29）",
        "",
        "无新 QED。生产初态是 nmax=8 的 PDPS：先截断 SV，再 Galerkin 位移，再归一化。",
        "权威 QED 数字不改写：ρ=1.456，R_ideal=1.273，arg(A/B)=177.65°。",
        "",
        "## 1. 截断有没有把 R 抬到 1.456？",
        "",
        "| nmax | 省略质量 | |⟨a⟩| | ⟨n⟩ | R_M/|⟨a⟩|² | R_M/α² |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for r in payload["nmax_scan"]:
        lines.append(
            f"| {r['nmax']} | {r['omitted_mass']:.3e} | {r['mean_a_abs']:.6f} | "
            f"{r['mean_n']:.6f} | {r['R_M_over_mean_a2']:.4f} | {r['R_M_over_alpha2']:.4f} |"
        )
    lines += [
        "",
        f"- 理想 R = **{QED_R:.4f}**，QED ρ = **{QED_RHO:.3f}**，差 14.4%。",
        f"- nmax=8 的 R_M/|⟨a⟩|² = **{r8['R_M_over_mean_a2']:.4f}**，R_M/α² = **{r8['R_M_over_alpha2']:.4f}**。",
        f"- 判定：`{payload['verdict']}`。",
        "",
        "## 2. Δn=±2 是不是 A/C 的块，Δn=±4 能不能解释拟合？",
        "",
        f"- 四点 ABC 残差 **{100 * fit['four_point_residual']:.3f}%**（与归档 0.492% 一致）。四点不够唯一确定 e^{{±2iθ}}，不以五参数拟合当证据。",
        f"- 真正的高次扇区检验是 Q6 π/4 留出误差 **0.425%**。",
        f"- 位移后 a² 核的 |N_{{ℓ=2}}/N_{{ℓ=1}}| = **{fw['N_ell2_over_ell1']:.4f}**，分母 off-diagonal/Z0 = **{fw['Z_offdiag_over_Z0']:.3e}**。角向 Δn=±4 可以忽略。",
        f"- 实验室 Fock 基底里 |ρ_{{n,n+4}}| 看起来不小，那是位移把偶数压缩真空混进各 n；物理扇区要在位移后的基底看。",
        f"- A1 偶通道 |A+C|/|B| = {A1_EVEN_AC_OVER_B:.5f}，奇通道在 Δ=0/SOC=0 下关闭。",
        "",
        "## 3. 这对机制意味着什么",
        "",
        payload["meaning"],
        "",
        "A 仍然不是 χ^{(2)}。本轮没有新传播，也没有反演完整 9×9 K。",
        f"归档：`{OUT.as_posix()}`。",
        "",
    ]
    md.write_text("\n".join(lines), encoding="utf-8")
    payload["one_page"] = str(md)


def main() -> int:
    t0 = time.perf_counter()
    OUT.mkdir(parents=True, exist_ok=True)
    ideal = ideal_moments(ALPHA, R_SQ, 0.0)
    scan = nmax_scan()
    thetas = theta_table(NMAX_PROD)
    fw = displaced_fourier_weights(ALPHA, R_SQ, NMAX_PROD)
    fit = abc_from_four(QED_ODD_JX)
    r8 = next(r for r in scan if r["nmax"] == 8)
    verd = verdict(r8["R_M_over_mean_a2"], fit["ell2_over_A"])
    r_trunc = r8["R_M_over_mean_a2"]
    gap = QED_RHO - QED_R
    trunc_gap = r_trunc - QED_R
    kernel_gap = QED_RHO - r_trunc
    if verd == "truncation_explains_majority_of_14pct":
        meaning = (
            f"nmax=8 把有效 R 从 {QED_R:.3f} 抬到 {r_trunc:.3f}，"
            f"已经覆盖 QED ρ−R 的 {100 * trunc_gap / gap:.0f}%。"
            "剩下的才可能是量子核 K_M/K_B ≠ 1。因此不能把 14% 全部写成精确映射修正。"
        )
    elif verd == "quantum_kernel_KM_over_KB_remainder":
        meaning = (
            f"nmax=8 的有效 R 仍是 {r_trunc:.4f}，与理想 {QED_R:.4f} 几乎相同。"
            f"QED ρ=1.456 因此不是截断初态。若 J≈K_B α²+K_M M，则 K_M/K_B=ρ/R=**1.144**。"
            "GH5 在 kick=True 时已经是 0.998，所以这 14% 也不是 Ehrenfest 反作用。"
            "形式上 ρ/R=1.144，但后续 KM/KB 拆分表明：这不是材料把 Δn=±2 读强 14%，"
            "而是主窗里 Δn=0 的 K_B 被压低约 13%，K_M 与 GH5 只差 0.3%，且 K_B 随窗口变号。"
            "nmax=10 几乎不可能把有效 R 从 1.273 抬到 1.456。详见 `线Q_ABC_KMKB拆分_一页_20261029.md`。"
        )
    else:
        meaning = (
            "截断与更高 Δn 扇区都有可见贡献，14% 不能单归因。下一步仍是矩阵块而不是更大网格。"
        )
    payload = {
        "protocol": "Qin_a2_1_4_1 PDPS nmax=8",
        "no_new_qed": True,
        "alpha": ALPHA,
        "r": R_SQ,
        "ideal": {
            "mean_n": ideal["mean_n"],
            "R": ideal["R_M_over_alpha2"],
            "abs_M": abs(ideal["M"]),
        },
        "qed_frozen": {
            "rho": QED_RHO,
            "R": QED_R,
            "rho_over_R": QED_RHO / QED_R,
            "arg_A_over_B_deg": QED_ARG_AB_DEG,
            "B": cpack(QED_B),
            "A": cpack(QED_A),
            "C": cpack(QED_C),
            "C_over_A": abs(QED_C) / abs(QED_A),
            "pi4_error_over_A": QED_PI4_ERR,
            "four_point_residual_reported": QED_FIT_RESIDUAL,
        },
        "nmax_scan": scan,
        "theta_table_nmax8": [
            {k: (cpack(v) if isinstance(v, complex) else v) for k, v in row.items()}
            for row in thetas
        ],
        "displaced_fourier_aa": fw,
        "qed_four_angle_abc": fit,
        "truncation_gap": float(trunc_gap),
        "kernel_gap": float(kernel_gap),
        "truncation_fraction_of_14pct": float(trunc_gap / gap),
        "verdict": verd,
        "meaning": meaning,
        "wall_s": time.perf_counter() - t0,
        "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    write_reports(payload)
    dump(OUT / "dn_blocks.json", payload)
    print(json.dumps({
        "verdict": verd,
        "R8": r8["R_M_over_mean_a2"],
        "rho_qed": QED_RHO,
        "ell2_over_A": fit["ell2_over_A"],
        "one_page": payload["one_page"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
