#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Source copy of C:\Users\26507\Downloads\q_cov4_reference_check.py — formulas/self-tests/VARIANTS unchanged.
"""
q_cov4_reference_check.py — 两偏振 4×4 协方差的独立参考实现与自检（只做后处理，不传播）

约定（与 线Q_G1 文件一致）
  b = U a + V a†，U,V 形状 (2, Ncol)，输入列 a_j 为规范真空
  n_{μν} = <b_μ† b_ν> = (V* Vᵀ)_{μν}      m_{μν} = <b_μ b_ν> = (U Vᵀ)_{μν}
  C = U U† − V V† = I + R_N               R_A = U Vᵀ − V Uᵀ
  X = (b + b†)/√2,  P = (b − b†)/(i√2),  X(θ) = X cosθ + P sinθ
  排序 r = (X0, X1, P0, P1)，σ = ½<{Δr, Δrᵀ}>，真空 σ = I/2
  Q_μ = n_μμ + C_μμ/2 − |m_μμ| = 第 μ 偏振 2×2 块的最小本征值
  θ_min = (arg m_μμ + π)/2 (mod π)

两条独立路径
  路径 S ：S = [[Re(U+V), −Im(U−V)], [Im(U+V), Re(U−V)]]，σ = ½ S Sᵀ，Ω_eff = S Ω_in Sᵀ
  路径 nm：由 2×2 的 n, m, C 解析装配
    XX = Re n + Re m_s + ½Re C，PP = Re n − Re m_s + ½Re C，XP = Im m_s + Im n − ½Im C，
    m_s = (m + mᵀ)/2，σ = [[XX, XP], [XPᵀ, PP]]

必须成立
  (1) 两路径逐元一致；(2) λ_min(σ) ≤ min_μ Q_μ（Cauchy 交错）；(3) σ + (i/2)Ω_eff ≥ 0；
  (4) b_μ → e^{iχ_μ} b_μ 时，σ 本征值、辛本征值、部分转置辛本征值都不变。
  (4) 是对任意装配代码的黑盒检验：把生产 U,V 的某一行乘 e^{iχ} 再跑一遍，谱变了就是装配错。

用法
  python3 q_cov4_reference_check.py                       # 只跑解析自检
  python3 q_cov4_reference_check.py --npz F.npz --u-key u_prod --v-key v_prod --reported-min 81.408
  Python：from q_cov4_reference_check import analyze_uv, analyze_nm
"""
import argparse
import json
import sys

import numpy as np

OMEGA0 = np.block([[np.zeros((2, 2)), np.eye(2)], [-np.eye(2), np.zeros((2, 2))]])
T_PT = np.diag([1.0, 1.0, 1.0, -1.0])          # 对偏振 1 做部分转置：P1 → −P1
VARIANTS = ["drop_imag", "half_xp", "xp_sign", "conj_m", "n_transpose", "xp_lower_not_T"]
CHI_SETS = [(0.37, -1.10), (1.90, 0.60), (-2.40, 2.80)]


# ---------------------------------------------------------------- 基本构造
def S_from_uv(U, V):
    U = np.asarray(U, complex)
    V = np.asarray(V, complex)
    top = np.hstack([(U + V).real, -(U - V).imag])
    bot = np.hstack([(U + V).imag, (U - V).real])
    return np.vstack([top, bot])


def sigma_omega_from_uv(U, V):
    S = S_from_uv(U, V)
    k = S.shape[1] // 2
    Sx, Sp = S[:, :k], S[:, k:]
    return 0.5 * (S @ S.T), Sx @ Sp.T - Sp @ Sx.T


def nm_from_uv(U, V):
    U = np.asarray(U, complex)
    V = np.asarray(V, complex)
    n = V.conj() @ V.T
    m = U @ V.T
    C = U @ U.conj().T - V @ V.conj().T
    RA = U @ V.T - V @ U.T
    return n, m, C, RA


def sigma_from_nm(n, m, C, variant="reference"):
    n = np.asarray(n, complex)
    m = np.asarray(m, complex)
    C = np.asarray(C, complex)
    if variant == "conj_m":
        m = m.conj()
    if variant == "n_transpose":
        n = n.T
    ms = 0.5 * (m + m.T)
    XX = n.real + ms.real + 0.5 * C.real
    PP = n.real - ms.real + 0.5 * C.real
    XP = ms.imag + n.imag - 0.5 * C.imag
    if variant == "drop_imag":
        XP = np.zeros_like(XP)
    if variant == "half_xp":
        XP = 0.5 * XP
    if variant == "xp_sign":
        XP = -XP
    lower = XP.T
    if variant == "xp_lower_not_T":
        lower = XP
    sig = np.block([[XX, XP], [lower, PP]])
    if variant == "xp_lower_not_T":
        sig = 0.5 * (sig + sig.T)
    return sig


def rotate_nm(n, m, C, chi):
    D = np.diag(np.exp(1j * np.asarray(chi, float)))
    return D.conj() @ n @ D, D @ m @ D, D @ C @ D.conj()


# ---------------------------------------------------------------- 谱量
def symp_eigs_left_multiply(sig, om=OMEGA0):
    """旧字段：i Ω σ。仅当 Ω=Ω0 时因 Ω0^{-1}=-Ω0 可与广义辛谱重合。"""
    ev = np.sort(np.abs(np.linalg.eigvals(1j * np.asarray(om) @ sig).real))
    return ev[::2]


def symp_eigs(sig, om=OMEGA0):
    """广义辛谱：|eig(i Ω^{-1} σ)| 的偶位（与 williamson_prod 同一侧）。"""
    inv = np.linalg.inv(np.asarray(om, float))
    ev = np.sort(np.abs(np.linalg.eigvals(1j * (inv @ sig)).real))
    return ev[::2]


def spectra(sig, om=OMEGA0):
    return dict(eig=np.linalg.eigvalsh(sig), nu=symp_eigs(sig, om),
                nu_pt=symp_eigs(T_PT @ sig @ T_PT, om))


def q_values(n, m, C):
    return [float(n[mu, mu].real + 0.5 * C[mu, mu].real - abs(m[mu, mu])) for mu in (0, 1)]


def phase_invariance(n, m, C, variant="reference"):
    base = spectra(sigma_from_nm(n, m, C, variant))
    worst = 0.0
    for chi in CHI_SETS:
        sp = spectra(sigma_from_nm(*rotate_nm(n, m, C, chi), variant))
        for key in ("eig", "nu", "nu_pt"):
            rel = np.max(np.abs(sp[key] - base[key]) / np.maximum(np.abs(base[key]), 1e-300))
            worst = max(worst, float(rel))
    return worst


# ---------------------------------------------------------------- 报告
def analyze_nm(n, m, C, sigma_S=None, omega_eff=None, reported_min=None, verbose=True):
    n = np.asarray(n, complex)
    m = np.asarray(m, complex)
    C = np.asarray(C, complex)
    sig_nm = sigma_from_nm(n, m, C)
    sig = sig_nm if sigma_S is None else sigma_S
    om = OMEGA0 if omega_eff is None else omega_eff
    scale = max(1.0, float(np.max(np.abs(sig))))
    tol = 1e-10 * scale

    ev, evec = np.linalg.eigh(sig)
    Q = q_values(n, m, C)
    blocks = []
    for mu in (0, 1):
        idx = [mu, 2 + mu]
        blk = sig[np.ix_(idx, idx)]
        th = float(((np.angle(m[mu, mu]) + np.pi) / 2) % np.pi)
        v = np.array([np.cos(th), np.sin(th)])
        blocks.append(dict(mode=mu, block=blk.tolist(), block_min_eig=float(np.linalg.eigvalsh(blk)[0]),
                           Q=Q[mu], theta_min=th, var_at_theta_min=float(v @ blk @ v)))
    sp0 = spectra(sig, OMEGA0)
    out = dict(
        n=[[str(z) for z in row] for row in n], m=[[str(z) for z in row] for row in m],
        R_N_max=float(np.max(np.abs(C - np.eye(2)))),
        Q=Q, eig_sigma=ev.tolist(), min_eig_vector_X0X1P0P1=evec[:, 0].tolist(),
        symplectic_Omega0=sp0["nu"].tolist(), symplectic_Omega_eff=symp_eigs(sig, om).tolist(),
        pt_symplectic_Omega0=sp0["nu_pt"].tolist(),
        min_eig_sigma_plus_iOmega0_over2=float(np.linalg.eigvalsh(sig + 0.5j * OMEGA0)[0]),
        min_eig_sigma_plus_iOmegaEff_over2=float(np.linalg.eigvalsh(sig + 0.5j * om)[0]),
        max_abs_Omega_eff_minus_Omega0=float(np.max(np.abs(om - OMEGA0))),
        interlacing_ok=bool(ev[0] <= min(Q) + tol),
        blocks=blocks,
        phase_invariance_reference=phase_invariance(n, m, C),
    )
    if sigma_S is not None:
        out["max_abs_sigma_nm_minus_S"] = float(np.max(np.abs(sig_nm - sigma_S)))
    var = {}
    for vname in VARIANTS:
        s = sigma_from_nm(n, m, C, vname)
        lm = float(np.linalg.eigvalsh(s)[0])
        var[vname] = dict(min_eig=lm, interlacing_ok=bool(lm <= min(Q) + tol),
                          phase_invariance=phase_invariance(n, m, C, vname))
        if reported_min is not None:
            var[vname]["diff_to_reported"] = lm - reported_min
    out["bug_variants"] = var
    if verbose:
        print_report(out, reported_min)
    return out


def analyze_uv(U, V, reported_min=None, verbose=True):
    U = np.asarray(U, complex)
    V = np.asarray(V, complex)
    if U.shape[0] != 2 and U.ndim == 2 and U.shape[1] == 2:
        U, V = U.T, V.T
    sig_S, om = sigma_omega_from_uv(U, V)
    n, m, C, RA = nm_from_uv(U, V)
    out = analyze_nm(n, m, C, sigma_S=sig_S, omega_eff=om, reported_min=reported_min, verbose=False)
    out["R_A_max"] = float(np.max(np.abs(RA)))
    out["ncol"] = int(U.shape[1])
    if verbose:
        print_report(out, reported_min)
    return out


def print_report(o, reported_min=None):
    f = lambda x: f"{x:.10g}"
    print(f"  Q0, Q1                      = {f(o['Q'][0])}, {f(o['Q'][1])}")
    print(f"  σ 本征值                    = {', '.join(f(x) for x in o['eig_sigma'])}")
    print(f"  交错 λmin(σ) ≤ min Q        : {'通过' if o['interlacing_ok'] else '违反'}")
    for b in o["blocks"]:
        print(f"  偏振{b['mode']} 块最小本征值 = {f(b['block_min_eig'])}  (Q={f(b['Q'])}, "
              f"Var[X(θmin)]={f(b['var_at_theta_min'])}, θmin={b['theta_min']:.6f})")
    print(f"  辛本征值(Ω0)                = {', '.join(f(x) for x in o['symplectic_Omega0'])}")
    print(f"  部分转置辛本征值(仅灵敏度)  = {', '.join(f(x) for x in o['pt_symplectic_Omega0'])}")
    print(f"  min eig σ+iΩ0/2, σ+iΩeff/2  = {o['min_eig_sigma_plus_iOmega0_over2']:.3e}, "
          f"{o['min_eig_sigma_plus_iOmegaEff_over2']:.3e}")
    print(f"  |Ωeff−Ω0|max, |R_N|max      = {o['max_abs_Omega_eff_minus_Omega0']:.3e}, {o['R_N_max']:.3e}"
          + (f", |R_A|max = {o['R_A_max']:.3e}" if "R_A_max" in o else ""))
    if "max_abs_sigma_nm_minus_S" in o:
        print(f"  |σ_nm − σ_S|max             = {o['max_abs_sigma_nm_minus_S']:.3e}")
    print(f"  参考装配相位不变性(相对)    = {o['phase_invariance_reference']:.2e}")
    print("  常见错误装配：")
    for k, v in o["bug_variants"].items():
        extra = f", 与报告值差 {v['diff_to_reported']:+.4g}" if reported_min is not None else ""
        print(f"    {k:15s} λmin={f(v['min_eig'])}  交错{'通过' if v['interlacing_ok'] else '违反'}  "
              f"相位不变性 {v['phase_invariance']:.1e}{extra}")


# ---------------------------------------------------------------- 解析自检
def random_unitary(k, rng):
    Z = (rng.normal(size=(k, k)) + 1j * rng.normal(size=(k, k))) / np.sqrt(2)
    Qm, R = np.linalg.qr(Z)
    d = np.diag(R)
    return Qm * (d / np.abs(d))


def random_bogoliubov(r, rng):
    k = len(r)
    W1, W2 = random_unitary(k, rng), random_unitary(k, rng)
    return W1 @ np.diag(np.cosh(r)) @ W2, W1 @ np.diag(np.sinh(r)) @ W2.conj()


def selftest():
    ok = True

    def check(name, cond, detail=""):
        nonlocal ok
        ok &= bool(cond)
        print(f"  [{'OK' if cond else 'FAIL'}] {name} {detail}")

    rng = np.random.default_rng(20261018)
    pad = lambda A, k: np.hstack([A, np.zeros((2, k - A.shape[1]), complex)])

    print("自检 1：真空")
    o = analyze_uv(pad(np.eye(2, dtype=complex), 5), np.zeros((2, 5), complex), verbose=False)
    check("σ = I/2", np.allclose(o["eig_sigma"], 0.5, atol=1e-15))
    check("Ωeff = Ω0", o["max_abs_Omega_eff_minus_Omega0"] < 1e-15)

    print("自检 2：两个独立单模压缩，带复相位")
    r, ph = np.array([0.7, 1.3]), np.array([0.4, 2.1])
    o = analyze_uv(pad(np.diag(np.cosh(r)).astype(complex), 4),
                   pad(np.diag(np.exp(1j * ph) * np.sinh(r)), 4), verbose=False)
    want = np.sort(np.concatenate([np.exp(-2 * r), np.exp(2 * r)]) / 2)
    check("σ 本征值 = e^{±2r}/2", np.allclose(o["eig_sigma"], want, rtol=1e-12))
    check("Q_μ = e^{-2r_μ}/2", np.allclose(o["Q"], np.exp(-2 * r) / 2, rtol=1e-12))
    check("Var[X(θmin)] = Q", all(abs(b["var_at_theta_min"] - b["Q"]) < 1e-12 for b in o["blocks"]))
    check("辛本征值 = 1/2", np.allclose(o["symplectic_Omega0"], 0.5, atol=1e-12))

    print("自检 3：双模压缩 + 局部相位 + 复相位分束器")
    r, ph = 1.1, 0.9
    X = np.array([[0, 1], [1, 0]], complex)
    U0, V0 = np.cosh(r) * np.eye(2, dtype=complex), np.exp(1j * ph) * np.sinh(r) * X
    Dl = np.diag(np.exp(1j * np.array([0.3, -1.2])))
    t = 0.61
    W = np.array([[np.cos(t), -np.exp(-1j * 0.8) * np.sin(t)], [np.exp(1j * 0.8) * np.sin(t), np.cos(t)]])
    o_t = analyze_uv(pad(U0, 3), pad(V0, 3), verbose=False)
    o_l = analyze_uv(pad(Dl @ U0, 3), pad(Dl @ V0, 3), verbose=False)
    o_b = analyze_uv(pad(W @ Dl @ U0, 3), pad(W @ Dl @ V0, 3), verbose=False)
    want = np.sort([np.exp(-2 * r) / 2] * 2 + [np.exp(2 * r) / 2] * 2)
    for lab, oo in (("TMSV", o_t), ("局部相位", o_l), ("分束器", o_b)):
        check(f"{lab}: σ 本征值不变 = e^{{±2r}}/2", np.allclose(oo["eig_sigma"], want, rtol=1e-12))
        check(f"{lab}: 辛本征值 = 1/2", np.allclose(oo["symplectic_Omega0"], 0.5, atol=1e-12))
    check("TMSV/局部相位: Q_μ = cosh2r/2", np.allclose(o_t["Q"] + o_l["Q"], np.cosh(2 * r) / 2, rtol=1e-12))
    check("TMSV/局部相位: 部分转置 ν̃- = e^{-2r}/2",
          abs(o_t["pt_symplectic_Omega0"][0] - np.exp(-2 * r) / 2) < 1e-12
          and abs(o_l["pt_symplectic_Omega0"][0] - np.exp(-2 * r) / 2) < 1e-12)

    print("自检 4：随机多列混合态（含生产量级 n~8e4、Q~20 的强相消情形）")
    for lab, rr in (("一般", rng.uniform(0, 2.5, 7)), ("强增益", np.array([6.3, 3.1, 2.2, 1.4, 0.9, 0.5, 0.2, 0.1]))):
        U, V = random_bogoliubov(rr, rng)
        o = analyze_uv(U[:2], V[:2], verbose=False)
        check(f"{lab}: σ_nm = σ_S", o["max_abs_sigma_nm_minus_S"] <= 1e-11 * max(1, max(o["eig_sigma"])),
              f"(差 {o['max_abs_sigma_nm_minus_S']:.1e}, λmax {max(o['eig_sigma']):.3g}, Q {o['Q'][0]:.4g}/{o['Q'][1]:.4g})")
        check(f"{lab}: 交错 λmin ≤ min Q", o["interlacing_ok"])
        check(f"{lab}: 辛本征值 ≥ 1/2", min(o["symplectic_Omega0"]) >= 0.5 - 1e-9)
        check(f"{lab}: σ+iΩ/2 ≥ 0", o["min_eig_sigma_plus_iOmegaEff_over2"] >= -1e-9 * max(o["eig_sigma"]))
        check(f"{lab}: 参考装配相位不变", o["phase_invariance_reference"] < 1e-9)
        bad = [k for k, v in o["bug_variants"].items() if k != "xp_sign" and v["phase_invariance"] < 1e-6]
        check(f"{lab}: 错误装配都会被相位不变性测出（xp_sign 只翻转相位方向，谱不变）", not bad, str(bad) if bad else "")

    print("自检 5：非标准 Ω_eff 必须用 Ω^{-1}（不重跑 Slice D）")
    sig = 0.5 * np.eye(4)
    om2 = 2.0 * OMEGA0
    nu0_new = symp_eigs(sig, OMEGA0)
    nu0_old = symp_eigs_left_multiply(sig, OMEGA0)
    check("Ω0 上 iΩ^{-1}σ 与 iΩσ 一致", np.allclose(nu0_new, nu0_old, atol=1e-12), str(nu0_new))
    nu2_new = symp_eigs(sig, om2)
    nu2_old = symp_eigs_left_multiply(sig, om2)
    check("Ω=2Ω0 上两种写法必须分开", float(np.max(np.abs(nu2_new - nu2_old))) > 1e-6,
          f"inv {nu2_new} left {nu2_old}")
    will = np.sort(np.abs(np.real(np.linalg.eigvals(1j * (np.linalg.inv(om2) @ sig)))))
    check("Ω=2Ω0 的 inv 路径对上 williamson_prod 全谱偶位", np.allclose(will[::2], nu2_new, atol=1e-12))

    print("自检全部通过" if ok else "自检有失败项")
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz")
    ap.add_argument("--u-key", default="u_prod")
    ap.add_argument("--v-key", default="v_prod")
    ap.add_argument("--reported-min", type=float)
    ap.add_argument("--out", default="q_cov4_report.json")
    a = ap.parse_args()
    ok = selftest()
    if a.npz:
        d = np.load(a.npz)
        print(f"\n生产数据 {a.npz} [{a.u_key}, {a.v_key}]")
        o = analyze_uv(d[a.u_key], d[a.v_key], reported_min=a.reported_min)
        with open(a.out, "w", encoding="utf-8") as fh:
            json.dump(o, fh, ensure_ascii=False, indent=1)
        print(f"  已写 {a.out}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
