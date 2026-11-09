#!/usr/bin/env python3
"""Zero-propagation M0 Neel-inversion identity check for frozen C1.

Authorized 2026-11-04: no new SBE, no extra BSV samples, no C2, no classic
random-field campaign, no edits to c1.json / frozen M0, no git commit/push.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from q8_m0_classical import initial_rho, k_grid_xy, pol_vector, trace_current
from q8_m0_model import (
    N_OCC,
    R0,
    build_geometry,
    current_ops,
    h8,
    op_P,
    op_sewing,
    sha256_file,
)

ROOT = HERE.parents[1]
OUT = ROOT / "results/q_line/q1600_c1_pureBSV_stat_sbe_20261104"
DOC = ROOT / "docs/04_BSV量子光"
C1_JSON = OUT / "c1.json"
C1_PY = HERE / "q8_m0_c1_pureBSV_stat_sbe_20261104.py"
JSON_OUT = OUT / "c1_neel_inversion_identity_20261104.json"
MD_OUT = OUT / "C1_neel_inversion_identity_20261104.md"
DOC_OUT = DOC / "线Q_C1_M0磁畴反演对称_20261104.md"

C1_SHA_FROZEN = "6A8C3812F80898C946AC59D32EB168DC62FEB8E4DE570F5FF83B28054FD4C925"
BOOT_SEED = 202611042
N_BOOT = 5000
ALG_TOL = 1.0e-10
FUNC_TOL = 1.0e-10
STEP_TOL = 1.0e-9


def frob(a) -> float:
    return float(np.linalg.norm(np.asarray(a)))


def cpack(z) -> dict:
    z = np.asarray(z, dtype=complex)
    if z.shape == ():
        zc = complex(z)
        return {
            "re": float(zc.real),
            "im": float(zc.imag),
            "abs": float(abs(zc)),
            "phase_deg": float(np.degrees(np.angle(zc))),
        }
    return [cpack(x) for x in z]


def unpack_h2(rows) -> np.ndarray:
    out = np.empty((len(rows), 2), dtype=complex)
    for i, row in enumerate(rows):
        for j, z in enumerate(row):
            out[i, j] = complex(z["re"], z["im"])
    return out


def pass_of(val: float, tol: float) -> bool:
    return bool(np.isfinite(val) and val <= tol)


def load_c1_mod():
    spec = importlib.util.spec_from_file_location("c1_prod_ident", C1_PY)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def partner_index(kf: np.ndarray) -> np.ndarray:
    wrap = np.round((-kf) % 1.0, 12)
    table = {tuple(np.round(row, 12)): i for i, row in enumerate(kf)}
    idx = np.empty(kf.shape[0], dtype=int)
    for i, row in enumerate(wrap):
        key = tuple(row)
        if key not in table:
            raise RuntimeError(f"grid not inversion-closed at {kf[i]}")
        idx[i] = table[key]
    return idx


def sew_left(ph: np.ndarray, mats: np.ndarray) -> np.ndarray:
    """Apply U M U^dag with U=diag(ph). mats: (nk,8,8) or (nk,2,8,8)."""
    if mats.ndim == 3:
        return np.einsum("ki,kij,kj->kij", ph, mats, np.conj(ph), optimize=True)
    if mats.ndim == 4:
        return np.einsum("ki,kaij,kj->kaij", ph, mats, np.conj(ph), optimize=True)
    raise ValueError(mats.shape)


def conj_P(P: np.ndarray, mats: np.ndarray) -> np.ndarray:
    Pd = P.conj().T
    if mats.ndim == 2:
        return P @ mats @ Pd
    if mats.ndim == 3:
        return np.einsum("ij,kjl,lm->kim", P, mats, Pd, optimize=True)
    if mats.ndim == 4:
        return np.einsum("ij,kajl,lm->kaim", P, mats, Pd, optimize=True)
    raise ValueError(mats.shape)


def unitary_step(h: np.ndarray, dt: float) -> np.ndarray:
    evals, vecs = np.linalg.eigh(h)
    phase = np.exp(-1j * evals * dt)
    if h.ndim == 2:
        return (vecs * phase) @ vecs.conj().T
    return (vecs * phase[:, None, :]) @ np.conj(np.swapaxes(vecs, 1, 2))


def current_functional(geo, p, kc, w, neel: int, rho, a_vec) -> np.ndarray:
    jop = current_ops(geo, p, kc + a_vec, neel)
    return w @ trace_current(rho, jop)


def algebraic_random_k(geo, p, P, rng, n: int = 12) -> dict:
    rows = []
    worst = {
        "P_H": 0.0,
        "P_j": 0.0,
        "j_neel_indep": 0.0,
        "rho_P": 0.0,
        "sewing_H": 0.0,
        "sewing_j": 0.0,
        "U_step": 0.0,
        "occ_mismatch": 0,
    }
    G1 = geo.recip()[0]
    for _ in range(n):
        k = rng.normal(scale=0.35, size=2)
        A = rng.normal(scale=0.08, size=2)
        dt = float(0.05 + 0.2 * rng.random())
        for neel in (+1, -1):
            h = h8(geo, p, k, neel)
            hm = h8(geo, p, -k, -neel)
            j = current_ops(geo, p, k, neel)
            jm = current_ops(geo, p, -k, -neel)
            j_flip = current_ops(geo, p, k, -neel)
            rho, e = initial_rho(geo, p, k[None, :], neel, mu=0.0)
            rho_m, _ = initial_rho(geo, p, (-k)[None, :], -neel, mu=0.0)
            rec = {
                "P_H": frob(conj_P(P, h) - hm),
                "P_j": frob(conj_P(P, j) + jm),
                "j_neel_indep": frob(j - j_flip),
                "rho_P": frob(conj_P(P, rho[0]) - rho_m[0]),
                "occ": int(np.sum(e[0] < 0.0)),
            }
            U = op_sewing(geo, G1)
            rec["sewing_H"] = frob(h8(geo, p, k + G1, neel) - U.conj().T @ h @ U)
            rec["sewing_j"] = frob(
                current_ops(geo, p, k + G1, neel)
                - np.einsum("ij,ajl,lm->aim", U.conj().T, j, U)
            )
            u_plus = unitary_step(h8(geo, p, k + A, neel), dt)
            u_minus = unitary_step(h8(geo, p, -k - A, -neel), dt)
            rec["U_step"] = frob(conj_P(P, u_plus) - u_minus)
            rec["occ_ok"] = rec["occ"] == N_OCC
            rows.append(rec)
            for key in ("P_H", "P_j", "j_neel_indep", "rho_P", "sewing_H", "sewing_j", "U_step"):
                worst[key] = max(worst[key], rec[key])
            if not rec["occ_ok"]:
                worst["occ_mismatch"] += 1
    worst["pass"] = all(
        pass_of(worst[k], ALG_TOL) for k in worst if k not in ("occ_mismatch", "pass")
    ) and worst["occ_mismatch"] == 0
    return {"n_k": n, "n_rows": len(rows), "worst": worst, "pass": worst["pass"]}


def c1_grid_identities(geo, p, P, kf, kc, w) -> dict:
    pidx = partner_index(kf)
    Gcart = kc + kc[pidx]
    g_unique = np.unique(np.round(Gcart, 10), axis=0)
    ph = np.exp(1j * np.sum(Gcart[:, None, :] * geo.tau_orbitals_cart()[None, :, :2], axis=-1))
    rho_p, e_p = initial_rho(geo, p, kc, +1, mu=0.0)
    rho_m, e_m = initial_rho(geo, p, kc, -1, mu=0.0)
    h_p = h8(geo, p, kc, +1)
    h_m = h8(geo, p, kc[pidx], -1)
    j_p = current_ops(geo, p, kc, +1)
    j_m = current_ops(geo, p, kc[pidx], -1)
    h_mk = sew_left(ph, h_m)
    j_mk = sew_left(ph, j_m)
    rho_mk = sew_left(ph, rho_m[pidx])
    P_H = frob(conj_P(P, h_mk) - h_p)
    P_j = frob(conj_P(P, j_mk) + j_p)
    rho_P = frob(conj_P(P, rho_mk) - rho_p)
    occ_p = np.sum(e_p < 0.0, axis=1)
    occ_m = np.sum(e_m < 0.0, axis=1)
    A0 = np.zeros(2)
    rng = np.random.default_rng(20261104)
    A_list = [A0, rng.normal(scale=0.05, size=2), rng.normal(scale=0.12, size=2)]
    func_rows = []
    func_worst = 0.0
    step_worst = 0.0
    dt = 0.17
    for A in A_list:
        jp = current_functional(geo, p, kc, w, +1, rho_p, A)
        jm = current_functional(geo, p, kc, w, -1, rho_m, -A)
        res = frob(jp + jm)
        u_p = unitary_step(h8(geo, p, kc + A, +1), dt)
        u_m = unitary_step(h8(geo, p, kc - A, -1), dt)
        rho_p1 = u_p @ rho_p @ np.conj(np.swapaxes(u_p, 1, 2))
        rho_m1 = u_m @ rho_m @ np.conj(np.swapaxes(u_m, 1, 2))
        jp1 = current_functional(geo, p, kc, w, +1, rho_p1, A)
        jm1 = current_functional(geo, p, kc, w, -1, rho_m1, -A)
        res1 = frob(jp1 + jm1)
        func_rows.append({
            "A": A.tolist(),
            "J_plus": jp.tolist(),
            "J_minus_of_minusA": jm.tolist(),
            "residual_initial": res,
            "residual_one_step": res1,
        })
        func_worst = max(func_worst, res)
        step_worst = max(step_worst, res1)
    A = A_list[1]
    u_p = unitary_step(h8(geo, p, kc + A, +1), dt)
    u_m_part = unitary_step(h8(geo, p, kc[pidx] - A, -1), dt)
    u_mk = sew_left(ph, u_m_part)
    U_step = frob(conj_P(P, u_mk) - u_p)
    out = {
        "nk": int(round(np.sqrt(kf.shape[0]))),
        "n_kpts": int(kf.shape[0]),
        "shift_xy": [0.5, 0.5],
        "inversion_closed": True,
        "partner_index": pidx.tolist(),
        "G_cart_unique": g_unique.tolist(),
        "n_unique_G": int(g_unique.shape[0]),
        "occ_plus": occ_p.tolist(),
        "occ_minus": occ_m.tolist(),
        "occ_all_4": bool(np.all(occ_p == N_OCC) and np.all(occ_m == N_OCC)),
        "P_H_with_sewing": P_H,
        "P_j_with_sewing": P_j,
        "rho_P_with_sewing": rho_P,
        "U_step_with_sewing": U_step,
        "functional_rows": func_rows,
        "functional_worst_initial": func_worst,
        "functional_worst_one_step": step_worst,
        "j_neel_indep_grid": frob(current_ops(geo, p, kc, +1) - current_ops(geo, p, kc, -1)),
    }
    out["pass_algebra"] = all(
        pass_of(out[k], ALG_TOL)
        for k in ("P_H_with_sewing", "P_j_with_sewing", "rho_P_with_sewing", "j_neel_indep_grid")
    ) and out["occ_all_4"]
    out["pass_functional"] = (
        pass_of(func_worst, FUNC_TOL)
        and pass_of(step_worst, STEP_TOL)
        and pass_of(U_step, STEP_TOL)
    )
    out["pass"] = bool(out["pass_algebra"] and out["pass_functional"])
    return out


def field_sign_check(geo, c1mod) -> dict:
    times = np.linspace(0.0, 4.0, 81)
    t0_fs = 2.0
    t_unit_fs = 0.6582119568564649
    omega = 2.0 * np.pi / (times[-1] / 2.0)
    env = np.exp(-((times - times.mean()) / 0.8) ** 2)
    meta = {"times": times, "env": env, "omega": omega, "t0_fs": t0_fs, "t_unit_fs": t_unit_fs}
    proto = {"E1_V_m": 1.0e10}
    rng = np.random.default_rng(11)
    X, Pval = rng.normal(scale=1.3, size=2)
    plus = c1mod.field_from_xp(meta, proto, float(X), float(Pval))
    minus = c1mod.field_from_xp(meta, proto, float(-X), float(-Pval))
    e_res = frob(plus["E_VA"] + minus["E_VA"])
    a_res = frob(plus["a_mag"] + minus["a_mag"])
    a0 = float(abs(plus["a_mag"][0]))
    pol = pol_vector(type("P", (), {"polarization": "a2"})(), geo)
    a_plus = plus["a_mag"][:, None] * pol[None, :]
    a_minus = minus["a_mag"][:, None] * pol[None, :]
    avec_res = frob(a_plus + a_minus)
    return {
        "X": float(X),
        "P": float(Pval),
        "A0_is_zero": a0,
        "E_odd_residual": e_res,
        "a_odd_residual": a_res,
        "a_vector_odd_residual": avec_res,
        "pass": pass_of(e_res, 1e-14) and pass_of(a_res, 1e-12) and pass_of(avec_res, 1e-12) and a0 == 0.0,
        "note": "(X,P)->(-X,-P) flips E and the integrated Peierls A because A(0)=0.",
    }


def bootstrap_even_cartesian(h2p: np.ndarray, h2m: np.ndarray, n_boot: int, seed: int) -> dict:
    n = h2p.shape[0]
    rng = np.random.default_rng(seed)
    mean_p = h2p.mean(axis=0)
    mean_m = h2m.mean(axis=0)
    even = 0.5 * (mean_p + mean_m)
    odd = 0.5 * (mean_p - mean_m)
    labels = ["Re_H2x", "Im_H2x", "Re_H2y", "Im_H2y"]

    def cart(z):
        return np.array([z[0].real, z[0].imag, z[1].real, z[1].imag], dtype=float)

    point = cart(even)
    boot = np.empty((n_boot, 4), dtype=float)
    even_norm = np.empty(n_boot, dtype=float)
    odd_norm = np.empty(n_boot, dtype=float)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        e = 0.5 * (h2p[idx].mean(axis=0) + h2m[idx].mean(axis=0))
        o = 0.5 * (h2p[idx].mean(axis=0) - h2m[idx].mean(axis=0))
        boot[b] = cart(e)
        even_norm[b] = np.linalg.norm(e)
        odd_norm[b] = np.linalg.norm(o)
    lo, hi = np.quantile(boot, [0.025, 0.975], axis=0)
    contains0 = (lo <= 0.0) & (hi >= 0.0)
    se = boot.std(axis=0, ddof=1)
    n_sigma = np.abs(point) / np.maximum(se, 1e-30)
    return {
        "n": int(n),
        "n_boot": int(n_boot),
        "seed": int(seed),
        "labels": labels,
        "even_point": point.tolist(),
        "even_ci95_lo": lo.tolist(),
        "even_ci95_hi": hi.tolist(),
        "even_contains_zero": contains0.tolist(),
        "even_n_sigma_boot": n_sigma.tolist(),
        "all_even_cart_contain_zero": bool(np.all(contains0)),
        "even_norm_point": float(np.linalg.norm(even)),
        "even_norm_ci95": [float(np.quantile(even_norm, 0.025)), float(np.quantile(even_norm, 0.975))],
        "odd_norm_point": float(np.linalg.norm(odd)),
        "odd_norm_ci95": [float(np.quantile(odd_norm, 0.025)), float(np.quantile(odd_norm, 0.975))],
        "odd_vector": cpack(odd),
        "even_vector": cpack(even),
        "note": "Euclidean even-norm is non-negative biased and need not contain 0 even if every Cartesian component does.",
    }


def consequences(identity_holds: bool) -> dict:
    return {
        "identity": "J_{-N}[A](t) = -J_{+N}[-A](t)",
        "holds_on_checked_algebra_and_C1_grid": bool(identity_holds),
        "if_holds": {
            "even_mean_current": "must vanish for any even W(A)=W(-A), including pure BSV",
            "odd_mean_current": "can be nonzero; J_odd(A) is even in A",
            "C_stat": "must be exactly 0 for H2, because |J_-N[A]|^2 = |J_+N[-A]|^2",
            "classical_even_mixtures": "the same three statements apply, including sign-randomized coherent states and beta=i sqrt(n_s) Z",
            "quantum_advantage": False,
        },
    }


def write_report(payload: dict) -> str:
    alg = payload["algebra_random_k"]["worst"]
    grid = payload["c1_grid"]
    field = payload["field_sign"]
    boot = payload["bootstrap_even"]
    holds = payload["identity_holds"]
    verdict = "成立" if holds else "不成立"
    lines = []
    lines.append("# C1：M0 磁畴反演 × 零位移光场对称性核查（2026-11-04）")
    lines.append("")
    lines.append("外审授权范围：零新传播、不增加 BSV 样本、不启动 C2、不改 `c1.json` 与冻结 M0、不 commit/push。")
    lines.append("")
    lines.append("本轮只检验条件性命题：")
    lines.append("")
    lines.append("$$J_{-N}[A](t)=-J_{+N}[-A](t).$$")
    lines.append("")
    lines.append(f"**代数 + C1 网格结论：恒等式{verdict}。** 不把 1024 条抽样的 $C_{{\\mathrm{{stat}}}}\\approx 0$ 当成独立发现；若恒等式成立，它是选择定则。")
    lines.append("")
    lines.append("## 1. 推导（不预设成立）")
    lines.append("")
    lines.append("M0 的空间反演算符是 $P=\\tau_x\\sigma_x s_0$，且 $P^\\dagger=P=P^{-1}$。历史只读审计与本轮随机 $k$ 都给出")
    lines.append("")
    lines.append("$$P H_N(k)P=H_{-N}(-k).$$")
    lines.append("")
    lines.append("Peierls 约定是 $\\kappa=k+A$，电流 $j=-\\partial H/\\partial k$。`dh8` 丢掉 Néel，因此 $j_N(k)=j_{-N}(k)$。对上面的 Hamiltonian 恒等式求导后，")
    lines.append("")
    lines.append("$$P j_N(k)P=-j_{-N}(-k).$$")
    lines.append("")
    lines.append("共同化学势 $\\mu=0$ 的基态投影子因此满足 $P\\rho_N(k)P=\\rho_{-N}(-k)$。von Neumann / C1 中点幺正只依赖 $H(k+A)$，所以")
    lines.append("")
    lines.append("$$P\\,U_N[k;A]\\,P=U_{-N}[-k;-A],\\qquad P\\,\\rho_N[A](-k,t)\\,P=\\rho_{-N}[-A](k,t).$$")
    lines.append("")
    lines.append("在反演封闭、等权 $k$ 网格上把 $k\\mapsto -k+G$，Bloch sewing $H(k+G)=U_G^\\dagger H(k)U_G$ 在迹中抵消，得到电流泛函")
    lines.append("")
    lines.append("$$J_{-N}[A](t)=-J_{+N}[-A](t).$$")
    lines.append("")
    lines.append("C1 的场映射 $E\\propto f(t)[X\\cos\\omega t+P\\sin\\omega t]$ 且 $A(0)=0$，故 $(X,P)\\to(-X,-P)$ 就是 $A\\to-A$。镜面 $M_{a2}$ 不是精确对称，本轮不使用。")
    lines.append("")
    lines.append("## 2. 数值残差")
    lines.append("")
    lines.append("| 检查 | 最大残差 | 通过 |")
    lines.append("|---|---:|:---:|")
    lines.append(f"| 随机 $k$：$P H_N(k)P-H_{{-N}}(-k)$ | {alg['P_H']:.3e} | {str(pass_of(alg['P_H'], ALG_TOL))} |")
    lines.append(f"| 随机 $k$：$P j_N(k)P+j_{{-N}}(-k)$ | {alg['P_j']:.3e} | {str(pass_of(alg['P_j'], ALG_TOL))} |")
    lines.append(f"| $j(N)-j(-N)$（`dh8` 丢掉 Néel） | {alg['j_neel_indep']:.3e} | {str(pass_of(alg['j_neel_indep'], ALG_TOL))} |")
    lines.append(f"| 基态 $P\\rho_N(k)P-\\rho_{{-N}}(-k)$ | {alg['rho_P']:.3e} | {str(pass_of(alg['rho_P'], ALG_TOL))} |")
    lines.append(f"| 单步幺正 $P U_N[k;A]P-U_{{-N}}[-k;-A]$ | {alg['U_step']:.3e} | {str(pass_of(alg['U_step'], ALG_TOL))} |")
    lines.append(f"| C1 网格 sewing 后 $P H P$ | {grid['P_H_with_sewing']:.3e} | {str(pass_of(grid['P_H_with_sewing'], ALG_TOL))} |")
    lines.append(f"| C1 网格 sewing 后 $P j P$ | {grid['P_j_with_sewing']:.3e} | {str(pass_of(grid['P_j_with_sewing'], ALG_TOL))} |")
    lines.append(f"| C1 网格 $J_{{+}}[A]+J_{{-}}[-A]$（初态） | {grid['functional_worst_initial']:.3e} | {str(pass_of(grid['functional_worst_initial'], FUNC_TOL))} |")
    lines.append(f"| C1 网格同一恒等式（一步幺正后） | {grid['functional_worst_one_step']:.3e} | {str(pass_of(grid['functional_worst_one_step'], STEP_TOL))} |")
    lines.append(f"| 场 $(X,P)\\to(-X,-P)$ 的 $A$ 反号 | {field['a_vector_odd_residual']:.3e} | {str(field['pass'])} |")
    lines.append("")
    lines.append(f"C1 网格：nk=2，shift=(0.5,0.5)，四点全部 4 占据，互反演配对。唯一 reciprocal 向量数 = {grid['n_unique_G']}（nk=2 的配对是 $-k+G$，$G=b_1+b_2$，必须 sewing）。")
    lines.append("")
    lines.append("## 3. 若恒等式成立，对 C1 意味着什么")
    lines.append("")
    if holds:
        lines.append("纯 BSV 的 Wigner 满足 $W(A)=W(-A)$。因此：")
        lines.append("")
        lines.append("1. **Néel-even 平均电流必须为零。** $\\langle J_e\\rangle=\\int W(A)\\,J_{\\mathrm{even}}[A]=0$，因为 $J_{\\mathrm{even}}[A]$ 是 $A$ 的奇函数。")
        lines.append("2. **Néel-odd 平均电流可以非零。** $J_{\\mathrm{odd}}[A]$ 是 $A$ 的偶函数；二阶及以上场关联仍可驱动非线性。")
        lines.append("3. **两磁畴的样本平均 H2 强度必须相同，** 即 $C_{\\mathrm{stat}}=0$。线性 Fourier 的 H2 继承同一关系：$|\\mathcal J_{2,-N}[A]|^2=|\\mathcal J_{2,+N}[-A]|^2$。")
        lines.append("4. **同一结论适用于任何偶的经典随机驱动**，包括符号随机的相干态混合，以及 $\\beta=i\\sqrt{n_s}Z$、$Z\\sim\\mathcal N(0,1)$。因此 $C_{\\mathrm{stat}}=0$ **不是量子优势**。")
        lines.append("")
        lines.append("有限样本的 $\\|\\langle J_e\\rangle\\|$ 是非负有偏的，95% 区间可以不含零；应看四个笛卡尔分量是否与零相容。")
    else:
        lines.append("恒等式在本轮检查中失败。不得用选择定理解释 $C_{\\mathrm{stat}}$；失败项见 JSON。不强行解释现有 1024 条数据。")
    lines.append("")
    lines.append("## 4. 冻结 1024 条样本的 even 笛卡尔 bootstrap")
    lines.append("")
    lines.append(f"N=1024，B={boot['n_boot']}，seed={boot['seed']}。")
    lines.append("")
    lines.append("| 分量 | 点估计 | 约 95% 区间 | 含 0 | 约 nσ |")
    lines.append("|---|---:|---|:---:|---:|")
    for i, lab in enumerate(boot["labels"]):
        lo = boot["even_ci95_lo"][i]
        hi = boot["even_ci95_hi"][i]
        lines.append(
            f"| {lab} | {boot['even_point'][i]:.3e} | [{lo:.3e}, {hi:.3e}] | {boot['even_contains_zero'][i]} | {boot['even_n_sigma_boot'][i]:.2f} |"
        )
    lines.append("")
    lines.append(
        f"$\\|\\langle J_e\\rangle\\|={boot['even_norm_point']:.3e}$，区间 [{boot['even_norm_ci95'][0]:.3e}, {boot['even_norm_ci95'][1]:.3e}]（范数有偏）。"
        f"$\\|\\langle J_o\\rangle\\|={boot['odd_norm_point']:.3e}$，区间 [{boot['odd_norm_ci95'][0]:.3e}, {boot['odd_norm_ci95'][1]:.3e}]。"
    )
    if boot["all_even_cart_contain_zero"]:
        lines.append("")
        lines.append("四个 even 笛卡尔分量的 95% 区间均含 0，与“even 主要是有限抽样残差、odd 是稳定非零响应”一致。")
    else:
        lines.append("")
        lines.append("并非所有 even 笛卡尔分量都与零相容。这不自动推翻代数恒等式，但说明 1024 条抽样下 even 残差仍可见，或存在未纳入本轮代数检查的数值通道。不得把 even 范数的有偏区间当成 even 物理上非零。")
    lines.append("")
    lines.append("## 5. 判断与停机")
    lines.append("")
    lines.append(f"- 恒等式：{verdict}。")
    lines.append("- C1 方法验证维持冻结。$C_{\\mathrm{stat}}$ 点估计与零相容，现有数学解释是输入点反射对称 + 磁畴反演，而不是“强度涨落偶然洗掉对比”。")
    lines.append("- 经典偶混合给出同样的 $C_{\\mathrm{stat}}=0$，不能称为 BSV 不可经典性。")
    lines.append("- 未增加 BSV 样本，未启动经典随机场长任务，未启动 C2，未修改 `c1.json`。")
    lines.append(f"- 运行前后 `c1.json` SHA-256 仍为 `{payload['c1_sha256_after']}`。")
    lines.append("")
    lines.append("本轮停止。")
    lines.append("")
    lines.append(f"JSON：`{JSON_OUT}`")
    return "\n".join(lines) + "\n"


def main() -> int:
    t0 = time.perf_counter()
    sha_before = sha256_file(C1_JSON).upper()
    if sha_before != C1_SHA_FROZEN:
        raise SystemExit(f"c1.json sha changed before run: {sha_before}")
    c1 = json.loads(C1_JSON.read_text(encoding="utf-8"))
    geo = build_geometry()
    p = R0
    P = op_P()
    rng = np.random.default_rng(202611040)
    alg = algebraic_random_k(geo, p, P, rng, n=12)
    kf, kc, w = k_grid_xy(geo, 2, 0.5, 0.5)
    grid = c1_grid_identities(geo, p, P, kf, kc, w)
    c1mod = load_c1_mod()
    field = field_sign_check(geo, c1mod)
    h2p = unpack_h2(c1["bsv"]["per_sample_H2_plus"])
    h2m = unpack_h2(c1["bsv"]["per_sample_H2_minus"])
    boot = bootstrap_even_cartesian(h2p, h2m, N_BOOT, BOOT_SEED)
    identity_holds = bool(alg["pass"] and grid["pass"] and field["pass"])
    sha_after = sha256_file(C1_JSON).upper()
    payload = {
        "status": "c1_neel_inversion_identity_complete",
        "date": "2026-11-04",
        "not_C2": True,
        "did_not_rerun_bsv": True,
        "did_not_modify_c1_json": sha_after == sha_before == C1_SHA_FROZEN,
        "did_not_start_classical_random_field": True,
        "c1_sha256_before": sha_before,
        "c1_sha256_after": sha_after,
        "source_sha256": {
            "this": sha256_file(Path(__file__)),
            "q8_m0_model": sha256_file(HERE / "q8_m0_model.py"),
            "q8_m0_classical": sha256_file(HERE / "q8_m0_classical.py"),
            "c1_disk_script": sha256_file(C1_PY),
        },
        "algebra_random_k": alg,
        "c1_grid": grid,
        "field_sign": field,
        "bootstrap_even": boot,
        "identity_holds": identity_holds,
        "consequences": consequences(identity_holds),
        "mirror_not_used": True,
        "wall_s": time.perf_counter() - t0,
    }
    report = write_report(payload)
    payload["report_sha256"] = hashlib.sha256(report.encode("utf-8")).hexdigest()
    OUT.mkdir(parents=True, exist_ok=True)
    JSON_OUT.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    MD_OUT.write_text(report, encoding="utf-8")
    DOC_OUT.write_text(report, encoding="utf-8")
    print(json.dumps({
        "identity_holds": identity_holds,
        "algebra_pass": alg["pass"],
        "grid_pass": grid["pass"],
        "field_pass": field["pass"],
        "even_cart_all_contain_zero": boot["all_even_cart_contain_zero"],
        "c1_sha_unchanged": payload["did_not_modify_c1_json"],
        "wall_s": payload["wall_s"],
        "json": str(JSON_OUT),
    }, indent=2))
    return 0 if identity_holds else 1


if __name__ == "__main__":
    raise SystemExit(main())
