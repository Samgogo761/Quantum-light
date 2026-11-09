"""A2: finite-time second-order Peierls kernel on the Q6 1+4+1 window.

Classical drive: photon quadratures rotate at omega0, no Ehrenfest kick.
Electrons evolve under h8(k + g s(t) X(t) ehat). Extract A^2 by
Richardson finite difference, then linear_exact Fourier on [2T0,4T0].
This is not chi^(2) certification and not a quantum output kernel.
"""
from __future__ import annotations

import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import numpy as np
from scipy.linalg import expm

import q8_m0_abc_microscopic_kernel_20261029 as adia
import q8_m0_qinput_dynamics_20261026 as dyn
import q8_m0_qinput_minimal_20261020 as qmin
import q8_m0_qinput_shared_kk_dyn_preflight_20261030 as pf
import q8_m0_qinput_shared_kk_mtef_map_20261108 as mp
import q8_m0_qinput_shared_kk_static_20261029 as sh
import qinput_spectrum_reanalysis_20261028 as spec
from q8_m0_model import M0Params, R0, build_geometry, current_ops, h8

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results/q_line/abc_a2_finite_time_kernel_20261029"
FIG = OUT / "figures"
DOCS = ROOT / "docs/04_BSV量子光"
N_FULL = 11154
SAMPLE_EVERY = 32
EPS = 0.05
MAIN_WIN = (2.0, 4.0)
QS = (1, 2, 3, 4, 5)
GAMMA_ADIA = 0.03

# Frozen QED/GH5 window phases from 线Q_ABC微观机制_20261029.md. Not recomputed.
FROZEN_ARG_B_DEG = {
    "eta0.5_GH5": 156.1,
    "eta1.0_GH5": 167.7,
    "eta1.0_QED": 163.9,
    "eta1.5_GH5": -141.5,
}
FROZEN_ADIA_DEG = {
    "eta0.5": -117.6,
    "eta1.0": -117.2,
    "eta1.5": -110.8,
}


def dump(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(adia.jsonable(obj) if hasattr(adia, "jsonable") else obj,
                              ensure_ascii=False, indent=2, default=_default) + "\n",
                   encoding="utf-8")
    os.replace(tmp, path)


def _default(x):
    if isinstance(x, (np.floating, np.integer)):
        return x.item()
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, complex):
        return {"re": x.real, "im": x.imag, "abs": abs(x), "arg_deg": float(np.degrees(np.angle(x)))}
    raise TypeError(type(x))


def progress(msg: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    line = f"{time.strftime('%H:%M:%S')} {msg}"
    with (OUT / "progress.txt").open("a", encoding="utf-8") as fh:
        fh.write(line + "\n")
    print(line, flush=True)


def cpack(z):
    z = complex(z)
    return {"re": float(z.real), "im": float(z.imag), "abs": float(abs(z)),
            "arg_deg": float(np.degrees(np.angle(z)))}


def wrap(phi: float) -> float:
    return float((phi + np.pi) % (2.0 * np.pi) - np.pi)


def angdiff_deg(a: float, b: float) -> float:
    return float(np.degrees(wrap(np.radians(float(a) - float(b)))))


def params_eta(eta: float) -> M0Params:
    return M0Params(t_h=R0.t_h, lam_so=R0.lam_so, delta=float(eta) * R0.delta,
                    u=R0.u, v=R0.v, label=f"R0_eta{eta:g}")


def params_ctrl(name: str) -> M0Params:
    if name.startswith("eta"):
        return params_eta(float(name[3:]))
    if name == "d0":
        return M0Params(t_h=R0.t_h, lam_so=R0.lam_so, delta=0.0, u=R0.u, v=R0.v, label="R0_d0")
    if name == "s0":
        return M0Params(t_h=R0.t_h, lam_so=0.0, delta=R0.delta, u=R0.u, v=R0.v, label="R0_s0")
    raise KeyError(name)


def sample_js(n_steps: int):
    js = list(range(0, n_steps, SAMPLE_EVERY))
    if not js or js[-1] != n_steps - 1:
        js.append(n_steps - 1)
    return js


def occupied(geo, p, k, neel: int):
    ev, vec = np.linalg.eigh(h8(geo, p, k, neel))
    return vec[:, :4].copy(), ev


def propagate_classical(geo, p, k0, neel, ehat, A0, env, n_steps, js):
    """Prescribed X(t)=A0 cos(omega t) via rotate_xp; no current kick."""
    dt = env.t_total / n_steps
    want = set(int(j) for j in js)
    psi, _ = occupied(geo, p, k0, neel)
    X, P = float(A0), 0.0
    j_rows, t_rows = [], []
    max_leak = 0.0
    g = float(pf.G_ABS)
    ehat = np.asarray(ehat, float)
    for j in range(n_steps):
        X, P = mp.rotate_xp(X, P, 0.5 * dt, pf.OMEGA0)
        s = env.s((j + 0.5) * dt)
        kshift = np.asarray(k0, float) + g * s * X * ehat
        H = h8(geo, p, kshift, neel)
        psi = expm(-1j * dt * H) @ psi
        nrm = np.linalg.norm(psi, axis=0)
        max_leak = max(max_leak, float(np.max(np.abs(nrm - 1.0))))
        X, P = mp.rotate_xp(X, P, 0.5 * dt, pf.OMEGA0)
        if j in want:
            t = (j + 1) * dt
            s_t = env.s(t)
            k_t = np.asarray(k0, float) + g * s_t * X * ehat
            jop = current_ops(geo, p, k_t, neel)
            jvec = np.zeros(2, complex)
            for a in range(2):
                jvec[a] = np.einsum("in,ij,jn->", psi.conj(), jop[a], psi)
            j_rows.append(jvec)
            t_rows.append(t)
    return np.asarray(j_rows, complex), np.asarray(t_rows, float), max_leak


def pair_current(geo, p, kcart, neel, ehat, A0, env, n_steps, js):
    jk, t, lk = propagate_classical(geo, p, kcart, neel, ehat, A0, env, n_steps, js)
    jm, tm, lm = propagate_classical(geo, p, -np.asarray(kcart, float), neel, ehat, A0, env, n_steps, js)
    if np.max(np.abs(t - tm)) > 1e-12:
        raise RuntimeError("k/-k time mismatch")
    return jk + jm, t, max(lk, lm)


def richardson_second(jmap, eps: float):
    j0 = jmap[0.0]
    jp, jm = jmap[eps], jmap[-eps]
    j2p, j2m = jmap[2.0 * eps], jmap[-2.0 * eps]
    first = (jp + jm - 2.0 * j0) / (eps * eps)
    second = (j2p + j2m - 2.0 * j0) / (4.0 * eps * eps)
    rich = (4.0 * first - second) / 3.0
    return first, second, rich


def fourier_pack(t, j):
    T0 = 2.0 * np.pi / pf.OMEGA0
    out = {}
    for name, (a, b) in (("main", MAIN_WIN), ("front", (1.0, 3.0)), ("back", (3.0, 5.0))):
        jq = spec.spectrum(t, j, a * T0, b * T0, QS, method="linear_exact", omega=pf.OMEGA0)
        out[name] = {f"H{q}": [cpack(jq[i, 0]), cpack(jq[i, 1])] for i, q in enumerate(QS)}
        if name == "main":
            out["main_arr"] = jq
    return out


def odd_even(jp, jm):
    return 0.5 * (jp - jm), 0.5 * (jp + jm)


def adiabatic_odd_x(geo, p, kcart, ehat, omega, gamma):
    rec = adia.odd_even_kernel(geo, p, kcart, ehat, np.array([1.0, 0.0]),
                               omega + 1j * gamma, omega + 1j * gamma)
    return 0.5 * rec["odd"]


def run_one(name: str, geo, kcart, ehat, env, n_steps, js, eps: float) -> dict:
    p = params_ctrl(name)
    t0 = time.perf_counter()
    amps = (0.0, eps, -eps, 2.0 * eps, -2.0 * eps)
    series = {}
    leaks = {}
    t_arr = None
    for neel in (1, -1):
        series[neel] = {}
        for A0 in amps:
            j, t, leak = pair_current(geo, p, kcart, neel, ehat, A0, env, n_steps, js)
            series[neel][float(A0)] = j
            leaks[f"N{neel}_A{A0:g}"] = leak
            t_arr = t
            progress(f"{name} N={neel:+d} A0={A0:g} leak={leak:.2e} wall={time.perf_counter()-t0:.1f}s")
    first_p, second_p, rich_p = richardson_second(series[1], eps)
    first_m, second_m, rich_m = richardson_second(series[-1], eps)
    odd_rich, even_rich = odd_even(rich_p, rich_m)
    pack_o = fourier_pack(t_arr, odd_rich)
    pack_e = fourier_pack(t_arr, even_rich)
    jq_o = pack_o.pop("main_arr")
    jq_e = pack_e.pop("main_arr")
    Kx = adiabatic_odd_x(geo, p, kcart, ehat, pf.OMEGA0, GAMMA_ADIA)
    ft_phase = float(np.degrees(np.angle(jq_o[1, 0])))
    ad_phase = float(np.degrees(np.angle(Kx)))
    rec = {
        "name": name, "label": p.label, "delta": float(p.delta), "lam_so": float(p.lam_so),
        "eps": eps, "n_steps": n_steps, "wall_s": time.perf_counter() - t0,
        "max_leak": max(leaks.values()), "leaks": leaks,
        "odd_H2_x": cpack(jq_o[1, 0]), "odd_H2_y": cpack(jq_o[1, 1]),
        "even_H2_x": cpack(jq_e[1, 0]), "even_H2_y": cpack(jq_e[1, 1]),
        "odd_norm2D": float(np.linalg.norm(jq_o[1])),
        "even_norm2D": float(np.linalg.norm(jq_e[1])),
        "adiabatic_odd_x": cpack(Kx),
        "finite_time_arg_deg": ft_phase,
        "adiabatic_arg_deg": ad_phase,
        "finite_minus_adiabatic_deg": angdiff_deg(ft_phase, ad_phase),
        "windows_odd": pack_o, "windows_even": pack_e,
        "richardson_rel_odd_H2": float(
            abs(np.linalg.norm(0.5 * (fourier_pack(t_arr, 0.5 * (first_p - first_m)).pop("main_arr")[1]
                                      )) )
        ) if False else None,
    }
    # Richardson vs first-order FD relative change of odd H2 x
    jq_first = fourier_pack(t_arr, 0.5 * (first_p - first_m))["main_arr"]
    rec["fd_eps_odd_H2_x"] = cpack(jq_first[1, 0])
    rec["richardson_vs_fd_rel"] = float(
        abs(jq_o[1, 0] - jq_first[1, 0]) / max(abs(jq_first[1, 0]), 1e-30))
    rec["front_back_odd_H2_x_rel"] = float(
        abs(complex(pack_o["front"]["H2"][0]["re"], pack_o["front"]["H2"][0]["im"])
            - complex(pack_o["back"]["H2"][0]["re"], pack_o["back"]["H2"][0]["im"]))
        / max(abs(jq_o[1, 0]), 1e-30)
    )
    return rec


def write_reports(rec: dict) -> None:
    one = DOCS / "线Q_A2有限时间核_一页_20261029.md"
    longp = DOCS / "线Q_A2有限时间核_20261029.md"
    eta1 = rec["cases"].get("eta1.0", rec["cases"].get("eta1"))
    lines = [
        "# 线 Q：A2 有限时间二阶核一页结论（2026-10-29）",
        "",
        "与 Q6 相同的 1+4+1 开关和主窗 [2T0,4T0]，经典弱场、无光子反作用。",
        "不是 χ^{(2)} 认证，也不是量子输出核。",
        "",
        "## 1. 能否解释至少一半的 79° 相位差？",
        "",
    ]
    if eta1:
        qed = FROZEN_ARG_B_DEG["eta1.0_QED"]
        ad = eta1["adiabatic_arg_deg"]
        ft = eta1["finite_time_arg_deg"]
        d_ad = angdiff_deg(qed, ad)
        d_ft = angdiff_deg(qed, ft)
        explained = abs(d_ad) - abs(d_ft)
        frac = explained / max(abs(d_ad), 1e-12)
        rec["eta1_qed_minus_adiabatic_deg"] = d_ad
        rec["eta1_qed_minus_finite_deg"] = d_ft
        rec["eta1_explained_fraction_of_79"] = frac
        lines += [
            f"- 绝热核相位 {ad:.2f}°，有限时间核 {ft:.2f}°，QED arg(B) {qed:.2f}°。",
            f"- 绝热相对 QED：{d_ad:.2f}°；有限时间相对 QED：{d_ft:.2f}°。",
            f"- 解释了原相位差的 **{100*frac:.1f}%**（合格线是 50%）。",
            f"- Richardson 相对 ε 差分 {eta1['richardson_vs_fd_rel']:.3e}；邻窗相对变化 {eta1['front_back_odd_H2_x_rel']:.3g}。",
            "",
        ]
        if abs(frac) >= 0.5 and abs(d_ft) < abs(d_ad):
            rec["verdict"] = "finite_time_explains_half"
        else:
            rec["verdict"] = "adiabatic_still_insufficient"
            rec["next_step"] = "承认绝热/弱场核不够。A/B/C 统计层仍可写；不要把 χ^{(2)} 叫完成。"
    lines += ["## 2. 交换强度 η 的相位旋转", ""]
    for key in ("eta0.5", "eta1.0", "eta1.5"):
        if key in rec["cases"]:
            c = rec["cases"][key]
            lines.append(
                f"- {key}: 有限时间 {c['finite_time_arg_deg']:.2f}°，绝热 {c['adiabatic_arg_deg']:.2f}°。"
            )
    lines += ["", "## 3. 门控抽检", ""]
    for key in ("d0", "s0"):
        if key in rec["cases"]:
            c = rec["cases"][key]
            lines.append(
                f"- {key}: 奇 H2 2D={c['odd_norm2D']:.3e}，偶={c['even_norm2D']:.3e}。"
            )
    lines += [
        "",
        rec.get("next_step", "若有限时间核仍解释不了 79°，保持 A/B/C 统计层，转入材料抽检 B1 或停下扩核。"),
        "",
        f"归档：`{OUT.as_posix()}`。",
        "",
    ]
    one.write_text("\n".join(lines), encoding="utf-8")
    longp.write_text(
        "# 八带 M0：A2 有限时间二阶核（2026-10-29）\n\n"
        + json.dumps(rec, ensure_ascii=False, indent=2, default=_default) + "\n",
        encoding="utf-8",
    )
    rec["reports"] = {"one_page": str(one), "long": str(longp)}


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    geo = build_geometry()
    kcart = np.asarray(geo.k_cart(sh.K_FRAC), float)
    ehat = qmin.a2_polarization(geo)
    env = dyn.Envelope(pf.OMEGA0, ramp_cycles=1.0, flat_cycles=4.0)
    js = sample_js(N_FULL)
    names = ("eta1.0", "eta0.5", "eta1.5", "d0", "s0")
    cases = {}
    for name in names:
        progress(f"start {name}")
        cases[name] = run_one(name, geo, kcart, ehat, env, N_FULL, js, EPS)
        dump(OUT / f"case_{name}.json", cases[name])
    rec = {
        "task_tag": "20261029", "protocol": "Qin_a2_1_4_1",
        "status": "ok", "eps": EPS, "n_steps": N_FULL,
        "k_frac": list(sh.K_FRAC), "g": pf.G_ABS, "omega0": pf.OMEGA0,
        "gamma_adiabatic": GAMMA_ADIA,
        "no_ehrenfest_kick": True, "used_linear_exact": True,
        "not_chi2_certification": True, "not_quantum_output_kernel": True,
        "frozen_arg_B_deg": FROZEN_ARG_B_DEG,
        "cases": cases,
        "wall_s": time.perf_counter() - t0,
        "utc": datetime.now(timezone.utc).isoformat(),
        "next_step": "若解释不足一半，保持 A/B/C 统计层；不要把 χ^{(2)} 叫完成。",
    }
    write_reports(rec)
    dump(OUT / "a2_finite_time.json", rec)
    progress(f"done verdict={rec.get('verdict')} wall={rec['wall_s']:.1f}s")
    print(json.dumps({
        "status": rec["status"], "verdict": rec.get("verdict"),
        "eta1": {k: cases["eta1.0"][k] for k in (
            "finite_time_arg_deg", "adiabatic_arg_deg", "odd_norm2D",
            "richardson_vs_fd_rel") if k in cases["eta1.0"]},
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
