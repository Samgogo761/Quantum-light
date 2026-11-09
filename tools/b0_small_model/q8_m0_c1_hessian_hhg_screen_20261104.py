#!/usr/bin/env python3
"""C1 same-frequency Hessian screen for magnetic H2–H8 (2026-11-04).

Authorized: 1600 nm / 25 fs / 0.1 TW/cm^2 coherent pump + weak same-mode
quadrature perturbations. Deterministic SBE curvature, not BSV Monte Carlo,
not QED, not C2, not CrI3, does not modify frozen c1.json.

Observables follow the external-review corrections:
- save both Néel domains' complex H2–H8, then odd/even
- even q: focus Néel-odd; odd q: focus Néel-even; cross channels diagnostic
- two finite-difference steps; H8 time-step check (nt/cycle 32 vs 64)
- judge (i) structure beyond trivial 2q(q-1) J / X0^2, (ii) ns needed for 1%
"""
from __future__ import annotations

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

from q8_m0_c1_pureBSV_stat_sbe_20261104 import (
    I0_DESIGN_W_CM2,
    FROZEN_FIT_SHA,
    a_vector,
    build_times,
    inversion_closed,
    propagate_a,
    traveling_protocol,
)
from q8_m0_classical import initial_rho, k_grid_xy
from q8_m0_model import R0, build_geometry, sha256_file
from qinput_spectrum_reanalysis_20261028 import clean, packed, spectrum

ROOT = HERE.parents[1]
C1_JSON = ROOT / "results/q_line/q1600_c1_pureBSV_stat_sbe_20261104/c1.json"
OUT = ROOT / "results/q_line/q1600_c1_hessian_hhg_screen_20261104"
DOC = ROOT / "docs/04_BSV量子光"
QS = list(range(2, 9))
REL_STEPS = (0.005, 0.01)
PRIMARY_REL = 0.005
GATE_H2_REL = 1.0e-8
HESSIAN_STEP_TOL = 0.5
BEYOND_TRIVIAL = 0.35
NS_WEAK_FRAC = 0.3
TARGET_REL = 0.01


def dump(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(clean(obj), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def unpack(z) -> complex:
    if isinstance(z, dict):
        return complex(z["re"], z["im"])
    return complex(z)


def cvec2(row) -> np.ndarray:
    return np.array([unpack(row[0]), unpack(row[1])], complex)


def project_qs(times, jxy, omega, qs=QS) -> np.ndarray:
    lo, hi = float(times[0]), float(times[-1])
    return spectrum(times, np.asarray(jxy, complex), lo, hi, qs, method="linear_exact", omega=omega)


def pack_qs(arr: np.ndarray) -> dict:
    return {str(q): packed(arr[i]) for i, q in enumerate(QS)}


def dominant_kind(q: int) -> str:
    return "odd" if q % 2 == 0 else "even"


def split_pm(plus: np.ndarray, minus: np.ndarray) -> dict:
    even = 0.5 * (plus + minus)
    odd = 0.5 * (plus - minus)
    return {"plus": plus, "minus": minus, "even": even, "odd": odd}


def load_c1_coherent_h2() -> dict:
    rec = json.loads(C1_JSON.read_text(encoding="utf-8"))
    mc = rec["coherent"]["analysis"]["mean_current"]
    return {
        "status": rec["status"],
        "plus": cvec2(mc["plus"]),
        "minus": cvec2(mc["minus"]),
        "odd_norm2D": float(mc["odd_norm2D"]),
        "nbar": float(rec["protocol"]["nbar"]),
        "X": float(rec["protocol"]["coherent_mean_X"]),
    }


def run_xp(geo, p, kc, w, meta, proto, X, P, rhos) -> dict:
    a_t, _fld = a_vector(meta, proto, geo, float(X), float(P))
    out = {}
    herms = []
    t0 = time.perf_counter()
    for neel in (+1, -1):
        r = propagate_a(geo, p, kc, w, neel, a_t, meta["times"], rhos[neel])
        herms.append(r["herm"])
        out[neel] = project_qs(meta["times"], r["J"], meta["omega"])
    wall = time.perf_counter() - t0
    return {
        "X": float(X),
        "P": float(P),
        "nt_per_cycle": int(meta["nt_per_cycle"]),
        "plus": out[+1],
        "minus": out[-1],
        "wall_s": wall,
        "max_herm": float(np.max(herms)),
    }


def fd2(jp, j0, jm, h: float) -> np.ndarray:
    return (jp - 2.0 * j0 + jm) / (h * h)


def ns_for_target(dK: np.ndarray, J: np.ndarray, target: float = TARGET_REL) -> float:
    mag_dk = float(np.linalg.norm(dK))
    mag_j = float(np.linalg.norm(J))
    if mag_dk < 1.0e-30 or mag_j < 1.0e-30:
        return float("inf")
    need_sinh = target * mag_j / mag_dk
    s = (0.5 * need_sinh) ** 2
    return float(0.5 * (-1.0 + np.sqrt(1.0 + 4.0 * s)))


def channel_metrics(q: int, J: np.ndarray, dK: np.ndarray, X0: float) -> dict:
    factor = 2.0 * q * (q - 1) / (X0 * X0)
    trivial = factor * J
    mag_j = float(np.linalg.norm(J))
    mag_dk = float(np.linalg.norm(dK))
    mag_tr = float(np.linalg.norm(trivial))
    excess = float(np.linalg.norm(dK - trivial) / mag_dk) if mag_dk > 0 else None
    xi_2d = float(mag_dk / mag_tr) if mag_tr > 0 else None
    align = None
    if mag_dk > 0 and mag_tr > 0:
        align = float(np.real(np.vdot(trivial, dK)) / (mag_tr * mag_dk))
    xi_comp = []
    for a, b in zip(dK, trivial):
        if abs(b) > 0:
            xi_comp.append(complex(a / b))
        else:
            xi_comp.append(None)
    return {
        "q": int(q),
        "kind": dominant_kind(q),
        "J": packed(J),
        "dK": packed(dK),
        "trivial": packed(trivial),
        "abs_J": mag_j,
        "abs_dK": mag_dk,
        "abs_trivial": mag_tr,
        "xi_2d": xi_2d,
        "xi_minus_1": None if xi_2d is None else float(abs(xi_2d - 1.0)),
        "alignment": align,
        "excess": excess,
        "xi_x": None if xi_comp[0] is None else packed(xi_comp[0]),
        "xi_y": None if xi_comp[1] is None else packed(xi_comp[1]),
        "ns_for_1pct": ns_for_target(dK, J),
        "trivial_factor_2q_qm1_over_X0sq": factor,
    }


def unpack_row(z) -> np.ndarray:
    return np.array([unpack(z[0]), unpack(z[1])], complex)


def hessian_from_fields(center, x_plus, x_minus, p_plus, p_minus, h: float, X0: float) -> dict:
    out = {"h": float(h), "h_over_X0": float(h / X0), "channels": {}}
    for kind in ("plus", "minus", "even", "odd"):
        c = split_pm(center["plus"], center["minus"])[kind]
        xp = split_pm(x_plus["plus"], x_plus["minus"])[kind]
        xm = split_pm(x_minus["plus"], x_minus["minus"])[kind]
        pp = split_pm(p_plus["plus"], p_plus["minus"])[kind]
        pm = split_pm(p_minus["plus"], p_minus["minus"])[kind]
        kxx = fd2(xp, c, xm, h)
        kpp = fd2(pp, c, pm, h)
        dk = kxx - kpp
        rows = []
        for i, q in enumerate(QS):
            rows.append(
                {
                    "q": int(q),
                    "J": packed(c[i]),
                    "Kxx": packed(kxx[i]),
                    "Kpp": packed(kpp[i]),
                    "dK": packed(dk[i]),
                    "abs_J": float(np.linalg.norm(c[i])),
                    "abs_dK": float(np.linalg.norm(dk[i])),
                }
            )
        out["channels"][kind] = rows
    dom = []
    cross = []
    for i, q in enumerate(QS):
        kind = dominant_kind(q)
        other = "even" if kind == "odd" else "odd"
        J = split_pm(center["plus"], center["minus"])[kind][i]
        dK = unpack_row(out["channels"][kind][i]["dK"])
        met = channel_metrics(q, J, dK, X0)
        met["Kxx"] = out["channels"][kind][i]["Kxx"]
        met["Kpp"] = out["channels"][kind][i]["Kpp"]
        dom.append(met)
        Jx = split_pm(center["plus"], center["minus"])[other][i]
        dKx = unpack_row(out["channels"][other][i]["dK"])
        cross.append(
            {
                "q": int(q),
                "kind": other,
                "abs_J": float(np.linalg.norm(Jx)),
                "abs_dK": float(np.linalg.norm(dKx)),
                "J": packed(Jx),
                "dK": packed(dKx),
                "note": "cross channel; 25 fs Gaussian is not half-cycle periodic",
            }
        )
    out["dominant"] = dom
    out["cross"] = cross
    return out


def step_consistency(a, b, floor: float) -> dict:
    rels = []
    for ra, rb in zip(a["dominant"], b["dominant"]):
        da = unpack_row(ra["dK"])
        db = unpack_row(rb["dK"])
        denom = max(float(np.linalg.norm(da)), float(np.linalg.norm(db)), floor)
        rel = float(np.linalg.norm(da - db) / denom)
        rels.append({"q": ra["q"], "rel": rel, "abs_dK_primary": ra["abs_dK"], "abs_dK_check": rb["abs_dK"]})
    rel_max = max(x["rel"] for x in rels) if rels else None
    return {"per_q": rels, "rel_max": rel_max, "pass": bool(rel_max is not None and rel_max <= HESSIAN_STEP_TOL)}


def judge(primary, check, n_pump: float, j_floor: float) -> dict:
    dks = [m["abs_dK"] for m in primary["dominant"]]
    cons = step_consistency(primary, check, floor=max(1.0e-30, 1.0e-4 * max(dks) if dks else 0.0))
    valid = [m for m in primary["dominant"] if m["abs_J"] >= j_floor]
    xi_m1 = [m["xi_minus_1"] for m in valid if m["xi_minus_1"] is not None]
    excess = [m["excess"] for m in valid if m["excess"] is not None]
    med_xi = float(np.median(xi_m1)) if xi_m1 else None
    med_ex = float(np.median(excess)) if excess else None
    beyond = bool(
        valid
        and med_xi is not None
        and med_ex is not None
        and med_xi > BEYOND_TRIVIAL
        and med_ex > BEYOND_TRIVIAL
    )
    ns_list = [m["ns_for_1pct"] for m in valid if np.isfinite(m["ns_for_1pct"])]
    ns_min = float(min(ns_list)) if ns_list else float("inf")
    frac = ns_min / n_pump if np.isfinite(ns_min) else float("inf")
    weak_ok = bool(np.isfinite(frac) and frac < NS_WEAK_FRAC)
    if not cons["pass"]:
        verdict = "stop_unreliable_hessian"
        reason = (
            f"两步长主导通道 dK 最大相对差 {cons['rel_max']:.3f} > {HESSIAN_STEP_TOL}，"
            "曲率可能被高阶非线性或数值消去污染。"
        )
    elif not valid:
        verdict = "stop_trivial_or_absent_structure"
        reason = "主导通道泵浦谐波低于数值地板，无法判断阶次结构。"
    elif not beyond:
        verdict = "stop_trivial_or_absent_structure"
        reason = (
            f"主导通道相对平凡因子 2q(q-1)J/X0^2 的偏离不足："
            f"median |ξ|-1 = {med_xi}，median excess = {med_ex}（门槛 {BEYOND_TRIVIAL}）。"
        )
    elif not weak_ok:
        verdict = "stop_aux_field_not_weak"
        reason = (
            f"即使最敏感主导通道，1% 相对变化也需要 n_s/n_pump = {frac:.3e} "
            f"（≥ {NS_WEAK_FRAC}），超出同频弱扰动适用范围。"
        )
    else:
        verdict = "candidate_keep_narrow"
        reason = (
            "存在超出平凡阶次因子的曲率结构，且按 C1 归一化后，"
            f"最敏感通道 1% 变化约需 n_s/n_pump = {frac:.3e}。仍不是量子优势。"
        )
    return {
        "verdict": verdict,
        "reason": reason,
        "beyond_trivial": beyond,
        "median_abs_xi_minus_1": med_xi,
        "median_excess": med_ex,
        "n_valid_dominant": len(valid),
        "j_floor": j_floor,
        "ns_1pct_min": ns_min,
        "ns_1pct_min_over_npump": frac,
        "step_consistency": cons,
        "not_quantum_advantage": True,
    }


def write_report(rec: dict) -> Path:
    lines = []
    lines.append("# C1 同频 Hessian 多谐波筛查（2026-11-04）\n\n")
    lines.append("确定性 SBE 曲率筛查，不是压缩态传播，不是 QED，不是 C2，不是 CrI₃。")
    lines.append("未修改冻结 `c1.json`。\n\n")
    lines.append("## 协议\n\n")
    lines.append("- 泵浦：C1 的 1600 nm、25 fs、0.1 TW/cm² 相干 CEP=0（X=X0，P=0）\n")
    lines.append("- 扰动：同频同偏振 δX、δP，相对步长 0.005 与 0.01\n")
    lines.append("- 网格：M0，nk=2 反演封闭四点；nt/cycle=32，H8 另做 64 对照\n")
    lines.append("- 偶数阶看 Néel-odd，奇数阶看 Néel-even；交叉通道只作诊断\n\n")
    g = rec["gates"]
    lines.append("## 闸门\n\n")
    lines.append(f"- 泵浦 H2 相对 c1.json：max_rel = {g['h2_max_rel']:.3e}，")
    lines.append(f"{'通过' if g['h2_reproduced'] else '失败'}（门槛 {GATE_H2_REL:.0e}）\n")
    lines.append(f"- 反演封闭：{rec['grid']['inversion_closed']}\n")
    lines.append(f"- 最大非厄米：{rec['max_herm']:.3e}\n\n")
    if rec.get("status") == "gate_failed":
        lines.append("**泵浦复现失败，未做 Hessian。**\n")
        path = OUT / "hessian_report.md"
        path.write_text("".join(lines), encoding="utf-8")
        return path
    dt = rec["time_check"]
    lines.append("## 时间离散（泵浦-only，32 vs 64）\n\n")
    lines.append("| q | 主导通道 | |J|_32 | |J|_64 | 相对差 |\n|---|---|---|---|---|\n")
    for row in dt["per_q"]:
        lines.append(
            f"| {row['q']} | {row['kind']} | {row['abs_32']:.4e} | {row['abs_64']:.4e} | {row['rel']:.3e} |\n"
        )
    lines.append(f"\nH8 相对差 {dt['h8_rel']:.3e}。")
    lines.append(f"{'可作筛查' if dt['h8_ok'] else 'H8 时间离散不可靠，仅作诊断'}。\n\n")
    prim = rec["hessian"][str(PRIMARY_REL)]
    lines.append("## 主导通道曲率（相对步长 0.005）\n\n")
    lines.append("| q | 通道 | \\|J\\| | \\|dK\\| | \\|ξ\\| | \\|ξ\\|-1 | excess | n_s(1%)/n_pump |\n")
    lines.append("|---|---|---|---|---|---|---|---|\n")
    n_pump = rec["protocol"]["nbar"]
    for m in prim["dominant"]:
        xi = m["xi_2d"]
        xi_s = "—" if xi is None else f"{xi:.3f}"
        xm = m["xi_minus_1"]
        xm_s = "—" if xm is None else f"{xm:.3f}"
        ex = m["excess"]
        ex_s = "—" if ex is None else f"{ex:.3f}"
        frac = m["ns_for_1pct"] / n_pump if np.isfinite(m["ns_for_1pct"]) else float("inf")
        frac_s = "inf" if not np.isfinite(frac) else f"{frac:.3e}"
        lines.append(
            f"| {m['q']} | {m['kind']} | {m['abs_J']:.4e} | {m['abs_dK']:.4e} | "
            f"{xi_s} | {xm_s} | {ex_s} | {frac_s} |\n"
        )
    lines.append("\n交叉通道（诊断，不参与正结果）：\n\n")
    lines.append("| q | 通道 | \\|J\\| | \\|dK\\| |\n|---|---|---|---|\n")
    for m in prim["cross"]:
        lines.append(f"| {m['q']} | {m['kind']} | {m['abs_J']:.4e} | {m['abs_dK']:.4e} |\n")
    jdg = rec["judgment"]
    lines.append("\n## 判断\n\n")
    lines.append(f"- 两步长一致性：rel_max = {jdg['step_consistency']['rel_max']}\n")
    lines.append(f"- median |ξ|-1 = {jdg['median_abs_xi_minus_1']}\n")
    lines.append(f"- median excess = {jdg['median_excess']}\n")
    lines.append(f"- 最敏感 1% 所需 n_s/n_pump = {jdg['ns_1pct_min_over_npump']}\n")
    lines.append(f"- **裁决：`{jdg['verdict']}`**\n\n")
    lines.append(jdg["reason"] + "\n\n")
    lines.append("这不是量子优势，也不是实验协议。同频同模合成态在物理上接近强位移压缩（C2），")
    lines.append("Hessian 可算不等于压缩光实验可做。\n\n")
    lines.append(f"JSON：`{OUT / 'hessian.json'}`\n")
    text = "".join(lines)
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "hessian_report.md"
    path.write_text(text, encoding="utf-8")
    doc = DOC / "线Q_C1同频Hessian筛查_20261104.md"
    doc.write_text(text, encoding="utf-8")
    return path


def main() -> int:
    t_all = time.perf_counter()
    OUT.mkdir(parents=True, exist_ok=True)
    geo = build_geometry()
    p = R0
    proto = traveling_protocol(I0_DESIGN_W_CM2)
    meta32 = build_times(32)
    meta64 = build_times(64)
    kf, kc, w = k_grid_xy(geo, 2, 0.5, 0.5)
    closed = inversion_closed(kf)
    rhos32 = {s: initial_rho(geo, p, kc, s, mu=0.0)[0] for s in (+1, -1)}
    rhos64 = {s: initial_rho(geo, p, kc, s, mu=0.0)[0] for s in (+1, -1)}
    X0 = float(proto["coherent_mean_X"])
    P0 = float(proto["coherent_mean_P"])
    ref = load_c1_coherent_h2()

    print(f"pump-only nt=32 X0={X0:.6f} ...", flush=True)
    center = run_xp(geo, p, kc, w, meta32, proto, X0, P0, rhos32)
    h2p = center["plus"][0]
    h2m = center["minus"][0]
    rel_p = float(np.linalg.norm(h2p - ref["plus"]) / max(np.linalg.norm(ref["plus"]), 1e-30))
    rel_m = float(np.linalg.norm(h2m - ref["minus"]) / max(np.linalg.norm(ref["minus"]), 1e-30))
    h2_max_rel = max(rel_p, rel_m)
    h2_ok = bool(h2_max_rel < GATE_H2_REL)
    print(f"  H2 gate max_rel={h2_max_rel:.3e} odd={0.5 * np.linalg.norm(h2p - h2m):.6e}", flush=True)

    rec = {
        "date": "2026-11-04",
        "not_CrI3": True,
        "not_QED": True,
        "not_C2": True,
        "not_BSV_MonteCarlo": True,
        "did_not_modify_c1_json": True,
        "frozen_fit_sha256": FROZEN_FIT_SHA,
        "protocol": {
            "lambda_nm": proto["lambda_nm"],
            "fwhm_I_fs": proto["fwhm_I_fs"],
            "I0_W_cm2": proto["I0_W_cm2"],
            "nbar": proto["nbar"],
            "alpha_coh": proto["alpha_coh"],
            "coherent_mean_X": X0,
            "coherent_mean_P": P0,
            "cep": 0.0,
        },
        "grid": {
            "nk": 2,
            "n_kpts": int(kc.shape[0]),
            "inversion_closed": closed,
            "kfrac": kf.tolist(),
        },
        "time": {k: v for k, v in meta32.items() if k not in ("times", "times_fs", "env")},
        "gates": {
            "h2_plus_rel": rel_p,
            "h2_minus_rel": rel_m,
            "h2_max_rel": h2_max_rel,
            "h2_reproduced": h2_ok,
            "inversion_closed": closed,
        },
        "source_sha256": {
            "this": sha256_file(Path(__file__)),
            "c1_script_on_disk": sha256_file(HERE / "q8_m0_c1_pureBSV_stat_sbe_20261104.py"),
            "q8_m0_model": sha256_file(HERE / "q8_m0_model.py"),
            "q8_m0_classical": sha256_file(HERE / "q8_m0_classical.py"),
        },
        "max_herm": center["max_herm"],
        "pump": {
            "plus": pack_qs(center["plus"]),
            "minus": pack_qs(center["minus"]),
            "odd": pack_qs(0.5 * (center["plus"] - center["minus"])),
            "even": pack_qs(0.5 * (center["plus"] + center["minus"])),
            "wall_s": center["wall_s"],
        },
    }
    if not (h2_ok and closed):
        rec["status"] = "gate_failed"
        rec["wall_s_total"] = time.perf_counter() - t_all
        dump(OUT / "hessian.json", rec)
        write_report(rec)
        print("GATE FAILED", OUT / "hessian.json")
        return 2

    print("pump-only nt=64 ...", flush=True)
    center64 = run_xp(geo, p, kc, w, meta64, proto, X0, P0, rhos64)
    rec["max_herm"] = max(rec["max_herm"], center64["max_herm"])
    time_rows = []
    for i, q in enumerate(QS):
        kind = dominant_kind(q)
        j32 = split_pm(center["plus"], center["minus"])[kind][i]
        j64 = split_pm(center64["plus"], center64["minus"])[kind][i]
        rel = float(np.linalg.norm(j32 - j64) / max(np.linalg.norm(j64), 1e-30))
        time_rows.append(
            {
                "q": int(q),
                "kind": kind,
                "abs_32": float(np.linalg.norm(j32)),
                "abs_64": float(np.linalg.norm(j64)),
                "rel": rel,
            }
        )
    h8_rel = time_rows[-1]["rel"]
    rec["time_check"] = {
        "nt_per_cycle": [32, 64],
        "per_q": time_rows,
        "h8_rel": h8_rel,
        "h8_ok": bool(h8_rel < 0.35),
        "pump64_wall_s": center64["wall_s"],
    }
    print(f"  H8 32vs64 rel={h8_rel:.3e}", flush=True)

    fields = {}
    hessian = {}
    for rel in REL_STEPS:
        h = rel * X0
        print(f"rel={rel} h={h:.6e} ...", flush=True)
        xp = run_xp(geo, p, kc, w, meta32, proto, X0 + h, P0, rhos32)
        xm = run_xp(geo, p, kc, w, meta32, proto, X0 - h, P0, rhos32)
        pp = run_xp(geo, p, kc, w, meta32, proto, X0, P0 + h, rhos32)
        pm = run_xp(geo, p, kc, w, meta32, proto, X0, P0 - h, rhos32)
        rec["max_herm"] = max(rec["max_herm"], xp["max_herm"], xm["max_herm"], pp["max_herm"], pm["max_herm"])
        fields[str(rel)] = {
            "h": h,
            "+X": {"plus": pack_qs(xp["plus"]), "minus": pack_qs(xp["minus"]), "wall_s": xp["wall_s"]},
            "-X": {"plus": pack_qs(xm["plus"]), "minus": pack_qs(xm["minus"]), "wall_s": xm["wall_s"]},
            "+P": {"plus": pack_qs(pp["plus"]), "minus": pack_qs(pp["minus"]), "wall_s": pp["wall_s"]},
            "-P": {"plus": pack_qs(pm["plus"]), "minus": pack_qs(pm["minus"]), "wall_s": pm["wall_s"]},
        }
        hessian[str(rel)] = hessian_from_fields(center, xp, xm, pp, pm, h, X0)
        print(f"  wall={xp['wall_s']+xm['wall_s']+pp['wall_s']+pm['wall_s']:.2f}s", flush=True)

    rec["fields"] = fields
    rec["hessian"] = hessian
    j_abs = [m["abs_J"] for m in hessian[str(PRIMARY_REL)]["dominant"]]
    j_floor = 1.0e-4 * max(j_abs) if j_abs else 0.0
    rec["judgment"] = judge(hessian[str(PRIMARY_REL)], hessian["0.01"], proto["nbar"], j_floor)
    rec["status"] = rec["judgment"]["verdict"]
    rec["wall_s_total"] = time.perf_counter() - t_all
    dump(OUT / "hessian.json", rec)
    report = write_report(rec)
    print("STATUS", rec["status"])
    print("JSON", OUT / "hessian.json")
    print("REPORT", report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
