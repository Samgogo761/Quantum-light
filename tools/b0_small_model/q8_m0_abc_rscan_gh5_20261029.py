"""R-scan: product GH5, Q6 protocol, eta=1, kick=True, theta=0/pi.

Vary (alpha, r) at fixed Hamiltonian. Do not edit A1 hardcoded ALPHA/R_SQ.
Local analytic_xp_theta / gh5_nodes only. Reuse A2b one_traj with kick=True.
Two-angle proxy cannot split A vs C.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
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
from numpy.polynomial.hermite import hermgauss

import q8_m0_abc_a1_gate_gh5_20261029 as a1
import q8_m0_abc_a2b_ehrenfest_kick_20261029 as a2b
import q8_m0_qinput_dynamics_20261026 as dyn
import q8_m0_qinput_shared_kk_dyn_preflight_20261030 as pf
from q8_m0_model import sha256_file

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results/q_line/abc_rscan_gh5_20261029"
FIG = OUT / "figures"
DOCS = ROOT / "docs/04_BSV量子光"
TASK_TAG = "20261029"
RECORD_DATE = "2026-10-29"
PROTOCOL = "Qin_a2_1_4_1"
DRIVER_WALL_S = 3600.0
JOB_WALL_S = 1800.0
N_FULL = a1.N_FULL
ORDER = a1.ORDER
THETA_0 = a1.THETA_0
THETA_PI = a1.THETA_PI
QED_RHO = 1.456
QED_R = 1.2733
QED_ARG_AB = 177.65

CASES = (
    ("a05r01", 0.5, 0.1),
    ("a05r02", 0.5, 0.2),
    ("a05r03", 0.5, 0.3),
    ("a05r05", 0.5, 0.5),
    ("a08r03", 0.8, 0.3),
    ("a12r03", 1.2, 0.3),
)
JOBS = []
for prefix, alpha, rsq in CASES:
    JOBS.extend((
        (f"{prefix}_p_0", prefix, alpha, rsq, 1, THETA_0),
        (f"{prefix}_m_0", prefix, alpha, rsq, -1, THETA_0),
        (f"{prefix}_p_pi", prefix, alpha, rsq, 1, THETA_PI),
        (f"{prefix}_m_pi", prefix, alpha, rsq, -1, THETA_PI),
    ))
JOBS = tuple(JOBS)
JOB_MAP = {tag: (prefix, alpha, rsq, neel, theta)
           for tag, prefix, alpha, rsq, neel, theta in JOBS}
BATCHES = tuple(
    (prefix, (f"{prefix}_p_0", f"{prefix}_m_0", f"{prefix}_p_pi", f"{prefix}_m_pi"))
    for prefix, _, _ in CASES
)

jsonable = a1.jsonable
cpack = a1.cpack
cz = a1.cz
wrap = a1.wrap
angdiff_deg = a1.angdiff_deg
sample_js_for = a1.sample_js_for
h2_from_series = a1.h2_from_series
two_angle_proxy = a1.two_angle_proxy
one_traj = a2b.one_traj


def dump(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(jsonable(obj), ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                   encoding="utf-8")
    os.replace(tmp, path)


def progress(msg: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    line = f"{time.strftime('%H:%M:%S')} {msg}"
    with (OUT / "progress.txt").open("a", encoding="utf-8") as fh:
        fh.write(line + "\n")
    print(line, flush=True)


def field_R(alpha: float, r: float) -> float:
    return float(np.sinh(r) * np.cosh(r) / (alpha * alpha))


def field_nbar(alpha: float, r: float) -> float:
    return float(alpha * alpha + np.sinh(r) ** 2)


def analytic_xp_theta(theta: float, alpha: float, r: float) -> dict:
    a = complex(float(alpha))
    mx, mp = 2.0 * a.real, 2.0 * a.imag
    rr = float(r)
    th = float(theta)
    if abs(th) < 1e-15:
        vx, vp = float(np.exp(-2.0 * rr)), float(np.exp(2.0 * rr))
        axis = "X_squeezed"
    elif abs(th - np.pi) < 1e-12:
        vx, vp = float(np.exp(2.0 * rr)), float(np.exp(-2.0 * rr))
        axis = "P_squeezed"
    else:
        raise ValueError("R-scan ships theta=0 or pi only; product GH requires Cov=0")
    nbar = field_nbar(float(alpha), rr)
    return {
        "kind": "ds", "alpha": pf.cpack(a), "r": rr, "theta": th,
        "mean_X": mx, "mean_P": mp, "var_X": vx, "var_P": vp,
        "cov_XP": 0.0, "mean_n_analytic": float(nbar),
        "mean_n_Weyl": 0.25 * (mx * mx + mp * mp + vx + vp) - 0.5,
        "product_GH_valid": True, "squeeze_axis": axis,
        "R_M_over_alpha2": field_R(float(alpha), rr),
        "did_not_call_a1_analytic_xp_theta": True,
    }


def gh5_nodes(theta: float, alpha: float, r: float) -> dict:
    xi, w1 = hermgauss(ORDER)
    mom = analytic_xp_theta(theta, alpha, r)
    sx, sp = np.sqrt(mom["var_X"]), np.sqrt(mom["var_P"])
    nodes = []
    for i, x in enumerate(xi):
        for j, y in enumerate(xi):
            wt = float(w1[i] * w1[j] / np.pi)
            nodes.append({
                "xi": float(x), "eta": float(y), "weight": wt,
                "X": mom["mean_X"] + sx * np.sqrt(2.0) * float(x),
                "P": mom["mean_P"] + sp * np.sqrt(2.0) * float(y),
            })
    wsum = float(sum(n["weight"] for n in nodes))
    mx = sum(n["weight"] * n["X"] for n in nodes)
    mpv = sum(n["weight"] * n["P"] for n in nodes)
    vx = sum(n["weight"] * n["X"] ** 2 for n in nodes) - mx * mx
    vp = sum(n["weight"] * n["P"] ** 2 for n in nodes) - mpv * mpv
    cxp = sum(n["weight"] * n["X"] * n["P"] for n in nodes) - mx * mpv
    return {
        "kind": "ds", "theta": float(theta), "order": ORDER, "n_nodes": len(nodes),
        "weight_sum": wsum, "quad_mean_X": float(mx), "quad_mean_P": float(mpv),
        "quad_var_X": float(vx), "quad_var_P": float(vp), "quad_cov_XP": float(cxp),
        "target_mean_X": mom["mean_X"], "target_mean_P": mom["mean_P"],
        "target_var_X": mom["var_X"], "target_var_P": mom["var_P"],
        "nodes": nodes, "analytic": mom,
        "product_GH": True, "did_not_call_a1_gh5_nodes": True,
        "alpha": float(alpha), "r": float(r),
    }


def ckpt_path(tag: str) -> Path:
    return OUT / f"ckpt_{tag}.npz"


def job_path(tag: str) -> Path:
    return OUT / f"job_{tag}.json"


def series_path(tag: str) -> Path:
    return OUT / f"series_{tag}.npz"


def save_ckpt(tag, prefix, alpha, rsq, neel, theta, n_done, n_nodes, acc, wsum, max_n, t_arr) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    acc = np.asarray(acc, complex)
    np.savez_compressed(
        ckpt_path(tag),
        acc_re=acc.real, acc_im=acc.imag, wsum=float(wsum),
        n_done=int(n_done), n_nodes=int(n_nodes), max_n=float(max_n),
        t=np.asarray(t_arr, float), neel=int(neel), theta=float(theta),
        alpha=float(alpha), r=float(rsq), prefix=np.asarray(prefix),
    )


def load_ckpt(tag: str):
    p = ckpt_path(tag)
    if not p.is_file():
        return None
    z = np.load(p, allow_pickle=False)
    acc = np.asarray(z["acc_re"], float) + 1j * np.asarray(z["acc_im"], float)
    return {
        "n_done": int(z["n_done"]), "n_nodes": int(z["n_nodes"]),
        "acc": acc, "wsum": float(z["wsum"]), "max_n": float(z["max_n"]),
        "t": np.asarray(z["t"], float),
        "theta": float(z["theta"]) if "theta" in z.files else None,
        "neel": int(z["neel"]) if "neel" in z.files else None,
        "alpha": float(z["alpha"]) if "alpha" in z.files else None,
        "r": float(z["r"]) if "r" in z.files else None,
    }


def load_job(tag: str):
    p = job_path(tag)
    if not p.is_file():
        return None
    return json.loads(p.read_text(encoding="utf-8"))


def job_finished_ok(tag: str) -> bool:
    rec = load_job(tag)
    return bool(rec and rec.get("status") == "ok" and series_path(tag).is_file())


def save_series(tag: str, t, j) -> None:
    j = np.asarray(j, complex)
    np.savez_compressed(series_path(tag), t=np.asarray(t, float), Jx=j[:, 0], Jy=j[:, 1])


def load_series(tag: str):
    z = np.load(series_path(tag))
    t = np.asarray(z["t"], float)
    j = np.stack([z["Jx"], z["Jy"]], axis=1)
    return t, j


def run_one_job(tag: str) -> dict:
    if tag not in JOB_MAP:
        raise ValueError(tag)
    prefix, alpha, rsq, neel, theta = JOB_MAP[tag]
    if job_finished_ok(tag):
        rec = load_job(tag)
        progress(f"{tag} already complete n_run={rec.get('n_run')}")
        return rec
    OUT.mkdir(parents=True, exist_ok=True)
    _, _, p, models, grounds, env = a2b.make_bundle(1.0)
    gh = gh5_nodes(theta, alpha, rsq)
    if abs(gh["quad_cov_XP"]) > 1e-10:
        rec = {"tag": tag, "status": "cov_not_zero_abort",
               "quad_cov_XP": gh["quad_cov_XP"], "stopped": True}
        dump(job_path(tag), rec)
        return rec
    nodes = gh["nodes"]
    ek0, emk0, mass = grounds[neel]
    model = models[neel]
    js = sample_js_for(N_FULL)
    ck = load_ckpt(tag)
    i0, acc, wsum, max_n, t_arr = 0, None, 0.0, 0.0, None
    if ck is not None:
        if ck.get("alpha") is not None and abs(ck["alpha"] - alpha) > 1e-12:
            rec = {"tag": tag, "status": "ckpt_alpha_mismatch", "stopped": True}
            dump(job_path(tag), rec)
            return rec
        if ck.get("r") is not None and abs(ck["r"] - rsq) > 1e-12:
            rec = {"tag": tag, "status": "ckpt_r_mismatch", "stopped": True}
            dump(job_path(tag), rec)
            return rec
        i0, acc, wsum, max_n, t_arr = ck["n_done"], ck["acc"], ck["wsum"], ck["max_n"], ck["t"]
        progress(f"{tag} resume n_done={i0}/{len(nodes)}")
    clock = dyn.WallClock(JOB_WALL_S)
    progress(
        f"{tag} start alpha={alpha} r={rsq} R={field_R(alpha, rsq):.4f} "
        f"neel={neel} theta={theta:.6f} nodes={len(nodes)}"
    )
    timed_out = False
    timeout_msg = None
    try:
        for i, node in enumerate(nodes):
            if i < i0:
                continue
            j_rows, t_rows, nrm, _, _ = one_traj(
                model, ek0, emk0, node["X"], node["P"], env, N_FULL, js, clock, True)
            w = float(node["weight"])
            acc = w * j_rows if acc is None else acc + w * j_rows
            wsum += w
            max_n = max(max_n, nrm)
            t_arr = t_rows
            save_ckpt(tag, prefix, alpha, rsq, neel, theta, i + 1, len(nodes), acc, wsum, max_n, t_arr)
            if (i + 1) == len(nodes) or (i + 1) % 5 == 0:
                progress(f"{tag} node {i + 1}/{len(nodes)} wall={clock.elapsed():.1f}s")
    except TimeoutError as exc:
        timed_out = True
        timeout_msg = str(exc)
        progress(f"{tag} timeout {timeout_msg}")
    n_done = 0 if acc is None else int(load_ckpt(tag)["n_done"] if ckpt_path(tag).is_file() else 0)
    if timed_out or n_done < len(nodes):
        rec = {
            "tag": tag, "prefix": prefix, "alpha": float(alpha), "r": float(rsq),
            "R": field_R(alpha, rsq), "nbar": field_nbar(alpha, rsq),
            "neel": int(neel), "theta": float(theta), "status": "timeout_partial",
            "n_nodes": len(nodes), "n_done": n_done, "weight_sum": wsum,
            "max_abs_norm_minus_1": float(max_n), "wall_s": clock.elapsed(),
            "timeout_msg": timeout_msg, "n_max": 0, "protocol": PROTOCOL,
            "n_steps": N_FULL, "stopped": True, "electron": mass,
            "quad_cov_XP": gh["quad_cov_XP"], "kick": True, "eta": 1.0,
        }
        dump(job_path(tag), rec)
        return rec
    avg = acc / (wsum + 1e-30)
    save_series(tag, t_arr, avg)
    rec = {
        "tag": tag, "prefix": prefix, "alpha": float(alpha), "r": float(rsq),
        "R": field_R(alpha, rsq), "nbar": field_nbar(alpha, rsq),
        "neel": int(neel), "order": ORDER, "theta": float(theta), "status": "ok",
        "n_nodes": len(nodes), "n_run": len(nodes), "n_done": len(nodes),
        "weight_sum": wsum, "max_abs_norm_minus_1": float(max_n),
        "wall_s": clock.elapsed(), "n_max": 0, "protocol": PROTOCOL,
        "n_steps": N_FULL, "dt": env.t_total / N_FULL,
        "n_samples": int(len(t_arr)), "electron": mass, "stopped": True,
        "kick": True, "eta": 1.0, "wigner": analytic_xp_theta(theta, alpha, rsq),
        "quad_cov_XP": gh["quad_cov_XP"], "used_product_GH": True,
        "did_not_edit_a1_alpha_r": True, "did_not_mix_1600nm_8020": True,
        "operator_order": "D(alpha)S(xi)|0>, xi=r exp(i theta)",
    }
    dump(job_path(tag), rec)
    progress(f"{tag} done wall={clock.elapsed():.1f}s")
    return rec


def load_pm_h2(prefix: str, theta: float):
    th = "0" if abs(theta) < 1e-15 else "pi"
    tp, jp = load_series(f"{prefix}_p_{th}")
    tm, jm = load_series(f"{prefix}_m_{th}")
    if np.max(np.abs(tp - tm)) > 1e-12:
        raise RuntimeError(f"time grid mismatch {prefix} theta={th}")
    qp, pack_p = h2_from_series(tp, jp)
    qm, pack_m = h2_from_series(tm, jm)
    odd_j = 0.5 * (qp - qm)
    even_j = 0.5 * (qp + qm)
    return {
        "theta": float(theta),
        "odd_H2": [cpack(odd_j[1, 0]), cpack(odd_j[1, 1])],
        "even_H2": [cpack(even_j[1, 0]), cpack(even_j[1, 1])],
        "odd_norm2D": float(np.linalg.norm(odd_j[1])),
        "even_norm2D": float(np.linalg.norm(even_j[1])),
        "windows": {"plus": pack_p, "minus": pack_m},
    }


def summarize_prefix(prefix: str, alpha: float, rsq: float) -> dict:
    a0 = load_pm_h2(prefix, THETA_0)
    api = load_pm_h2(prefix, THETA_PI)
    z0 = cz(a0["odd_H2"][0])
    zpi = cz(api["odd_H2"][0])
    odd_x = two_angle_proxy(z0, zpi)
    R = field_R(alpha, rsq)
    rho = float(odd_x["abs_AC_over_B"])
    pred_amp = abs(1.0 + R) / max(abs(1.0 - R), 1e-30)
    pred_amp_rho = abs(1.0 + rho) / max(abs(1.0 - rho), 1e-30)
    B = cz(odd_x["B_proxy"])
    AC = cz(odd_x["A_plus_C_proxy"])
    arg_ac_over_b = float(np.degrees(wrap(np.angle(AC) - np.angle(B))))
    return {
        "prefix": prefix, "alpha": float(alpha), "r": float(rsq),
        "R": R, "nbar": field_nbar(alpha, rsq),
        "theta0": a0, "theta_pi": api, "odd_x_proxy": odd_x,
        "rho_proxy_AC_over_B": rho,
        "material_factor_rho_over_R": float(rho / max(R, 1e-30)),
        "arg_B_deg": odd_x["B_proxy"]["arg_deg"],
        "arg_AC_over_B_deg": arg_ac_over_b,
        "odd_norm2D_0": a0["odd_norm2D"],
        "odd_norm2D_pi": api["odd_norm2D"],
        "amp2D_pi_over_0": float(api["odd_norm2D"] / max(a0["odd_norm2D"], 1e-30)),
        "amp_Jx_pi_over_0": float(odd_x["amp_ratio_pi_over_0"]),
        "pred_amp_from_R": float(pred_amp),
        "pred_amp_from_rho": float(pred_amp_rho),
        "cannot_separate_A_and_C": True,
    }


def write_reports(rec: dict) -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    one = DOCS / "线Q_ABC_R扫描_一页_20261029.md"
    longp = DOCS / "线Q_ABC_R扫描_20261029.md"
    rows = rec.get("summaries", [])
    lines = [
        "# 线 Q：A/B/C 的 R=|M|/α² 扫描一页（2026-10-29）",
        "",
        "Q6 原协议乘积 GH5，η=1，kick=True，θ=0/π。只改位移压缩输入的 (α,r)。",
        "两角代理只能给出 B 与 (A+C)，不能拆开 A 与 C。不是 1600 nm 的 80%/20% DS。",
        "权威 QED 数字不改写：ρ=1.456，R=1.273，arg(A/B)=177.65°。",
        "",
        "| 点 | α | r | R | ⟨n⟩ | ρ=|(A+C)/B| | ρ/R | 2D π/0 | 预测 |1+R|/|1−R| | arg((A+C)/B) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['prefix']} | {row['alpha']:.1f} | {row['r']:.1f} | {row['R']:.3f} | "
            f"{row['nbar']:.3f} | {row['rho_proxy_AC_over_B']:.3f} | "
            f"{row['material_factor_rho_over_R']:.3f} | {row['amp2D_pi_over_0']:.2f} | "
            f"{row['pred_amp_from_R']:.2f} | {row['arg_AC_over_B_deg']:.1f}° |"
        )
    d = rec.get("diagnostics", {})
    lines += [
        "",
        "## 判定",
        "",
        rec.get("verdict_text", rec.get("verdict", "")),
        "",
        f"- GH5 材料因子 ρ/R 均值 {d.get('mean_f', 'n/a')}，相对散度 {d.get('rel_spread_f', 'n/a')}。",
        f"- QED 材料因子 1.456/1.273 = 1.144，本轮是乘积 GH5，不能直接改写该值。",
        f"- a05r03 应对齐 A2b e10_on：ρ≈1.268，2D π/0≈8.02。",
        "",
        f"归档：`{OUT.as_posix()}`。",
        "",
    ]
    one.write_text("\n".join(lines), encoding="utf-8")
    longp.write_text(
        "# 八带 M0：A/B/C 的 R 扫描（2026-10-29）\n\n"
        + "```json\n"
        + json.dumps(jsonable({
            "verdict": rec.get("verdict"),
            "diagnostics": rec.get("diagnostics"),
            "summaries": rec.get("summaries"),
            "notes": rec.get("notes"),
        }), ensure_ascii=False, indent=2)
        + "\n```\n",
        encoding="utf-8",
    )
    rec["reports"] = {"one_page": str(one), "long": str(longp)}


def analyze() -> dict:
    missing = [tag for tag, *_ in JOBS if not job_finished_ok(tag)]
    summaries = []
    for prefix, alpha, rsq in CASES:
        tags = [f"{prefix}_p_0", f"{prefix}_m_0", f"{prefix}_p_pi", f"{prefix}_m_pi"]
        if all(job_finished_ok(t) for t in tags):
            summaries.append(summarize_prefix(prefix, alpha, rsq))
    notes = []
    d = {}
    if summaries:
        fs = np.array([row["material_factor_rho_over_R"] for row in summaries], float)
        rs = np.array([row["R"] for row in summaries], float)
        rhos = np.array([row["rho_proxy_AC_over_B"] for row in summaries], float)
        args = np.array([row["arg_AC_over_B_deg"] for row in summaries], float)
        d["mean_f"] = float(np.mean(fs))
        d["std_f"] = float(np.std(fs))
        d["rel_spread_f"] = float(np.std(fs) / max(abs(np.mean(fs)), 1e-30))
        d["max_abs_arg_minus_180"] = float(np.max(np.abs((args + 180.0 + 180.0) % 360.0 - 180.0)))
        d["max_rel_rho_minus_R"] = float(np.max(np.abs(rhos - rs) / np.maximum(rs, 1e-30)))
        q6 = next((row for row in summaries if row["prefix"] == "a05r03"), None)
        if q6 is not None:
            d["a05r03_rho"] = q6["rho_proxy_AC_over_B"]
            d["a05r03_amp2D"] = q6["amp2D_pi_over_0"]
            d["a05r03_vs_a2b_rho"] = abs(q6["rho_proxy_AC_over_B"] - 1.268)
        mean_f = d["mean_f"]
        spread = d["rel_spread_f"]
        if spread < 0.08 and abs(mean_f - 1.0) < 0.08:
            verdict = "rho_tracks_R_field_dominated"
            notes.append("GH5 上 ρ≈R，材料因子接近 1 且随 R 稳定。Q6 的 14% QED 修正不在本层。")
        elif spread < 0.08:
            verdict = "material_factor_stable_across_R"
            notes.append(f"材料因子 ρ/R≈{mean_f:.3f} 随 R 稳定，可当作输入无关的电子修正。")
        else:
            verdict = "material_factor_R_dependent"
            notes.append("ρ/R 随 R 变化，不能把单一 1.14 因子外推到所有位移压缩窗口。")
        if d["max_abs_arg_minus_180"] < 15.0:
            notes.append("arg((A+C)/B) 接近 180°，两角干涉结构保留。")
        else:
            notes.append("相位不再锁在 180°，R 扫描改变了干涉几何。")
    else:
        verdict = "incomplete"
    rec = {
        "task_tag": TASK_TAG, "record_date": RECORD_DATE, "protocol": PROTOCOL,
        "status": "ok" if summaries and not missing else "partial",
        "verdict": verdict, "notes": notes, "verdict_text": " ".join(notes) or verdict,
        "summaries": summaries, "diagnostics": d, "missing_jobs": missing,
        "qed_not_rerun": True, "did_not_overwrite_q6": True,
        "used_product_GH": True, "kick": True, "eta": 1.0,
        "did_not_edit_a1": True, "did_not_mix_1600nm_8020": True,
        "cannot_separate_A_and_C": True,
        "utc": datetime.now(timezone.utc).isoformat(),
        "source_file": Path(__file__).name,
        "source_sha256": sha256_file(Path(__file__)),
    }
    write_reports(rec)
    dump(OUT / "abc_rscan.json", rec)
    return rec


def launch_batch(tags, t0: float) -> dict:
    procs, logs = {}, {}
    for tag in tags:
        if job_finished_ok(tag):
            progress(f"drive skip {tag} already ok")
            continue
        logf = (OUT / f"log_{tag}.txt").open("a", encoding="utf-8")
        logs[tag] = logf
        envp = os.environ.copy()
        for k in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
            envp[k] = "1"
        cmd = [sys.executable, "-u", str(Path(__file__).resolve()), "--job", tag]
        progress(f"launch {tag}")
        procs[tag] = subprocess.Popen(
            cmd, cwd=str(HERE), env=envp, stdout=logf, stderr=subprocess.STDOUT)
    while True:
        elapsed = time.perf_counter() - t0
        alive = {tag: p for tag, p in procs.items() if p.poll() is None}
        if not alive:
            break
        if elapsed >= DRIVER_WALL_S:
            progress(f"driver wall {DRIVER_WALL_S}s: terminating {list(alive)}")
            for tag, p in alive.items():
                p.terminate()
            time.sleep(2.0)
            for tag, p in alive.items():
                if p.poll() is None:
                    p.kill()
            break
        time.sleep(2.0)
    for fh in logs.values():
        fh.close()
    return {tag: (procs[tag].returncode if tag in procs else 0) for tag in tags}


def drive() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    codes = {}
    for name, tags in BATCHES:
        remaining = DRIVER_WALL_S - (time.perf_counter() - t0)
        if remaining < 60.0:
            progress(f"skip {name}: remaining {remaining:.1f}s < 60")
            break
        progress(f"batch {name} remaining={remaining:.1f}s")
        codes.update(launch_batch(tags, t0))
        if time.perf_counter() - t0 >= DRIVER_WALL_S:
            progress("driver wall reached after batch")
            break
    progress(f"jobs return {codes} driver_wall={time.perf_counter()-t0:.1f}s")
    done = [tag for tag, *_ in JOBS if job_finished_ok(tag)]
    if done:
        rec = analyze()
        progress(f"analyze done verdict={rec.get('verdict')}")
        return 0 if rec.get("status") in ("ok", "partial") else 1
    rec = {
        "status": "timeout_partial", "stopped": True,
        "driver_wall_s": time.perf_counter() - t0, "returncodes": codes,
        "did_not_analyze_incomplete": True, "qed_not_rerun": True,
    }
    dump(OUT / "abc_rscan.json", rec)
    progress("incomplete, no full analyze")
    return 1


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--job", type=str, default="")
    ap.add_argument("--analyze-only", action="store_true")
    args = ap.parse_args(argv)
    OUT.mkdir(parents=True, exist_ok=True)
    if args.job:
        rec = run_one_job(args.job)
        return 0 if rec.get("status") in ("ok", "timeout_partial") else 1
    if args.analyze_only:
        rec = analyze()
        print(json.dumps(jsonable({
            "status": rec.get("status"), "verdict": rec.get("verdict"),
            "notes": rec.get("notes"), "diagnostics": rec.get("diagnostics"),
        }), ensure_ascii=False, indent=2))
        return 0 if rec.get("status") in ("ok", "partial") else 1
    return drive()


if __name__ == "__main__":
    raise SystemExit(main())
