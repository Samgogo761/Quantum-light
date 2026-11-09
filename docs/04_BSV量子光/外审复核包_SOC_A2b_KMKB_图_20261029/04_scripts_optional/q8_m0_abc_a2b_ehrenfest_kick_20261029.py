"""A2b: product GH5 Ehrenfest kick on/off at eta=1.0 and 1.5.

Same Q6 protocol as A1: D(0.5)S(0.3 e^{i theta})|0>, theta=0/pi, k/-k.
Do not call Q2 correlated gh5_nodes. Do not reuse Q4 (correlated, pi/2).
Do not edit q8_m0_model.py. No new QED, no nmax=10, no M52 grid.
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

import q8_m0_abc_a1_gate_gh5_20261029 as a1
import q8_m0_qinput_dynamics_20261026 as dyn
import q8_m0_qinput_shared_kk_mtef_map_20261108 as mp
import q8_m0_qinput_shared_kk_static_20261029 as sh
from q8_m0_model import M0Params, R0, sha256_file

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results/q_line/abc_a2b_ehrenfest_kick_20261029"
FIG = OUT / "figures"
DOCS = ROOT / "docs/04_BSV量子光"
TASK_TAG = "20261029"
RECORD_DATE = "2026-10-29"
PROTOCOL = "Qin_a2_1_4_1"
DRIVER_WALL_S = 3600.0
JOB_WALL_S = 1800.0
N_FULL = a1.N_FULL
THETA_0 = a1.THETA_0
THETA_PI = a1.THETA_PI
QED_ARG_B = 163.90
FT_ARG_B = -175.36145486637295
FROZEN_GH5_ARG_B = 167.7

ETAS = (1.0, 1.5)
JOBS = (
    ("e10_on_p_0", 1.0, True, 1, THETA_0),
    ("e10_on_m_0", 1.0, True, -1, THETA_0),
    ("e10_on_p_pi", 1.0, True, 1, THETA_PI),
    ("e10_on_m_pi", 1.0, True, -1, THETA_PI),
    ("e10_off_p_0", 1.0, False, 1, THETA_0),
    ("e10_off_m_0", 1.0, False, -1, THETA_0),
    ("e10_off_p_pi", 1.0, False, 1, THETA_PI),
    ("e10_off_m_pi", 1.0, False, -1, THETA_PI),
    ("e15_on_p_0", 1.5, True, 1, THETA_0),
    ("e15_on_m_0", 1.5, True, -1, THETA_0),
    ("e15_on_p_pi", 1.5, True, 1, THETA_PI),
    ("e15_on_m_pi", 1.5, True, -1, THETA_PI),
    ("e15_off_p_0", 1.5, False, 1, THETA_0),
    ("e15_off_m_0", 1.5, False, -1, THETA_0),
    ("e15_off_p_pi", 1.5, False, 1, THETA_PI),
    ("e15_off_m_pi", 1.5, False, -1, THETA_PI),
)
JOB_MAP = {tag: (eta, kick, neel, theta) for tag, eta, kick, neel, theta in JOBS}
BATCHES = (
    ("e10_on", ("e10_on_p_0", "e10_on_m_0", "e10_on_p_pi", "e10_on_m_pi")),
    ("e10_off", ("e10_off_p_0", "e10_off_m_0", "e10_off_p_pi", "e10_off_m_pi")),
    ("e15_on", ("e15_on_p_0", "e15_on_m_0", "e15_on_p_pi", "e15_on_m_pi")),
    ("e15_off", ("e15_off_p_0", "e15_off_m_0", "e15_off_p_pi", "e15_off_m_pi")),
)

jsonable = a1.jsonable
cpack = a1.cpack
cz = a1.cz
wrap = a1.wrap
angdiff_deg = a1.angdiff_deg
sample_js_for = a1.sample_js_for
make_env = a1.make_env
analytic_xp_theta = a1.analytic_xp_theta
gh5_nodes = a1.gh5_nodes
h2_from_series = a1.h2_from_series
two_angle_proxy = a1.two_angle_proxy


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


def params_eta(eta: float) -> M0Params:
    return M0Params(
        t_h=R0.t_h, lam_so=R0.lam_so, delta=float(eta) * R0.delta,
        u=R0.u, v=R0.v, label=f"R0_eta{eta:g}",
    )


def one_traj(model, ek0, emk0, X0, P0, env, n_steps, js, clock, kick: bool):
    ek, emk = np.array(ek0, complex), np.array(emk0, complex)
    X, P = float(X0), float(P0)
    dt = env.t_total / n_steps
    want = set(int(j) for j in js)
    j_rows, t_rows = [], []
    max_n = 0.0
    for j in range(n_steps):
        if clock is not None:
            clock.check()
        s = env.s((j + 0.5) * dt)
        ek, emk, X, P, _ = mp.step_ehrenfest(ek, emk, X, P, model, s, dt, kick=bool(kick))
        nk = abs(np.linalg.norm(ek) - 1.0)
        nm = abs(np.linalg.norm(emk) - 1.0)
        max_n = max(max_n, nk, nm)
        if j in want:
            t = (j + 1) * dt
            jvec = mp.j_total(ek, emk, model, env.s(t), X)
            j_rows.append(jvec)
            t_rows.append(t)
    return np.asarray(j_rows, complex), np.asarray(t_rows, float), max_n, X, P


def make_bundle(eta: float):
    geo = sh.build_geometry() if hasattr(sh, "build_geometry") else None
    from q8_m0_model import build_geometry
    geo = build_geometry()
    k = geo.k_cart(sh.K_FRAC)
    p = params_eta(eta)
    models, grounds = {}, {}
    for neel in (1, -1):
        models[neel] = sh.build_shared(
            geo, p, k, neel, n_max=0, omega0=a1.pf.OMEGA0, g=a1.pf.G_ABS)
        grounds[neel] = a1.ground_pair_ctrl(geo, p, k, neel)
    return geo, k, p, models, grounds, make_env()


def ckpt_path(tag: str) -> Path:
    return OUT / f"ckpt_{tag}.npz"


def job_path(tag: str) -> Path:
    return OUT / f"job_{tag}.json"


def series_path(tag: str) -> Path:
    return OUT / f"series_{tag}.npz"


def save_ckpt(tag, eta, kick, neel, theta, n_done, n_nodes, acc, wsum, max_n, t_arr) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    acc = np.asarray(acc, complex)
    np.savez_compressed(
        ckpt_path(tag),
        acc_re=acc.real, acc_im=acc.imag, wsum=float(wsum),
        n_done=int(n_done), n_nodes=int(n_nodes), max_n=float(max_n),
        t=np.asarray(t_arr, float), eta=float(eta), neel=int(neel),
        theta=float(theta), kick=np.asarray(int(bool(kick))),
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
        "eta": float(z["eta"]) if "eta" in z.files else None,
        "kick": bool(int(z["kick"])) if "kick" in z.files else None,
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
    eta, kick, neel, theta = JOB_MAP[tag]
    if job_finished_ok(tag):
        rec = load_job(tag)
        progress(f"{tag} already complete n_run={rec.get('n_run')}")
        return rec
    OUT.mkdir(parents=True, exist_ok=True)
    _, _, p, models, grounds, env = make_bundle(eta)
    gh = gh5_nodes(theta)
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
        if ck.get("kick") is not None and bool(ck["kick"]) != bool(kick):
            rec = {"tag": tag, "status": "ckpt_kick_mismatch", "stopped": True}
            dump(job_path(tag), rec)
            return rec
        i0, acc, wsum, max_n, t_arr = ck["n_done"], ck["acc"], ck["wsum"], ck["max_n"], ck["t"]
        progress(f"{tag} resume n_done={i0}/{len(nodes)}")
    clock = dyn.WallClock(JOB_WALL_S)
    progress(f"{tag} start eta={eta} kick={int(kick)} neel={neel} theta={theta:.6f} nodes={len(nodes)}")
    timed_out = False
    timeout_msg = None
    try:
        for i, node in enumerate(nodes):
            if i < i0:
                continue
            j_rows, t_rows, nrm, _, _ = one_traj(
                model, ek0, emk0, node["X"], node["P"], env, N_FULL, js, clock, kick)
            w = float(node["weight"])
            acc = w * j_rows if acc is None else acc + w * j_rows
            wsum += w
            max_n = max(max_n, nrm)
            t_arr = t_rows
            save_ckpt(tag, eta, kick, neel, theta, i + 1, len(nodes), acc, wsum, max_n, t_arr)
            if (i + 1) == len(nodes) or (i + 1) % 5 == 0:
                progress(f"{tag} node {i + 1}/{len(nodes)} wall={clock.elapsed():.1f}s")
    except TimeoutError as exc:
        timed_out = True
        timeout_msg = str(exc)
        progress(f"{tag} timeout {timeout_msg}")
    n_done = 0 if acc is None else int(load_ckpt(tag)["n_done"] if ckpt_path(tag).is_file() else 0)
    if timed_out or n_done < len(nodes):
        rec = {
            "tag": tag, "eta": float(eta), "kick": bool(kick), "neel": int(neel),
            "theta": float(theta), "status": "timeout_partial",
            "n_nodes": len(nodes), "n_done": n_done, "weight_sum": wsum,
            "max_abs_norm_minus_1": float(max_n), "wall_s": clock.elapsed(),
            "timeout_msg": timeout_msg, "n_max": 0, "protocol": PROTOCOL,
            "n_steps": N_FULL, "stopped": True, "electron": mass,
            "quad_cov_XP": gh["quad_cov_XP"], "delta": float(p.delta),
        }
        dump(job_path(tag), rec)
        return rec
    avg = acc / (wsum + 1e-30)
    save_series(tag, t_arr, avg)
    rec = {
        "tag": tag, "eta": float(eta), "kick": bool(kick), "neel": int(neel),
        "order": a1.ORDER, "theta": float(theta), "status": "ok",
        "n_nodes": len(nodes), "n_run": len(nodes), "n_done": len(nodes),
        "weight_sum": wsum, "max_abs_norm_minus_1": float(max_n),
        "wall_s": clock.elapsed(), "n_max": 0, "protocol": PROTOCOL,
        "n_steps": N_FULL, "dt": env.t_total / N_FULL,
        "n_samples": int(len(t_arr)), "electron": mass, "stopped": True,
        "delta": float(p.delta), "lam_so": float(p.lam_so),
        "wigner": analytic_xp_theta(theta), "quad_cov_XP": gh["quad_cov_XP"],
        "used_product_GH": True, "used_correlated_GH": False,
        "did_not_call_q2_gh5_nodes": True, "did_not_reuse_q4": True,
        "operator_order": "D(alpha)S(xi)|0>, xi=r exp(i theta)",
    }
    dump(job_path(tag), rec)
    progress(f"{tag} done wall={clock.elapsed():.1f}s")
    return rec


def load_pm_h2(eta: float, kick: bool, theta: float):
    prefix = f"e{int(eta * 10):02d}_{'on' if kick else 'off'}"
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
        "theta": float(theta), "eta": float(eta), "kick": bool(kick),
        "odd_H2": [cpack(odd_j[1, 0]), cpack(odd_j[1, 1])],
        "even_H2": [cpack(even_j[1, 0]), cpack(even_j[1, 1])],
        "odd_norm2D": float(np.linalg.norm(odd_j[1])),
        "even_norm2D": float(np.linalg.norm(even_j[1])),
        "windows": {"plus": pack_p, "minus": pack_m},
    }


def summarize_pair(eta: float, kick: bool) -> dict:
    a0 = load_pm_h2(eta, kick, THETA_0)
    api = load_pm_h2(eta, kick, THETA_PI)
    odd_x = two_angle_proxy(cz(a0["odd_H2"][0]), cz(api["odd_H2"][0]))
    return {
        "eta": float(eta), "kick": bool(kick),
        "delta": float(eta) * float(R0.delta),
        "theta0": a0, "theta_pi": api, "odd_x_proxy": odd_x,
        "arg_B_deg": odd_x["B_proxy"]["arg_deg"],
        "odd_norm2D_0": a0["odd_norm2D"],
        "odd_norm2D_pi": api["odd_norm2D"],
    }


def write_reports(rec: dict) -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    one = DOCS / "线Q_A2b_Ehrenfest反作用_一页_20261029.md"
    longp = DOCS / "线Q_A2b_Ehrenfest反作用_20261029.md"
    s = rec.get("summaries", {})
    d = rec.get("diagnostics", {})
    lines = [
        "# 线 Q：A2b Ehrenfest kick on/off 一页结论（2026-10-29）",
        "",
        "Q6 原协议乘积 GH5，θ=0/π，η=1.0 与 1.5。kick=False 仍是有限振幅 Wigner 样本，不是 A2 弱场核。",
        "不使用 Q4 相关 GH。权威 QED 数字不改写。",
        "",
        "## 1. η=1.0 能否复现冻结乘积 GH5？",
        "",
        f"- 冻结 GH5 arg(B) = {FROZEN_GH5_ARG_B:.1f}°；QED arg(B) = {QED_ARG_B:.2f}°；有限时间核 = {FT_ARG_B:.2f}°。",
    ]
    e10_on = s.get("e10_on")
    e10_off = s.get("e10_off")
    e15_on = s.get("e15_on")
    e15_off = s.get("e15_off")
    if e10_on:
        lines.append(
            f"- 本轮 kick=True arg(B) = {e10_on['arg_B_deg']:.2f}°，"
            f"相对冻结 {angdiff_deg(e10_on['arg_B_deg'], FROZEN_GH5_ARG_B):.2f}°。"
        )
    if e10_off:
        lines.append(f"- 本轮 kick=False arg(B) = {e10_off['arg_B_deg']:.2f}°。")
    lines += ["", "## 2. 剩余 20.7° 是否来自反作用？", ""]
    if e10_on and e10_off:
        lines.append(
            f"- kick on−off 相位差 {angdiff_deg(e10_on['arg_B_deg'], e10_off['arg_B_deg']):.2f}°。"
        )
        lines.append(
            f"- kick=False 相对有限时间核 {angdiff_deg(e10_off['arg_B_deg'], FT_ARG_B):.2f}°，"
            f"相对 QED {angdiff_deg(e10_off['arg_B_deg'], QED_ARG_B):.2f}°。"
        )
    lines += ["", "## 3. η=1.5 旋转在 kick=False 下是否还在？", ""]
    if e10_on and e15_on:
        lines.append(f"- kick=True 旋转 {d.get('rot_on_deg')}°（η1.5−η1.0 的 arg B）。")
    if e10_off and e15_off:
        lines.append(f"- kick=False 旋转 {d.get('rot_off_deg')}°。")
    if e15_on:
        lines.append(f"- η=1.5 kick=True arg(B) = {e15_on['arg_B_deg']:.2f}°。")
    if e15_off:
        lines.append(f"- η=1.5 kick=False arg(B) = {e15_off['arg_B_deg']:.2f}°。")
    lines += [
        "",
        "## 4. 判定",
        "",
        rec.get("verdict_text", rec.get("verdict", "")),
        "",
        f"归档：`{OUT.as_posix()}`。",
        "",
    ]
    one.write_text("\n".join(lines), encoding="utf-8")
    longp.write_text(
        "# 八带 M0：A2b Ehrenfest 反作用（2026-10-29）\n\n"
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
    summaries = {}
    for eta, kick, key in (
        (1.0, True, "e10_on"), (1.0, False, "e10_off"),
        (1.5, True, "e15_on"), (1.5, False, "e15_off"),
    ):
        tags = [t for t, e, k, *_ in JOBS if abs(e - eta) < 1e-12 and bool(k) is bool(kick)]
        if all(job_finished_ok(t) for t in tags):
            summaries[key] = summarize_pair(eta, kick)
    d = {}
    notes = []
    if "e10_on" in summaries and "e15_on" in summaries:
        d["rot_on_deg"] = angdiff_deg(summaries["e15_on"]["arg_B_deg"],
                                      summaries["e10_on"]["arg_B_deg"])
    if "e10_off" in summaries and "e15_off" in summaries:
        d["rot_off_deg"] = angdiff_deg(summaries["e15_off"]["arg_B_deg"],
                                       summaries["e10_off"]["arg_B_deg"])
    if "e10_on" in summaries and "e10_off" in summaries:
        d["kick_on_minus_off_eta1_deg"] = angdiff_deg(
            summaries["e10_on"]["arg_B_deg"], summaries["e10_off"]["arg_B_deg"])
        d["off_minus_finite_time_deg"] = angdiff_deg(
            summaries["e10_off"]["arg_B_deg"], FT_ARG_B)
        d["off_minus_qed_deg"] = angdiff_deg(
            summaries["e10_off"]["arg_B_deg"], QED_ARG_B)
        d["on_minus_frozen_deg"] = angdiff_deg(
            summaries["e10_on"]["arg_B_deg"], FROZEN_GH5_ARG_B)
    rot_on = abs(d.get("rot_on_deg", 0.0))
    rot_off = abs(d.get("rot_off_deg", 0.0))
    kick_d = abs(d.get("kick_on_minus_off_eta1_deg", 0.0))
    off_ft = abs(d.get("off_minus_finite_time_deg", 99.0))
    verdict = "incomplete"
    if summaries and not missing:
        if rot_on > 25.0 and rot_off < 12.0:
            verdict = "ehrenfest_explains_eta_rotation"
            notes.append("η 旋转在 kick=False 后消失，反作用是候选。")
        elif rot_on > 25.0 and abs(rot_off - rot_on) < 12.0:
            verdict = "eta_rotation_survives_no_kick"
            notes.append("η 旋转在 kick=False 下仍在，不是 χ^{(2)} 也不是 Ehrenfest。")
        elif rot_on < 15.0:
            verdict = "product_GH_0pi_no_large_eta_rotation"
            notes.append("θ=0/π 乘积 GH5 未复现 Q4 约 50° 旋转；该旋转可能依赖 π/2 协议。")
        else:
            verdict = "mixed_partial_backaction"
            notes.append("反作用只解释部分 η 旋转。")
        if kick_d < 8.0 and off_ft > 12.0:
            notes.append("剩余 21° 不是 Ehrenfest：有限振幅无踢仍远离弱场核。")
        elif kick_d > 12.0 and off_ft < 12.0:
            notes.append("kick=False 靠近有限时间核，剩余 21° 可由反作用解释。")
    rec = {
        "task_tag": TASK_TAG, "record_date": RECORD_DATE, "protocol": PROTOCOL,
        "status": "ok" if summaries and not missing else "partial",
        "verdict": verdict, "notes": notes, "verdict_text": " ".join(notes) or verdict,
        "summaries": summaries, "diagnostics": d, "missing_jobs": missing,
        "qed_not_rerun": True, "did_not_overwrite_q6": True,
        "used_product_GH": True, "did_not_reuse_q4": True,
        "input_state": "D(0.5)S(0.3 exp(i theta))|0>",
        "utc": datetime.now(timezone.utc).isoformat(),
        "source_file": Path(__file__).name,
        "source_sha256": sha256_file(Path(__file__)),
    }
    write_reports(rec)
    dump(OUT / "a2b_ehrenfest.json", rec)
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
    dump(OUT / "a2b_ehrenfest.json", rec)
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
