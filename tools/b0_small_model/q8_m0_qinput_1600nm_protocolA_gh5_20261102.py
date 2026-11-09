#!/usr/bin/env python3
"""Protocol A: same quantum statistics, Q6 M0 at 1600 nm (GH5 first).

Keep D(0.5)S(0.3 e^{i theta})|0>, R=1.273, abstract g=0.005 A^{-1},
static M0, same k/-k. Change only omega0: 0.05 -> 0.775 (explicit).
Do not silently use pf.OMEGA0=0.05. Do not scale g ~ omega^{-1/2}.
Do not copy 4.98 / 110.6 deg. GH5 is not Cohen MTEF.
1+4+1 at 1600 nm is ~32 fs coupling switch, not a 25 fs Gaussian pulse.
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

import q8_m0_qinput_dynamics_20261026 as dyn
import q8_m0_qinput_qin_a2_141_gh5_q2_covsign_20261120 as q2
import q8_m0_qinput_shared_kk_ds_odd_20261102 as odd
import q8_m0_qinput_shared_kk_dyn_preflight_20261030 as pf
import q8_m0_qinput_shared_kk_static_20261029 as sh
from q8_m0_model import R0, build_geometry, sha256_file

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results/q_line/qinput_1600nm_protocolA_gh5_20261102"
FIG = OUT / "figures"
DOCS = ROOT / "docs/04_BSV量子光"
TASK_TAG = "20261102"
RECORD_DATE = "2026-11-02"
PROTOCOL = "Qin_a2_1_4_1_1600nm_protocolA"
OMEGA0_Q6 = 0.05
OMEGA0_1600 = 0.775
G_ABS = 0.005
ALPHA = float(np.real(odd.ALPHA))
R_SQ = float(odd.R_SQ)
HBAR_EV_FS = 0.6582119568564649
ORDER = q2.ORDER
N_NODES = q2.N_NODES
N_FULL = q2.N_FULL
N_SMOKE = 800
SAMPLE_EVERY = q2.SAMPLE_EVERY
NMAX_FOCK_REF = q2.NMAX_FOCK_REF
DRIVER_WALL_S = 600.0
JOB_WALL_S = 570.0
THETA_0 = 0.0
THETA_Q = 0.25 * np.pi
THETA_H = 0.5 * np.pi
THETA_PI = float(np.pi)
THETA_T = 1.5 * np.pi
THETA_KEYS = ("0", "h", "pi", "t")
THETAS_FOUR = (THETA_0, THETA_H, THETA_PI, THETA_T)
LEAVEOUT_MARK = 0.02
AMP_RATIO_MARK = 1.20
EXPECTED_HASHES = dict(q2.EXPECTED_HASHES)
FROZEN = tuple(HERE / name for name in EXPECTED_HASHES)

jsonable = q2.jsonable
cz = q2.cz
pack_c = q2.pack_c
wrap = q2.wrap
angdiff = q2.angdiff
sample_js_for = q2.sample_js_for
analytic_xp_theta = q2.analytic_xp_theta
one_traj = q2.one_traj
rect_jq = q2.rect_jq
slot_pack = q2.slot_pack
h2_jx = q2.h2_jx
fit_angle_model = q2.fit_angle_model
pack_win = q2.pack_win
fock_xp_full = q2.fock_xp_full


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


def source_hashes() -> dict:
    out = {p.name: sha256_file(p) for p in FROZEN}
    out[Path(__file__).name] = sha256_file(Path(__file__))
    return out


def hash_abort(hashes: dict) -> list:
    bad = []
    for name, expect in EXPECTED_HASHES.items():
        got = hashes.get(name)
        if got != expect:
            bad.append({"file": name, "expected": expect, "got": got})
    return bad


def assert_protocol_omega(omega0: float) -> None:
    if abs(float(omega0) - OMEGA0_1600) > 1e-15:
        raise RuntimeError(f"protocol A requires omega0={OMEGA0_1600}, got {omega0}")
    if abs(float(pf.OMEGA0) - OMEGA0_Q6) > 1e-15:
        raise RuntimeError(f"frozen pf.OMEGA0 changed: {pf.OMEGA0}")
    if abs(float(sh.OMEGA0) - OMEGA0_Q6) > 1e-15:
        raise RuntimeError(f"frozen sh.OMEGA0 changed: {sh.OMEGA0}")
    if abs(float(sh.G_ABS) - G_ABS) > 1e-15:
        raise RuntimeError(f"frozen g changed: {sh.G_ABS}")


def make_env(omega0: float = OMEGA0_1600):
    assert_protocol_omega(omega0)
    return dyn.Envelope(float(omega0), ramp_cycles=1.0, flat_cycles=4.0)


def make_bundle(omega0: float = OMEGA0_1600):
    assert_protocol_omega(omega0)
    geo = build_geometry()
    k = geo.k_cart(sh.K_FRAC)
    models, grounds = {}, {}
    for neel in (1, -1):
        models[neel] = sh.build_shared(
            geo, R0, k, neel, n_max=0, omega0=float(omega0), g=G_ABS)
        if abs(float(models[neel].omega0) - OMEGA0_1600) > 1e-15:
            raise RuntimeError("build_shared did not keep explicit 1600 nm omega0")
        grounds[neel] = odd.ground_pair_neel(geo, k, neel)
    env = make_env(omega0)
    return geo, k, models, grounds, env


def envelope_meta(env) -> dict:
    t_h = float(getattr(R0, "t_h", 1.0))
    T0 = float(env.period)
    return {
        "omega0": float(env.omega0),
        "t_h_eV": t_h,
        "period_model": T0,
        "period_fs": T0 * HBAR_EV_FS / t_h,
        "t_total_model": float(env.t_total),
        "t_total_fs": float(env.t_total) * HBAR_EV_FS / t_h,
        "ramp_cycles": 1.0,
        "flat_cycles": 4.0,
        "not_25fs_gaussian": True,
        "not_free_space_pulse": True,
        "used_pf_OMEGA0": False,
        "g_abs": G_ABS,
        "g_note": sh.g_normalization_note(2, G_ABS),
        "k_frac": list(sh.K_FRAC),
        "two_hbar_omega_eV": 2.0 * float(env.omega0) * t_h,
        "three_hbar_omega_eV": 3.0 * float(env.omega0) * t_h,
        "local_gap_note_eV": 1.91,
    }


def gh5_nodes_any(theta: float) -> dict:
    mom = analytic_xp_theta(theta)
    if abs(mom["cov_XP"]) < 1e-10:
        xi, w1 = hermgauss(ORDER)
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
        used_corr, used_prod = False, True
    else:
        gh = q2.gh5_nodes(theta)
        nodes = gh["nodes"]
        used_corr, used_prod = True, False
    wsum = float(sum(n["weight"] for n in nodes))
    mx = sum(n["weight"] * n["X"] for n in nodes)
    mpv = sum(n["weight"] * n["P"] for n in nodes)
    vx = sum(n["weight"] * n["X"] ** 2 for n in nodes) - mx * mx
    vp = sum(n["weight"] * n["P"] ** 2 for n in nodes) - mpv * mpv
    cxy = sum(n["weight"] * n["X"] * n["P"] for n in nodes) - mx * mpv
    return {
        "kind": "ds", "theta": float(theta), "order": ORDER, "n_nodes": len(nodes),
        "weight_sum": wsum, "quad_mean_X": float(mx), "quad_mean_P": float(mpv),
        "quad_var_X": float(vx), "quad_var_P": float(vp), "quad_cov_XP": float(cxy),
        "target_mean_X": mom["mean_X"], "target_mean_P": mom["mean_P"],
        "target_var_X": mom["var_X"], "target_var_P": mom["var_P"],
        "target_cov_XP": mom["cov_XP"],
        "used_correlated_GH": used_corr, "used_product_GH": used_prod,
        "nodes": nodes, "analytic": mom,
        "did_not_call_mp_gauss_hermite_nodes_ds": True,
    }


def job_spec(tag: str) -> dict:
    table = {
        "smoke_p_0": {"neel": 1, "theta": THETA_0, "n_steps": N_SMOKE, "family": "smoke"},
        "smoke_m_0": {"neel": -1, "theta": THETA_0, "n_steps": N_SMOKE, "family": "smoke"},
        "p_0": {"neel": 1, "theta": THETA_0, "n_steps": N_FULL, "family": "fourangle"},
        "m_0": {"neel": -1, "theta": THETA_0, "n_steps": N_FULL, "family": "fourangle"},
        "p_h": {"neel": 1, "theta": THETA_H, "n_steps": N_FULL, "family": "fourangle"},
        "m_h": {"neel": -1, "theta": THETA_H, "n_steps": N_FULL, "family": "fourangle"},
        "p_pi": {"neel": 1, "theta": THETA_PI, "n_steps": N_FULL, "family": "fourangle"},
        "m_pi": {"neel": -1, "theta": THETA_PI, "n_steps": N_FULL, "family": "fourangle"},
        "p_t": {"neel": 1, "theta": THETA_T, "n_steps": N_FULL, "family": "fourangle"},
        "m_t": {"neel": -1, "theta": THETA_T, "n_steps": N_FULL, "family": "fourangle"},
        "p_q": {"neel": 1, "theta": THETA_Q, "n_steps": N_FULL, "family": "pi4"},
        "m_q": {"neel": -1, "theta": THETA_Q, "n_steps": N_FULL, "family": "pi4"},
    }
    if tag not in table:
        raise ValueError(tag)
    spec = dict(table[tag])
    spec["tag"] = tag
    return spec


def tags_for_mode(mode: str) -> tuple[str, ...]:
    if mode == "smoke":
        return ("smoke_p_0", "smoke_m_0")
    if mode == "fourangle":
        return ("p_0", "m_0", "p_h", "m_h", "p_pi", "m_pi", "p_t", "m_t")
    if mode == "pi4":
        return ("p_q", "m_q")
    raise ValueError(mode)


def ckpt_path(tag: str) -> Path:
    return OUT / f"ckpt_{tag}.npz"


def job_path(tag: str) -> Path:
    return OUT / f"job_{tag}.json"


def series_path(tag: str) -> Path:
    return OUT / f"series_{tag}.npz"


def save_ckpt(tag, neel, theta, n_done, n_nodes, acc, wsum, max_n, t_arr,
              n_steps, omega0) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    acc = np.asarray(acc, complex)
    np.savez_compressed(
        ckpt_path(tag),
        acc_re=acc.real, acc_im=acc.imag, wsum=float(wsum),
        n_done=int(n_done), n_nodes=int(n_nodes), max_n=float(max_n),
        t=np.asarray(t_arr, float), neel=int(neel), theta=float(theta),
        n_steps=int(n_steps), omega0=float(omega0),
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
        "n_steps": int(z["n_steps"]) if "n_steps" in z.files else None,
        "omega0": float(z["omega0"]) if "omega0" in z.files else None,
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


def ckpt_mismatch(ck, theta, n_steps, omega0) -> str | None:
    if ck is None:
        return None
    if ck.get("theta") is None or abs(float(ck["theta"]) - float(theta)) > 1e-12:
        return "ckpt_theta_mismatch"
    if ck.get("n_steps") is None or int(ck["n_steps"]) != int(n_steps):
        return "ckpt_n_steps_mismatch"
    if ck.get("omega0") is None or abs(float(ck["omega0"]) - float(omega0)) > 1e-12:
        return "ckpt_omega0_mismatch"
    return None


def resource_row(theta: float) -> dict:
    ana = analytic_xp_theta(theta)
    raw = dyn.displaced_squeezed_coefficients(ALPHA, R_SQ, float(theta), NMAX_FOCK_REF)
    fock = fock_xp_full(raw)
    gh = gh5_nodes_any(theta)
    r_stat = abs(complex(ana["M"]["re"], ana["M"]["im"])) / (ALPHA ** 2)
    return {
        "theta": float(theta),
        "analytic": ana,
        "fock_nmax": NMAX_FOCK_REF,
        "fock": fock,
        "R": float(r_stat),
        "gh5_quad_mean_X": gh["quad_mean_X"],
        "gh5_quad_mean_P": gh["quad_mean_P"],
        "gh5_quad_var_X": gh["quad_var_X"],
        "gh5_quad_var_P": gh["quad_var_P"],
        "gh5_quad_cov_XP": gh["quad_cov_XP"],
        "gh5_weight_sum": gh["weight_sum"],
        "gh5_used_correlated_GH": gh["used_correlated_GH"],
        "gh5_used_product_GH": gh["used_product_GH"],
        "gh5_n_nodes": gh["n_nodes"],
        "abs_d_mean_X": abs(fock["mean_X"] - ana["mean_X"]),
        "abs_d_mean_P": abs(fock["mean_P"] - ana["mean_P"]),
        "abs_d_var_X": abs(fock["var_X"] - ana["var_X"]),
        "abs_d_var_P": abs(fock["var_P"] - ana["var_P"]),
        "abs_d_cov_XP": abs(fock["cov_XP"] - ana["cov_XP"]),
        "abs_d_nbar": abs(fock["mean_n"] - ana["mean_n_analytic"]),
        "abs_d_quad_cov": abs(gh["quad_cov_XP"] - ana["cov_XP"]),
        "abs_d_quad_var_X": abs(gh["quad_var_X"] - ana["var_X"]),
        "abs_d_quad_var_P": abs(gh["quad_var_P"] - ana["var_P"]),
    }


def collect_resources() -> dict:
    hashes = source_hashes()
    bad = hash_abort(hashes)
    env = make_env()
    rec = {
        "task_tag": TASK_TAG, "record_date": RECORD_DATE, "protocol": PROTOCOL,
        "status": "resources", "source_hashes": hashes, "hash_abort": bad,
        "operator_order": "D(alpha)S(xi)|0>, xi=r exp(i theta)",
        "alpha": ALPHA, "r": R_SQ, "R_target": 1.273,
        "omega0": OMEGA0_1600, "omega0_q6_frozen": OMEGA0_Q6,
        "g": G_ABS, "n_max_electronic_fock": 0,
        "gh5_is_not_cohen_mtef": True,
        "did_not_scale_g_with_omega": True,
        "did_not_copy_q6_4p98": True,
        "envelope": envelope_meta(env),
        "n_full": N_FULL, "n_smoke": N_SMOKE,
        "stopped": True,
    }
    if bad:
        rec["status"] = "hash_abort"
        dump(OUT / "resources.json", rec)
        return rec
    rows = {key: resource_row(th) for key, th in zip(
        ("0", "q", "h", "pi", "t"), (THETA_0, THETA_Q, THETA_H, THETA_PI, THETA_T))}
    rec["rows"] = rows
    rec["nbar"] = rows["0"]["analytic"]["mean_n_analytic"]
    rec["R_from_M"] = rows["0"]["R"]
    rec["same_nbar"] = all(abs(rows[k]["analytic"]["mean_n_analytic"] - rec["nbar"]) < 1e-14
                           for k in rows)
    rec["product_at_0_pi"] = bool(rows["0"]["gh5_used_product_GH"] and rows["pi"]["gh5_used_product_GH"])
    rec["correlated_at_h_t"] = bool(rows["h"]["gh5_used_correlated_GH"] and rows["t"]["gh5_used_correlated_GH"])
    rec["correlated_at_pi4"] = bool(rows["q"]["gh5_used_correlated_GH"])
    rec["resource_match_ok"] = bool(
        rec["same_nbar"] and rec["product_at_0_pi"] and rec["correlated_at_h_t"]
        and rec["correlated_at_pi4"] and abs(rec["R_from_M"] - 1.273) < 5e-3
        and abs(rec["envelope"]["omega0"] - OMEGA0_1600) < 1e-15
        and abs(rec["envelope"]["period_fs"] - 5.34) < 0.05
    )
    dump(OUT / "resources.json", rec)
    progress(
        f"resources match={rec['resource_match_ok']} omega0={OMEGA0_1600} "
        f"T0_fs={rec['envelope']['period_fs']:.4f} nbar={rec['nbar']:.6f} "
        f"R={rec['R_from_M']:.6f} used_pf_OMEGA0=False"
    )
    return rec


def run_one_job(tag: str, n_steps_override: int | None = None) -> dict:
    spec = job_spec(tag)
    neel, theta = spec["neel"], spec["theta"]
    n_steps = int(n_steps_override) if n_steps_override else int(spec["n_steps"])
    if spec["family"] != "smoke" and n_steps != N_FULL:
        rec = {"tag": tag, "status": "n_steps_guard", "stopped": True,
               "n_steps": n_steps, "expected": N_FULL}
        dump(job_path(tag), rec)
        return rec
    if spec["family"] == "smoke" and tag in ("p_0", "m_0"):
        rec = {"tag": tag, "status": "smoke_tag_collision", "stopped": True}
        dump(job_path(tag), rec)
        return rec
    hashes = source_hashes()
    bad = hash_abort(hashes)
    if bad:
        rec = {"tag": tag, "status": "hash_abort", "hash_mismatch": bad, "stopped": True}
        dump(job_path(tag), rec)
        return rec
    if job_finished_ok(tag):
        rec = load_job(tag)
        if rec.get("omega0") != OMEGA0_1600 or rec.get("n_steps") != n_steps:
            rec = {"tag": tag, "status": "finished_contract_mismatch", "stopped": True,
                   "have": rec, "want_omega0": OMEGA0_1600, "want_n_steps": n_steps}
            dump(job_path(tag), rec)
            return rec
        progress(f"{tag} already complete n_run={rec.get('n_run')}")
        return rec
    OUT.mkdir(parents=True, exist_ok=True)
    _, _, models, grounds, env = make_bundle(OMEGA0_1600)
    if abs(float(env.omega0) - OMEGA0_1600) > 1e-15:
        raise RuntimeError("env silently used Q6 omega0")
    gh = gh5_nodes_any(theta)
    nodes = gh["nodes"]
    ek0, emk0, mass = grounds[neel]
    model = models[neel]
    js = sample_js_for(n_steps)
    ck = load_ckpt(tag)
    mismatch = ckpt_mismatch(ck, theta, n_steps, OMEGA0_1600)
    if mismatch:
        rec = {"tag": tag, "status": mismatch, "stopped": True,
               "ckpt": {k: ck.get(k) for k in ("theta", "n_steps", "omega0", "n_done")},
               "job_theta": float(theta), "job_n_steps": int(n_steps),
               "job_omega0": OMEGA0_1600}
        dump(job_path(tag), rec)
        return rec
    i0, acc, wsum, max_n, t_arr = 0, None, 0.0, 0.0, None
    if ck is not None:
        i0, acc, wsum, max_n, t_arr = ck["n_done"], ck["acc"], ck["wsum"], ck["max_n"], ck["t"]
        progress(f"{tag} resume n_done={i0}/{len(nodes)}")
    clock = dyn.WallClock(JOB_WALL_S)
    progress(f"{tag} start neel={neel} theta={theta:.6f} omega0={OMEGA0_1600} "
             f"n_steps={n_steps} T0_fs={envelope_meta(env)['period_fs']:.4f} "
             f"nodes={len(nodes)} gh_corr={gh['used_correlated_GH']}")
    timed_out = False
    timeout_msg = None
    max_weyl_n = 0.0
    try:
        for i, node in enumerate(nodes):
            if i < i0:
                continue
            j_rows, t_rows, nrm, Xf, Pf = one_traj(
                model, ek0, emk0, node["X"], node["P"], env, n_steps, js, clock)
            w = float(node["weight"])
            acc = w * j_rows if acc is None else acc + w * j_rows
            wsum += w
            max_n = max(max_n, nrm)
            max_weyl_n = max(max_weyl_n, 0.25 * (Xf * Xf + Pf * Pf) - 0.5)
            t_arr = t_rows
            save_ckpt(tag, neel, theta, i + 1, len(nodes), acc, wsum, max_n, t_arr,
                      n_steps, OMEGA0_1600)
            if (i + 1) == len(nodes) or (i + 1) % 5 == 0:
                progress(f"{tag} node {i + 1}/{len(nodes)} wall={clock.elapsed():.1f}s")
    except TimeoutError as exc:
        timed_out = True
        timeout_msg = str(exc)
        progress(f"{tag} timeout {timeout_msg}")
    n_done = 0 if acc is None else int(load_ckpt(tag)["n_done"] if ckpt_path(tag).is_file() else 0)
    if timed_out or n_done < len(nodes):
        rec = {
            "tag": tag, "neel": int(neel), "theta": float(theta),
            "status": "timeout_partial", "n_nodes": len(nodes), "n_done": n_done,
            "weight_sum": wsum, "max_abs_norm_minus_1": float(max_n),
            "max_weyl_n": float(max_weyl_n),
            "wall_s": clock.elapsed(), "timeout_msg": timeout_msg,
            "omega0": OMEGA0_1600, "n_steps": int(n_steps),
            "used_pf_OMEGA0": False, "g": G_ABS, "family": spec["family"],
            "stopped": True,
        }
        dump(job_path(tag), rec)
        return rec
    j = acc / wsum
    save_series(tag, t_arr, j)
    rec = {
        "tag": tag, "neel": int(neel), "theta": float(theta), "status": "ok",
        "n_nodes": len(nodes), "n_run": n_done, "weight_sum": wsum,
        "max_abs_norm_minus_1": float(max_n), "max_weyl_n": float(max_weyl_n),
        "wall_s": clock.elapsed(), "omega0": OMEGA0_1600, "n_steps": int(n_steps),
        "dt": float(env.t_total / n_steps),
        "used_pf_OMEGA0": False, "g": G_ABS, "family": spec["family"],
        "used_correlated_GH": gh["used_correlated_GH"],
        "used_product_GH": gh["used_product_GH"],
        "mass": mass, "envelope": envelope_meta(env),
        "gh5_is_not_cohen_mtef": True,
        "not_macroscopic_BSV": True,
    }
    dump(job_path(tag), rec)
    progress(f"{tag} done wall={clock.elapsed():.1f}s omega0={OMEGA0_1600} n_steps={n_steps}")
    return rec


def pred_from_bac(B, A, C, theta: float) -> complex:
    return B + A * np.exp(1j * theta) + C * np.exp(-1j * theta)


def analyze_smoke() -> dict:
    missing = [tag for tag in tags_for_mode("smoke") if not job_finished_ok(tag)]
    rec = {
        "status": "ok" if not missing else "missing_jobs",
        "mode": "smoke", "missing": missing,
        "omega0": OMEGA0_1600, "n_steps": N_SMOKE,
        "jobs": {tag: load_job(tag) for tag in tags_for_mode("smoke")},
        "did_not_fit_ABC_on_smoke": True,
        "did_not_copy_q6_numbers": True,
        "utc": datetime.now(timezone.utc).isoformat(),
    }
    if not missing:
        t_p, j_p = load_series("smoke_p_0")
        t_m, j_m = load_series("smoke_m_0")
        rec["time_grid_match"] = bool(np.allclose(t_p, t_m, rtol=0, atol=1e-9))
        rec["n_samples"] = int(t_p.size)
        rec["t_last"] = float(t_p[-1])
        env = make_env()
        rec["envelope"] = envelope_meta(env)
        rec["omega0_guard"] = abs(rec["jobs"]["smoke_p_0"]["omega0"] - OMEGA0_1600) < 1e-15
        rec["not_q6_omega"] = abs(rec["jobs"]["smoke_p_0"]["omega0"] - OMEGA0_Q6) > 0.5
        even, oddj = odd.even_odd(j_p, j_m)
        rec["odd_rms"] = float(np.sqrt(np.mean(np.abs(oddj) ** 2)))
        rec["even_rms"] = float(np.sqrt(np.mean(np.abs(even) ** 2)))
        rec["smoke_pass"] = bool(
            rec["time_grid_match"] and rec["omega0_guard"] and rec["not_q6_omega"]
            and rec["jobs"]["smoke_p_0"]["n_steps"] == N_SMOKE
        )
        rec["status"] = "ok" if rec["smoke_pass"] else "smoke_guard_fail"
    dump(OUT / "smoke.json", rec)
    return rec


def load_pair(tag_p: str, tag_m: str):
    t_p, j_p = load_series(tag_p)
    t_m, j_m = load_series(tag_m)
    if not np.allclose(t_p, t_m, rtol=0, atol=1e-9):
        raise ValueError(f"time_grid_mismatch {tag_p} {tag_m}")
    return t_p, j_p, j_m


def analyze_fourangle() -> dict:
    hashes = source_hashes()
    bad = hash_abort(hashes)
    if bad:
        rec = {"status": "hash_abort", "hash_mismatch": bad, "stopped": True}
        dump(OUT / "protocolA_gh5.json", rec)
        return rec
    missing = [tag for tag in tags_for_mode("fourangle") if not job_finished_ok(tag)]
    if missing:
        rec = {"status": "missing_jobs", "missing": missing, "stopped": True}
        dump(OUT / "protocolA_gh5.json", rec)
        return rec
    env = make_env()
    T0 = env.period
    omega0 = OMEGA0_1600
    t0, jp0, jm0 = load_pair("p_0", "m_0")
    th, jph, jmh = load_pair("p_h", "m_h")
    tpi, jppi, jmpi = load_pair("p_pi", "m_pi")
    tt, jpt, jmt = load_pair("p_t", "m_t")
    t = t0
    for lab, tt_ in (("h", th), ("pi", tpi), ("t", tt)):
        if not np.allclose(tt_, t, rtol=0, atol=1e-9):
            rec = {"status": "time_grid_mismatch", "tag": lab, "stopped": True}
            dump(OUT / "protocolA_gh5.json", rec)
            return rec
    pairs = {"0": (jp0, jm0), "h": (jph, jmh), "pi": (jppi, jmpi), "t": (jpt, jmt)}
    arrs = {}
    for key, (jp, jm) in pairs.items():
        ev, od = odd.even_odd(jp, jm)
        arrs[f"J_odd_{key}"] = od
        arrs[f"J_even_{key}"] = ev
    arrs["dJ_odd"] = arrs["J_odd_t"] - arrs["J_odd_h"]
    main = pack_win("plateau_center_2T", t, omega0, T0, 2.0 * T0, 4.0 * T0, arrs)
    front = pack_win("front_2T", t, omega0, T0, 1.0 * T0, 3.0 * T0, arrs)
    back = pack_win("back_2T", t, omega0, T0, 3.0 * T0, 5.0 * T0, arrs)

    def take(winobj, key):
        return h2_jx(winobj["slots"], key)

    z0 = take(main, "J_odd_0")
    zh = take(main, "J_odd_h")
    zpi = take(main, "J_odd_pi")
    zt = take(main, "J_odd_t")
    zs = [z0[0], zh[0], zpi[0], zt[0]]
    fit = fit_angle_model(list(THETAS_FOUR), zs)
    B, A, C = cz(fit["B"]), cz(fit["A"]), cz(fit["C"])
    amp_ratio = float(abs(zpi[0]) / max(abs(z0[0]), 1e-30))
    rho = float(abs(A) / max(abs(B), 1e-30))
    arg_ab = float(np.degrees(np.angle(A / B))) if abs(B) > 0 else None
    leave = None
    if job_finished_ok("p_q") and job_finished_ok("m_q"):
        tq, jpq, jmq = load_pair("p_q", "m_q")
        if np.allclose(tq, t, rtol=0, atol=1e-9):
            _, odd_q = odd.even_odd(jpq, jmq)
            jq = rect_jq(t, odd_q, omega0, 2.0 * T0, 4.0 * T0, np.array([2.0]))[0, 0]
            pred = pred_from_bac(B, A, C, THETA_Q)
            leave = {
                "actual": pack_c(jq), "predicted": pack_c(pred),
                "rel_err": float(abs(jq - pred) / max(abs(jq), abs(pred), 1e-30)),
                "needs_delta_n_pm4": None,
            }
            leave["needs_delta_n_pm4"] = bool(leave["rel_err"] > LEAVEOUT_MARK)
    angle_enters = bool(amp_ratio >= AMP_RATIO_MARK or fit["rel_resid"] < 0.15 and rho > 0.2)
    if not angle_enters:
        verdict = "negative_squeeze_angle_does_not_enter_neel_odd_H2"
        answer = ("负结果：在相同 R 的 1600 nm 协议 A 下，压缩角未进入 Néel-odd H2。"
                  "不调 SOC/g 抢救。")
    elif leave and leave["needs_delta_n_pm4"]:
        verdict = "abc_insufficient_need_delta_n_pm4"
        answer = "A/B/C 三项不足以描述 π/4 留出，需要承认 Δn=±4。"
    else:
        verdict = "mechanism_survives_frequency_bridge_gh5_only"
        answer = ("GH5 下压缩角仍进入 Néel-odd H2，且 A/B/C 暂可描述。"
                  "这不是精确 QED，也不是实验产额。")
    rec = {
        "status": "ok", "mode": "fourangle", "verdict": verdict, "answer": answer,
        "omega0": omega0, "envelope": envelope_meta(env),
        "h2_main": {
            "theta0": pack_c(z0[0]), "theta_h": pack_c(zh[0]),
            "theta_pi": pack_c(zpi[0]), "theta_t": pack_c(zt[0]),
            "amp_ratio_pi_over_0": amp_ratio,
            "did_not_copy_q6_4p98": True,
        },
        "fit_ABC": fit, "rho_absA_over_absB": rho, "arg_A_over_B_deg": arg_ab,
        "leaveout_pi4": leave,
        "windows": {"main": main, "front": front, "back": back},
        "gh5_is_not_qed": True, "gh5_is_not_cohen_mtef": True,
        "g_not_Wcm2": True, "utc": datetime.now(timezone.utc).isoformat(),
    }
    dump(OUT / "protocolA_gh5.json", rec)
    write_report(rec)
    return rec


def write_report(rec: dict) -> None:
    DOCS.mkdir(parents=True, exist_ok=True)
    lines = [
        "# 线 Q：1600 nm 协议 A GH5 频率桥接（2026-11-02）\n",
        "相同量子统计：`D(0.5)S(0.3 e^{iθ})|0⟩`，`R=1.273`，抽象 `g=0.005 Å^{-1}`。",
        f"`ω=0.775`（显式传入，**没有**使用 `pf.OMEGA0=0.05`）。",
        "1+4+1 在 1600 nm 下约 32 fs 耦合开关，**不是** 25 fs 高斯脉冲。",
        "GH5 ≠ 精确 QED ≠ Cohen MTEF。禁止把 4.98 / 110.6° 抄过来。\n",
        f"**结论：{rec.get('answer', rec.get('status'))}**\n",
        f"- verdict: `{rec.get('verdict')}`",
        f"- amp_ratio π/0: {rec.get('h2_main', {}).get('amp_ratio_pi_over_0')}",
        f"- ρ=|A|/|B|: {rec.get('rho_absA_over_absB')}",
        f"- arg(A/B): {rec.get('arg_A_over_B_deg')}",
        f"- π/4 留出: {rec.get('leaveout_pi4')}",
        "\n未做：协议 B、纯 BSV、精确 QED、全 BZ、SOC/Δ 扫描、git push。\n",
    ]
    # Do not overwrite the hand-written protocol-A report.
    path = DOCS / "线Q_1600nm_协议A_GH5自动摘要_20261102.md"
    path.write_text("\n".join(lines), encoding="utf-8")


def launch_tags(tags, n_steps_override: int | None = None) -> dict:
    t0 = time.perf_counter()
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
        if n_steps_override:
            cmd.extend(["--n-steps", str(int(n_steps_override))])
        progress(f"launch {' '.join(cmd[-4:])}")
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
    return {
        "returncodes": {tag: (procs[tag].returncode if tag in procs else 0) for tag in tags},
        "driver_wall_s": time.perf_counter() - t0,
    }


def drive(mode: str, n_steps: int | None) -> int:
    hashes = source_hashes()
    bad = hash_abort(hashes)
    if bad:
        dump(OUT / "protocolA_gh5.json", {"status": "hash_abort", "hash_mismatch": bad, "stopped": True})
        progress("hash abort, no propagation")
        return 3
    OUT.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    res = collect_resources()
    if not res.get("resource_match_ok"):
        dump(OUT / "protocolA_gh5.json", {"status": "resource_mismatch", "resources": res, "stopped": True})
        progress("resource mismatch, no propagation")
        return 3
    tags = tags_for_mode(mode)
    launch = launch_tags(tags, n_steps)
    progress(f"jobs return {launch['returncodes']} driver_wall={launch['driver_wall_s']:.1f}s")
    if not all(job_finished_ok(tag) for tag in tags):
        rec = {"status": "timeout_partial", "mode": mode, "stopped": True, **launch,
               "job_status": {tag: None if load_job(tag) is None else load_job(tag).get("status")
                              for tag in tags}}
        dump(OUT / f"{mode}.json", rec)
        return 1
    if mode == "smoke":
        rec = analyze_smoke()
        progress(f"smoke analyze status={rec.get('status')} pass={rec.get('smoke_pass')}")
        return 0 if rec.get("smoke_pass") else 1
    rec = analyze_fourangle()
    progress(f"analyze done verdict={rec.get('verdict')}")
    return 0 if rec.get("status") == "ok" else 1


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=("smoke", "fourangle", "pi4"), default="")
    ap.add_argument("--job", type=str, default="")
    ap.add_argument("--analyze-only", action="store_true")
    ap.add_argument("--resources-only", action="store_true")
    ap.add_argument("--n-steps", type=int, default=0)
    args = ap.parse_args(argv)
    OUT.mkdir(parents=True, exist_ok=True)
    n_steps = int(args.n_steps) if args.n_steps else None
    if args.resources_only:
        rec = collect_resources()
        print(json.dumps(jsonable({
            "status": rec.get("status"),
            "resource_match_ok": rec.get("resource_match_ok"),
            "omega0": rec.get("omega0"),
            "period_fs": None if "envelope" not in rec else rec["envelope"]["period_fs"],
            "t_total_fs": None if "envelope" not in rec else rec["envelope"]["t_total_fs"],
            "nbar": rec.get("nbar"),
            "R_from_M": rec.get("R_from_M"),
            "used_pf_OMEGA0": False,
            "hash_abort": rec.get("hash_abort"),
        }), ensure_ascii=False, indent=2))
        return 0 if rec.get("status") == "resources" and rec.get("resource_match_ok") else 3
    if args.job:
        rec = run_one_job(args.job, n_steps)
        return 0 if rec.get("status") in ("ok", "timeout_partial") else 1
    if args.analyze_only:
        if args.mode == "smoke":
            rec = analyze_smoke()
        else:
            rec = analyze_fourangle()
        print(json.dumps(jsonable({
            "status": rec.get("status"), "verdict": rec.get("verdict"),
            "smoke_pass": rec.get("smoke_pass"),
        }), ensure_ascii=False, indent=2))
        return 0 if rec.get("status") == "ok" else 1
    mode = args.mode or "smoke"
    return drive(mode, n_steps)


if __name__ == "__main__":
    raise SystemExit(main())
