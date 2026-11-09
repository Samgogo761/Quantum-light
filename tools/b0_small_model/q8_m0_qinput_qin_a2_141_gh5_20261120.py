#!/usr/bin/env python3
"""Q1a GH5/MTEF vs frozen exact QED on Qin_a2_1_4_1.

Same-alpha C vs DS, N=+/-1, shared k/-k, GH5 Wigner nodes.
Does not rerun QED. Does not start Cn, theta=pi, SV, G1/H2, D2,
Floquet, extra k, nmax scan, or server. Integer Fourier is not
certified HHG. GH5 agreement is not classical light.
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

import q8_m0_qinput_dynamics_20261026 as dyn
import q8_m0_qinput_shared_kk_ds_odd_20261102 as odd
import q8_m0_qinput_shared_kk_dyn_preflight_20261030 as pf
import q8_m0_qinput_shared_kk_mtef_map_20261108 as mp
import q8_m0_qinput_shared_kk_static_20261029 as sh
from q8_m0_model import R0, build_geometry, sha256_file

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results/q_line/qinput_qin_a2_141_gh5_20261120"
FIG = OUT / "figures"
QED_DIR = ROOT / "results/q_line/qinput_qin_a2_141_q1a_20261120"
QED_JSON = QED_DIR / "q1a.json"
DOCS = ROOT / "docs/04_BSV量子光"
TASK_TAG = "20261120"
RECORD_DATE = "2026-10-22"
PROTOCOL = "Qin_a2_1_4_1"
ORDER = 5
N_NODES = ORDER * ORDER
N_FULL = 11154
SAMPLE_EVERY = 32
DRIVER_WALL_S = 900.0
JOB_WALL_S = 870.0
PREFLIGHT_WALL_S = 180.0
JOBS = odd.JOBS
JOB_MAP = {tag: (kind, neel) for tag, kind, neel in JOBS}
EXPECTED_HASHES = {
    "q8_m0_qinput_dynamics_20261026.py":
        "bf6d4e407bf55f1e6ebc3767f1b2bd6075ac7ef564f195abc1217c7d51671f94",
    "q8_m0_qinput_minimal_20261020.py":
        "b60c3b7e53e413e614d4a8ec784956f4d16001a906465314dea963a17ae9bc97",
    "q8_m0_qinput_shared_kk_static_20261029.py":
        "872347c94656ef74d2d0e3cce961e8fb1f1b47d30610bf0bb534b9ad80925345",
    "q8_m0_qinput_shared_kk_dyn_preflight_20261030.py":
        "1640a0598b9731cd9468b7156ff632401f245438af8fc8d0f8e13cd726d23540",
    "q8_m0_qinput_shared_kk_ds_odd_20261102.py":
        "cfc6d8c5dee162ab53e5d2c0344fc3b35b03cd77621fbfe8eb963069ecb7c41c",
    "q8_m0_model.py":
        "5d53090cabb60238c6c16a74b92ddc061d5adf2d8081e48b46b9bdf0089261d3",
    "q8_m0_qinput_shared_kk_mtef_map_20261108.py":
        "cc69ae6aa963cb4ac8af9bcc308c24950c6db01dc9b94ca6a971253f4bf1b34c",
    "q8_m0_qinput_shared_kk_mtef_compare_20261109.py":
        "391a75b37470e2670d9e782bf580191fa9c67d786515fd17df5f2569b192de19",
    "q8_m0_qinput_shared_kk_mtef_gh5_20261110.py":
        "3af3c05b33c304238b252d601124a2f6016e162b234aaa1821221e7aac4a3c14",
    "q8_m0_qinput_qin_a2_141_q1a_20261120.py":
        "4a148766f46091dfc8f3ebefa2252b3ebb0b447e3bf73220755ad543a81df04b",
}
FROZEN = tuple(HERE / name for name in EXPECTED_HASHES)
REL_REPRO = 0.20
PHASE_REPRO_DEG = 20.0


def jsonable(x):
    return odd.jsonable(x)


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


def cz(pack) -> complex:
    return complex(pack["re"], pack["im"])


def pack_c(z):
    z = np.asarray(z, complex)
    if z.ndim == 0:
        return pf.cpack(complex(z))
    return [pf.cpack(complex(v)) for v in z]


def wrap(phi: float) -> float:
    return float((phi + np.pi) % (2.0 * np.pi) - np.pi)


def angdiff(a: float, b: float) -> float:
    return wrap(float(a) - float(b))


def sample_js_for(n_steps: int):
    js = list(range(0, n_steps, SAMPLE_EVERY))
    if not js or js[-1] != n_steps - 1:
        js.append(n_steps - 1)
    return js


def make_env():
    return dyn.Envelope(pf.OMEGA0, ramp_cycles=1.0, flat_cycles=4.0)


def make_bundle():
    geo = build_geometry()
    k = geo.k_cart(sh.K_FRAC)
    models, grounds = {}, {}
    for neel in (1, -1):
        models[neel] = sh.build_shared(
            geo, R0, k, neel, n_max=0, omega0=pf.OMEGA0, g=pf.G_ABS)
        grounds[neel] = odd.ground_pair_neel(geo, k, neel)
    env = make_env()
    return geo, k, models, grounds, env


def one_traj(model, ek0, emk0, X0, P0, env, n_steps, js, clock):
    """Qin_a2_1_4_1 Ehrenfest. dt = t_total/n_steps, not t_ramp/n_steps."""
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
        ek, emk, X, P, _ = mp.step_ehrenfest(ek, emk, X, P, model, s, dt, kick=True)
        nk = abs(np.linalg.norm(ek) - 1.0)
        nm = abs(np.linalg.norm(emk) - 1.0)
        max_n = max(max_n, nk, nm)
        if j in want:
            t = (j + 1) * dt
            jvec = mp.j_total(ek, emk, model, env.s(t), X)
            j_rows.append(jvec)
            t_rows.append(t)
    return np.asarray(j_rows, complex), np.asarray(t_rows, float), max_n, X, P


def ckpt_path(tag: str) -> Path:
    return OUT / f"ckpt_{tag}.npz"


def job_path(tag: str) -> Path:
    return OUT / f"job_{tag}.json"


def series_path(tag: str) -> Path:
    return OUT / f"series_{tag}.npz"


def save_ckpt(tag, kind, neel, n_done, n_nodes, acc, wsum, max_n, t_arr) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    acc = np.asarray(acc, complex)
    np.savez_compressed(
        ckpt_path(tag),
        acc_re=acc.real, acc_im=acc.imag, wsum=float(wsum),
        n_done=int(n_done), n_nodes=int(n_nodes), max_n=float(max_n),
        t=np.asarray(t_arr, float), kind=str(kind), neel=int(neel),
    )


def load_ckpt(tag: str):
    p = ckpt_path(tag)
    if not p.is_file():
        return None
    z = np.load(p, allow_pickle=False)
    acc = np.asarray(z["acc_re"], float) + 1j * np.asarray(z["acc_im"], float)
    return {
        "n_done": int(z["n_done"]),
        "n_nodes": int(z["n_nodes"]),
        "acc": acc,
        "wsum": float(z["wsum"]),
        "max_n": float(z["max_n"]),
        "t": np.asarray(z["t"], float),
    }


def load_job(tag: str):
    p = job_path(tag)
    if not p.is_file():
        return None
    rec = json.loads(p.read_text(encoding="utf-8"))
    return rec


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


def window_mask(t, t0, t1):
    return (t >= t0 - 1e-12) & (t <= t1 + 1e-12)


def rect_jq(t, j, omega0, t0, t1, qs):
    t = np.asarray(t, float)
    j = np.asarray(j, complex)
    m = window_mask(t, t0, t1)
    tw, jw = t[m], j[m]
    span = float(t1 - t0)
    out = []
    for q in qs:
        ker = np.exp(1j * float(q) * omega0 * tw)
        slot = np.array([np.trapezoid(ker * jw[:, a], tw) for a in range(j.shape[1])])
        out.append(slot / span)
    return np.asarray(out, complex)


def run_one_job(tag: str) -> dict:
    if tag not in JOB_MAP:
        raise ValueError(tag)
    kind, neel = JOB_MAP[tag]
    hashes = source_hashes()
    bad = hash_abort(hashes)
    if bad:
        rec = {"tag": tag, "status": "hash_abort", "hash_mismatch": bad, "stopped": True}
        dump(job_path(tag), rec)
        return rec
    if job_finished_ok(tag):
        rec = load_job(tag)
        progress(f"{tag} already complete n_run={rec.get('n_run')}")
        return rec
    OUT.mkdir(parents=True, exist_ok=True)
    _, _, models, grounds, env = make_bundle()
    nodes = mp.gauss_hermite_nodes(kind, ORDER)["nodes"]
    ek0, emk0, mass = grounds[neel]
    model = models[neel]
    js = sample_js_for(N_FULL)
    ck = load_ckpt(tag)
    i0, acc, wsum, max_n, t_arr = 0, None, 0.0, 0.0, None
    if ck is not None:
        i0, acc, wsum, max_n, t_arr = ck["n_done"], ck["acc"], ck["wsum"], ck["max_n"], ck["t"]
        progress(f"{tag} resume n_done={i0}/{len(nodes)}")
    clock = dyn.WallClock(JOB_WALL_S)
    progress(f"{tag} start kind={kind} neel={neel} nodes={len(nodes)} n_steps={N_FULL}")
    timed_out = False
    timeout_msg = None
    try:
        for i, node in enumerate(nodes):
            if i < i0:
                continue
            j_rows, t_rows, nrm, _, _ = one_traj(
                model, ek0, emk0, node["X"], node["P"], env, N_FULL, js, clock)
            w = float(node["weight"])
            acc = w * j_rows if acc is None else acc + w * j_rows
            wsum += w
            max_n = max(max_n, nrm)
            t_arr = t_rows
            save_ckpt(tag, kind, neel, i + 1, len(nodes), acc, wsum, max_n, t_arr)
            if (i + 1) == len(nodes) or (i + 1) % 5 == 0:
                progress(f"{tag} node {i + 1}/{len(nodes)} wall={clock.elapsed():.1f}s")
    except TimeoutError as exc:
        timed_out = True
        timeout_msg = str(exc)
        progress(f"{tag} timeout n_done={0 if acc is None else load_ckpt(tag)['n_done']} {timeout_msg}")
    n_done = 0 if acc is None else int(load_ckpt(tag)["n_done"] if ckpt_path(tag).is_file() else 0)
    if timed_out or n_done < len(nodes):
        rec = {
            "tag": tag, "kind": kind, "neel": int(neel), "order": ORDER,
            "status": "timeout_partial", "n_nodes": len(nodes), "n_done": n_done,
            "weight_sum": wsum, "max_abs_norm_minus_1": float(max_n),
            "wall_s": clock.elapsed(), "timeout_msg": timeout_msg,
            "n_max": 0, "protocol": PROTOCOL, "n_steps": N_FULL,
            "dt": env.t_total / N_FULL, "stopped": True,
            "electron": mass,
        }
        dump(job_path(tag), rec)
        return rec
    avg = acc / (wsum + 1e-30)
    save_series(tag, t_arr, avg)
    rec = {
        "tag": tag, "kind": kind, "neel": int(neel), "order": ORDER,
        "status": "ok", "n_nodes": len(nodes), "n_run": len(nodes),
        "n_done": len(nodes), "weight_sum": wsum,
        "max_abs_norm_minus_1": float(max_n),
        "wall_s": clock.elapsed(), "n_max": 0, "protocol": PROTOCOL,
        "n_steps": N_FULL, "dt": env.t_total / N_FULL,
        "n_samples": int(len(t_arr)), "t0": float(t_arr[0]), "t_end": float(t_arr[-1]),
        "electron": mass, "stopped": True,
        "wigner": mp.analytic_xp(kind),
        "ds_not_rescaled_coherent": True,
    }
    dump(job_path(tag), rec)
    progress(f"{tag} done wall={clock.elapsed():.1f}s")
    return rec


def preflight() -> dict:
    hashes = source_hashes()
    bad = hash_abort(hashes)
    rec = {
        "task_tag": TASK_TAG, "record_date": RECORD_DATE, "protocol": PROTOCOL,
        "status": "preflight", "source_hashes": hashes, "hash_abort": bad,
        "stopped": True,
    }
    if bad:
        rec["status"] = "hash_abort"
        dump(OUT / "preflight.json", rec)
        progress("hash abort, no preflight traj")
        return rec
    _, _, models, grounds, env = make_bundle()
    node = mp.gauss_hermite_nodes("ds", ORDER)["nodes"][0]
    ek, emk, _ = grounds[1]
    js = sample_js_for(N_FULL)
    mom_c = mp.analytic_xp("coherent")
    mom_d = mp.analytic_xp("ds")
    clock = dyn.WallClock(PREFLIGHT_WALL_S)
    t0 = time.perf_counter()
    one_traj(models[1], ek, emk, node["X"], node["P"], env, N_FULL, js, clock)
    per = time.perf_counter() - t0
    est_job = per * N_NODES
    est_serial = est_job * 4.0
    rec.update({
        "per_traj_s": per,
        "n_nodes": N_NODES,
        "n_steps": N_FULL,
        "dt": env.t_total / N_FULL,
        "t_total": env.t_total,
        "flat_cycles": 4.0,
        "est_one_job_s": est_job,
        "est_serial_s": est_serial,
        "est_four_parallel_s": est_job,
        "driver_wall_s": DRIVER_WALL_S,
        "under_cap_if_four_parallel": bool(est_job < DRIVER_WALL_S),
        "serial_may_exceed_cap": bool(est_serial > DRIVER_WALL_S),
        "will_still_run_four_parallel": True,
        "did_not_shorten_platform_or_drop_to_GH3": True,
        "var_X_coherent": mom_c["var_X"],
        "var_X_ds": mom_d["var_X"],
        "ds_uses_e_minus_2r": True,
        "n_max": 0,
        "qed_archive": str(QED_JSON.as_posix()),
        "qed_not_rerun": True,
    })
    dump(OUT / "preflight.json", rec)
    progress(
        f"preflight per_traj={per:.3f}s est_job={est_job:.1f}s "
        f"est_serial={est_serial:.1f}s cap={DRIVER_WALL_S} "
        f"parallel_under={rec['under_cap_if_four_parallel']}"
    )
    return rec


def slot_pack(jq):
    return {
        "Jx": pack_c(jq[:, 0]),
        "Jy": pack_c(jq[:, 1]),
        "abs": [float(np.linalg.norm(v)) for v in jq],
        "phase_x": [float(np.angle(v[0])) for v in jq],
        "phase_y": [float(np.angle(v[1])) for v in jq],
    }


def pack_win(name, t, omega0, T0, w0, w1, arrs):
    slots = {key: slot_pack(rect_jq(t, arr, omega0, w0, w1, np.arange(1, 7, dtype=float)))
             for key, arr in arrs.items()}
    return {
        "name": name, "t0": w0, "t1": w1,
        "t0_over_T0": w0 / T0, "t1_over_T0": w1 / T0,
        "complete": True,
        "convention": "J_q = 1/(t1-t0) int exp(+i q omega0 t) J(t) dt, rectangular",
        "not_certified_HHG": True,
        "slots": slots,
    }


def h2_from_slots(slots, key):
    jx = cz(slots[key]["Jx"][1])
    jy = cz(slots[key]["Jy"][1])
    return jx, jy, float(slots[key]["abs"][1]), float(slots[key]["phase_x"][1])


def qed_h2(qed, win, key):
    return h2_from_slots(qed["windows"][win]["slots"], key)


def compare_delta(jx_q, jy_q, jx_m, jy_m):
    vq = np.array([jx_q, jy_q], complex)
    vm = np.array([jx_m, jy_m], complex)
    den = max(float(np.linalg.norm(vq)), 1e-30)
    rel = float(np.linalg.norm(vq - vm) / den)
    dphi = angdiff(np.angle(jx_m), np.angle(jx_q))
    dphi_deg = abs(dphi) * 180.0 / np.pi
    mag_q = float(np.linalg.norm(vq))
    mag_m = float(np.linalg.norm(vm))
    if rel < REL_REPRO and dphi_deg < PHASE_REPRO_DEG:
        kind = "reproduced_within_thresholds"
    elif rel < REL_REPRO and dphi_deg >= PHASE_REPRO_DEG:
        kind = "magnitude_close_phase_mismatch"
    elif rel >= REL_REPRO and dphi_deg < PHASE_REPRO_DEG:
        kind = "phase_close_magnitude_mismatch"
    else:
        kind = "not_reproduced"
    return {
        "abs_QED": mag_q, "abs_MTEF": mag_m,
        "rel_vec": rel,
        "phase_x_QED": float(np.angle(jx_q)),
        "phase_x_MTEF": float(np.angle(jx_m)),
        "dphi_x_rad": dphi,
        "dphi_x_deg_abs": dphi_deg,
        "Jx_QED": pack_c(jx_q), "Jy_QED": pack_c(jy_q),
        "Jx_MTEF": pack_c(jx_m), "Jy_MTEF": pack_c(jy_m),
        "kind": kind,
    }


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
    return str(path.as_posix())


def map_xy(z, lim, x0, y0, w, h):
    px = x0 + (z.real / lim * 0.5 + 0.5) * w
    py = y0 + (0.5 - z.imag / lim * 0.5) * h
    return px, py


def arrow(z, lim, x0, y0, w, h, color, label, lx, ly):
    cx, cy = x0 + w / 2, y0 + h / 2
    px, py = map_xy(z, lim, x0, y0, w, h)
    return (
        f'<line x1="{cx:.1f}" y1="{cy:.1f}" x2="{px:.1f}" y2="{py:.1f}" '
        f'stroke="{color}" stroke-width="2"/>'
        f'<circle cx="{px:.1f}" cy="{py:.1f}" r="4" fill="{color}"/>'
        f'<text x="{lx}" y="{ly}" font-size="11" fill="{color}">{label}</text>'
    )


def figure_h2_plane(qed_c, qed_d, qed_dj, m_c, m_d, m_dj, path):
    zs = [qed_c, qed_d, qed_dj, m_c, m_d, m_dj]
    lim = max(abs(z) for z in zs) * 1.25
    x0, y0, w, h = 90, 40, 740, 430
    body = svg_axes(x0, y0, w, h, "Re Jx (H2 odd)", "Im Jx (H2 odd)",
                    "H2 odd Jx: QED vs GH5/MTEF")
    cx, cy = x0 + w / 2, y0 + h / 2
    body += f'<line x1="{x0}" y1="{cy:.1f}" x2="{x0+w}" y2="{cy:.1f}" stroke="#8a9096"/>'
    body += f'<line x1="{cx:.1f}" y1="{y0}" x2="{cx:.1f}" y2="{y0+h}" stroke="#8a9096"/>'
    body += arrow(qed_c, lim, x0, y0, w, h, "#4c6b8a", "QED C", 80, 36)
    body += arrow(qed_d, lim, x0, y0, w, h, "#c45c26", "QED DS", 150, 36)
    body += arrow(qed_dj, lim, x0, y0, w, h, "#2f9e44", "QED Δ", 230, 36)
    body += arrow(m_c, lim, x0, y0, w, h, "#7aa2c4", "MTEF C", 80, 52)
    body += arrow(m_d, lim, x0, y0, w, h, "#e08a5a", "MTEF DS", 150, 52)
    body += arrow(m_dj, lim, x0, y0, w, h, "#6fc27a", "MTEF Δ", 230, 52)
    body += (
        '<text x="460" y="548" font-size="10" fill="#5c5148">'
        "same alpha; not same nbar; not certified HHG; GH5 agreement is not classical light</text>"
    )
    return write_svg(path, body)


def figure_windows(rows, path):
    """rows: list of (label, z, color)."""
    lim = max(abs(z) for _, z, _ in rows) * 1.25
    x0, y0, w, h = 90, 40, 740, 430
    body = svg_axes(x0, y0, w, h, "Re ΔJx (H2 odd)", "Im ΔJx (H2 odd)",
                    "H2 ΔJ_odd three windows")
    cx, cy = x0 + w / 2, y0 + h / 2
    body += f'<line x1="{x0}" y1="{cy:.1f}" x2="{x0+w}" y2="{cy:.1f}" stroke="#8a9096"/>'
    body += f'<line x1="{cx:.1f}" y1="{y0}" x2="{cx:.1f}" y2="{y0+h}" stroke="#8a9096"/>'
    lx = 80
    for label, z, color in rows:
        body += arrow(z, lim, x0, y0, w, h, color, label, lx, 36 if lx < 400 else 52)
        lx += 130
    body += (
        '<text x="460" y="548" font-size="10" fill="#5c5148">'
        "QED |Δ| stable ~0.8%; complex δ~25%. Do not upgrade |Δ| stability to spectral convergence.</text>"
    )
    return write_svg(path, body)


def figure_cont(q, yc, yd, ym, path):
    xmin, xmax = float(q[0]), float(q[-1])
    ys = np.concatenate([yc, yd, ym])
    ymin, ymax = 0.0, float(np.max(ys)) * 1.12
    x0, y0, w, h = 70, 40, 800, 430
    body = svg_axes(x0, y0, w, h, "Ω / ω0", "|ΔJ^o(Ω)|",
                    "Main-window continuum |ΔJ_odd|")

    def poly(xs, ys_, color, sw):
        pts = []
        for x, y in zip(xs, ys_):
            px = x0 + (float(x) - xmin) / (xmax - xmin) * w
            py = y0 + (1.0 - (float(y) - ymin) / max(ymax - ymin, 1e-30)) * h
            pts.append(f"{px:.2f},{py:.2f}")
        return (f'<polyline fill="none" stroke="{color}" stroke-width="{sw}" '
                f'points="{" ".join(pts)}"/>')

    body += poly(q, yc, "#2f9e44", 1.4)
    body += poly(q, ym, "#6fc27a", 1.4)
    for n in range(1, 7):
        px = x0 + (n - xmin) / (xmax - xmin) * w
        body += (f'<line x1="{px:.1f}" y1="{y0}" x2="{px:.1f}" y2="{y0+h}" '
                 f'stroke="#8a9096" stroke-dasharray="3 4" stroke-width="0.8"/>')
    body += '<text x="80" y="36" font-size="11" fill="#2f9e44">QED Δ</text>'
    body += '<text x="150" y="36" font-size="11" fill="#6fc27a">MTEF Δ</text>'
    return write_svg(path, body, width=920, height=560)


def write_reports(rec: dict) -> None:
    DOCS.mkdir(parents=True, exist_ok=True)
    h2 = rec["h2_main"]
    qed_ro = rec["qed_readonly_delta_complex"]
    cmpk = rec["compare_main"]
    v = rec["verdict"]
    ans = rec["answer"]
    full = []
    full.append("# 量子光多周期磁序响应：Q1a 的 GH5/MTEF 对照\n")
    full.append(f"`{TASK_TAG}`；执行 {RECORD_DATE}。协议 `{PROTOCOL}`。")
    full.append("冻结 Q1a 四路精确 QED，不重跑。本轮仅 GH5/MTEF，n_max=0 经典 X。")
    full.append("Cα α=0.5 与 DS D(0.5)S(0.3,θ=0)|0⟩，各 ±N，共享 k/−k。`stopped = true`。\n")
    full.append(f"**结论：{ans}**\n")
    full.append("| 量 | QED | GH5/MTEF |\n|---|---|---|")
    full.append(
        f"| 主窗 H2 \\(\\lvert\\Delta\\mathcal J_2^o\\rvert\\) | "
        f"{h2['abs_dJ_QED']:.3e} | {h2['abs_dJ_MTEF']:.3e} |"
    )
    full.append(
        f"| 主窗 arg(ΔJx) | {h2['phase_x_dJ_QED']:.3f} rad | {h2['phase_x_dJ_MTEF']:.3f} rad |"
    )
    full.append(
        f"| \\(\\lvert\\Delta_\\mathrm{{QED}}-\\Delta_\\mathrm{{MTEF}}\\rvert/"
        f"\\lvert\\Delta_\\mathrm{{QED}}\\rvert\\) | — | {cmpk['rel_vec']:.3f} |"
    )
    full.append(
        f"| 相位偏差 \\(|\\Delta\\phi_x|\\) | — | {cmpk['dphi_x_deg_abs']:.1f}° |"
    )
    full.append(
        f"| H2 \\(\\lvert\\mathcal J_2^{{o,C}}\\rvert\\) | "
        f"{h2['abs_C_QED']:.3e} | {h2['abs_C_MTEF']:.3e} |"
    )
    full.append(
        f"| H2 \\(\\lvert\\mathcal J_2^{{o,DS}}\\rvert\\) | "
        f"{h2['abs_DS_QED']:.3e} | {h2['abs_DS_MTEF']:.3e} |"
    )
    full.append(
        f"| C 相位 | {h2['phase_x_C_QED']:.3f} | {h2['phase_x_C_MTEF']:.3f} |"
    )
    full.append(
        f"| DS 相位 | {h2['phase_x_DS_QED']:.3f} | {h2['phase_x_DS_MTEF']:.3f} |"
    )
    full.append("\nQED 只读补报（本轮不改写 Q1a 档案）：\n")
    full.append("| 窗 | \\(\\lvert\\Delta\\mathcal J_2^o\\rvert\\) | arg(ΔJx) |\n|---|---|---|")
    full.append(
        f"| 前 [1,3]T0 | {qed_ro['abs_front']:.3e} | {qed_ro['phase_front']:.3f} rad |"
    )
    full.append(
        f"| 主 [2,4]T0 | {qed_ro['abs_main']:.3e} | {qed_ro['phase_main']:.3f} rad |"
    )
    full.append(
        f"| 后 [3,5]T0 | {qed_ro['abs_back']:.3e} | {qed_ro['phase_back']:.3f} rad |"
    )
    full.append(
        f"\n模长相对变化 0.8%，但 "
        f"\\(\\delta_\\mathrm{{complex}}=\\lvert\\Delta Jx_\\mathrm{{rear}}-\\Delta Jx_\\mathrm{{front}}\\rvert"
        f"/\\lvert\\Delta Jx_\\mathrm{{main}}\\rvert={qed_ro['delta_complex']:.3f}\\)（约 25%）。"
        "禁止把“模长稳”升级为“复谱已收敛”。\n"
    )
    full.append("## 解释边界\n")
    full.append("- GH5 一致 ≠ 输入光是经典态（压缩态可有正 Wigner，通常无正 P 分布）。")
    full.append("- GH5 不一致 ≠ 输出纠缠。")
    full.append("- 同 α 对照 ≠ 同平均光子数对照。")
    full.append("- 两周期矩形 Fourier ≠ 已认证 HHG。")
    full.append("- 磁序奇电流相对偶背景仍约 \(10^{-6}\)。")
    full.append("- 低占据 \(A^*_\\mathrm{mean}\\approx0.0202\) 不得与经典 \(A^*=0.8\) 拼接。\n")
    full.append("本轮未启动 Cn、θ=π、SV、新 QED、G1/H2、额外 k、nmax、D2、Floquet 或服务器。\n")
    full.append("完成后停机。不自动扩大参数范围。\n")
    (DOCS / "线Q_量子光多周期磁序响应_Q1a_GH5_20261120.md").write_text(
        "\n".join(full) + "\n", encoding="utf-8")
    one = []
    one.append("# 量子光多周期磁序响应：Q1a GH5/MTEF（一页）\n")
    one.append(f"`{TASK_TAG}`；执行 {RECORD_DATE}。协议 `{PROTOCOL}`。冻结 Q1a QED，不重跑。`stopped = true`。\n")
    one.append(f"**{ans}**\n")
    one.append("| 量 | 结果 |\n|---|---|")
    one.append(f"| 对照判据 | rel_vec={cmpk['rel_vec']:.3f}（阈 0.20）；\\|Δφx\\|={cmpk['dphi_x_deg_abs']:.1f}°（阈 20°） |")
    one.append(f"| 分类 | `{cmpk['kind']}` / verdict `{v}` |")
    one.append(f"| 主窗 \\(\\lvert\\Delta\\mathcal J_2^o\\rvert\\) QED / MTEF | {h2['abs_dJ_QED']:.3e} / {h2['abs_dJ_MTEF']:.3e} |")
    one.append(f"| QED δ_complex（前后复差） | {qed_ro['delta_complex']:.3f}（模长变化 0.8%） |")
    one.append("| 同 α ≠ 同 ⟨n⟩ | C 0.250 vs DS 0.343 |")
    one.append("| 服务器 | 现在不需要 |\n")
    one.append("**外审两问：**\n")
    one.append(f"1. Q1a 的 H2 复响应差能否由相同 Wigner 统计的半经典电子–光场动力学重现？**{rec['answer_short']}**")
    one.append("2. 模长稳是否等于复谱收敛？**否**（δ_complex≈25%）。\n")
    one.append("停机。不自动开 Cn、θ=π、SV、新 QED、G1/H2 或服务器。\n")
    (DOCS / "线Q_量子光多周期磁序响应_Q1a_GH5_一页_20261120.md").write_text(
        "\n".join(one) + "\n", encoding="utf-8")


def analyze() -> dict:
    hashes = source_hashes()
    bad = hash_abort(hashes)
    if bad:
        rec = {"status": "hash_abort", "hash_mismatch": bad, "stopped": True}
        dump(OUT / "gh5.json", rec)
        return rec
    missing = [tag for tag, _, _ in JOBS if not job_finished_ok(tag)]
    if missing:
        rec = {"status": "missing_jobs", "missing": missing, "stopped": True,
               "job_status": {tag: None if load_job(tag) is None else load_job(tag).get("status")
                              for tag, _, _ in JOBS}}
        dump(OUT / "gh5.json", rec)
        return rec
    if not QED_JSON.is_file():
        rec = {"status": "missing_qed_archive", "path": str(QED_JSON), "stopped": True}
        dump(OUT / "gh5.json", rec)
        return rec
    qed = json.loads(QED_JSON.read_text(encoding="utf-8"))
    env = make_env()
    T0 = env.period
    omega0 = pf.OMEGA0
    series = {}
    for tag, _, _ in JOBS:
        t, j = load_series(tag)
        series[tag] = {"t": t, "j": j}
    t = series["coh_p"]["t"]
    for tag, _, _ in JOBS:
        if not np.allclose(series[tag]["t"], t, rtol=0, atol=1e-9):
            rec = {"status": "time_grid_mismatch", "tag": tag, "stopped": True}
            dump(OUT / "gh5.json", rec)
            return rec
    even_c, odd_c = odd.even_odd(series["coh_p"]["j"], series["coh_m"]["j"])
    even_d, odd_d = odd.even_odd(series["ds_p"]["j"], series["ds_m"]["j"])
    d_odd = odd_d - odd_c
    d_even = even_d - even_c
    arrs = {
        "J_odd_coh": odd_c, "J_odd_ds": odd_d, "dJ_odd": d_odd,
        "J_even_coh": even_c, "J_even_ds": even_d, "dJ_even": d_even,
    }
    win_main = (2.0 * T0, 4.0 * T0)
    win_front = (1.0 * T0, 3.0 * T0)
    win_back = (3.0 * T0, 5.0 * T0)
    main = pack_win("plateau_center_2T", t, omega0, T0, *win_main, arrs)
    front = pack_win("front_2T", t, omega0, T0, *win_front, arrs)
    back = pack_win("back_2T", t, omega0, T0, *win_back, arrs)
    qs_c = np.linspace(0.0, 6.5, 326)
    cont_m = rect_jq(t, d_odd, omega0, *win_main, qs_c)
    abs_cont_m = np.array([float(np.linalg.norm(v)) for v in cont_m], float)
    abs_cont_q = np.array(qed["continuum"]["abs_dJodd"], float)
    q_cont = np.array(qed["continuum"]["q"], float)
    peak_i = int(np.argmax(abs_cont_m[1:])) + 1
    peak_q_m = float(qs_c[peak_i])

    def h2pack(winobj):
        c = h2_from_slots(winobj["slots"], "J_odd_coh")
        d = h2_from_slots(winobj["slots"], "J_odd_ds")
        dj = h2_from_slots(winobj["slots"], "dJ_odd")
        return c, d, dj

    m_main_c, m_main_d, m_main_dj = h2pack(main)
    m_front_c, m_front_d, m_front_dj = h2pack(front)
    m_back_c, m_back_d, m_back_dj = h2pack(back)
    q_main_c = qed_h2(qed, "main", "J_odd_coh")
    q_main_d = qed_h2(qed, "main", "J_odd_ds")
    q_main_dj = qed_h2(qed, "main", "dJ_odd")
    q_front_dj = qed_h2(qed, "front", "dJ_odd")
    q_back_dj = qed_h2(qed, "back", "dJ_odd")
    cmp_main = compare_delta(q_main_dj[0], q_main_dj[1], m_main_dj[0], m_main_dj[1])
    cmp_front = compare_delta(q_front_dj[0], q_front_dj[1], m_front_dj[0], m_front_dj[1])
    cmp_back = compare_delta(q_back_dj[0], q_back_dj[1], m_back_dj[0], m_back_dj[1])
    dphi_cd_q = abs(angdiff(q_main_c[3], q_main_d[3]))
    dphi_cd_m = abs(angdiff(m_main_c[3], m_main_d[3]))
    near_pi_q = abs(dphi_cd_q - np.pi)
    near_pi_m = abs(dphi_cd_m - np.pi)
    qed_delta_complex = abs(q_back_dj[0] - q_front_dj[0]) / max(abs(q_main_dj[0]), 1e-30)
    mtef_delta_complex = abs(m_back_dj[0] - m_front_dj[0]) / max(abs(m_main_dj[0]), 1e-30)
    kind = cmp_main["kind"]
    if kind == "reproduced_within_thresholds":
        verdict = "h2_complex_difference_reproduced_by_GH5_not_certified_HHG"
        answer = (
            "是：在预设阈值内，GH5/MTEF 重现了 Q1a 主窗 H2 磁序奇复差的模长与相位。"
            "这只说明该观测量可由相应半经典相空间动力学近似描述，不等于输入光是经典态，"
            "也不等于已认证 HHG 或已分离的非经典优势。"
        )
        answer_short = "是（阈值内重现，非认证 HHG）"
    elif kind == "magnitude_close_phase_mismatch":
        verdict = "h2_magnitude_close_phase_not_reproduced"
        answer = (
            "否：主窗 H2 差向量模长接近，但复相位偏差超过 20°。"
            "不能把模长相近判为一致。Q1a 的近反相结构未被半经典动力学定量重现。"
        )
        answer_short = "否（模长近、相位不重现）"
    elif kind == "phase_close_magnitude_mismatch":
        verdict = "h2_phase_close_magnitude_not_reproduced"
        answer = (
            "否：主窗 H2 差向量相位接近，但模长相对偏差超过 0.20。"
            "不能仅凭相位相近宣称重现。"
        )
        answer_short = "否（相位近、模长不重现）"
    else:
        verdict = "h2_complex_difference_not_reproduced_by_GH5"
        answer = (
            "否：相同 Wigner 统计下的 GH5/MTEF 未能重现 Q1a 主窗 H2 磁序奇复差。"
            "这削弱了“输入涨落的半经典电子–光场动力学已足够”的解释，"
            "但还不等于证明输出纠缠。"
        )
        answer_short = "否（模长与相位均未过阈）"
    FIG.mkdir(parents=True, exist_ok=True)
    f_plane = figure_h2_plane(
        q_main_c[0], q_main_d[0], q_main_dj[0],
        m_main_c[0], m_main_d[0], m_main_dj[0],
        FIG / "h2_complex_plane_qed_vs_mtef.svg")
    f_win = figure_windows(
        [
            ("QED front", q_front_dj[0], "#4c6b8a"),
            ("QED main", q_main_dj[0], "#2f9e44"),
            ("QED back", q_back_dj[0], "#1b6b3a"),
            ("MTEF front", m_front_dj[0], "#7aa2c4"),
            ("MTEF main", m_main_dj[0], "#6fc27a"),
            ("MTEF back", m_back_dj[0], "#8fd19a"),
        ],
        FIG / "h2_delta_three_windows.svg",
    )
    f_cont = figure_cont(q_cont, abs_cont_q, abs_cont_q, abs_cont_m,
                         FIG / "h2_continuum_delta.svg")
    jobs = {tag: load_job(tag) for tag, _, _ in JOBS}
    rec = {
        "task_tag": TASK_TAG,
        "record_date": RECORD_DATE,
        "protocol": PROTOCOL,
        "status": "ok",
        "stopped": True,
        "verdict": verdict,
        "answer": answer,
        "answer_short": answer_short,
        "compare_kind": kind,
        "n_max": 0,
        "order": ORDER,
        "n_nodes": N_NODES,
        "n_steps": N_FULL,
        "dt": env.t_total / N_FULL,
        "qed_archive": str(QED_JSON.as_posix()),
        "qed_not_rerun": True,
        "ds_not_rescaled_coherent": True,
        "same_alpha_not_same_nbar": True,
        "source_hashes": hashes,
        "hash_abort": [],
        "windows": {"main": main, "front": front, "back": back},
        "compare_main": cmp_main,
        "compare_front": cmp_front,
        "compare_back": cmp_back,
        "h2_main": {
            "abs_C_QED": q_main_c[2], "abs_DS_QED": q_main_d[2],
            "abs_dJ_QED": q_main_dj[2],
            "abs_C_MTEF": m_main_c[2], "abs_DS_MTEF": m_main_d[2],
            "abs_dJ_MTEF": m_main_dj[2],
            "phase_x_C_QED": q_main_c[3], "phase_x_DS_QED": q_main_d[3],
            "phase_x_dJ_QED": q_main_dj[3],
            "phase_x_C_MTEF": m_main_c[3], "phase_x_DS_MTEF": m_main_d[3],
            "phase_x_dJ_MTEF": m_main_dj[3],
            "dphi_C_DS_QED_rad": dphi_cd_q,
            "dphi_C_DS_MTEF_rad": dphi_cd_m,
            "near_pi_residual_QED_rad": near_pi_q,
            "near_pi_residual_MTEF_rad": near_pi_m,
            "mtef_predicts_near_antiphase": bool(near_pi_m < 40.0 * np.pi / 180.0),
        },
        "qed_readonly_delta_complex": {
            "abs_front": q_front_dj[2], "abs_main": q_main_dj[2], "abs_back": q_back_dj[2],
            "phase_front": q_front_dj[3], "phase_main": q_main_dj[3], "phase_back": q_back_dj[3],
            "delta_complex": float(qed_delta_complex),
            "mod_rel_front_back": float(
                abs(q_back_dj[2] - q_front_dj[2]) / max(q_main_dj[2], 1e-30)),
            "note": "delta_complex uses Delta Jx, not 2D |Delta|. |Delta| stability is not complex convergence.",
        },
        "mtef_delta_complex": float(mtef_delta_complex),
        "continuum": {
            "q": qs_c.tolist(),
            "abs_dJodd_MTEF": abs_cont_m.tolist(),
            "peak_q_MTEF": peak_q_m,
            "peak_q_QED": qed["continuum"]["peak_q"],
            "peak_abs_dJ_QED": qed["continuum"]["peak_abs_dJ"],
            "peak_abs_dJ_MTEF": float(abs_cont_m[peak_i]),
        },
        "jobs": {tag: {k: jobs[tag].get(k) for k in
                       ("status", "n_run", "n_done", "wall_s", "max_abs_norm_minus_1",
                        "weight_sum", "kind", "neel")}
                 for tag, _, _ in JOBS},
        "figures": {"complex_plane": f_plane, "three_windows": f_win, "continuum": f_cont},
        "physics_boundaries": {
            "GH5_agreement_is_not_classical_light": True,
            "GH5_disagreement_is_not_output_entanglement": True,
            "same_alpha_is_not_same_nbar": True,
            "integer_Fourier_is_not_certified_HHG": True,
            "odd_current_is_tiny_versus_even": True,
            "A_star_mean_not_classical_A_star_0p8": True,
            "modulus_stability_is_not_complex_convergence": True,
        },
        "did_not": [
            "rerun QED", "Cn", "theta=pi", "squeezed vacuum", "G1", "H2 extra k",
            "nmax scan", "D2", "Floquet", "T2", "server", "shorten plateau", "drop to GH3",
        ],
        "utc": datetime.now(timezone.utc).isoformat(),
    }
    dump(OUT / "gh5.json", rec)
    write_reports(rec)
    return rec


def drive() -> int:
    hashes = source_hashes()
    bad = hash_abort(hashes)
    if bad:
        dump(OUT / "gh5.json", {"status": "hash_abort", "hash_mismatch": bad, "stopped": True})
        progress("hash abort, no propagation")
        return 3
    OUT.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    procs, logs = {}, {}
    for tag, kind, neel in JOBS:
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
    codes = {tag: (procs[tag].returncode if tag in procs else 0) for tag, _, _ in JOBS}
    progress(f"jobs return {codes} driver_wall={time.perf_counter()-t0:.1f}s")
    if all(job_finished_ok(tag) for tag, _, _ in JOBS):
        rec = analyze()
        progress(f"analyze done verdict={rec.get('verdict')}")
        return 0
    rec = {
        "status": "timeout_partial",
        "stopped": True,
        "driver_wall_s": time.perf_counter() - t0,
        "returncodes": codes,
        "job_status": {tag: None if load_job(tag) is None else load_job(tag).get("status")
                       for tag, _, _ in JOBS},
        "jobs_n_done": {tag: None if load_job(tag) is None else load_job(tag).get("n_done")
                        for tag, _, _ in JOBS},
        "did_not_analyze_incomplete": True,
        "qed_not_rerun": True,
    }
    dump(OUT / "gh5.json", rec)
    progress("incomplete, no full analyze")
    return 1


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--preflight", action="store_true")
    ap.add_argument("--job", type=str, default="")
    ap.add_argument("--analyze-only", action="store_true")
    args = ap.parse_args(argv)
    OUT.mkdir(parents=True, exist_ok=True)
    if args.preflight:
        rec = preflight()
        print(json.dumps(jsonable({
            "status": rec.get("status"),
            "per_traj_s": rec.get("per_traj_s"),
            "est_one_job_s": rec.get("est_one_job_s"),
            "est_serial_s": rec.get("est_serial_s"),
            "under_cap_if_four_parallel": rec.get("under_cap_if_four_parallel"),
        }), ensure_ascii=False, indent=2))
        return 0 if rec.get("status") == "preflight" else 3
    if args.job:
        rec = run_one_job(args.job)
        return 0 if rec.get("status") in ("ok", "timeout_partial") else 1
    if args.analyze_only:
        rec = analyze()
        print(json.dumps(jsonable({
            "status": rec.get("status"), "verdict": rec.get("verdict"),
            "answer_short": rec.get("answer_short"),
            "compare_kind": rec.get("compare_kind"),
        }), ensure_ascii=False, indent=2))
        return 0 if rec.get("status") == "ok" else 1
    return drive()


if __name__ == "__main__":
    raise SystemExit(main())
