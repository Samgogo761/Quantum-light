#!/usr/bin/env python3
"""Q1a: full Qin_a2_1_4_1 four-way exact QED.

Approved after Q0. Local 900 s hard cap. Same-alpha C vs DS, ±N.
Wave 2: --resume-timeout continues timeout_partial jobs from checkpoints.
Do not restart from t=0. Analyze if all jobs reach 4T0; prefer 5T0;
if 5T0 is reached but 6T0 is not, mark full_switch_incomplete.
Does not start Q1b/c, Cn, theta=pi, SV, GH5/MTEF, G1/H2, extra k,
nmax scan, vacuum, D2, Gamma192, T2, Floquet, or server.
Integer Fourier is not certified HHG. Same alpha is not same nbar.
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
import q8_m0_qinput_shared_kk_static_20261029 as sh
from q8_m0_model import R0, build_geometry, sha256_file

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results/q_line/qinput_qin_a2_141_q1a_20261120"
FIG = OUT / "figures"
TASK_TAG = "20261120"
RECORD_DATE = "2026-10-22"
PROTOCOL = "Qin_a2_1_4_1"
MU = -0.009010620372198276
NMAX = odd.NMAX
ALPHA = odd.ALPHA
R_SQ = odd.R_SQ
THETA = odd.THETA
N_FULL = 11154
SAMPLE_EVERY = 32
CHECKPOINT_EVERY = 400
DRIVER_WALL_S = 900.0
JOB_WALL_S = 870.0
FOCK_ALARM = 0.05
CLASSICAL_A_STAR = 0.8
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
}
FROZEN = tuple(HERE / name for name in EXPECTED_HASHES)


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


def a_star_mean(geo) -> float:
    return 2.0 * float(geo.a_nn_mean_ang) * float(pf.G_ABS) * abs(complex(ALPHA))


def ckpt_paths(tag: str):
    return OUT / f"ckpt_{tag}.npz", OUT / f"ckpt_{tag}_samples.json"


def save_ckpt(tag: str, psi, next_j: int, samples) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    p_npz, p_js = ckpt_paths(tag)
    np.savez_compressed(p_npz, psi=np.asarray(psi, complex), next_j=np.int64(next_j))
    p_js.write_text(json.dumps(jsonable(samples), ensure_ascii=False), encoding="utf-8")


def load_ckpt(tag: str):
    p_npz, p_js = ckpt_paths(tag)
    if not p_npz.is_file() or not p_js.is_file():
        return None
    z = np.load(p_npz)
    samples = json.loads(p_js.read_text(encoding="utf-8"))
    return np.asarray(z["psi"], complex), int(z["next_j"]), samples


def dedup_samples_by_step(samples):
    out = []
    seen = set()
    for rec in list(samples or []):
        st = int(rec.get("step", -1))
        if st in seen:
            continue
        seen.add(st)
        out.append(rec)
    return out


def load_job(tag: str):
    path = OUT / f"job_{tag}.json"
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def sample_js_for(n_steps: int):
    js = list(range(0, n_steps, SAMPLE_EVERY))
    if not js or js[-1] != n_steps - 1:
        js.append(n_steps - 1)
    return js


def propagate_ckpt(model, psi, t0, t1, env, n_steps, sample_js, clock, tag,
                   start_j: int = 0, samples=None, ckpt_every: int = CHECKPOINT_EVERY):
    dt = (t1 - t0) / n_steps
    sample_set = set(int(j) for j in sample_js)
    samples = dedup_samples_by_step(list(samples or []))
    worst_last = 0.0
    j = int(start_j)
    timed_out = False
    timeout_msg = None
    t_loop0 = time.perf_counter()
    try:
        while j < n_steps:
            clock.check()
            tm = t0 + (j + 0.5) * dt
            s = env.s(tm)
            Ds = dyn.peierls_at(model.q_units, s, model.n_max)
            psi, last = pf.taylor_step(model, psi, s, dt, Ds=Ds)
            worst_last = max(worst_last, last)
            if j in sample_set:
                t_now = t0 + (j + 1) * dt
                obs = pf.snapshot(model, psi, env.s(t_now), Ds=None)
                obs["t"] = t_now
                dhds = pf.dHds_expect(model, psi, env.s(t_now))
                dsdt = env.ds(t_now)
                obs["dHds_expect"] = pf.cpack(dhds)
                obs["dsdt"] = float(dsdt)
                obs["power_dHds_dsdt"] = float((dhds * dsdt).real)
                obs["step"] = j + 1
                samples.append(obs)
            if (j + 1) % 32 == 0 or j == n_steps - 1:
                nrm = float(np.linalg.norm(psi))
                if (not np.isfinite(psi).all()) or nrm > 2.0 or nrm < 0.5:
                    raise RuntimeError(f"norm {nrm} at step {j}")
            j += 1
            if ckpt_every and j % ckpt_every == 0:
                save_ckpt(tag, psi, j, samples)
                progress(f"{tag} ckpt j={j}/{n_steps} wall={clock.elapsed():.1f}s")
    except TimeoutError as exc:
        timed_out = True
        timeout_msg = str(exc)
        save_ckpt(tag, psi, j, samples)
    work = 0.0
    if len(samples) >= 2:
        for a, b in zip(samples[:-1], samples[1:]):
            work += 0.5 * (a["power_dHds_dsdt"] + b["power_dHds_dsdt"]) * (b["t"] - a["t"])
    return psi, {
        "steps_requested": int(n_steps),
        "steps_completed": int(j),
        "dt": float(dt),
        "t0": float(t0),
        "t1_requested": float(t1),
        "t1_reached": float(t0 + j * dt),
        "order": pf.ORDER,
        "max_last_taylor_term": float(worst_last),
        "samples": samples,
        "work_trap_dHds_dsdt": float(work),
        "timed_out": bool(timed_out),
        "timeout_msg": timeout_msg,
        "next_j": int(j),
        "loop_seconds": float(time.perf_counter() - t_loop0),
    }


def series_from_samples(samples):
    t = np.array([s["t"] for s in samples], float)
    j = np.array(
        [[s["mean_J"][a]["re"] + 1j * s["mean_J"][a]["im"] for a in range(2)]
         for s in samples], complex)
    n_ph = np.array([s["n_ph"] for s in samples], float)
    top = np.array([s["top_fock_mass"] for s in samples], float)
    nxt = np.array([s["p_n"][-2] for s in samples], float)
    nrm = np.array([s["norm"] for s in samples], float)
    ss = np.array([s["s"] for s in samples], float)
    return t, j, n_ph, top, nxt, nrm, ss


def save_series_npz(tag: str, samples) -> None:
    t, j, n_ph, top, nxt, nrm, ss = series_from_samples(samples)
    np.savez_compressed(
        OUT / f"series_{tag}.npz",
        t=t, Jx=j[:, 0], Jy=j[:, 1], n_ph=n_ph, top_fock=top,
        next_top_fock=nxt, norm=nrm, s=ss)


def run_one_job(tag: str, resume_timeout: bool = False) -> dict:
    if tag not in JOB_MAP:
        raise ValueError(tag)
    kind, neel = JOB_MAP[tag]
    hashes = source_hashes()
    bad = hash_abort(hashes)
    if bad:
        report = {
            "tag": tag, "status": "hash_abort", "hash_mismatch": bad,
            "source_hashes": hashes, "stopped": True,
        }
        dump(OUT / f"job_{tag}.json", report)
        return report

    prev = load_job(tag)
    if prev and prev.get("status") in ("ok", "ok_fock_alarm"):
        progress(f"{tag} already {prev['status']}, skip")
        return prev
    if prev and prev.get("status") == "timeout_partial":
        if not resume_timeout:
            progress(f"{tag} timeout_partial from a finished wave, do not resume")
            return prev

    resume_ckpt = None
    if resume_timeout and prev and prev.get("status") == "timeout_partial":
        resume_ckpt = load_ckpt(tag)
        if resume_ckpt is None or int(resume_ckpt[1]) <= 0:
            report = {
                "tag": tag, "status": "resume_abort",
                "reason": "missing_or_nonpositive_checkpoint",
                "stopped": True, "resume_timeout": True,
            }
            dump(OUT / f"job_{tag}.json", report)
            progress(f"{tag} resume_abort missing checkpoint")
            return report
        progress(f"{tag} timeout_partial resume enabled next_j={int(resume_ckpt[1])}")

    geo = build_geometry()
    k = geo.k_cart(sh.K_FRAC)
    env = dyn.Envelope(pf.OMEGA0, ramp_cycles=1.0, flat_cycles=4.0)
    n_steps = N_FULL
    dt = env.t_total / n_steps
    ek, emk, masses = odd.ground_pair_neel(geo, k, neel)
    raw = odd.photon_raw(kind, NMAX)
    raw_diag = odd.photon_moments(raw)
    c = raw / np.sqrt(raw_diag["unnormalized_mass"])
    model = sh.build_shared(geo, R0, k, neel, n_max=NMAX, omega0=pf.OMEGA0, g=pf.G_ABS)
    psi, prod_n = odd.product_normalized(ek, emk, c)

    start_j = 0
    samples = []
    resumed = False
    want_ckpt = (
        prev is None
        or prev.get("status") in ("running", "failed", "incomplete")
        or (resume_timeout and prev.get("status") == "timeout_partial")
    )
    if want_ckpt:
        loaded = resume_ckpt if resume_ckpt is not None else load_ckpt(tag)
        if loaded is not None:
            psi, start_j, samples = loaded
            samples = dedup_samples_by_step(samples)
            resumed = True
            progress(f"{tag} resume next_j={start_j} n_samples={len(samples)}")

    if resume_timeout and prev and prev.get("status") == "timeout_partial":
        if (not resumed) or start_j <= 0:
            report = {
                "tag": tag, "status": "resume_abort",
                "reason": "checkpoint_not_applied",
                "stopped": True, "resume_timeout": True,
            }
            dump(OUT / f"job_{tag}.json", report)
            progress(f"{tag} resume_abort checkpoint not applied")
            return report

    dump(OUT / f"job_{tag}.json", {
        "tag": tag, "kind": kind, "neel": int(neel), "status": "running",
        "resumed": resumed, "resume_timeout": bool(resume_timeout),
        "next_j": start_j, "utc": datetime.now(timezone.utc).isoformat(),
    })
    resume_obs = pf.snapshot(model, psi, env.s(start_j * dt))
    if resumed and prev and prev.get("initial"):
        s0 = prev["initial"]
    else:
        s0 = resume_obs
    progress(f"{tag} start kind={kind} neel={neel} dim={model.dim} j0={start_j}/{n_steps}")
    clock = dyn.WallClock(JOB_WALL_S)
    finite_ok = True
    err = None
    try:
        psi, led = propagate_ckpt(
            model, psi, 0.0, env.t_total, env, n_steps, sample_js_for(n_steps),
            clock, tag, start_j=start_j, samples=samples)
    except Exception as exc:
        finite_ok = False
        err = f"{type(exc).__name__}: {exc}"
        led = {
            "steps_requested": n_steps, "steps_completed": start_j, "dt": float(dt),
            "samples": samples, "timed_out": False, "timeout_msg": None,
            "max_last_taylor_term": None, "work_trap_dHds_dsdt": None,
            "next_j": start_j, "t1_reached": start_j * dt, "loop_seconds": clock.elapsed(),
        }
        save_ckpt(tag, psi, start_j, samples)
        progress(f"{tag} exception {err}")

    wall = clock.elapsed()
    samples = led.get("samples") or []
    if samples:
        save_series_npz(tag, samples)
        save_ckpt(tag, psi, led.get("next_j", start_j), samples)
    send = samples[-1] if samples else pf.snapshot(model, psi, env.s(led.get("t1_reached", 0.0)))
    tops = [s["top_fock_mass"] for s in samples] + [send["top_fock_mass"]]
    nxts = [s["p_n"][-2] for s in samples] + [send["p_n"][-2]]
    norms = [s["norm"] for s in samples] + [send["norm"]]
    nphs = [s["n_ph"] for s in samples] + [send["n_ph"]]
    max_top = float(np.max(tops))
    max_next = float(np.max(nxts))
    max_dn = float(np.max(np.abs(np.asarray(norms) - 1.0)))
    fock_alarm = bool(max_top > FOCK_ALARM)
    timed_out = bool(led.get("timed_out"))
    completed = int(led.get("steps_completed") or 0)
    t_reach = float(led.get("t1_reached") or 0.0)
    t_over = t_reach / env.period
    if not finite_ok:
        status = "failed"
    elif timed_out:
        status = "timeout_partial"
    elif completed < n_steps:
        status = "incomplete"
    elif fock_alarm:
        status = "ok_fock_alarm"
    else:
        status = "ok"

    report = {
        "task_tag": TASK_TAG,
        "record_date": RECORD_DATE,
        "protocol": PROTOCOL,
        "tag": tag,
        "kind": kind,
        "neel": int(neel),
        "status": status,
        "resumed": resumed,
        "resume_timeout": bool(resume_timeout),
        "resume_from_next_j": int(start_j) if resumed else 0,
        "finite_state": finite_ok,
        "error": err,
        "fock_alarm_0p05": fock_alarm,
        "source_hashes": hashes,
        "definition": {
            "n_max": NMAX, "g": pf.G_ABS, "omega0": pf.OMEGA0,
            "alpha": {"re": float(ALPHA.real), "im": float(ALPHA.imag)},
            "r": R_SQ, "theta": THETA,
            "k_fractional": list(sh.K_FRAC),
            "mu_half_filling_documented": MU,
            "n_full": n_steps, "dt": float(dt),
            "sample_every": SAMPLE_EVERY,
            "period": env.period, "t_total": env.t_total,
            "plateau": list(env.plateau),
            "A_star_mean": a_star_mean(geo),
            "classical_A_star": CLASSICAL_A_STAR,
            "s_is_coupling_switch_not_free_space_pulse": True,
        },
        "electron": masses,
        "photon_unnormalized": raw_diag,
        "product_norm_before_renorm": prod_n,
        "initial": s0,
        "endpoint": send,
        "ledger": {k: led[k] for k in led if k != "samples"},
        "max_top_fock": max_top,
        "max_next_top_fock": max_next,
        "max_n_ph": float(np.max(nphs)),
        "min_n_ph": float(np.min(nphs)),
        "max_abs_norm_minus_1": max_dn,
        "t_reached": t_reach,
        "t_reached_over_T0": t_over,
        "covers_5T0": bool(t_over + 1e-9 >= 5.0),
        "covers_4T0": bool(t_over + 1e-9 >= 4.0),
        "covers_6T0": bool(t_over + 1e-9 >= 6.0),
        "wave": 2 if resume_timeout else 1,
        "wall_seconds": wall,
        "seconds_per_step": (wall / max(completed - start_j, 1)),
        "time_series": pf.pack_samples(samples) if samples else {},
        "n_samples": len(samples),
        "series_npz": str((OUT / f"series_{tag}.npz").as_posix()),
        "same_alpha_is_not_same_nbar": True,
        "integer_Fourier_is_not_certified_HHG": True,
        "q0_fock_does_not_certify_plateau": True,
    }
    if resumed:
        report["resume_snapshot"] = resume_obs
    dump(OUT / f"job_{tag}.json", report)
    progress(f"{tag} {status} steps={completed}/{n_steps} t/T0={t_over:.3f} "
             f"top={max_top:.3e} wall={wall:.1f}s")
    return report


def cvec(pack):
    return np.array([z["re"] + 1j * z["im"] for z in pack], complex)


def pack_c(z):
    z = np.asarray(z, complex)
    if z.ndim == 0:
        return pf.cpack(complex(z))
    return [pf.cpack(complex(v)) for v in z]


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


def amp_phase(z):
    z = np.asarray(z, complex)
    return np.abs(z), np.angle(z)


def rel_change(a, b):
    a = np.asarray(a, complex)
    b = np.asarray(b, complex)
    den = np.maximum(np.abs(a), 1e-30)
    return np.abs(b - a) / den


def load_series(tag: str):
    path = OUT / f"series_{tag}.npz"
    if path.is_file():
        z = np.load(path)
        t = np.asarray(z["t"], float)
        j = np.stack([z["Jx"], z["Jy"]], axis=1)
        return t, j, z
    job = load_job(tag)
    ts = job["time_series"]
    t = np.asarray(ts["t"], float)
    j = np.array([[c["re"] + 1j * c["im"] for c in row] for row in ts["mean_J"]], complex)
    return t, j, None


def svg_poly(xs, ys, xmin, xmax, ymin, ymax, x0, y0, w, h, color, sw=1.4):
    if xmax <= xmin:
        xmax = xmin + 1.0
    if ymax <= ymin:
        ymax = ymin + 1e-30
    pts = []
    for x, y in zip(xs, ys):
        px = x0 + (float(x) - xmin) / (xmax - xmin) * w
        py = y0 + (1.0 - (float(y) - ymin) / (ymax - ymin)) * h
        pts.append(f"{px:.2f},{py:.2f}")
    return (f'<polyline fill="none" stroke="{color}" stroke-width="{sw}" '
            f'points="{" ".join(pts)}"/>')


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


def figure_time(tT, yc, yd, ydlt, title, ylabel, path):
    xmin, xmax = float(tT[0]), float(tT[-1])
    ys = np.concatenate([yc, yd, ydlt])
    ymin, ymax = float(np.min(ys)), float(np.max(ys))
    pad = 0.08 * max(ymax - ymin, 1e-30)
    ymin, ymax = ymin - pad, ymax + pad
    x0, y0, w, h = 70, 40, 800, 430
    body = svg_axes(x0, y0, w, h, "t / T0", ylabel, title)
    body += svg_poly(tT, yc, xmin, xmax, ymin, ymax, x0, y0, w, h, "#4c6b8a", 1.5)
    body += svg_poly(tT, yd, xmin, xmax, ymin, ymax, x0, y0, w, h, "#c45c26", 1.5)
    body += svg_poly(tT, ydlt, xmin, xmax, ymin, ymax, x0, y0, w, h, "#2f9e44", 1.1)
    body += (
        '<text x="80" y="36" font-size="11" fill="#4c6b8a">Cα</text>'
        '<text x="120" y="36" font-size="11" fill="#c45c26">DS</text>'
        '<text x="160" y="36" font-size="11" fill="#2f9e44">Δ</text>'
        '<text x="460" y="548" font-size="10" fill="#5c5148">'
        'Q1a source current; not certified HHG; same α ≠ same nbar</text>'
    )
    return write_svg(path, body)


def figure_bars(qs, ac, ad, title, path):
    x0, y0, w, h = 70, 40, 800, 430
    ymax = max(float(np.max(ac)), float(np.max(ad)), 1e-30) * 1.15
    body = svg_axes(x0, y0, w, h, "harmonic q", "|J_q^o|", title)
    n = len(qs)
    bw = w / (n * 3.2)
    for i, q in enumerate(qs):
        cx = x0 + (i + 0.5) * (w / n)
        hc = float(ac[i]) / ymax * h
        hd = float(ad[i]) / ymax * h
        body += (f'<rect x="{cx - bw:.1f}" y="{y0 + h - hc:.1f}" width="{bw:.1f}" '
                 f'height="{hc:.1f}" fill="#4c6b8a"/>')
        body += (f'<rect x="{cx + 2:.1f}" y="{y0 + h - hd:.1f}" width="{bw:.1f}" '
                 f'height="{hd:.1f}" fill="#c45c26"/>')
        body += (f'<text x="{cx:.1f}" y="{y0 + h + 16:.1f}" text-anchor="middle" '
                 f'font-size="11" fill="#243038">{int(q)}</text>')
    body += '<text x="80" y="36" font-size="11" fill="#4c6b8a">Cα</text>'
    body += '<text x="120" y="36" font-size="11" fill="#c45c26">DS</text>'
    return write_svg(path, body)


def figure_cont(q, yc, yd, ydlt, title, path):
    xmin, xmax = float(q[0]), float(q[-1])
    ys = np.concatenate([yc, yd, ydlt])
    ymin, ymax = 0.0, float(np.max(ys)) * 1.12
    x0, y0, w, h = 70, 40, 800, 430
    body = svg_axes(x0, y0, w, h, "Ω / ω0", "|J^o(Ω)|", title)
    body += svg_poly(q, yc, xmin, xmax, ymin, ymax, x0, y0, w, h, "#4c6b8a", 1.4)
    body += svg_poly(q, yd, xmin, xmax, ymin, ymax, x0, y0, w, h, "#c45c26", 1.4)
    body += svg_poly(q, ydlt, xmin, xmax, ymin, ymax, x0, y0, w, h, "#2f9e44", 1.2)
    for n in range(1, 7):
        px = 70 + (n - xmin) / (xmax - xmin) * 800
        body += (f'<line x1="{px:.1f}" y1="40" x2="{px:.1f}" y2="470" '
                 f'stroke="#8a9096" stroke-dasharray="3 4" stroke-width="0.8"/>')
    return write_svg(path, body)


def analyze() -> dict:
    hashes = source_hashes()
    jobs = {tag: load_job(tag) for tag, _, _ in JOBS}
    missing = [tag for tag, rec in jobs.items() if rec is None]
    if missing:
        return {"status": "missing_jobs", "missing": missing, "stopped": True}

    geo = build_geometry()
    env = dyn.Envelope(pf.OMEGA0, ramp_cycles=1.0, flat_cycles=4.0)
    T0 = env.period
    omega0 = pf.OMEGA0
    series = {}
    for tag, _, _ in JOBS:
        t, j, raw = load_series(tag)
        series[tag] = {"t": t, "j": j, "raw": raw}

    t = series["coh_p"]["t"]
    for tag, _, _ in JOBS:
        if not np.allclose(series[tag]["t"], t, rtol=0, atol=1e-9):
            n = min(len(series[tag]["t"]) for tag, _, _ in JOBS)
            t0 = series["coh_p"]["t"][:n]
            aligned = True
            for tag2, _, _ in JOBS:
                if not np.allclose(series[tag2]["t"][:n], t0, rtol=0, atol=1e-9):
                    aligned = False
                    break
            if not aligned or n < 8:
                return {"status": "time_grid_mismatch", "tag": tag, "stopped": True}
            t = t0
            for tag2, _, _ in JOBS:
                series[tag2]["t"] = series[tag2]["t"][:n]
                series[tag2]["j"] = series[tag2]["j"][:n]
            break

    jp_c, jm_c = series["coh_p"]["j"], series["coh_m"]["j"]
    jp_d, jm_d = series["ds_p"]["j"], series["ds_m"]["j"]
    even_c, odd_c = odd.even_odd(jp_c, jm_c)
    even_d, odd_d = odd.even_odd(jp_d, jm_d)
    d_odd = odd_d - odd_c
    d_even = even_d - even_c
    imag_floor = max(float(np.max(np.abs(x.imag))) for x in (jp_c, jm_c, jp_d, jm_d))
    tT = t / T0

    win_main = (2.0 * T0, 4.0 * T0)
    win_front = (1.0 * T0, 3.0 * T0)
    win_back = (3.0 * T0, 5.0 * T0)
    qs = np.arange(1, 7, dtype=float)
    q_cont = np.linspace(0.0, 6.5, 326)

    t_min_jobs = min(float(jobs[tag]["t_reached_over_T0"]) for tag, _, _ in JOBS)
    covers_4 = bool(t_min_jobs + 1e-9 >= 4.0)
    covers_5 = bool(t_min_jobs + 1e-9 >= 5.0)
    covers_6 = bool(t_min_jobs + 1e-9 >= 6.0)
    main_complete = covers_4
    front_complete = bool(t_min_jobs + 1e-9 >= 3.0)
    back_complete = covers_5

    def pack_win(name, w0, w1, arrs, complete: bool):
        if not complete:
            return {
                "name": name, "t0": w0, "t1": w1,
                "t0_over_T0": w0 / T0, "t1_over_T0": w1 / T0,
                "complete": False,
                "not_used_as_2T0_integral": True,
                "reason": "coverage_short_of_window_end",
                "convention": "J_q = 1/(t1-t0) int exp(+i q omega0 t) J(t) dt, rectangular",
                "not_certified_HHG": True,
                "slots": {},
            }
        slots = {}
        for key, arr in arrs.items():
            jq = rect_jq(t, arr, omega0, w0, w1, qs)
            slots[key] = {
                "Jx": pack_c(jq[:, 0]),
                "Jy": pack_c(jq[:, 1]),
                "abs": [float(np.linalg.norm(v)) for v in jq],
                "phase_x": [float(np.angle(v[0])) for v in jq],
                "phase_y": [float(np.angle(v[1])) for v in jq],
            }
        return {
            "name": name, "t0": w0, "t1": w1, "t0_over_T0": w0 / T0, "t1_over_T0": w1 / T0,
            "complete": True,
            "convention": "J_q = 1/(t1-t0) int exp(+i q omega0 t) J(t) dt, rectangular",
            "not_certified_HHG": True,
            "slots": slots,
        }

    arrs = {
        "J_odd_coh": odd_c, "J_odd_ds": odd_d, "dJ_odd": d_odd,
        "J_even_coh": even_c, "J_even_ds": even_d, "dJ_even": d_even,
    }
    main = pack_win("plateau_center_2T", *win_main, arrs, main_complete)
    front = pack_win("front_2T", *win_front, arrs, front_complete)
    back = pack_win("back_2T", *win_back, arrs, back_complete)

    if not main_complete:
        dump(OUT / "q1a.json", {
            "status": "insufficient_coverage",
            "t_min_over_T0": t_min_jobs,
            "covers_4T0": covers_4,
            "covers_5T0": covers_5,
            "stopped": True,
            "windows": {"main": main, "front": front, "back": back},
        })
        return {
            "status": "insufficient_coverage",
            "t_min_over_T0": t_min_jobs,
            "stopped": True,
        }

    cont = {}
    for key, arr in arrs.items():
        jq = rect_jq(t, arr, omega0, *win_main, q_cont)
        cont[key] = {
            "abs": [float(np.linalg.norm(v)) for v in jq],
            "abs_x": np.abs(jq[:, 0]).tolist(),
            "abs_y": np.abs(jq[:, 1]).tolist(),
        }

    abs_c = np.array(main["slots"]["J_odd_coh"]["abs"], float)
    abs_d = np.array(main["slots"]["J_odd_ds"]["abs"], float)
    abs_dj = np.array(main["slots"]["dJ_odd"]["abs"], float)
    scale = np.maximum(np.maximum(abs_c, abs_d), 1e-30)
    rel = abs_dj / scale
    front_abs = None
    back_abs = None
    win_rel = None
    if front_complete:
        front_abs = np.array(front["slots"]["dJ_odd"]["abs"], float)
    if back_complete:
        back_abs = np.array(back["slots"]["dJ_odd"]["abs"], float)
    if front_complete and back_complete:
        win_rel = np.maximum(rel_change(front_abs, back_abs), rel_change(back_abs, front_abs))

    abs_cont_dj = np.array(cont["dJ_odd"]["abs"], float)
    peak_i = int(np.argmax(abs_cont_dj[1:])) + 1
    peak_q = float(q_cont[peak_i])
    nearest = float(np.round(peak_q)) if peak_q >= 0.5 else 0.0
    dq = abs(peak_q - nearest)

    channels = []
    for i, q in enumerate(qs):
        near_int = bool(dq <= 0.15 and int(nearest) == int(q))
        win_rel_i = None if win_rel is None else float(win_rel[i])
        identifiable = bool(
            back_complete
            and rel[i] > 0.05
            and abs_dj[i] > 10.0 * max(imag_floor, 1e-18)
            and near_int
            and win_rel_i is not None
            and win_rel_i < 0.5
        )
        if identifiable:
            kind_ch = "integer_plus_continuum_candidate_not_certified_HHG"
        elif not back_complete:
            kind_ch = "main_window_only_back_window_incomplete"
        elif rel[i] > 0.05 and dq > 0.15:
            kind_ch = "off_integer_beat_or_transient"
        elif rel[i] > 0.05 and win_rel_i is not None and win_rel_i >= 0.5:
            kind_ch = "window_inconsistent"
        else:
            kind_ch = "no_identifiable_input_stat_change"
        channels.append({
            "q": int(q),
            "abs_Jodd_C": float(abs_c[i]),
            "abs_Jodd_DS": float(abs_d[i]),
            "abs_dJodd": float(abs_dj[i]),
            "rel_dJ_over_J": float(rel[i]),
            "phase_x_C": main["slots"]["J_odd_coh"]["phase_x"][i],
            "phase_x_DS": main["slots"]["J_odd_ds"]["phase_x"][i],
            "phase_y_C": main["slots"]["J_odd_coh"]["phase_y"][i],
            "phase_y_DS": main["slots"]["J_odd_ds"]["phase_y"][i],
            "front_back_rel_dJ": win_rel_i,
            "kind": kind_ch,
            "identifiable_candidate": identifiable,
        })

    n_ident = sum(1 for ch in channels if ch["identifiable_candidate"])
    n_beat = sum(1 for ch in channels if ch["kind"] == "off_integer_beat_or_transient")
    if n_ident:
        verdict = "identifiable_frequency_resolved_magnetization_odd_change_not_certified_HHG"
        answer = (
            "Yes, same-alpha DS vs C produces an identifiable frequency-resolved "
            "magnetization-odd source change in at least one integer-adjacent band. "
            "This is a candidate, not certified HHG, and not energy-matched nonclassicality."
        )
    elif not back_complete:
        verdict = "main_window_computed_back_window_incomplete"
        answer = (
            "Central two-cycle Fourier is available, but the back window [3T0,5T0] "
            "is incomplete, so front/back consistency and identifiable-candidate "
            "certification are not claimed. Integer Fourier is not certified HHG."
        )
    elif n_beat or any(ch["rel_dJ_over_J"] > 0.05 for ch in channels):
        verdict = "difference_present_but_transient_or_window_unstable"
        answer = (
            "A DS–C magnetization-odd difference is visible, but it is not a stable "
            "integer-harmonic peak under the predeclared windows. Treat as joint "
            "electron-photon nonequilibrium, not certified HHG."
        )
    else:
        verdict = "no_reliable_input_state_dependence_at_this_low_occupation"
        answer = (
            "No reliable, frequency-resolved magnetization-odd response change is "
            "identified at this low-occupation same-alpha protocol."
        )

    t_min = t_min_jobs
    incomplete = not covers_6
    covers = covers_5
    if covers_6:
        status = "ok"
    elif covers_5:
        status = "ok_full_switch_incomplete"
    elif covers_4:
        status = "ok_main_window_no_back"
    else:
        status = "insufficient_coverage"

    figures = []
    figures.append(figure_time(
        tT, odd_c[:, 0].real, odd_d[:, 0].real, d_odd[:, 0].real,
        "J_odd,x (Cα vs DS)", "Re J_odd,x", FIG / "figure_q1a_time_odd_x.svg"))
    figures.append(figure_time(
        tT, odd_c[:, 1].real, odd_d[:, 1].real, d_odd[:, 1].real,
        "J_odd,y (Cα vs DS)", "Re J_odd,y", FIG / "figure_q1a_time_odd_y.svg"))
    figures.append(figure_time(
        tT, even_c[:, 0].real, even_d[:, 0].real, d_even[:, 0].real,
        "J_even,x (Cα vs DS)", "Re J_even,x", FIG / "figure_q1a_time_even_x.svg"))
    figures.append(figure_bars(
        qs, abs_c, abs_d, "H1–H6 |J_q^o| center [2T0,4T0]", FIG / "figure_q1a_harmonics.svg"))
    figures.append(figure_cont(
        q_cont, np.array(cont["J_odd_coh"]["abs"]), np.array(cont["J_odd_ds"]["abs"]),
        abs_cont_dj, "continuum |J^o| on [2T0,4T0]", FIG / "figure_q1a_continuum.svg"))
    if front_complete and back_complete:
        figures.append(figure_bars(
            qs, front_abs, back_abs, "ΔJ_q^o front [1,3] vs back [3,5]",
            FIG / "figure_q1a_windows.svg"))

    nbar_c = jobs["coh_p"]["photon_unnormalized"]["mean_n"]
    nbar_d = jobs["ds_p"]["photon_unnormalized"]["mean_n"]
    report = {
        "task_tag": TASK_TAG,
        "record_date": RECORD_DATE,
        "utc": datetime.now(timezone.utc).isoformat(),
        "protocol": PROTOCOL,
        "status": status,
        "wave": 2,
        "stopped": True,
        "claim_level": "same_alpha_DS_vs_C_magnetization_odd_spectrum_not_HHG_not_advantage",
        "verdict": verdict,
        "answer": answer,
        "did_not": [
            "Q1b", "Q1c", "Cn", "theta=pi", "squeezed vacuum", "GH5", "MTEF",
            "G1", "H2", "extra k", "nmax scan", "vacuum switch", "D2",
            "Gamma192", "T2", "Floquet", "server", "classical A*=0.8 splice",
        ],
        "physics_boundaries": {
            "integer_Fourier_is_not_certified_HHG": True,
            "plateau_H_time_independent_but_mode_still_oscillates": True,
            "coupling_switch_can_prepare_transients": True,
            "same_alpha_is_not_same_nbar": True,
            "A_star_mean_not_classical_A_star_0p8": True,
            "QED_current_is_not_output_squeezing_or_entanglement": True,
            "no_vacuum_control_this_round": True,
        },
        "coverage": {
            "t_min_over_T0": t_min,
            "covers_4T0": covers_4,
            "covers_5T0": covers_5,
            "covers_6T0": covers_6,
            "covers_analysis_windows_to_5T0": covers,
            "incomplete_fall_edge": incomplete,
            "job_status": {tag: jobs[tag]["status"] for tag, _, _ in JOBS},
            "job_wall": {tag: jobs[tag]["wall_seconds"] for tag, _, _ in JOBS},
            "max_top_fock": {tag: jobs[tag]["max_top_fock"] for tag, _, _ in JOBS},
            "max_next_top_fock": {tag: jobs[tag]["max_next_top_fock"] for tag, _, _ in JOBS},
            "max_abs_norm_minus_1": {tag: jobs[tag]["max_abs_norm_minus_1"] for tag, _, _ in JOBS},
            "max_last_taylor": {tag: jobs[tag]["ledger"].get("max_last_taylor_term") for tag, _, _ in JOBS},
            "fock_alarm": {tag: jobs[tag]["fock_alarm_0p05"] for tag, _, _ in JOBS},
        },
        "resources": {
            "alpha": 0.5, "r": R_SQ, "theta": THETA,
            "nbar_coherent_code": nbar_c,
            "nbar_ds_code": nbar_d,
            "nbar_not_matched": True,
            "A_star_mean": a_star_mean(geo),
            "classical_A_star": CLASSICAL_A_STAR,
        },
        "time_series_summary": {
            "t": t.tolist(),
            "J_odd_coh": [[pf.cpack(z) for z in row] for row in odd_c],
            "J_odd_ds": [[pf.cpack(z) for z in row] for row in odd_d],
            "dJ_odd": [[pf.cpack(z) for z in row] for row in d_odd],
            "J_even_coh": [[pf.cpack(z) for z in row] for row in even_c],
            "J_even_ds": [[pf.cpack(z) for z in row] for row in even_d],
            "max_abs": {
                "dJ_odd_x": float(np.max(np.abs(d_odd[:, 0]))),
                "dJ_odd_y": float(np.max(np.abs(d_odd[:, 1]))),
                "J_odd_coh_x": float(np.max(np.abs(odd_c[:, 0]))),
                "J_odd_ds_x": float(np.max(np.abs(odd_d[:, 0]))),
                "imag_J_floor": imag_floor,
            },
        },
        "windows": {"main": main, "front": front, "back": back},
        "continuum": {
            "q": q_cont.tolist(),
            "abs_Jodd_C": cont["J_odd_coh"]["abs"],
            "abs_Jodd_DS": cont["J_odd_ds"]["abs"],
            "abs_dJodd": cont["dJ_odd"]["abs"],
            "peak_q": peak_q,
            "peak_nearest_integer": nearest,
            "peak_dq": dq,
            "peak_abs_dJ": float(abs_cont_dj[peak_i]),
        },
        "channels_H1_H6": channels,
        "n_identifiable_candidates": n_ident,
        "figures": figures,
        "source_hashes": hashes,
        "hash_abort": hash_abort(hashes),
    }
    dump(OUT / "q1a.json", report)
    dump(OUT / "q1a_summary.json", {
        "status": report["status"],
        "verdict": verdict,
        "n_identifiable_candidates": n_ident,
        "peak_q": peak_q,
        "job_status": report["coverage"]["job_status"],
        "max_top_fock": report["coverage"]["max_top_fock"],
        "rel_H1_H6": [ch["rel_dJ_over_J"] for ch in channels],
        "kinds": [ch["kind"] for ch in channels],
    })
    return report


def job_finished_ok(tag: str) -> bool:
    rec = load_job(tag)
    return bool(rec and rec.get("status") in ("ok", "ok_fock_alarm", "timeout_partial", "incomplete"))


def compact_job(rec):
    if rec is None:
        return None
    led = rec.get("ledger") or {}
    return {
        "status": rec.get("status"),
        "t_reached_over_T0": rec.get("t_reached_over_T0"),
        "t_reached": rec.get("t_reached"),
        "steps_completed": led.get("steps_completed"),
        "next_j": led.get("next_j"),
        "max_top_fock": rec.get("max_top_fock"),
        "max_next_top_fock": rec.get("max_next_top_fock"),
        "max_abs_norm_minus_1": rec.get("max_abs_norm_minus_1"),
        "max_last_taylor_term": led.get("max_last_taylor_term"),
        "max_n_ph": rec.get("max_n_ph"),
        "min_n_ph": rec.get("min_n_ph"),
        "fock_alarm_0p05": rec.get("fock_alarm_0p05"),
        "wall_seconds": rec.get("wall_seconds"),
        "seconds_per_step": rec.get("seconds_per_step"),
        "resumed": rec.get("resumed"),
        "resume_timeout": rec.get("resume_timeout"),
        "resume_from_next_j": rec.get("resume_from_next_j"),
        "covers_4T0": rec.get("covers_4T0"),
        "covers_5T0": rec.get("covers_5T0"),
        "covers_6T0": rec.get("covers_6T0"),
        "wave": rec.get("wave"),
        "finite_state": rec.get("finite_state"),
        "error": rec.get("error"),
        "n_samples": rec.get("n_samples"),
    }


def remaining_estimate(recs, s_per_step=0.150):
    out = {}
    for tag, _, _ in JOBS:
        rec = recs.get(tag) if recs else None
        done = 0
        if rec:
            done = int((rec.get("ledger") or {}).get("steps_completed") or 0)
        left = max(N_FULL - done, 0)
        out[tag] = {
            "steps_completed": done,
            "steps_left_to_6T0": left,
            "est_seconds_to_4T0_if_0p150": max((7436 - done) * s_per_step, 0.0),
            "est_seconds_to_5T0_if_0p150": max((9295 - done) * s_per_step, 0.0),
            "est_seconds_to_6T0_if_0p150": left * s_per_step,
            "prediction_not_guarantee": True,
        }
    return out


def preserve_wave1_json() -> None:
    src = OUT / "q1a.json"
    dst = OUT / "q1a_wave1.json"
    if src.is_file() and not dst.is_file():
        dst.write_text(src.read_text(encoding="utf-8"), encoding="utf-8")


def should_launch(tag: str, resume_timeout: bool = False) -> bool:
    rec = load_job(tag)
    if rec is None:
        return True
    if rec.get("status") in ("ok", "ok_fock_alarm"):
        return False
    if rec.get("status") == "timeout_partial":
        return bool(resume_timeout)
    return True


def drive(resume_timeout: bool = False) -> int:
    hashes = source_hashes()
    bad = hash_abort(hashes)
    if bad:
        dump(OUT / "q1a.json", {"status": "hash_abort", "hash_mismatch": bad, "stopped": True})
        progress("hash abort, no propagation")
        return 3
    OUT.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    if resume_timeout:
        preserve_wave1_json()
        progress("wave 2 resume-timeout from existing checkpoints; do not restart t=0")
    t0 = time.perf_counter()
    procs = {}
    logs = {}
    for tag, kind, neel in JOBS:
        if not should_launch(tag, resume_timeout=resume_timeout):
            progress(f"drive skip {tag} status={load_job(tag).get('status')}")
            continue
        logf = (OUT / f"log_{tag}.txt").open("a", encoding="utf-8")
        logs[tag] = logf
        envp = os.environ.copy()
        for k in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
            envp[k] = "1"
        cmd = [sys.executable, "-u", str(Path(__file__).resolve()), "--job", tag]
        if resume_timeout:
            cmd.append("--resume-timeout")
        progress(f"launch {tag} {' '.join(cmd[-2:])}")
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
    recs = {tag: load_job(tag) for tag, _, _ in JOBS}
    t_vals = [
        0.0 if recs[tag] is None else float(recs[tag].get("t_reached_over_T0") or 0.0)
        for tag, _, _ in JOBS
    ]
    t_min = min(t_vals) if t_vals else 0.0
    covers_4 = bool(t_min + 1e-9 >= 4.0)
    covers_5 = bool(t_min + 1e-9 >= 5.0)
    covers_6 = bool(t_min + 1e-9 >= 6.0)
    if covers_4:
        progress(f"analyze start t_min/T0={t_min:.3f} covers4={covers_4} covers5={covers_5} covers6={covers_6}")
        rec = analyze()
        dump(OUT / "q1a_wave2.json", {
            "status": rec.get("status"),
            "wave": 2,
            "verdict": rec.get("verdict"),
            "answer": rec.get("answer"),
            "t_min_over_T0": t_min,
            "covers_4T0": covers_4,
            "covers_5T0": covers_5,
            "covers_6T0": covers_6,
            "n_identifiable_candidates": rec.get("n_identifiable_candidates"),
            "job_status": {tag: None if recs[tag] is None else recs[tag].get("status") for tag, _, _ in JOBS},
            "jobs": {tag: compact_job(recs[tag]) for tag, _, _ in JOBS},
            "returncodes": codes,
            "driver_wall_s": time.perf_counter() - t0,
            "stopped": True,
        })
        progress("analyze done")
        return 0
    timeout_rep = {
        "task_tag": TASK_TAG,
        "record_date": RECORD_DATE,
        "protocol": PROTOCOL,
        "status": "timeout_partial_no_analyze",
        "wave": 2,
        "resume_timeout": bool(resume_timeout),
        "jobs_incomplete_no_analyze": True,
        "stopped": True,
        "analyze_ran": False,
        "spectra_computed": False,
        "t_min_over_T0": t_min,
        "covers_4T0": covers_4,
        "covers_5T0": covers_5,
        "covers_6T0": covers_6,
        "claim_level": "q1a_wave2_timeout_incomplete_coverage_not_frequency_resolved_discovery",
        "driver_wall_s": time.perf_counter() - t0,
        "driver_wall_cap_s": DRIVER_WALL_S,
        "job_wall_cap_s": JOB_WALL_S,
        "returncodes": codes,
        "job_status": {tag: None if recs[tag] is None else recs[tag].get("status") for tag, _, _ in JOBS},
        "jobs": {tag: compact_job(recs[tag]) for tag, _, _ in JOBS},
        "remaining_estimate_if_0p150_s_per_step": remaining_estimate(recs),
        "wave1_physics_preserved_in": str((OUT / "q1a_wave1.json").as_posix()),
        "physics_boundaries": {
            "integer_Fourier_is_not_certified_HHG": True,
            "time_domain_delta_Jodd_is_not_a_spectrum": True,
            "same_alpha_is_not_same_nbar": True,
            "QED_current_is_not_output_squeezing_or_entanglement": True,
            "A_star_mean_not_classical_A_star_0p8": True,
            "coupling_switch_can_prepare_transients": True,
        },
        "did_not": [
            "central two-cycle Fourier if 4T0 not reached",
            "Q1b", "Q1c", "Cn", "theta=pi", "squeezed vacuum", "GH5", "MTEF",
            "G1", "H2", "extra k", "nmax scan", "vacuum switch", "D2",
            "Gamma192", "T2", "Floquet", "server",
        ],
    }
    dump(OUT / "q1a_wave2.json", timeout_rep)
    dump(OUT / "q1a.json", timeout_rep)
    progress(f"not enough coverage for analyze t_min/T0={t_min:.3f}")
    return 1


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--job", type=str, default="")
    ap.add_argument("--analyze-only", action="store_true")
    ap.add_argument("--resume-timeout", action="store_true")
    args = ap.parse_args(argv)
    OUT.mkdir(parents=True, exist_ok=True)
    if args.job:
        rec = run_one_job(args.job, resume_timeout=args.resume_timeout)
        return 0 if rec.get("status") in ("ok", "ok_fock_alarm", "timeout_partial") else 1
    if args.analyze_only:
        rec = analyze()
        print(json.dumps(jsonable({
            "status": rec.get("status"), "verdict": rec.get("verdict"),
            "n_identifiable_candidates": rec.get("n_identifiable_candidates"),
        }), ensure_ascii=False, indent=2))
        return 0 if rec.get("status") in (
            "ok", "ok_incomplete_fall", "ok_full_switch_incomplete",
            "ok_main_window_no_back") else 1
    return drive(resume_timeout=args.resume_timeout)


if __name__ == "__main__":
    raise SystemExit(main())

