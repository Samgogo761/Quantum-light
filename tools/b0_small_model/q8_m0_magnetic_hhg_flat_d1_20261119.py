#!/usr/bin/env python3
"""D1: Gamma48 +/-N classical current on the approved 2+8+2 flat-top drive.

No CJJ, photon output, QED, squeezing, D2/shift48, Gamma192, or G1/H2.
Does not change the default eight-cycle sin^2 P1 path.
"""
from __future__ import annotations

import csv
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

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import q8_m0_classical as cl
import q8_m0_magnetic_hhg_aprime_20261118 as ap
import q8_m0_magnetic_hhg_source_20261114 as src
from q8_m0_model import R0, build_geometry, sha256_file
from q8_m0_s1_kernel import parallel_perp_basis

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results/q_line/magnetic_hhg_flat_d1_20261119"
ARCH_V48 = ROOT / "results/q_line/cjj_m0_p1_a2_s1_20260917/V48"
MU = -0.009010620372198276
NK = 48
NTPC = 400
PROP_BUDGET_S = 300.0
JOBS = (
    {"neel": 1, "sx": 0.0, "sy": 0.0},
    {"neel": -1, "sx": 0.0, "sy": 0.0},
)


def dump(path, obj):
    path = Path(path)
    if path.exists():
        path.unlink()
    src.dump(path, obj)


def job_stem(job):
    sx = str(job["sx"]).replace(".", "p")
    sy = str(job["sy"]).replace(".", "p")
    return f"G48_sx{sx}_sy{sy}_N{job['neel']:+d}"


def cycle_index(prot, cycle, ntpc=NTPC):
    return int(round(cycle * prot.period / (prot.period / ntpc)))


def rms_vec(x):
    x = np.asarray(x, dtype=float)
    if x.ndim == 1:
        return float(np.sqrt(np.mean(x * x))) if x.size else 0.0
    return float(np.sqrt(np.mean(np.sum(x * x, axis=-1)))) if x.size else 0.0


def half_period_block(t, jp, jm, geo, prot, cycle_lo, cycle_hi, sign=+1):
    dt = float(np.mean(np.diff(t)))
    n_half = int(round(0.5 * prot.period / dt))
    i0 = int(round(cycle_lo * prot.period / dt))
    i1 = int(round(cycle_hi * prot.period / dt))
    i1 = min(i1, len(t) - n_half)
    i0 = max(i0, 0)
    sl = slice(i0, i1)
    sls = slice(i0 + n_half, i1 + n_half)
    a = cl.kappa_shift(t, geo, prot, sign=sign)
    je = 0.5 * (jp + jm)
    jo = 0.5 * (jp - jm)
    rA = a[sls] + a[sl]
    r_map = jm[sl] + jp[sls]
    r_e = je[sls] + je[sl]
    r_o = jo[sls] - jo[sl]

    def pack(res, sig):
        num = rms_vec(res)
        den = rms_vec(sig)
        return num, den, (num / den if den > 0 else None)

    a_abs, a_den, a_rel = pack(rA, a[sl])
    e_abs, e_den, e_rel = pack(r_e, je[sl])
    o_abs, o_den, o_rel = pack(r_o, jo[sl])
    m_abs, m_den, m_rel = pack(r_map, jp[sls])
    return {
        "cycle_lo": cycle_lo,
        "cycle_hi": cycle_hi,
        "n_samples": int(i1 - i0),
        "n_half": n_half,
        "t_start": float(t[i0]),
        "t_end": float(t[i1 - 1]) if i1 > i0 else None,
        "drive_rel": a_rel,
        "drive_abs": a_abs,
        "drive_rms": a_den,
        "map_rel": m_rel,
        "even_rel": e_rel,
        "even_abs": e_abs,
        "rms_Je": e_den,
        "odd_rel": o_rel,
        "odd_abs": o_abs,
        "rms_Jo": o_den,
        "rms_Jo_over_Je": (o_den / e_den if e_den > 0 else None),
    }


def rect_fourier(times, signal, omega, t0, t1):
    """J_q = 1/(t1-t0) ∫_{t0}^{t1} J(t) exp(i omega t) dt."""
    t = np.asarray(times, dtype=float)
    sig = np.asarray(signal)
    dt = float(np.mean(np.diff(t)))
    i0 = int(round(t0 / dt))
    i1 = int(round(t1 / dt))
    i0 = max(i0, 0)
    i1 = min(i1, len(t) - 1)
    sl = slice(i0, i1 + 1)
    kern = np.exp(1j * omega * t[sl])
    if sig.ndim == 1:
        integ = np.trapezoid(sig[sl] * kern, t[sl])
    else:
        integ = np.trapezoid(sig[sl] * kern[:, None], t[sl], axis=0)
    return integ / (t1 - t0)


def windowed_rect_fourier(times, signal, omega, t0, t1, kind):
    t = np.asarray(times, dtype=float)
    sig = np.asarray(signal)
    dt = float(np.mean(np.diff(t)))
    i0 = int(round(t0 / dt))
    i1 = int(round(t1 / dt))
    i0 = max(i0, 0)
    i1 = min(i1, len(t) - 1)
    sl = slice(i0, i1 + 1)
    tau = t[sl]
    if kind == "rect":
        wgt = np.ones_like(tau)
    elif kind == "sin2":
        wgt = np.sin(np.pi * (tau - t0) / (t1 - t0)) ** 2
    else:
        raise ValueError(kind)
    kern = wgt * np.exp(1j * omega * tau)
    if sig.ndim == 1:
        integ = np.trapezoid(sig[sl] * kern, tau)
    else:
        integ = np.trapezoid(sig[sl] * kern[:, None], tau, axis=0)
    return integ / (t1 - t0)


def cvec(z):
    z = np.asarray(z, dtype=complex).reshape(-1)
    return {
        "re": [float(x.real) for x in z],
        "im": [float(x.imag) for x in z],
        "abs": [float(abs(x)) for x in z],
        "arg_deg": [float(np.angle(x, deg=True)) for x in z],
        "norm": float(np.linalg.norm(z)),
    }


def domain_phase(jp, jm):
    a = np.asarray(jp, dtype=complex).reshape(-1)
    b = np.asarray(jm, dtype=complex).reshape(-1)
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na == 0.0 or nb == 0.0:
        ov = None
        ph = None
    else:
        ovl = np.vdot(a / na, b / nb)
        ov = float(abs(ovl))
        ph = float(np.angle(ovl, deg=True))
    return {
        "plus": cvec(a),
        "minus": cvec(b),
        "overlap": ov,
        "phase_deg": ph,
        "norm_rel": float(abs(na - nb) / max(na, nb, 1e-30)),
    }


def job_identity(job, prot, hashes):
    return {
        "task_label": "20261119_D1",
        "sx": job["sx"],
        "sy": job["sy"],
        "neel": job["neel"],
        "field_sign": 1,
        "nk": NK,
        "nt_per_cycle": NTPC,
        "mu": MU,
        "params": R0.as_dict(),
        "protocol": prot.__dict__,
        "algorithm": cl.ALGORITHM,
        "object": "classical_J_sheet_only",
        "cjj_started": False,
        "photon_output": "not_run",
        "quantum_input": "not_run",
        "source_hashes": hashes,
    }


def resume_ok(path_meta, job, prot):
    if not path_meta.exists():
        return False
    meta = json.loads(path_meta.read_text(encoding="utf-8"))
    return (
        meta.get("sx") == job["sx"]
        and meta.get("sy") == job["sy"]
        and meta.get("neel") == job["neel"]
        and meta.get("nk") == NK
        and meta.get("nt_per_cycle") == NTPC
        and meta.get("mu") == MU
        and meta.get("field_sign") == 1
        and meta.get("protocol") == prot.__dict__
        and meta.get("cjj_started") is False
        and meta.get("params", {}).get("label") == "R0"
    )


def write_job(run, ident, path_npz, path_meta):
    np.savez_compressed(
        path_npz,
        times=run["times"],
        J_sheet=run["J_sheet"],
        A_over=run["A_over"],
        kf=run["kf"],
        w_k=run["w_k"],
        shift_xy=np.array(run["shift_xy"], dtype=float),
        neel=np.int32(run["neel"]),
        field_sign=np.int32(run["field_sign"]),
        dt=run["dt"],
        wall_seconds=run["wall_seconds"],
        U_e0=run["U_e0"],
        U_eT=run["U_eT"],
        delta_U_e=run["delta_U_e"],
        work_integral=run["work_integral"],
        J_equilibrium_t0=np.array(run["J_equilibrium_t0"]),
        diag_t=run["diag"]["t"],
        diag_n_exc=run["diag"]["n_exc"],
        diag_tr_rho=run["diag"]["tr_rho"],
        diag_herm=run["diag"]["herm"],
        diag_idem=run["diag"]["idem"],
    )
    dump(path_meta, ident)


def load_job(path_npz):
    with np.load(path_npz, allow_pickle=False) as z:
        return {
            "times": z["times"],
            "J_sheet": z["J_sheet"],
            "A_over": z["A_over"],
            "kf": z["kf"],
            "shift_xy": tuple(float(x) for x in z["shift_xy"]),
            "neel": int(z["neel"]),
            "wall_seconds": float(z["wall_seconds"]),
            "delta_U_e": float(z["delta_U_e"]),
            "work_integral": float(z["work_integral"]),
            "U_e0": float(z["U_e0"]),
            "diag_n_exc": z["diag_n_exc"],
            "diag_herm": z["diag_herm"],
            "diag_idem": z["diag_idem"],
            "diag_tr_rho": z["diag_tr_rho"],
            "J_equilibrium_t0": z["J_equilibrium_t0"],
        }


def run_jobs(geo, prot, hashes):
    OUT.mkdir(parents=True, exist_ok=True)
    records = []
    prop_t0 = time.perf_counter()
    timed_out = False
    for job in JOBS:
        stem = job_stem(job)
        path_npz = OUT / f"{stem}.npz"
        path_meta = OUT / f"{stem}.meta.json"
        rec = {"stem": stem, **job}
        if path_npz.exists() and resume_ok(path_meta, job, prot):
            rec.update(status="resumed", wall_s=0.0, path=str(path_npz))
            records.append(rec)
            print("resume", stem, flush=True)
            continue
        remaining = PROP_BUDGET_S - (time.perf_counter() - prop_t0)
        if remaining < 20.0:
            rec.update(status="skipped_budget", remaining_s=remaining)
            records.append(rec)
            timed_out = True
            print("budget stop before", stem, remaining, flush=True)
            break
        print(f"propagate {stem} remaining={remaining:.1f}s", flush=True)
        t1 = time.perf_counter()
        run = cl.propagate(
            geo,
            R0,
            prot,
            nk=NK,
            nt_per_cycle=NTPC,
            neel=job["neel"],
            sign=1,
            mu=MU,
            verbose=True,
            grid_shift=0.0,
            subspace=False,
            k_orders=(),
            shift_xy=(job["sx"], job["sy"]),
        )
        wall = time.perf_counter() - t1
        kf_expect, _kc, _w = cl.k_grid_xy(geo, NK, job["sx"], job["sy"])
        np.testing.assert_array_equal(run["kf"], kf_expect)
        ident = job_identity(job, prot, hashes)
        ident.update(
            {
                "wall_seconds": run["wall_seconds"],
                "wall_outer_s": wall,
                "U_e0": run["U_e0"],
                "U_eT": run["U_eT"],
                "delta_U_e": run["delta_U_e"],
                "work_integral": run["work_integral"],
                "J_equilibrium_t0": run["J_equilibrium_t0"],
                "shift_xy": list(run["shift_xy"]),
                "n_times": int(run["times"].shape[0]),
                "n_k": int(run["kf"].shape[0]),
            }
        )
        if path_npz.exists():
            path_npz.unlink()
        if path_meta.exists():
            path_meta.unlink()
        write_job(run, ident, path_npz, path_meta)
        rec.update(
            status="computed",
            wall_s=wall,
            path=str(path_npz),
            delta_U_e=run["delta_U_e"],
            n_k=int(run["kf"].shape[0]),
        )
        records.append(rec)
        print("done", stem, f"{wall:.1f}s", flush=True)
        if time.perf_counter() - prop_t0 > PROP_BUDGET_S:
            timed_out = True
            print("budget exhausted after", stem, flush=True)
            break
    return records, time.perf_counter() - prop_t0, timed_out


def harmonic_table(t, jp, jm, prot, t0, t1, kind, basis):
    jp_b = jp @ basis.T
    jm_b = jm @ basis.T
    je = 0.5 * (jp_b + jm_b)
    jo = 0.5 * (jp_b - jm_b)
    rows = []
    packed = {}
    for q in range(1, 10):
        omega = q * prot.omega0
        pe = windowed_rect_fourier(t, je, omega, t0, t1, kind)
        po = windowed_rect_fourier(t, jo, omega, t0, t1, kind)
        pp = windowed_rect_fourier(t, jp_b, omega, t0, t1, kind)
        pm = windowed_rect_fourier(t, jm_b, omega, t0, t1, kind)
        ph = domain_phase(pp, pm)
        rec = {
            "q": q,
            "parity": "even" if q % 2 == 0 else "odd",
            "window": kind,
            "t0_cycles": t0 / prot.period,
            "t1_cycles": t1 / prot.period,
            "J_even_norm": float(np.linalg.norm(pe)),
            "J_odd_norm": float(np.linalg.norm(po)),
            "odd_over_even": float(np.linalg.norm(po) / np.linalg.norm(pe)) if np.linalg.norm(pe) else None,
            "plus_minus_overlap": ph["overlap"],
            "plus_minus_phase_deg": ph["phase_deg"],
            "plus_minus_norm_rel": ph["norm_rel"],
            "J_even": cvec(pe),
            "J_odd": cvec(po),
            "J_plus": cvec(pp),
            "J_minus": cvec(pm),
        }
        rows.append(rec)
        packed[q] = rec
    return rows, packed


def old_integer_sin2(t, jp, jm, old_prot, basis):
    jp_b = jp @ basis.T
    jm_b = jm @ basis.T
    rows = []
    packed = {}
    for q in range(1, 10):
        pp = cl.project_harmonic(t, jp_b, q * old_prot.omega0, "sin2")
        pm = cl.project_harmonic(t, jm_b, q * old_prot.omega0, "sin2")
        pe = 0.5 * (pp + pm)
        po = 0.5 * (pp - pm)
        ph = domain_phase(pp, pm)
        rec = {
            "q": q,
            "parity": "even" if q % 2 == 0 else "odd",
            "window": "old_fullpulse_sin2",
            "J_even_norm": float(np.linalg.norm(pe)),
            "J_odd_norm": float(np.linalg.norm(po)),
            "odd_over_even": float(np.linalg.norm(po) / np.linalg.norm(pe)) if np.linalg.norm(pe) else None,
            "plus_minus_overlap": ph["overlap"],
            "plus_minus_phase_deg": ph["phase_deg"],
            "plus_minus_norm_rel": ph["norm_rel"],
            "J_even": cvec(pe),
            "J_odd": cvec(po),
        }
        rows.append(rec)
        packed[q] = rec
    return rows, packed


def classify(main_hp, old_hp, h2, h2_old):
    re = main_hp["even_rel"]
    ro = main_hp["odd_rel"]
    re_old = old_hp["even_rel"]
    ro_old = old_hp["odd_rel"]
    jo = main_hp["rms_Jo"]
    jo_old = old_hp["rms_Jo"]
    odd_src = h2["J_odd_norm"]
    even_src = h2["J_even_norm"]
    ratio = h2["odd_over_even"]
    identifiable = odd_src > 0.2 * h2_old["J_odd_norm"] and odd_src > 1e-9
    re_improved = re is not None and re_old is not None and re < 0.6 * re_old
    ro_improved = ro is not None and ro_old is not None and (ro <= 0.5 or ro < 0.6 * ro_old)
    ro_unchanged = ro is not None and ro_old is not None and ro > 0.8 * ro_old
    near_floor = (not identifiable) or (jo < 0.2 * jo_old and odd_src < 0.2 * h2_old["J_odd_norm"])
    if re_improved and ro_improved and identifiable:
        case = 1
        d2 = "eligible_after_report_not_auto"
        note = "Re and Ro both dropped and H2 odd source remains identifiable."
    elif re_improved and ro_unchanged:
        case = 2
        d2 = "blocked"
        note = "Plateau restored drive/even symmetry, but magnetic-odd current did not enter covariance."
    elif ro_improved and near_floor:
        case = 3
        d2 = "blocked"
        note = "Ro dropped but the odd source is near the numerical floor; not a selection rule."
    elif ro_improved and identifiable:
        case = 1
        d2 = "eligible_after_report_not_auto"
        note = "Ro improved with identifiable H2 odd source; Re reported separately."
    else:
        case = 2
        d2 = "blocked"
        note = "No substantial J_odd half-period improvement on Gamma48."
    return {
        "case": case,
        "d2": d2,
        "note": note,
        "re_new": re,
        "ro_new": ro,
        "re_old_gamma48": re_old,
        "ro_old_gamma48": ro_old,
        "re_improved": re_improved,
        "ro_improved": ro_improved,
        "identifiable_h2_odd": identifiable,
        "near_floor": near_floor,
        "h2_odd_over_even": ratio,
        "h2_odd_norm": odd_src,
        "h2_even_norm": even_src,
        "start_d2": False,
        "start_qed": False,
        "start_squeezing": False,
        "lengthen_pulse": False,
        "engineering_Ro_le_0.5": bool(ro is not None and ro <= 0.5),
        "not_a_universal_threshold": True,
    }


def write_tsv(path, rows):
    path = Path(path)
    if path.exists():
        path.unlink()
    if not rows:
        return
    fields = list(rows[0].keys())
    flat = []
    for r in rows:
        item = {}
        for k, v in r.items():
            if isinstance(v, dict):
                continue
            item[k] = v
        flat.append(item)
    fields = [k for k in rows[0] if k in flat[0]]
    with path.open("x", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, delimiter="\t")
        w.writeheader()
        w.writerows(flat)


def make_figures(t, jp, jm, prot, hp_cycles, main_rows, old_rows, old_hp):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.family": ["Microsoft YaHei", "DejaVu Sans"],
            "axes.unicode_minus": False,
            "font.size": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.15,
        }
    )
    cycles = [r["cycle_lo"] + 0.5 for r in hp_cycles]
    fig, ax = plt.subplots(figsize=(8.6, 4.6), layout="constrained")
    ax.plot(cycles, [r["drive_rel"] for r in hp_cycles], "o-", color="#444", label="drive A")
    ax.plot(cycles, [r["even_rel"] for r in hp_cycles], "s-", color="#1464a0", label=r"$J^e$")
    ax.plot(cycles, [r["odd_rel"] for r in hp_cycles], "^-", color="#b85b2a", label=r"$J^o$")
    ax.axhline(old_hp["even_rel"], color="#1464a0", ls="--", lw=0.8, label=r"old Γ48 $J^e$ 3–5")
    ax.axhline(old_hp["odd_rel"], color="#b85b2a", ls="--", lw=0.8, label=r"old Γ48 $J^o$ 3–5")
    ax.set_xlabel("plateau cycle")
    ax.set_ylabel("half-period relative RMS")
    ax.set_title("D1 Γ48：平台逐周期半周期残差")
    ax.legend(ncols=2, fontsize=8)
    fig.savefig(OUT / "figure_half_period.png", dpi=160)
    fig.savefig(OUT / "figure_half_period.pdf")
    plt.close(fig)

    qs = [r["q"] for r in main_rows]
    fig, axs = plt.subplots(1, 2, figsize=(10.4, 4.4), layout="constrained")
    axs[0].semilogy(qs, [r["J_even_norm"] for r in main_rows], "o-", color="#1464a0", label="D1 rect 6")
    axs[0].semilogy(qs, [r["J_even_norm"] for r in old_rows], "s--", color="#7c8797", label="old Γ48 sin²")
    axs[1].semilogy(qs, [r["J_odd_norm"] for r in main_rows], "o-", color="#b85b2a", label="D1 rect 6")
    axs[1].semilogy(qs, [r["J_odd_norm"] for r in old_rows], "s--", color="#7c8797", label="old Γ48 sin²")
    axs[0].set_title(r"$\|J^e_q\|$")
    axs[1].set_title(r"$\|J^o_q\|$")
    for ax in axs:
        ax.set_xlabel("harmonic q")
        ax.set_xticks(qs)
        ax.legend(fontsize=8)
    fig.suptitle("同网格对照：新协议矩形窗 ≠ 旧协议全脉冲 sin²，绝对值不可当增强")
    fig.savefig(OUT / "figure_even_odd_source.png", dpi=160)
    fig.savefig(OUT / "figure_even_odd_source.pdf")
    plt.close(fig)

    t0 = prot.period
    je = 0.5 * (jp + jm)
    jo = 0.5 * (jp - jm)
    keep = (t >= 2 * t0) & (t <= 10 * t0)
    fig, axs = plt.subplots(2, 1, figsize=(8.8, 5.6), sharex=True, layout="constrained")
    axs[0].plot(t[keep] / t0, je[keep, 0], color="#1464a0", lw=0.8, label="∥")
    axs[0].plot(t[keep] / t0, je[keep, 1], color="#5aa0d4", lw=0.8, label="⊥")
    axs[1].plot(t[keep] / t0, jo[keep, 0], color="#b85b2a", lw=0.8, label="∥")
    axs[1].plot(t[keep] / t0, jo[keep, 1], color="#e0a070", lw=0.8, label="⊥")
    axs[0].set_ylabel(r"$J^e$")
    axs[1].set_ylabel(r"$J^o$")
    axs[1].set_xlabel(r"$t/T_0$")
    axs[0].set_title("平台内磁序偶/奇电流（笛卡尔 x/y）")
    axs[0].legend(fontsize=8)
    axs[1].legend(fontsize=8)
    fig.savefig(OUT / "figure_plateau_current.png", dpi=160)
    fig.savefig(OUT / "figure_plateau_current.pdf")
    plt.close(fig)


def analyze(geo, prot, old_prot, hashes, jobs):
    plus = load_job(OUT / f"{job_stem(JOBS[0])}.npz")
    minus = load_job(OUT / f"{job_stem(JOBS[1])}.npz")
    t = plus["times"]
    jp = plus["J_sheet"]
    jm = minus["J_sheet"]
    np.testing.assert_array_equal(t, minus["times"])
    if abs(t[0]) > 1e-12 or abs(t[-1] - prot.t_pulse) > 1e-8:
        raise ValueError("flat-top time grid mismatch")
    par, perp = parallel_perp_basis(geo)
    basis = np.stack((par, perp))
    t0 = prot.period
    main_hp = half_period_block(t, jp, jm, geo, prot, 3.0, 9.0)
    plat_hp = half_period_block(t, jp, jm, geo, prot, 2.0, 10.0)
    hp_cycles = [half_period_block(t, jp, jm, geo, prot, float(c), float(c + 1)) for c in range(2, 9)]
    main_rows, main_pack = harmonic_table(t, jp, jm, prot, 3 * t0, 9 * t0, "rect", basis)
    plat_rows, plat_pack = harmonic_table(t, jp, jm, prot, 2 * t0, 10 * t0, "rect", basis)
    sm_rows, sm_pack = harmonic_table(t, jp, jm, prot, 3 * t0, 9 * t0, "sin2", basis)
    per_cycle_h = []
    for c in range(3, 9):
        rows, packed = harmonic_table(t, jp, jm, prot, c * t0, (c + 1) * t0, "rect", basis)
        for q in (2, 4, 6, 8):
            rec = dict(packed[q])
            rec["cycle"] = c
            per_cycle_h.append(rec)
    t_old, jp_old, jm_old, info_old, paths_old = src.load("Gamma48", geo, old_prot)
    old_hp = half_period_block(t_old, jp_old, jm_old, geo, old_prot, 3.0, 5.0)
    old_rows, old_pack = old_integer_sin2(t_old, jp_old, jm_old, old_prot, basis)
    verdict = classify(main_hp, old_hp, main_pack[2], old_pack[2])
    make_figures(t, jp, jm, prot, hp_cycles, main_rows, old_rows, old_hp)
    write_tsv(OUT / "half_period_cycles.tsv", hp_cycles)
    write_tsv(OUT / "harmonics_rect6.tsv", main_rows)
    write_tsv(OUT / "harmonics_rect8.tsv", plat_rows)
    write_tsv(OUT / "harmonics_sin2_6.tsv", sm_rows)
    write_tsv(OUT / "harmonics_old_gamma48_sin2.tsv", old_rows)
    write_tsv(OUT / "harmonics_per_cycle.tsv", per_cycle_h)
    diag = {
        "plus": {
            "n_exc_max": float(np.max(plus["diag_n_exc"])),
            "herm_max": float(np.max(plus["diag_herm"])),
            "idem_max": float(np.max(plus["diag_idem"])),
            "tr_dev": float(np.max(np.abs(plus["diag_tr_rho"] - 4.0))),
            "delta_U_e": plus["delta_U_e"],
            "work_integral": plus["work_integral"],
            "energy_balance_abs": abs(plus["delta_U_e"] - plus["work_integral"]),
            "wall_seconds": plus["wall_seconds"],
        },
        "minus": {
            "n_exc_max": float(np.max(minus["diag_n_exc"])),
            "herm_max": float(np.max(minus["diag_herm"])),
            "idem_max": float(np.max(minus["diag_idem"])),
            "tr_dev": float(np.max(np.abs(minus["diag_tr_rho"] - 4.0))),
            "delta_U_e": minus["delta_U_e"],
            "work_integral": minus["work_integral"],
            "energy_balance_abs": abs(minus["delta_U_e"] - minus["work_integral"]),
            "wall_seconds": minus["wall_seconds"],
        },
    }
    return {
        "main_half_period_3_to_9": main_hp,
        "plateau_half_period_2_to_10": plat_hp,
        "per_cycle_half_period": hp_cycles,
        "old_gamma48_half_period_3_to_5": old_hp,
        "harmonics_rect6": main_rows,
        "harmonics_rect8": plat_rows,
        "harmonics_sin2_6": sm_rows,
        "harmonics_old_gamma48_sin2": old_rows,
        "harmonics_even_per_cycle": per_cycle_h,
        "verdict": verdict,
        "diagnostics": diag,
        "old_gamma48_info": info_old,
        "old_paths": [str(p) for p in paths_old],
        "basis_parallel": par,
        "basis_perp": perp,
        "n_times": int(t.shape[0]),
        "n_k": int(plus["kf"].shape[0]),
        "jobs": jobs,
        "source_hashes": hashes,
    }


def run():
    start = time.perf_counter()
    OUT.mkdir(parents=True, exist_ok=True)
    geo = build_geometry()
    prot = cl.p1_a2_flat_282_protocol(0.8)
    old_prot = cl.p1_a2_protocol(0.8)
    hashes = {
        "model": sha256_file(HERE / "q8_m0_model.py"),
        "classical": sha256_file(HERE / "q8_m0_classical.py"),
        "d1_script": src.sha(Path(__file__)),
        "source_20261114": src.sha(HERE / "q8_m0_magnetic_hhg_source_20261114.py"),
        "aprime_20261118": src.sha(HERE / "q8_m0_magnetic_hhg_aprime_20261118.py"),
    }
    archive_paths = [ARCH_V48 / f"V48_N{n:+d}_A+1.npz" for n in (1, -1)]
    archive_paths += [p.with_suffix(".meta.json") for p in archive_paths]
    before = {str(p): src.sha(p) for p in archive_paths}
    kf0, _kc, _w = cl.k_grid(geo, NK, shift=0.0)
    kf1, _kc1, _w1 = cl.k_grid_xy(geo, NK, 0.0, 0.0)
    np.testing.assert_array_equal(kf0, kf1)
    jobs, prop_wall, timed_out = run_jobs(geo, prot, hashes)
    complete = (not timed_out) and all(r.get("status") in ("computed", "resumed") for r in jobs) and len(jobs) == 2
    payload = {
        "task_label": "20261119_D1",
        "executed_at_utc": datetime.now(timezone.utc).isoformat(),
        "protocol": prot.__dict__,
        "old_protocol": old_prot.__dict__,
        "jobs": jobs,
        "propagate_wall_s": prop_wall,
        "propagate_budget_s": PROP_BUDGET_S,
        "timed_out": timed_out,
        "complete_two_jobs": complete,
        "inputs_sha256_before": before,
        "source_hashes": hashes,
        "openblas_num_threads": os.environ.get("OPENBLAS_NUM_THREADS"),
        "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
        "cjj_started": False,
        "d2_started": False,
        "qed_started": False,
    }
    if not complete:
        payload.update(status="partial", analysis_started=False, wall_s=time.perf_counter() - start)
        dump(OUT / "d1_status.json", payload)
        print("partial stop", payload["status"], flush=True)
        return payload
    analysis = analyze(geo, prot, old_prot, hashes, jobs)
    after = {str(p): src.sha(p) for p in archive_paths}
    payload.update(
        {
            "status": "ok",
            "analysis_started": True,
            "analysis": analysis,
            "inputs_unchanged": before == after,
            "inputs_sha256_after": after,
            "wall_s": time.perf_counter() - start,
        }
    )
    dump(OUT / "d1_status.json", payload)
    print(
        "status",
        payload["status"],
        "case",
        analysis["verdict"]["case"],
        "Ro",
        analysis["verdict"]["ro_new"],
        "d2",
        analysis["verdict"]["d2"],
        "wall",
        payload["wall_s"],
        flush=True,
    )
    return payload


if __name__ == "__main__":
    run()
