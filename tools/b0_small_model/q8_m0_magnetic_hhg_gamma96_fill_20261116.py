#!/usr/bin/env python3
"""Fill missing 48x48 subgrids of Gamma96 and compare magnetic HHG candidates.

Classical sheet current only. No CJJ, photon output, QED, or G1/H2.
"""
from __future__ import annotations

import csv
import hashlib
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import q8_m0_classical as cl
import q8_m0_magnetic_hhg_bz_subgrid_20261115 as bz
import q8_m0_magnetic_hhg_source_20261114 as src
from q8_m0_model import R0, build_geometry, sha256_file
from q8_m0_s1_kernel import parallel_perp_basis

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results/q_line/magnetic_hhg_gamma96_fill_20261116"
ARCH_V48 = ROOT / "results/q_line/cjj_m0_p1_a2_s1_20260917/V48"
ARCH_V48S = ROOT / "results/q_line/cjj_m0_p1_a2_s1_20260917/V48S"
MU = -0.009010620372198276
NK = 48
NTPC = 400
PROP_BUDGET_S = 600.0
ARCHIVED_MODEL = "0dbcb90fad134a48f3b005022eee1f16b74a8529d195a494ecbdc68860e5509f"
JOBS = (
    {"tag": "odd_even", "sx": 0.5, "sy": 0.0, "neel": 1, "gamma96_parity": "odd-even"},
    {"tag": "odd_even", "sx": 0.5, "sy": 0.0, "neel": -1, "gamma96_parity": "odd-even"},
    {"tag": "even_odd", "sx": 0.0, "sy": 0.5, "neel": 1, "gamma96_parity": "even-odd"},
    {"tag": "even_odd", "sx": 0.0, "sy": 0.5, "neel": -1, "gamma96_parity": "even-odd"},
)
CHANNELS = bz.CHANNELS
HALFWIDTHS = bz.HALFWIDTHS
WINDOWS = bz.WINDOWS


def sha(path):
    return src.sha(path)


def dump(path, obj):
    path = Path(path)
    if path.exists():
        path.unlink()
    src.dump(path, obj)


def git_blob_sha256(rel: str) -> str:
    blob = subprocess.check_output(["git", "-C", str(ROOT), "show", f"HEAD:{rel}"])
    return hashlib.sha256(blob).hexdigest()


def git_diff_stat(rel: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(ROOT), "diff", "--", rel],
        text=True,
        encoding="utf-8",
    )


def job_stem(job):
    sx = str(job["sx"]).replace(".", "p")
    sy = str(job["sy"]).replace(".", "p")
    return f"G48_sx{sx}_sy{sy}_N{job['neel']:+d}"


def compatibility(geo, prot):
    model_live = sha256_file(HERE / "q8_m0_model.py")
    model_head = git_blob_sha256("tools/b0_small_model/q8_m0_model.py")
    classical_live = sha256_file(HERE / "q8_m0_classical.py")
    classical_head = git_blob_sha256("tools/b0_small_model/q8_m0_classical.py")
    model_diff = git_diff_stat("tools/b0_small_model/q8_m0_model.py")
    only_k_frac = "def k_frac" in model_diff and "def h8" not in model_diff and "current_ops" not in model_diff
    kf_a, kc_a, w_a = cl.k_grid(geo, NK, shift=0.0)
    kf_b, kc_b, w_b = cl.k_grid_xy(geo, NK, 0.0, 0.0)
    kf_c, kc_c, w_c = cl.k_grid(geo, NK, shift=0.5)
    kf_d, kc_d, w_d = cl.k_grid_xy(geo, NK, 0.5, 0.5)
    iso_ok = (
        np.array_equal(kf_a, kf_b)
        and np.array_equal(kc_a, kc_b)
        and np.array_equal(w_a, w_b)
        and np.array_equal(kf_c, kf_d)
        and np.array_equal(kc_c, kc_d)
        and np.array_equal(w_c, w_d)
    )
    metas = []
    for folder, name, shift in ((ARCH_V48, "V48", 0.0), (ARCH_V48S, "V48S", 0.5)):
        for neel in (1, -1):
            meta = json.loads((folder / f"{name}_N{neel:+d}_A+1.meta.json").read_text(encoding="utf-8"))
            metas.append(
                {
                    "name": name,
                    "neel": neel,
                    "protocol_matches": meta["protocol"] == prot.__dict__,
                    "nk": meta["nk"],
                    "nt_per_cycle": meta["nt_per_cycle"],
                    "mu": meta["mu"],
                    "grid_shift": meta["grid_shift"],
                    "expected_shift": shift,
                    "params_label": meta["params"]["label"],
                    "model_sha256": meta["source_hashes"]["model_module"]["sha256"],
                    "algorithm": meta["algorithm"],
                }
            )
    return {
        "model_live_sha256": model_live,
        "model_git_head_sha256": model_head,
        "model_archived_sha256": ARCHIVED_MODEL,
        "model_head_matches_archive": model_head == ARCHIVED_MODEL,
        "model_live_differs_only_by_k_frac": only_k_frac,
        "model_k_frac_is_coordinate_helper": True,
        "classical_live_sha256": classical_live,
        "classical_git_head_sha256": classical_head,
        "isotropic_k_grid_unchanged": bool(iso_ok),
        "propagate_algorithm": cl.ALGORITHM,
        "hamiltonian_current_peierls": "unchanged in q8_m0_model.py; live file only adds k_frac",
        "time_integrator": "midpoint_unitary_exponential_v1; inner eigh/current loop not rewritten",
        "need_h_skip_does_not_skip_current": True,
        "cjj_started": False,
        "archive_protocol_rows": metas,
        "archive_protocol_aligned": all(
            r["protocol_matches"] and r["nk"] == 48 and r["nt_per_cycle"] == 400 and r["mu"] == MU and r["params_label"] == "R0"
            for r in metas
        ),
        "physics_comparable_to_V48": bool(iso_ok and only_k_frac and model_head == ARCHIVED_MODEL),
    }


def membership(geo):
    g96 = bz.frac_grid(96, 0.0, 0.0).reshape(96, 96, 2)
    slices = {
        "Gamma48": (g96[0::2, 0::2], cl.k_grid_xy(geo, 48, 0.0, 0.0)[0].reshape(48, 48, 2)),
        "shift48": (g96[1::2, 1::2], cl.k_grid_xy(geo, 48, 0.5, 0.5)[0].reshape(48, 48, 2)),
        "odd_even": (g96[1::2, 0::2], cl.k_grid_xy(geo, 48, 0.5, 0.0)[0].reshape(48, 48, 2)),
        "even_odd": (g96[0::2, 1::2], cl.k_grid_xy(geo, 48, 0.0, 0.5)[0].reshape(48, 48, 2)),
    }
    maxabs = {name: float(np.max(np.abs(a - b))) for name, (a, b) in slices.items()}
    sets = {name: bz.as_set(b.reshape(-1, 2)) for name, (_a, b) in slices.items()}
    names = list(sets)
    disjoint = all(len(sets[names[i]] & sets[names[j]]) == 0 for i in range(4) for j in range(i + 1, 4))
    union = set().union(*sets.values())
    g96_set = bz.as_set(g96.reshape(-1, 2))
    kf96, kc96, w96 = cl.k_grid(geo, 96, shift=0.0)
    return {
        "fractional_maxabs": maxabs,
        "union_covers_gamma96": union == g96_set,
        "union_count": len(union),
        "four_subgrids_disjoint": disjoint,
        "inversion_closed": {name: bz.inversion_closed(b.reshape(-1, 2)) for name, (_a, b) in slices.items()},
        "w96": float(w96[0]),
        "merge_weight_identity": abs(0.25 / 2304.0 - float(w96[0])) < 1e-18,
        "identity_holds": all(v == 0.0 for v in maxabs.values()) and disjoint and union == g96_set,
        "n_k96": int(kf96.shape[0]),
    }


def write_job(job, run, ident, path_npz, path_meta):
    np.savez_compressed(
        path_npz,
        times=run["times"],
        J_sheet=run["J_sheet"],
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
            "kf": z["kf"],
            "shift_xy": tuple(float(x) for x in z["shift_xy"]),
            "neel": int(z["neel"]),
        }


def job_identity(job, prot, hashes):
    return {
        "task_label": "20261116",
        "tag": job["tag"],
        "gamma96_parity": job["gamma96_parity"],
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
        and meta.get("protocol") == prot.__dict__
        and meta.get("cjj_started") is False
    )


def run_jobs(geo, prot, hashes, memb):
    OUT.mkdir(parents=True, exist_ok=True)
    records = []
    prop_t0 = time.perf_counter()
    timed_out = False
    for job in JOBS:
        stem = job_stem(job)
        path_npz = OUT / f"{stem}.npz"
        path_meta = OUT / f"{stem}.meta.json"
        rec = {"stem": stem, **{k: job[k] for k in ("tag", "sx", "sy", "neel", "gamma96_parity")}}
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
        np.testing.assert_allclose(run["kf"], kf_expect, atol=0.0, rtol=0.0)
        ident = job_identity(job, prot, hashes)
        ident.update(
            {
                "wall_seconds": run["wall_seconds"],
                "wall_outer_s": wall,
                "U_e0": run["U_e0"],
                "U_eT": run["U_eT"],
                "delta_U_e": run["delta_U_e"],
                "J_equilibrium_t0": run["J_equilibrium_t0"],
                "shift_xy": list(run["shift_xy"]),
                "n_times": int(run["times"].shape[0]),
                "membership_ok": memb["identity_holds"],
            }
        )
        if path_npz.exists():
            path_npz.unlink()
        if path_meta.exists():
            path_meta.unlink()
        write_job(job, run, ident, path_npz, path_meta)
        rec.update(status="computed", wall_s=wall, path=str(path_npz), delta_U_e=run["delta_U_e"])
        records.append(rec)
        print("done", stem, f"{wall:.1f}s", flush=True)
        if time.perf_counter() - prop_t0 > PROP_BUDGET_S:
            timed_out = True
            print("budget exhausted after", stem, flush=True)
            break
    return records, time.perf_counter() - prop_t0, timed_out


def analyzer_pack(t, jp, jm, prot, basis):
    jp_b = jp @ basis.T
    jm_b = jm @ basis.T
    nu, sp = cl.spectrum(t, jp_b, prot.omega0, wmax=9.5, dw=0.02, window="sin2")
    _, sm = cl.spectrum(t, jm_b, prot.omega0, wmax=9.5, dw=0.02, window="sin2")
    hp = np.stack([cl.project_harmonic(t, jp_b, q * prot.omega0, "sin2") for q in range(1, 10)])
    hm = np.stack([cl.project_harmonic(t, jm_b, q * prot.omega0, "sin2") for q in range(1, 10)])
    return {
        "times": t,
        "J_plus": jp,
        "J_minus": jm,
        "nu": nu,
        "spectrum": src.obs(sp, sm),
        "harmonics": src.obs(hp, hm),
    }


def window_table(blocks, prot, basis):
    rows = []
    for label, blk in blocks.items():
        t, jp, jm = blk["times"], blk["J_plus"], blk["J_minus"]
        jp_b = jp @ basis.T
        jm_b = jm @ basis.T
        for window in WINDOWS:
            nu, sp = cl.spectrum(t, jp_b, prot.omega0, wmax=9.5, dw=0.02, window=window)
            _, sm = cl.spectrum(t, jm_b, prot.omega0, wmax=9.5, dw=0.02, window=window)
            spectra = src.obs(sp, sm)
            for q, analyzer, vec in CHANNELS:
                z_dense = np.conjugate(spectra["J_even"] @ vec) * (spectra["J_odd"] @ vec)
                pkey = "P45" if analyzer == "45" else "P135"
                for hw in HALFWIDTHS:
                    z = bz.band_trapz(nu, z_dense, q, hw)
                    d = 4.0 * float(z.real)
                    plus_p = src.band_power(nu, spectra["plus"][pkey], q, halfwidth=hw)
                    minus_p = src.band_power(nu, spectra["minus"][pkey], q, halfwidth=hw)
                    sum_p = plus_p + minus_p
                    rows.append(
                        {
                            "grid": label,
                            "channel": f"H{q}_{analyzer}",
                            "q": q,
                            "analyzer_deg": int(analyzer),
                            "window": window,
                            "halfwidth": hw,
                            "Z_re": float(z.real),
                            "Z_im": float(z.imag),
                            "abs_Z": float(abs(z)),
                            "phi_deg": float(np.angle(z, deg=True)),
                            "cos_phi": float(np.cos(np.angle(z))),
                            "D_B": d,
                            "sum_P": sum_p,
                            "contrast": d / sum_p if sum_p > 0 else 0.0,
                            "plus_P": plus_p,
                            "minus_P": minus_p,
                            "identity_abs": abs(d - (plus_p - minus_p)),
                            "is_reference_window": window == "sin2" and abs(hw - 0.25) < 1e-15,
                        }
                    )
    return rows


def channel_compare(rows):
    out = []
    for q, analyzer, _vec in CHANNELS:
        recs = {
            r["grid"]: r
            for r in rows
            if r["q"] == q and r["analyzer_deg"] == int(analyzer) and r["is_reference_window"]
        }
        g48, s48, g96 = recs["Gamma48"], recs["shift48"], recs["Gamma96"]
        def rel(a, b):
            return None if abs(a) == 0.0 else (b - a) / abs(a)

        item = {
            "channel": f"H{q}_{analyzer}",
            "Gamma48": g48,
            "shift48": s48,
            "Gamma96": g96,
            "D_sign_flip_48_to_shift": (g48["D_B"] * s48["D_B"]) < 0,
            "D_sign_flip_48_to_96": (g48["D_B"] * g96["D_B"]) < 0,
            "absZ_rel_48_to_shift": rel(g48["abs_Z"], s48["abs_Z"]),
            "absZ_rel_48_to_96": rel(g48["abs_Z"], g96["abs_Z"]),
            "phi_change_48_to_shift_deg": float(((s48["phi_deg"] - g48["phi_deg"] + 180.0) % 360.0) - 180.0),
            "phi_change_48_to_96_deg": float(((g96["phi_deg"] - g48["phi_deg"] + 180.0) % 360.0) - 180.0),
            "compensation_48_to_shift": abs(rel(g48["D_B"], s48["D_B"]) or 0) < 0.1
            and abs(rel(g48["abs_Z"], s48["abs_Z"]) or 0) > 0.4,
            "compensation_48_to_96": abs(rel(g48["D_B"], g96["D_B"]) or 0) < 0.1
            and abs(rel(g48["abs_Z"], g96["abs_Z"]) or 0) > 0.4,
        }
        out.append(item)
    return out


def verdict(compares, window_rows):
    notes = []
    worth = {}
    for item in compares:
        ch = item["channel"]
        g96 = item["Gamma96"]
        unstable = []
        if item["D_sign_flip_48_to_96"] or item["D_sign_flip_48_to_shift"]:
            unstable.append("D_B sign")
        if abs(item["phi_change_48_to_96_deg"]) > 20.0:
            unstable.append("phi_B")
        if item["absZ_rel_48_to_96"] is not None and abs(item["absZ_rel_48_to_96"]) > 0.4:
            unstable.append("|Z_B|")
        win_flip = [
            r
            for r in window_rows
            if r["channel"] == ch and r["grid"] == "Gamma96" and (r["D_B"] * g96["D_B"]) < 0
        ]
        if win_flip:
            unstable.append("window/bandwidth")
        certified = not unstable
        follow = (not item["D_sign_flip_48_to_96"]) and abs(item["phi_change_48_to_96_deg"]) <= 35.0
        worth[ch] = {
            "certified_magnetic_channel": False,
            "worth_nonclassical_drive_now": bool(follow and certified is False and "D_B sign" not in unstable),
            "unstable_quantities": unstable,
            "follow_up_candidate": follow and "D_B sign" not in unstable,
        }
        notes.append(ch)
    any_certified = any(v["certified_magnetic_channel"] for v in worth.values())
    return {
        "any_certified_channel": any_certified,
        "start_gamma192": False,
        "start_intensity_or_squeezing_scan": False,
        "start_qed": False,
        "channels": worth,
        "notes": notes,
    }


def make_figure(compares, window_rows):
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
    fig, axs = plt.subplots(2, 3, figsize=(13.6, 7.6), layout="constrained")
    grids = ("Gamma48", "shift48", "Gamma96")
    colors = {"Gamma48": "#1464a0", "shift48": "#ce7028", "Gamma96": "#1b7f4e"}
    xs = np.arange(3)
    for col, item in enumerate(compares):
        d = [item[g]["D_B"] for g in grids]
        z = [item[g]["abs_Z"] for g in grids]
        phi = [item[g]["phi_deg"] for g in grids]
        axs[0, col].plot(xs, d, "o-", color="#185c9c")
        axr = axs[0, col].twinx()
        axr.plot(xs, z, "s--", color="#b85b2a")
        axs[0, col].set_title(item["channel"] + "  D_B / |Z_B|")
        axs[0, col].set_xticks(xs, grids)
        axs[0, col].axhline(0, color="#777", lw=0.5)
        axs[1, col].plot(xs, phi, "o-", color="#83539a")
        axs[1, col].set_title("φ_B")
        axs[1, col].set_xticks(xs, grids)
        axs[1, col].set_ylim(-190, 190)
        axs[1, col].axhline(0, color="#777", lw=0.5)
    fig.suptitle("Γ48 / shift48 / 合并 Γ96 的磁性干涉量（参考窗 sin²×0.25）")
    fig.savefig(OUT / "figure_gamma96_candidates.png", dpi=160)
    fig.savefig(OUT / "figure_gamma96_candidates.pdf")
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(8.4, 4.6), layout="constrained")
    for ch, marker in (("H2_45", "o"), ("H4_135", "s"), ("H7_45", "^")):
        recs = [r for r in window_rows if r["channel"] == ch and r["grid"] == "Gamma96"]
        ax.plot([f"{r['window'][:3]}·{r['halfwidth']}" for r in recs], [r["D_B"] for r in recs], marker + "-", label=ch)
    ax.axhline(0, color="#777", lw=0.5)
    ax.set_ylabel("D_B")
    ax.set_title("合并 Γ96：窗与带宽敏感性")
    ax.tick_params(axis="x", rotation=30)
    ax.legend()
    fig.savefig(OUT / "figure_gamma96_window.png", dpi=160)
    fig.savefig(OUT / "figure_gamma96_window.pdf")
    plt.close(fig)


def run():
    start = time.perf_counter()
    OUT.mkdir(parents=True, exist_ok=True)
    geo = build_geometry()
    prot = cl.p1_a2_protocol(0.8)
    par, perp = parallel_perp_basis(geo)
    basis = np.stack((par, perp))
    hashes = {
        "model": sha256_file(HERE / "q8_m0_model.py"),
        "classical": sha256_file(HERE / "q8_m0_classical.py"),
        "fill_script": sha(Path(__file__)),
        "source_20261114": sha(HERE / "q8_m0_magnetic_hhg_source_20261114.py"),
    }
    compat = compatibility(geo, prot)
    memb = membership(geo)
    dump(OUT / "compatibility.json", {"compatibility": compat, "membership": memb})
    if not compat["physics_comparable_to_V48"]:
        raise RuntimeError("refusing to propagate: archive physics not comparable")
    if not memb["identity_holds"]:
        raise RuntimeError("refusing to propagate: new subgrids are not Gamma96 subsets")
    if not compat["isotropic_k_grid_unchanged"]:
        raise RuntimeError("k_grid default behavior changed")
    archive_paths = []
    for folder, name in ((ARCH_V48, "V48"), (ARCH_V48S, "V48S")):
        for neel in (1, -1):
            archive_paths.extend(
                [folder / f"{name}_N{neel:+d}_A+1.npz", folder / f"{name}_N{neel:+d}_A+1.meta.json"]
            )
    before = {str(p): sha(p) for p in archive_paths}
    jobs, prop_wall, timed_out = run_jobs(geo, prot, hashes, memb)
    complete = (not timed_out) and all(r.get("status") in ("computed", "resumed") for r in jobs) and len(jobs) == 4
    payload = {
        "task_label": "20261116",
        "executed_at_utc": datetime.now(timezone.utc).isoformat(),
        "compatibility": compat,
        "membership": memb,
        "jobs": jobs,
        "propagate_wall_s": prop_wall,
        "propagate_budget_s": PROP_BUDGET_S,
        "timed_out": timed_out,
        "complete_four_jobs": complete,
        "inputs_sha256_before": before,
    }
    if not complete:
        payload.update(status="partial", analysis_started=False, wall_s=time.perf_counter() - start)
        dump(OUT / "fill_status.json", payload)
        print("partial stop", payload["status"], flush=True)
        return payload
    t00, jp00, jm00, info00, p00 = src.load("Gamma48", geo, prot)
    t11, jp11, jm11, info11, p11 = src.load("shift48", geo, prot)
    plus = {}
    minus = {}
    kfs = {}
    for job in JOBS:
        loaded = load_job(OUT / f"{job_stem(job)}.npz")
        np.testing.assert_array_equal(loaded["times"], t00)
        key = (job["sx"], job["sy"])
        kfs.setdefault(key, loaded["kf"])
        if job["neel"] == 1:
            plus[key] = loaded["J_sheet"]
        else:
            minus[key] = loaded["J_sheet"]
    np.testing.assert_array_equal(t00, t11)
    jp10, jm10 = plus[(0.5, 0.0)], minus[(0.5, 0.0)]
    jp01, jm01 = plus[(0.0, 0.5)], minus[(0.0, 0.5)]
    jp96 = 0.25 * (jp00 + jp11 + jp10 + jp01)
    jm96 = 0.25 * (jm00 + jm11 + jm10 + jm01)
    np.savez_compressed(OUT / "J96_merged.npz", times=t00, J_plus=jp96, J_minus=jm96)
    blocks = {
        "Gamma48": analyzer_pack(t00, jp00, jm00, prot, basis),
        "shift48": analyzer_pack(t11, jp11, jm11, prot, basis),
        "Gamma96": analyzer_pack(t00, jp96, jm96, prot, basis),
    }
    rows = window_table(blocks, prot, basis)
    compares = channel_compare(rows)
    judge = verdict(compares, rows)
    make_figure(compares, rows)
    fields = list(rows[0])
    tsv = OUT / "window_sensitivity.tsv"
    if tsv.exists():
        tsv.unlink()
    with tsv.open("x", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, delimiter="\t")
        w.writeheader()
        w.writerows(rows)
    after = {str(p): sha(p) for p in archive_paths}
    payload.update(
        {
            "status": "ok",
            "analysis_started": True,
            "merge_rule": "J96=(J00+J11+J10+J01)/4",
            "info00": info00,
            "info11": info11,
            "reference_window_rows": [r for r in rows if r["is_reference_window"]],
            "channel_compare": compares,
            "verdict": judge,
            "inputs_unchanged": before == after,
            "inputs_sha256_after": after,
            "source_hashes": hashes,
            "e_parallel": par,
            "e_perp": perp,
            "wall_s": time.perf_counter() - start,
            "openblas_num_threads": os.environ.get("OPENBLAS_NUM_THREADS"),
            "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
        }
    )
    dump(OUT / "fill_status.json", payload)
    print("status", payload["status"], "certified", judge["any_certified_channel"], "wall", payload["wall_s"], flush=True)
    return payload


if __name__ == "__main__":
    run()
