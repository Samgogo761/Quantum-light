#!/usr/bin/env python3
"""Read-only BZ subgrid identity, window sensitivity, and classical cost quote.

No eight-cycle propagation. Does not modify production k_grid or archives.
"""
from __future__ import annotations

import csv
import hashlib
import inspect
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import q8_m0_classical as cl
import q8_m0_magnetic_hhg_source_20261114 as src
from q8_m0_model import R0, build_geometry, current_ops, h8, sha256_file
from q8_m0_s1_kernel import parallel_perp_basis

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results/q_line/magnetic_hhg_bz_subgrid_20261115"
ARCH_V48 = ROOT / "results/q_line/cjj_m0_p1_a2_s1_20260917/V48"
ARCH_V48S = ROOT / "results/q_line/cjj_m0_p1_a2_s1_20260917/V48S"
REF = ROOT / "results/q_line/magnetic_hhg_source_20261114"
CHANNELS = (
    (2, "45", np.array([1.0, 1.0]) / np.sqrt(2.0)),
    (4, "135", np.array([1.0, -1.0]) / np.sqrt(2.0)),
    (7, "45", np.array([1.0, 1.0]) / np.sqrt(2.0)),
)
HALFWIDTHS = (0.125, 0.25, 0.50)
WINDOWS = ("sin2", "boxcar")
ARCHIVED_CJJ = "a89592df0f9084dc794134862f686bd25c03b43173f371000a5f9a9f59278411"
ARCHIVED_MODEL = "0dbcb90fad134a48f3b005022eee1f16b74a8529d195a494ecbdc68860e5509f"


def serial(x):
    return src.serial(x)


def dump(path, obj):
    with Path(path).open("x", encoding="utf-8") as f:
        json.dump(serial(obj), f, ensure_ascii=False, indent=1, allow_nan=False)


def frac_grid(nk, sx, sy):
    g1 = (np.arange(nk) + float(sx)) / nk
    g2 = (np.arange(nk) + float(sy)) / nk
    return np.stack(np.meshgrid(g1, g2, indexing="ij"), axis=-1).reshape(-1, 2)


def as_set(kf, nd=12):
    x = np.mod(np.round(np.asarray(kf, dtype=float).reshape(-1, 2), nd), 1.0)
    x[np.abs(x - 1.0) < 10.0 ** (-nd)] = 0.0
    return set(tuple(row) for row in x.tolist())


def inversion_closed(kf):
    return as_set(kf) == as_set(np.mod(-np.asarray(kf, dtype=float), 1.0))


def band_trapz(nu, y, q, halfwidth):
    lo, hi = q - halfwidth, q + halfwidth
    keep = (nu > lo) & (nu < hi)
    y = np.asarray(y)
    xx = np.r_[lo, nu[keep], hi]

    def edge(x0):
        return np.interp(x0, nu, y.real) + 1j * np.interp(x0, nu, y.imag)

    yy = np.r_[edge(lo), y[keep], edge(hi)]
    return complex(np.trapezoid(yy.real, xx) + 1j * np.trapezoid(yy.imag, xx))


def proposed_k_grid_xy(geo, nk, shift_x=0.0, shift_y=0.0):
    """Audit-only anisotropic MP grid. Not wired into production propagate."""
    kf = frac_grid(nk, shift_x, shift_y)
    kc = geo.k_cart(kf)
    w = np.full(kc.shape[0], 1.0 / kc.shape[0])
    return kf, kc, w


def identity_block(geo):
    sig = inspect.signature(cl.k_grid)
    src_txt = inspect.getsource(cl.k_grid)
    g96 = frac_grid(96, 0.0, 0.0).reshape(96, 96, 2)
    g00 = frac_grid(48, 0.0, 0.0).reshape(48, 48, 2)
    g11 = frac_grid(48, 0.5, 0.5).reshape(48, 48, 2)
    g10 = frac_grid(48, 0.5, 0.0).reshape(48, 48, 2)
    g01 = frac_grid(48, 0.0, 0.5).reshape(48, 48, 2)
    even_even = g96[0::2, 0::2]
    odd_odd = g96[1::2, 1::2]
    odd_even = g96[1::2, 0::2]
    even_odd = g96[0::2, 1::2]
    max_ee = float(np.max(np.abs(even_even - g00)))
    max_oo = float(np.max(np.abs(odd_odd - g11)))
    max_oe = float(np.max(np.abs(odd_even - g10)))
    max_eo = float(np.max(np.abs(even_odd - g01)))
    union = as_set(g00) | as_set(g11) | as_set(g10) | as_set(g01)
    sets = [as_set(g00), as_set(g11), as_set(g10), as_set(g01)]
    disjoint = all(len(a & b) == 0 for i, a in enumerate(sets) for b in sets[i + 1 :])
    kf48, kc48, w48 = cl.k_grid(geo, 48, shift=0.0)
    kf48s, kc48s, w48s = cl.k_grid(geo, 48, shift=0.5)
    kf96, kc96, w96 = cl.k_grid(geo, 96, shift=0.0)
    cart_ee = float(np.max(np.abs(kc96.reshape(96, 96, 2)[0::2, 0::2] - kc48.reshape(48, 48, 2))))
    cart_oo = float(np.max(np.abs(kc96.reshape(96, 96, 2)[1::2, 1::2] - kc48s.reshape(48, 48, 2))))
    w_merge = 0.25 / 2304.0
    return {
        "k_grid_signature": str(sig),
        "k_grid_is_scalar_shift": "shift_y" not in src_txt and "shift_x" not in src_txt,
        "k_grid_source": src_txt,
        "gamma96_shift": 0.0,
        "existing_subgrids": {
            "Gamma48": {"nk": 48, "shift_xy": [0.0, 0.0], "gamma96_parity": "even-even", "present": True},
            "shift48": {"nk": 48, "shift_xy": [0.5, 0.5], "gamma96_parity": "odd-odd", "present": True},
        },
        "missing_subgrids": {
            "shift_odd_even": {"nk": 48, "shift_xy": [0.5, 0.0], "gamma96_parity": "odd-even", "present": False},
            "shift_even_odd": {"nk": 48, "shift_xy": [0.0, 0.5], "gamma96_parity": "even-odd", "present": False},
        },
        "fractional_identity_maxabs": {
            "even_even_vs_Gamma48": max_ee,
            "odd_odd_vs_shift48": max_oo,
            "odd_even_vs_proposed_0.5_0": max_oe,
            "even_odd_vs_proposed_0_0.5": max_eo,
        },
        "cartesian_identity_maxabs": {"even_even": cart_ee, "odd_odd": cart_oo},
        "union_covers_gamma96": union == as_set(g96.reshape(-1, 2)),
        "union_count": len(union),
        "gamma96_count": 96 * 96,
        "four_subgrids_disjoint": disjoint,
        "inversion_closed": {
            "Gamma48": inversion_closed(kf48),
            "shift48": inversion_closed(kf48s),
            "Gamma96": inversion_closed(kf96),
            "proposed_0.5_0": inversion_closed(g10.reshape(-1, 2)),
            "proposed_0_0.5": inversion_closed(g01.reshape(-1, 2)),
        },
        "weights": {
            "w48": float(w48[0]),
            "w96": float(w96[0]),
            "merge_rule": "J96=(J00+J11+J10+J01)/4 after current-level average",
            "merge_weight_identity": abs(w_merge - float(w96[0])) < 1e-18,
            "do_not_average_normalized_contrast": True,
        },
        "identity_holds": max_ee == 0.0 and max_oo == 0.0 and disjoint and union == as_set(g96.reshape(-1, 2)),
        "may_splice_now": False,
        "reason_not_splicing": "missing anisotropic subgrids (0.5,0) and (0,0.5); production k_grid is a scalar shift",
        "production_k_grid_unchanged": True,
        "proposed_helper_only_in_this_audit": "proposed_k_grid_xy",
    }


def protocol_alignment(prot):
    rows = []
    for folder, name, shift in ((ARCH_V48, "V48", 0.0), (ARCH_V48S, "V48S", 0.5)):
        for neel in (1, -1):
            meta = json.loads((folder / f"{name}_N{neel:+d}_A+1.meta.json").read_text(encoding="utf-8"))
            rows.append(
                {
                    "name": name,
                    "neel": neel,
                    "protocol": meta["protocol"],
                    "protocol_matches": meta["protocol"] == prot.__dict__,
                    "nk": meta["nk"],
                    "nt_per_cycle": meta["nt_per_cycle"],
                    "mu": meta["mu"],
                    "grid_shift": meta["grid_shift"],
                    "kf_kind": meta["kf_kind"],
                    "expected_shift": shift,
                    "cjj_module_sha256": meta.get("cjj_module_sha256"),
                    "model_sha256": meta["source_hashes"]["model_module"]["sha256"],
                    "params": meta["params"],
                }
            )
    mu = {r["mu"] for r in rows}
    nk = {r["nk"] for r in rows}
    ntpc = {r["nt_per_cycle"] for r in rows}
    return {
        "rows": rows,
        "all_protocol_match": all(r["protocol_matches"] for r in rows),
        "common_mu": next(iter(mu)) if len(mu) == 1 else sorted(mu),
        "mu_unique": len(mu) == 1,
        "nk_unique_48": nk == {48},
        "ntpc_unique_400": ntpc == {400},
        "params_R0": all(r["params"]["label"] == "R0" for r in rows),
    }


def module_hashes():
    classical = sha256_file(HERE / "q8_m0_classical.py")
    model = sha256_file(HERE / "q8_m0_model.py")
    cjj = sha256_file(HERE / "q8_m0_cjj.py")
    return {
        "live_classical": classical,
        "live_model": model,
        "live_cjj": cjj,
        "archived_cjj": ARCHIVED_CJJ,
        "archived_model": ARCHIVED_MODEL,
        "cjj_matches_archive": cjj == ARCHIVED_CJJ,
        "model_matches_archive": model == ARCHIVED_MODEL,
        "identity_basis": "formula + kf_kind + meta; not bit-identical live source",
    }


def load_ref_bands():
    data = json.loads((REF / "band_interpretation.json").read_text(encoding="utf-8"))
    out = {}
    for grid in ("Gamma48", "shift48"):
        for rec in data[grid]:
            for analyzer in ("45", "135"):
                out[(grid, rec["q"], analyzer)] = rec[analyzer]
    return out


def window_rows(geo, prot, basis, ref):
    rows = []
    input_hashes = {}
    for label in ("Gamma48", "shift48"):
        t, jp, jm, info, paths = src.load(label, geo, prot)
        for p in paths:
            input_hashes[str(p)] = src.sha(p)
        jp_b = jp @ basis.T
        jm_b = jm @ basis.T
        for window in WINDOWS:
            nu, sp = cl.spectrum(t, jp_b, prot.omega0, wmax=9.5, dw=0.02, window=window)
            _, sm = cl.spectrum(t, jm_b, prot.omega0, wmax=9.5, dw=0.02, window=window)
            spectra = src.obs(sp, sm)
            for q, analyzer, vec in CHANNELS:
                je = spectra["J_even"] @ vec
                jo = spectra["J_odd"] @ vec
                z_dense = np.conjugate(je) * jo
                pkey = "P45" if analyzer == "45" else "P135"
                for hw in HALFWIDTHS:
                    z = band_trapz(nu, z_dense, q, hw)
                    d = 4.0 * float(z.real)
                    phi = float(np.angle(z, deg=True))
                    plus_p = src.band_power(nu, spectra["plus"][pkey], q, halfwidth=hw)
                    minus_p = src.band_power(nu, spectra["minus"][pkey], q, halfwidth=hw)
                    sum_p = plus_p + minus_p
                    contrast = d / sum_p if sum_p > 0 else 0.0
                    ident = abs(d - (plus_p - minus_p))
                    key = (label, q, analyzer)
                    rec = ref.get(key)
                    is_ref = window == "sin2" and abs(hw - 0.25) < 1e-15
                    d_ref = rec["band_D"] if rec and is_ref else None
                    phi_ref = rec["effective_band_phase_deg"] if rec and is_ref else None
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
                            "phi_deg": phi,
                            "D_B": d,
                            "sum_P": sum_p,
                            "contrast": contrast,
                            "identity_abs": ident,
                            "is_20261114_reference_window": is_ref,
                            "D_ref": d_ref,
                            "phi_ref_deg": phi_ref,
                            "D_abs_err": None if d_ref is None else abs(d - d_ref),
                            "phi_abs_err_deg": None if phi_ref is None else abs(((phi - phi_ref + 180.0) % 360.0) - 180.0),
                            "n_times": info["n_times"],
                            "shift": info["shift"],
                        }
                    )
    return rows, input_hashes


def quote_inner(geo, prot, n_steps=12):
    nk, ntpc, neel, sign, mu = 48, 400, 1, 1, -0.009010620372198276
    _, kc, w = cl.k_grid(geo, nk, shift=0.0)
    t0 = time.perf_counter()
    rho, _e0 = cl.initial_rho(geo, R0, kc, neel, mu)
    t_init = time.perf_counter() - t0
    dt = prot.period / ntpc
    t1 = time.perf_counter()
    for n in range(n_steps):
        t = n * dt
        kap = kc + cl.kappa_shift(t, geo, prot, sign)
        j = current_ops(geo, R0, kap, neel)
        jk = cl.trace_current(rho, j)
        _sheet = w @ jk
        kmid = kc + cl.kappa_shift(t + 0.5 * dt, geo, prot, sign)
        hm = h8(geo, R0, kmid, neel)
        em, wm = np.linalg.eigh(hm)
        ph = np.exp(-1j * em * dt)
        u = (wm * ph[:, None, :]) @ np.conj(np.swapaxes(wm, 1, 2))
        rho = u @ rho @ np.conj(np.swapaxes(u, 1, 2))
    t_loop = time.perf_counter() - t1
    nt_full = prot.n_cycles * ntpc
    per = t_loop / n_steps
    one = t_init + per * nt_full
    four = 4.0 * one
    hist_one = 140.46757290000096
    hist_pair = 283.80736420000903
    return {
        "copied_from": "propagate inner loop, n_diag skipped, subspace=False, k_orders=()",
        "n_steps_timed": n_steps,
        "nk": nk,
        "nt_per_cycle": ntpc,
        "dt": dt,
        "n_cycles": prot.n_cycles,
        "nt_full": nt_full,
        "t_init_s": t_init,
        "t_loop_s": t_loop,
        "s_per_midpoint_step": per,
        "one_job_extrap_s": one,
        "missing_jobs": 4,
        "serial_four_jobs_s": four,
        "serial_four_jobs_min": four / 60.0,
        "historical_cjj_one_s": hist_one,
        "historical_cjj_pair_s": hist_pair,
        "historical_cjj_four_jobs_s": 4.0 * hist_one,
        "historical_cjj_four_jobs_min": 4.0 * hist_one / 60.0,
        "ten_minute_threshold_s": 600.0,
        "local_serial_gt_10min_classical": four > 600.0,
        "local_serial_gt_10min_cjj_upper": 4.0 * hist_one > 600.0,
        "propagate_now": False,
        "wrote_trajectory": False,
        "production_propagator_unchanged": True,
        "note": "quote copies the sparse current+eigh midpoint body only; not a full CJJ wall",
    }


def server_plan(quote):
    return {
        "when_needed": "if a later approved fill-in exceeds ~10 min local wall, or to keep the laptop free",
        "jobs": [
            {"grid": "48^2 shift_xy=(0.5,0)", "neel": "+1"},
            {"grid": "48^2 shift_xy=(0.5,0)", "neel": "-1"},
            {"grid": "48^2 shift_xy=(0,0.5)", "neel": "+1"},
            {"grid": "48^2 shift_xy=(0,0.5)", "neel": "-1"},
        ],
        "protocol": "P1_a2_A0.8, ntpc=400, mu frozen, classical J_sheet only",
        "parallelism": "four independent jobs; no k-splitting inside a job",
        "prerequisite": "add anisotropic (sx,sy) k_grid helper before launch; do not splice two grids as Gamma96",
        "memory": "rho (2304,8,8) complex128 ~2.3 MiB plus operators; CPU only",
        "this_round": "quote only; no sbatch",
        "classical_serial_min": quote["serial_four_jobs_min"],
        "cjj_upper_min": quote["historical_cjj_four_jobs_min"],
    }


def run():
    start = time.perf_counter()
    if OUT.exists() and any(OUT.iterdir()):
        raise FileExistsError("result directory is not empty")
    OUT.mkdir(parents=True, exist_ok=True)
    geo = build_geometry()
    prot = cl.p1_a2_protocol(0.8)
    par, perp = parallel_perp_basis(geo)
    basis = np.stack((par, perp))
    ident = identity_block(geo)
    align = protocol_alignment(prot)
    hashes = module_hashes()
    ref = load_ref_bands()
    rows, before = window_rows(geo, prot, basis, ref)
    quote = quote_inner(geo, prot, n_steps=12)
    after = {p: src.sha(p) for p in before}
    ref_err = [r for r in rows if r["is_20261114_reference_window"]]
    max_d_err = max(r["D_abs_err"] for r in ref_err)
    max_phi_err = max(r["phi_abs_err_deg"] for r in ref_err)
    max_ident = max(r["identity_abs"] for r in rows)
    payload = {
        "task_label": "20261115",
        "executed_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "ok",
        "new_propagations": 0,
        "propagate_now": False,
        "identity": ident,
        "protocol_alignment": align,
        "module_hashes": hashes,
        "window_reference_reproduction": {
            "window": "sin2",
            "halfwidth": 0.25,
            "max_D_abs_err": max_d_err,
            "max_phi_abs_err_deg": max_phi_err,
            "rows": ref_err,
        },
        "acceptance_observables": ["abs_Z", "phi_B", "D_B"],
        "do_not_certify_on_D_alone": True,
        "cost_quote": quote,
        "server_plan": server_plan(quote),
        "inputs_sha256_before": before,
        "inputs_unchanged": before == after,
        "band_identity_max_abs": max_ident,
        "source_sha256": src.sha(Path(__file__)),
        "wall_s": time.perf_counter() - start,
        "openblas_num_threads": os.environ.get("OPENBLAS_NUM_THREADS"),
        "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
    }
    dump(OUT / "identity_cost.json", payload)
    fields = [
        "grid",
        "channel",
        "q",
        "analyzer_deg",
        "window",
        "halfwidth",
        "Z_re",
        "Z_im",
        "abs_Z",
        "phi_deg",
        "D_B",
        "sum_P",
        "contrast",
        "identity_abs",
        "is_20261114_reference_window",
        "D_ref",
        "phi_ref_deg",
        "D_abs_err",
        "phi_abs_err_deg",
    ]
    with (OUT / "window_sensitivity.tsv").open("x", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, delimiter="\t", extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    print("status", payload["status"], "identity", ident["identity_holds"], "unchanged", payload["inputs_unchanged"], "wall", payload["wall_s"], flush=True)
    print("serial_four_min", quote["serial_four_jobs_min"], "cjj_upper_min", quote["historical_cjj_four_jobs_min"], flush=True)
    return payload


if __name__ == "__main__":
    run()
