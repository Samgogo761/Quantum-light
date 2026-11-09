#!/usr/bin/env python3
"""CPU profile of eight-band CJJ time-step pieces. No GPU kernel, no training.

Times H construct, eigh, U/ρ, X/Y, and a prefix-integral dummy on a
production-like k batch (nk=48, 80 steps), then sixk for contrast.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from q8_m0_cjj import dump, occup_empty_bases, x_from_bases  # noqa: E402
from q8_m0_classical import initial_rho, k_grid, kappa_shift, p1_a2_protocol  # noqa: E402
from q8_m0_model import R0, build_geometry, current_ops, h8, h8_and_current  # noqa: E402
from q8_m0_response import d_ops, mean_d_from_bases, prefix_state_dmu  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "results/q_line/stage1_io_20260921"
MU = -0.009010620372198276
N_STEPS = 80


def profile_batch(nk: int, n_steps: int) -> dict:
    geo = build_geometry()
    prot = p1_a2_protocol(0.8)
    kf, kc, w = k_grid(geo, nk, shift=0.0)
    dt = prot.period / 400
    rho, _ = initial_rho(geo, R0, kc, +1, MU)
    h0 = h8(geo, R0, kc, +1)
    vv, vc, _ = occup_empty_bases(h0, MU)
    acc = {k: 0.0 for k in ("h_and_j", "eigh_U", "rho_bases", "X", "D_mean", "prefix")}
    x_hist = []
    t0_all = time.perf_counter()
    for n in range(n_steps):
        t = n * dt
        kap = kc + kappa_shift(t, geo, prot, +1)
        t1 = time.perf_counter()
        _h, j = h8_and_current(geo, R0, kap, +1)
        acc["h_and_j"] += time.perf_counter() - t1
        t1 = time.perf_counter()
        kmid = kc + kappa_shift(t + 0.5 * dt, geo, prot, +1)
        hm = h8(geo, R0, kmid, +1)
        em, wm = np.linalg.eigh(hm)
        ph = np.exp(-1j * em * dt)
        u = (wm * ph[:, None, :]) @ np.conj(np.swapaxes(wm, 1, 2))
        acc["eigh_U"] += time.perf_counter() - t1
        t1 = time.perf_counter()
        rho = u @ rho @ np.conj(np.swapaxes(u, 1, 2))
        vv = u @ vv
        vc = u @ vc
        acc["rho_bases"] += time.perf_counter() - t1
        t1 = time.perf_counter()
        x = x_from_bases(j, vv, vc)
        acc["X"] += time.perf_counter() - t1
        x_hist.append(x)
        t1 = time.perf_counter()
        dab = d_ops(geo, R0, kap, +1)
        _ = np.einsum("k,kab->ab", w, mean_d_from_bases(dab, vv))
        acc["D_mean"] += time.perf_counter() - t1
    x_arr = np.stack(x_hist, axis=0)
    h = np.zeros((n_steps, 2))
    h[:, 0] = 1.0
    t1 = time.perf_counter()
    prefix_state_dmu(x_arr, w, h, dt)
    acc["prefix"] = time.perf_counter() - t1
    wall = time.perf_counter() - t0_all
    n_k = int(kc.shape[0])
    frac = {k: acc[k] / wall for k in acc}
    per_step = {k: acc[k] / n_steps for k in acc if k != "prefix"}
    per_step["prefix_full_window"] = acc["prefix"]
    return {
        "nk_lin": nk,
        "n_k": n_k,
        "n_steps": n_steps,
        "wall_s": wall,
        "seconds": acc,
        "fraction_of_wall": frac,
        "seconds_per_step": per_step,
        "extrapolate_3201_steps_s": wall * (3201 / n_steps),
        "gpu_status": "not_implemented_for_M0",
        "note": "CPU numpy batch over k; Python time loop remains. Not a GPU kernel.",
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rec = {
        "sixk": profile_batch(6, N_STEPS),
        "nk48_subset_steps": profile_batch(48, N_STEPS),
        "ai": {
            "training": "not_run",
            "FNO": "not_run",
            "signed_unresolved_labels": "forbidden",
            "next": "offline_k_budget_comparison_after_stage1_gates",
        },
    }
    dump(OUT / "m0_cpu_profile.json", rec)
    print(json.dumps({
        "wrote": str(OUT / "m0_cpu_profile.json"),
        "nk48_wall_80steps": rec["nk48_subset_steps"]["wall_s"],
        "nk48_frac": rec["nk48_subset_steps"]["fraction_of_wall"],
        "nk48_extrap_full": rec["nk48_subset_steps"]["extrapolate_3201_steps_s"],
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
