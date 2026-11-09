#!/usr/bin/env python3
"""Real-workload CPU profile of the frozen line-Q toy.  No artificial k inflation.

Answers one question: is the CPU two-time C_JJ already a bottleneck?
Writes results/q_line/profile_q_cpu_<date>.json (repo-relative) and prints a verdict.
"""
from __future__ import annotations

import json
import platform
import sys
import time
import tracemalloc
from datetime import date
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from q_backend import GPU_STATUS, toy  # noqa: E402  (q_backend puts b0_small_model on sys.path)

REPO = Path(__file__).resolve().parents[2]


def timed(fn, *a, **k):
    tracemalloc.start()
    t0 = time.perf_counter()
    out = fn(*a, **k)
    dt = time.perf_counter() - t0
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return out, dt, peak


def profile(p: toy.ToyParams) -> dict[str, object]:
    points = toy.k_grid(p.nk1d)
    nk = len(points)
    weights = np.full(nk, 1.0 / nk)
    times = toy.time_grid(p)
    ax = toy.vector_potential(times, p)
    nt = times.size

    (rho_p, j_p, steps_p), t_prop, m_prop = timed(toy.evolve_domain, p.delta_abs, p, times, ax)
    (_, t_mean, _) = timed(toy.bz_mean, j_p, rho_p, weights)
    (_, t_eq, _) = timed(toy.bz_equal_time, j_p, j_p, rho_p, weights)
    refs = np.arange(nt)
    (_c, t_two, m_two) = timed(
        toy._two_time_packet, times, weights, rho_p[0], steps_p, j_p, j_p, refs
    )
    (_tt, t_full, m_full) = timed(toy.two_time_cjj, p, blocks=True)

    n_pairs = nt * (nt + 1) // 2
    bytes_rho = rho_p.nbytes + j_p.nbytes + steps_p.nbytes
    nb = toy.N_ORB
    # (7.10): U(I-rho)B rho U^dagger then A@..., ~5 complex 4x4 GEMMs per (t,t',k) + U accumulation
    flops_two_time = 6 * n_pairs * nk * (8 * nb**3)
    return {
        "flops_two_time_one_pair_estimate": int(flops_two_time),
        "python_calls_two_time_one_pair": int(n_pairs * nk),
        "params": p.metadata()["params"],
        "nt": int(nt),
        "nk": int(nk),
        "n_bands": toy.N_ORB,
        "n_two_time_pairs": int(n_pairs),
        "seconds": {
            "propagation_one_domain": t_prop,
            "mean_current_one_domain": t_mean,
            "equal_time_wick_one_domain": t_eq,
            "two_time_wick_full_one_domain_one_operator_pair": t_two,
            "two_time_cjj_with_all_diagnostics_2_domains_14_packets": t_full,
        },
        "peak_bytes": {
            "propagation_one_domain": int(m_prop),
            "two_time_one_domain": int(m_two),
            "two_time_cjj_all": int(m_full),
            "trajectory_arrays_rho_j_steps": int(bytes_rho),
            "c_greater_matrix": int(nt * nt * 16),
        },
        "complexity_breakdown": {
            "propagation": f"O(nt*nk) eigh(4x4): {nt*nk} ops",
            "two_time_U_handling": f"O(nt^2/2 * nk) 4x4 GEMM: {n_pairs*nk} ops (recomputed per t', not stored)",
            "wick_GEMM": f"O(nt^2/2 * nk) x 4 GEMM + trace: {4*n_pairs*nk} ops",
            "storage": "C(t,t') dense nt^2 complex128; trajectories nt*nk*16*16B",
            "dominant_cost": "Python-loop overhead per 4x4 op, not FLOPs",
        },
    }


def main() -> None:
    default = toy.ToyParams()
    dev = toy.ToyParams(nk1d=6, ncyc=4.0, nt_per_cycle=24)
    rows = {"default_nk4": profile(default), "memo_max_nk6": profile(dev)}

    # Verdict is based on the physics quantity (one two-time operator pair, one domain)
    # on the memo-maximum grid (nk1d=6).  The 14-packet diagnostic bundle is a review aid.
    t_pair_nk6 = rows["memo_max_nk6"]["seconds"]["two_time_wick_full_one_domain_one_operator_pair"]
    flops = rows["memo_max_nk6"]["flops_two_time_one_pair_estimate"]
    if t_pair_nk6 < 60.0:
        verdict = "LOW"
    elif t_pair_nk6 < 600.0:
        verdict = "MEDIUM"
    else:
        verdict = "HIGH"
    effective_gflops = flops / t_pair_nk6 / 1.0e9

    out = {
        "label": toy.LABEL,
        "status": toy.STATUS,
        "not_a_physics_result": True,
        "date": date.today().isoformat(),
        "machine": {
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "platform": platform.platform(),
            "gpu_python_stack": "none installed (cupy/torch/jax absent)",
            "gpu_backend_status": GPU_STATUS,
        },
        "workload_rule": "existing frozen toy sizes only; no k inflation, no BSV samples",
        "profiles": rows,
        "GPU_PRIORITY": verdict,
        "verdict_rule": "LOW if one two-time operator pair (one domain) < 60 s on the memo-max grid nk1d=6",
        "effective_gflops_two_time_nk6": effective_gflops,
        "bottleneck": (
            "Python per-(t,t',k) call overhead in the B1-dev reference driver, not FLOPs; "
            "a k-batched numpy einsum of the same Eq. (7.10) is the first optimisation, "
            "GPU only after that if larger nt or parameter scans are actually needed"
        ),
    }
    dest = REPO / "results" / "q_line"
    dest.mkdir(parents=True, exist_ok=True)
    path = dest / f"profile_q_cpu_{date.today().strftime('%Y%m%d')}.json"
    path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in out.items() if k != "profiles"}, indent=2))
    for name, r in rows.items():
        print(name, "nt", r["nt"], "nk", r["nk"], json.dumps(r["seconds"], indent=2))
        print(name, "peak_bytes", json.dumps(r["peak_bytes"]))
    print("written", path)


if __name__ == "__main__":
    main()
