#!/usr/bin/env python3
"""Stage-1G: coarse uniform BZ qualification of probe6 cycle-1 tangent.

4x4, 4x4 half-shift, 6x6. First-cycle window only for amplification metrics.
Born A2/B2 use the same full-pulse H2 overlap as Stage-1B/1D.
No T2, no DC cut, no P1 change, no 48/96/192, no Fock, no server.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from q8_m0_cjj import dump, propagate_cjj  # noqa: E402
from q8_m0_classical import k_grid, p1_a2_protocol  # noqa: E402
from q8_m0_gsheet import gamma_self  # noqa: E402
from q8_m0_loop import ResummationNotChosenError, apply_resummation_to_output  # noqa: E402
from q8_m0_mode_io import close_output_moments, eta2_ncell  # noqa: E402
from q8_m0_model import R0, build_geometry  # noqa: E402
from q8_m0_profile_m0 import profile_batch  # noqa: E402
from q8_m0_response import predict_delta_mu  # noqa: E402
from q8_m0_runaway import (  # noqa: E402
    a_priori_windows,
    amplitude_compare,
    k_sets,
    propagate_seed_self,
)
from q8_m0_s1_kernel import pack_complex  # noqa: E402
from q8_m0_stage1 import stage1_spec  # noqa: E402
from q8_m0_volterra import (  # noqa: E402
    collect_A2_B2,
    h2_kernels,
    volterra_forward,
    volterra_response_to_complex_F,
)

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "results/q_line/stage1g_bzqual_20260922"
MU = -0.009010620372198276
EPS = 3.0e-5
NTPC = 400
G1G_ID = "coarse_bz_qualification_20260922"


def h2_seed(times, prot, geo, eps: float):
    hk = h2_kernels(times, prot.omega0, prot.t_pulse, geo)
    F = np.asarray(hk["F"])
    e_perp = np.asarray(hk["e_perp"])
    h = np.real(F)[:, None] * e_perp[None, :]
    peak = float(np.max(np.abs(h)))
    a_in = (eps / max(peak, 1e-30)) * h
    return a_in, hk


def _rel(a, b) -> float:
    a = np.asarray(a)
    b = np.asarray(b)
    return float(np.max(np.abs(a)) / max(float(np.max(np.abs(b))), 1e-30))


def perk_a2_state(x, mean_d, F, ehat, k_a, dt, pref) -> list:
    """Per-k Born A2 from one-hot weights. Contact uses the stored mean D.

    State piece is k-local from X. Contact is only approximately local.
    Sum_k w_k A2_k is not forced equal to the averaged A2.
    """
    from q8_m0_volterra import collect_overlap

    nk = int(x.shape[1])
    rows = []
    h_re = np.real(np.conj(F))[:, None] * ehat[None, :]
    for k in range(nk):
        w = np.zeros(nk, dtype=float)
        w[k] = 1.0
        rec = predict_delta_mu(x, w, h_re, dt, mean_d)
        i_fs = rec["total"]
        beta = collect_overlap(k_a, i_fs, dt)
        a2 = 1j * pref * beta
        rows.append({
            "k": k,
            "A2_state_proxy": pack_complex(a2),
            "maxabs_I": float(np.max(np.abs(i_fs))),
        })
    return rows


def cycle1_block(times, period, odd, even, a_self, kappa_p1, i_lin, geo, a_star):
    n = a_priori_windows(times, period)["cycle1"]
    sl = slice(0, n + 1)
    odd_w = np.asarray(odd)[sl]
    even_w = np.asarray(even)[sl]
    as_w = np.asarray(a_self)[sl]
    kp_w = np.asarray(kappa_p1)[sl]
    lin_w = np.asarray(i_lin)[sl]
    lin_ok = bool(np.isfinite(lin_w).all())
    amp = amplitude_compare(as_w, kappa_p1, geo, a_star)
    return {
        "n": n,
        "t": float(times[n]),
        "maxabs_odd": float(np.max(np.abs(odd_w))),
        "maxabs_even": float(np.max(np.abs(even_w))),
        "even_over_odd": float(np.max(np.abs(even_w)) / max(np.max(np.abs(odd_w)), 1e-30)),
        "maxabs_lin": float(np.max(np.abs(lin_w))) if lin_ok else None,
        "odd_minus_lin_rel": _rel(odd_w - lin_w, odd_w) if lin_ok else None,
        "maxmod_kappa_self": float(np.max(np.linalg.norm(as_w, axis=1))),
        "maxmod_kappa_p1": float(np.max(np.linalg.norm(np.asarray(kappa_p1), axis=1))),
        "ratio_kappa_self_over_p1_cycle1": float(
            np.max(np.linalg.norm(as_w, axis=1))
            / max(np.max(np.linalg.norm(np.asarray(kappa_p1), axis=1)), 1e-30)
        ),
        "ratio_kappa_self_over_p1_full_aself_vs_p1peak": amp["ratio_kappa_self_mod_over_p1"],
        "lin_finite_in_window": lin_ok,
    }


def one_grid(geo, prot, gamma, name, kf, labels, pref):
    t_grid = time.perf_counter()
    print(f"[stage1g] {name} nk={len(kf)}", flush=True)
    nt = prot.n_cycles * NTPC
    dt = prot.period / NTPC
    times = np.arange(nt + 1) * dt
    a_in, hk = h2_seed(times, prot, geo, EPS)
    F = hk["F"]
    e_perp = hk["e_perp"]
    k_a = hk["k_a"]
    print(f"  CJJ X+D", flush=True)
    t0 = time.perf_counter()
    r0 = propagate_cjj(
        geo, R0, prot, nt_per_cycle=NTPC, neel=+1, sign=+1, mu=MU,
        verbose=False, kf_frac=kf, save_full_x=True, save_mean_D=True,
    )
    cjj_s = time.perf_counter() - t0
    x = r0["X_full"]
    w = r0["w_k"]
    mean_d = r0["mean_D_t"]
    i_vv = np.asarray(r0["mu_per_cell"])

    t0 = time.perf_counter()
    closed = close_output_moments(x, w, i_vv, mean_d, times, dt, geo, prot)
    born_s = time.perf_counter() - t0
    A2 = complex(closed["A2"]["re"], closed["A2"]["im"])
    B2 = complex(closed["B2"]["re"], closed["B2"]["im"])

    t0 = time.perf_counter()
    vol0 = volterra_response_to_complex_F(x, w, mean_d, dt, 0.0, F, e_perp)
    coeff0 = collect_A2_B2(k_a, vol0["total"], dt, pref)
    vol_lin = volterra_forward(x, w, mean_d, dt, gamma, a_L_in=a_in)
    vol_s = time.perf_counter() - t0

    print(f"  NL I0 / 0 / +/-", flush=True)
    t0 = time.perf_counter()
    zeros = np.zeros_like(a_in)
    i0 = propagate_seed_self(
        geo, R0, prot, kf=kf, mu=MU, neel=+1, nt_per_cycle=NTPC,
        seed=zeros, i_ref=i_vv, gamma=0.0, verbose=False,
    )
    i_ref = i0["I"]
    zrun = propagate_seed_self(
        geo, R0, prot, kf=kf, mu=MU, neel=+1, nt_per_cycle=NTPC,
        seed=zeros, i_ref=i_ref, gamma=gamma, verbose=False,
    )
    prun = propagate_seed_self(
        geo, R0, prot, kf=kf, mu=MU, neel=+1, nt_per_cycle=NTPC,
        seed=a_in, i_ref=i_ref, gamma=gamma, verbose=False,
    )
    mrun = propagate_seed_self(
        geo, R0, prot, kf=kf, mu=MU, neel=+1, nt_per_cycle=NTPC,
        seed=-a_in, i_ref=i_ref, gamma=gamma, verbose=False,
    )
    nl_s = time.perf_counter() - t0
    odd = 0.5 * (prun["I"] - mrun["I"])
    even = 0.5 * (prun["I"] + mrun["I"]) - zrun["I"]
    zero_rel = _rel(zrun["I"] - i_ref, i_ref)
    c1 = cycle1_block(
        times, prot.period, odd, even, prun["A_self"], i0["kappa_p1"],
        vol_lin["I"], geo, prot.a0_dimless,
    )
    amp_full = amplitude_compare(prun["A_self"], i0["kappa_p1"], geo, prot.a0_dimless)

    t0 = time.perf_counter()
    perk = perk_a2_state(x, mean_d, F, e_perp, k_a, dt, pref)
    perk_s = time.perf_counter() - t0
    perk_re = np.array([p["A2_state_proxy"]["re"] for p in perk])
    perk_abs = np.array([p["A2_state_proxy"]["abs"] for p in perk])

    series = {
        "times": times,
        "odd": odd,
        "even": even,
        "A_self_plus": prun["A_self"],
        "kappa_p1": i0["kappa_p1"],
        "I_lin1": vol_lin["I"],
        "kf": np.asarray(kf),
    }
    np.savez(OUT / f"{name}_series.npz", **series)

    rec = {
        "name": name,
        "n_k": int(len(kf)),
        "labels": list(labels),
        "uniform_mp": name.startswith("mp"),
        "zero_seed_rel": zero_rel,
        "zero_seed_pass": bool(zero_rel <= 1.0e-3),
        "A2_born": pack_complex(A2),
        "B2_born": pack_complex(B2),
        "abs_A2_born": float(abs(A2)),
        "abs_B2_born": float(abs(B2)),
        "A2_volterra_gamma0": pack_complex(coeff0["A2"]),
        "dA2_vol0_minus_born": float(abs(coeff0["A2"] - A2)),
        "cycle1": c1,
        "full_kappa_self_over_p1": amp_full["ratio_kappa_self_mod_over_p1"],
        "plus_prop_halted": prun["halted"],
        "lin1_finite": bool(vol_lin["finite"]),
        "lin1_maxabs_I_full": vol_lin["maxabs_I"],
        "perk_A2_state_proxy": {
            "mean_re": float(np.mean(perk_re)),
            "std_re": float(np.std(perk_re)),
            "min_re": float(np.min(perk_re)),
            "max_re": float(np.max(perk_re)),
            "mean_abs": float(np.mean(perk_abs)),
            "frac_positive_re": float(np.mean(perk_re > 0.0)),
            "sum_re_over_nk": float(np.sum(perk_re) / max(len(kf), 1)),
        },
        "profile": {
            "cjj_s": cjj_s,
            "born_moments_s": born_s,
            "volterra_s": vol_s,
            "nl_four_runs_s": nl_s,
            "perk_s": perk_s,
            "grid_wall_s": time.perf_counter() - t_grid,
            "cjj_s_per_k": cjj_s / max(len(kf), 1),
            "nl_s_per_k": nl_s / max(len(kf), 1),
        },
        "gain_claim": "not_evaluated",
        "production_V": "Born_reference_diagnostic_not_qualified",
    }
    print(
        f"  |A2|={rec['abs_A2_born']:.4f} cycle1_odd={c1['maxabs_odd']:.3e} "
        f"ks/kp_c1={c1['ratio_kappa_self_over_p1_cycle1']:.3e} "
        f"zero={zero_rel:.1e} wall={rec['profile']['grid_wall_s']:.1f}s",
        flush=True,
    )
    return rec


def classify(rows: list) -> dict:
    by = {r["name"]: r for r in rows}
    probe = by["probe6"]
    uni = [by["mp4"], by["mp4_shift"], by["mp6"]]
    absA = np.array([r["abs_A2_born"] for r in uni])
    reA = np.array([r["A2_born"]["re"] for r in uni])
    odd = np.array([r["cycle1"]["maxabs_odd"] for r in uni])
    signs = np.array([np.sign(x) if abs(x) > 1e-4 else 0.0 for x in reA])
    abs_ratio = float(np.max(absA) / max(float(np.min(absA)), 1e-30))
    odd_ratio = float(np.max(odd) / max(float(np.min(odd)), 1e-30))
    sign_flip = bool(np.min(signs) * np.max(signs) < 0.0)
    fading = bool(float(absA[-1]) < 0.15 and float(odd[-1]) < 0.3 * probe["cycle1"]["maxabs_odd"])
    fading = fading or bool(
        float(np.max(absA)) < 0.15
        and float(np.max(odd)) < 0.3 * probe["cycle1"]["maxabs_odd"]
    )
    strong = bool(float(np.min(absA)) > 0.3 and odd_ratio < 4.0 and not sign_flip)
    if sign_flip or abs_ratio > 4.0 or odd_ratio > 8.0:
        kind = "C_quadrature_cancels_or_oscillates"
    elif strong:
        kind = "B_uniform_keeps_O1_feedback"
    elif fading:
        kind = "A_uniform_washes_out"
    elif abs_ratio > 2.5 or odd_ratio > 4.0:
        kind = "C_quadrature_cancels_or_oscillates"
    else:
        kind = "C_quadrature_cancels_or_oscillates"
    human = kind.startswith("B_")
    return {
        "kind": kind,
        "human_gate": human,
        "abs_A2_uniform": absA.tolist(),
        "re_A2_uniform": reA.tolist(),
        "cycle1_odd_uniform": odd.tolist(),
        "abs_A2_max_over_min": abs_ratio,
        "odd_max_over_min": odd_ratio,
        "re_A2_sign_flip": sign_flip,
        "probe6_abs_A2": probe["abs_A2_born"],
        "probe6_cycle1_odd": probe["cycle1"]["maxabs_odd"],
        "rule": (
            "B: all uniform |A2|>0.3, odd within x4, no Re(A2) sign flip. "
            "A: 6x6 |A2|<0.15 and cycle1 odd < 0.3 of probe6, or all weak. "
            "C: sign flip or |A2| or odd scatter. Human gate only on B."
        ),
    }


def make_figure(rows: list, cls: dict) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = [r["name"] for r in rows]
    a2 = [r["abs_A2_born"] for r in rows]
    odd = [r["cycle1"]["maxabs_odd"] for r in rows]
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.8))
    ax = axes[0]
    ax.bar(names, a2, color=["#888" if n == "probe6" else "#4C78A8" for n in names])
    ax.axhline(0.3, color="k", ls=":", lw=0.8)
    ax.set_ylabel(r"$|A_2|$ Born")
    ax.set_title("full-pulse H2 one-pass")
    ax.tick_params(axis="x", rotation=15)
    ax.grid(True, axis="y", alpha=0.3)
    ax = axes[1]
    ax.bar(names, odd, color=["#888" if n == "probe6" else "#F58518" for n in names])
    ax.set_ylabel(r"cycle-1 $\max|(I_+-I_-)/2|$")
    ax.set_title("tangent window")
    ax.tick_params(axis="x", rotation=15)
    ax.grid(True, axis="y", alpha=0.3)
    fig.suptitle(f"Stage-1G coarse BZ qual: {cls['kind']}", fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT / "stage1g_a2_cycle1.png", dpi=140)
    plt.close(fig)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    geo = build_geometry()
    prot = p1_a2_protocol(0.8)
    gamma = gamma_self(geo)
    pref = eta2_ncell(geo)
    blocked = False
    try:
        apply_resummation_to_output()
    except ResummationNotChosenError:
        blocked = True
    probe = k_sets(geo, R0, MU)["probe6"]
    kf4, _, _ = k_grid(geo, 4, shift=0.0)
    kf4s, _, _ = k_grid(geo, 4, shift=0.5)
    kf6, _, _ = k_grid(geo, 6, shift=0.0)
    grids = [
        ("probe6", probe["kf"], probe["labels"]),
        ("mp4", kf4, ["mp4"] * len(kf4)),
        ("mp4_shift", kf4s, ["mp4s"] * len(kf4s)),
        ("mp6", kf6, ["mp6"] * len(kf6)),
    ]
    rows = []
    for name, kf, lab in grids:
        rows.append(one_grid(geo, prot, gamma, name, kf, lab, pref))
    cls = classify(rows)
    print("[stage1g] CPU profile 80 steps", flush=True)
    prof = {
        "mp4_16k": profile_batch(4, 80),
        "mp6_36k": profile_batch(6, 80),
        "gpu": "not_implemented_for_M0",
        "note": "Python k-batch; eigh dominates. Not a GPU kernel.",
    }
    make_figure(rows, cls)
    payload = {
        "g1g_id": G1G_ID,
        "spec": stage1_spec(),
        "frozen": {
            "no_T2": True,
            "no_DC_subtraction": True,
            "P1_local_classical": True,
            "no_48_96_192": True,
            "no_fock_expansion": True,
            "production_V": "Born_reference_diagnostic_not_qualified",
        },
        "resummation_production_blocked": blocked,
        "nt_per_cycle": NTPC,
        "eps_seed_kappa": EPS,
        "window": "cycle1_a_priori_for_tangent_full_pulse_for_Born_A2",
        "grids": rows,
        "classification": cls,
        "profile": prof,
        "gain_claim": "not_evaluated",
        "squeezing_claim": "not_evaluated",
        "nk96": "not_run",
        "human_gate": cls["human_gate"],
        "next_if_not_B": "numerical_k_resolved_or_adaptive_not_48_96_192",
    }
    dump(OUT / "stage1g_bzqual.json", payload)
    print(json.dumps({
        "wrote": str(OUT / "stage1g_bzqual.json"),
        "kind": cls["kind"],
        "human_gate": cls["human_gate"],
        "abs_A2": {r["name"]: r["abs_A2_born"] for r in rows},
        "cycle1_odd": {r["name"]: r["cycle1"]["maxabs_odd"] for r in rows},
        "re_A2": {r["name"]: r["A2_born"]["re"] for r in rows},
        "walls": {r["name"]: r["profile"]["grid_wall_s"] for r in rows},
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
