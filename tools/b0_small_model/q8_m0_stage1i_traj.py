#!/usr/bin/env python3
"""Stage-1I: time-trajectory-aware bounded adaptive BZ quadrature.

Traditional ranking is the qualification baseline and looks at δI_k(t) / κ_k(t).
AI only orders tiles, with a multi-objective target (A2, B2, cycle-1 odd, κ(t)).
Same physical solver. No 48/96/192, no T2, no DC, no FNO-for-A2, no production V.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from q8_m0_adapt_bz import (  # noqa: E402
    ADAPT_ID,
    BUDGET,
    FEATURE_KEYS_TRAJ,
    ActiveRanker,
    PerkCache,
    assign_born,
    assign_traj,
    four_quantity_record,
    k_points_to_abs_error,
    leaf_arrays,
    multi_objective_target,
    pick_tiles,
    run_nl_block,
    split_leaves,
    start_leaves,
    tile_features,
    traj_l2,
    traj_verdict,
    traditional_score_traj,
)
from q8_m0_cjj import dump  # noqa: E402
from q8_m0_classical import p1_a2_protocol  # noqa: E402
from q8_m0_gsheet import gamma_self  # noqa: E402
from q8_m0_loop import ResummationNotChosenError, apply_resummation_to_output  # noqa: E402
from q8_m0_mode_io import eta2_ncell, response_to_complex_scalar  # noqa: E402
from q8_m0_model import R0, build_geometry  # noqa: E402
from q8_m0_s1_kernel import pack_complex  # noqa: E402
from q8_m0_stage1 import stage1_spec  # noqa: E402
from q8_m0_stage1g_bzqual import h2_seed  # noqa: E402
from q8_m0_volterra import collect_overlap  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "results/q_line/stage1i_traj_20260922"
MU = -0.009010620372198276
EPS = 3.0e-5
NTPC = 400
G1I_ID = "traj_aware_adapt_bz_20260922"


def identity_perk_vs_mean(packed, F, ehat, k_a, pref, a2, b2) -> dict:
    w = packed["w_k"]
    dt = packed["dt"]
    resp_f = response_to_complex_scalar(
        packed["X_full"], w, F, ehat, dt, packed["mean_D_t"],
    )
    resp_fs = response_to_complex_scalar(
        packed["X_full"], w, np.conj(F), ehat, dt, packed["mean_D_t"],
    )
    a2_mean = 1j * pref * collect_overlap(k_a, resp_fs["total"], dt)
    b2_mean = 1j * pref * collect_overlap(k_a, resp_f["total"], dt)
    a2_sum = complex(np.dot(w, a2))
    b2_sum = complex(np.dot(w, b2))
    return {
        "dA2_perk_minus_meanD": float(abs(a2_sum - a2_mean)),
        "dB2_perk_minus_meanD": float(abs(b2_sum - b2_mean)),
        "A2_perk": pack_complex(a2_sum),
        "A2_meanD": pack_complex(a2_mean),
    }


def evaluate_round(
    leaves, cache, geo, prot, gamma, pref, a_in, hk,
    *, mu, neel, ntpc, prev, parents=None, nl_cache=None, prev_kappa_t=None,
):
    kf, w = leaf_arrays(leaves)
    cache.fill(geo, R0, prot, kf, mu=mu, neel=neel, ntpc=ntpc)
    packed = cache.assemble(kf, w)
    a2, b2 = assign_born(leaves, packed, hk["F"], hk["e_perp"], hk["k_a"], pref)
    split_da = 0.0
    split_db = 0.0
    split_k = 0.0
    parent_rows = []
    nl = run_nl_block(geo, R0, prot, packed, a_in, gamma, mu, neel, ntpc, cache=nl_cache)
    assign_traj(leaves, nl["kappa_k"], nl["dI_k"], nl["dt"])
    dt = float(nl["dt"])
    if parents:
        for p in parents:
            ch = [
                t for t in leaves
                if t.level == p.level + 1 and t.ix // 2 == p.ix and t.iy // 2 == p.iy
            ]
            if len(ch) != 4:
                continue
            da = sum(t.area * t.a2 for t in ch) - p.area * p.a2
            db = sum(t.area * t.b2 for t in ch) - p.area * p.b2
            split_da += abs(da)
            split_db += abs(db)
            if p.kappa_traj is not None:
                parent_w = p.area * p.kappa_traj
                child_w = sum(t.area * t.kappa_traj for t in ch)
                dkap = traj_l2(child_w - parent_w, dt)
                split_k += dkap
                for t in ch:
                    t.inherited_disc = abs(t.a2 - p.a2) * t.area
                    t.inherited_kappa_disc = traj_l2(t.kappa_traj - p.kappa_traj, dt) * t.area
            parent_rows.append((p, float(abs(da) + abs(db))))
    ident = identity_perk_vs_mean(packed, hk["F"], hk["e_perp"], hk["k_a"], pref, a2, b2)
    for t, od in zip(leaves, nl["odd_abs_k"]):
        t.odd_abs = float(od)
    rec = four_quantity_record(
        a2=a2, b2=b2, w=w, odd=nl["odd"], even=nl["even"],
        a_self=nl["A_self"], kappa_p1=nl["kappa_p1"], i_perk_odd=nl["odd_k"],
        times=packed["times"], period=prot.period, geo=geo, a_star=prot.a0_dimless,
        prev=prev, split_delta_a2=split_da, split_delta_b2=split_db,
        kappa_k=nl["kappa_k"], dt=dt, prev_kappa_t=prev_kappa_t,
        split_delta_kappa_l2=split_k,
    )
    rec["identity"] = ident
    rec["zero_seed_rel"] = nl["zero_rel"]
    rec["plus_halted"] = nl["plus_halted"]
    rec["nl_wall_s"] = nl["wall_s"]
    rec["kappa_recon_err"] = nl["recon_err"]
    rec["kf"] = kf
    rec["w"] = w
    rec["a2"] = a2
    rec["b2"] = b2
    rec["_kappa_t"] = np.asarray(nl["A_self"])
    rec["_kappa_p1"] = np.asarray(nl["kappa_p1"])
    rec["_times"] = packed["times"]
    return rec, packed, parent_rows


def slim_round(rec: dict, method: str, round_id: int, rank_mode: str) -> dict:
    return {
        "method": method,
        "round": round_id,
        "rank_mode": rank_mode,
        "n_k": rec["n_k"],
        "A2": rec["A2"],
        "B2": rec["B2"],
        "I_odd_cycle1": rec["I_odd_cycle1"],
        "kappa_self": rec["kappa_self"],
        "delta": rec["delta"],
        "identity": rec["identity"],
        "zero_seed_rel": rec["zero_seed_rel"],
        "plus_halted": rec["plus_halted"],
        "nl_wall_s": rec["nl_wall_s"],
        "kappa_recon_err": rec["kappa_recon_err"],
        "round_wall_s": rec.get("round_wall_s"),
    }


def make_figures(hist_t, hist_a, leaves_t, leaves_a, full_t, full_a, verdict) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def scatter_tiles(ax, leaves, title, color_kappa=False):
        kf = np.array([t.center for t in leaves])
        area = np.array([t.area for t in leaves])
        if color_kappa:
            c = np.array([
                traj_l2(t.kappa_traj, t.traj_dt) if t.kappa_traj is not None else 0.0
                for t in leaves
            ])
            sc = ax.scatter(kf[:, 0], kf[:, 1], c=c, cmap="viridis", s=1800.0 * area)
            lab = "||kappa_k||_L2"
        else:
            re = np.array([complex(t.a2).real for t in leaves])
            sc = ax.scatter(
                kf[:, 0], kf[:, 1], c=re, cmap="coolwarm",
                s=1800.0 * area, vmin=-0.5, vmax=0.5,
            )
            lab = "Re a2(k)"
        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(-0.05, 1.05)
        ax.set_aspect("equal")
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        return sc, lab

    fig, axes = plt.subplots(2, 2, figsize=(9.6, 8.4), sharex=True, sharey=True)
    sc0, lab0 = scatter_tiles(axes[0, 0], leaves_t, f"trad Re a2  nk={len(leaves_t)}")
    scatter_tiles(axes[0, 1], leaves_a, f"AI Re a2  nk={len(leaves_a)}")
    sc1, lab1 = scatter_tiles(
        axes[1, 0], leaves_t, f"trad ||kappa_k||_L2  nk={len(leaves_t)}", color_kappa=True,
    )
    scatter_tiles(axes[1, 1], leaves_a, f"AI ||kappa_k||_L2  nk={len(leaves_a)}", color_kappa=True)
    fig.colorbar(sc0, ax=axes[0, :].ravel().tolist(), fraction=0.03, pad=0.02, label=lab0)
    fig.colorbar(sc1, ax=axes[1, :].ravel().tolist(), fraction=0.03, pad=0.02, label=lab1)
    fig.suptitle("Stage-1I trajectory-aware tiles (not 48/96/192)", fontsize=11)
    fig.savefig(OUT / "stage1i_tiles.png", dpi=140)
    plt.close(fig)

    nk_t = [h["n_k"] for h in hist_t]
    nk_a = [h["n_k"] for h in hist_a]
    fig, axes = plt.subplots(2, 2, figsize=(9.6, 7.2))
    pairs = [
        (axes[0, 0], [h["A2"]["net"]["abs"] for h in hist_t],
         [h["A2"]["net"]["abs"] for h in hist_a], "|A2| net"),
        (axes[0, 1], [h["B2"]["net"]["abs"] for h in hist_t],
         [h["B2"]["net"]["abs"] for h in hist_a], "|B2| net"),
        (axes[1, 0], [h["I_odd_cycle1"]["net"] for h in hist_t],
         [h["I_odd_cycle1"]["net"] for h in hist_a], "cycle-1 max|odd|"),
        (axes[1, 1], [h["kappa_self"]["ratio_full"] for h in hist_t],
         [h["kappa_self"]["ratio_full"] for h in hist_a],
         "max|kappa_self|/|kappa_P1|"),
    ]
    for ax, yt, ya, ylab in pairs:
        ax.plot(nk_t, yt, "o-", label="traditional (baseline)")
        ax.plot(nk_a, ya, "s--", label="AI rank")
        ax.set_xlabel("n_k")
        ax.set_ylabel(ylab)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
    fig.suptitle(
        f"traj-aware  born={verdict['born_restore_qualified']}  "
        f"kappa_agree={verdict['kappa_strategies_agree']}  "
        f"human_gate={verdict['human_gate']}",
        fontsize=10,
    )
    fig.tight_layout()
    fig.savefig(OUT / "stage1i_q_vs_nk.png", dpi=140)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.8))
    ax = axes[0]
    d_t = [h["delta"]["kappa_traj_rel_l2"] or np.nan for h in hist_t]
    d_a = [h["delta"]["kappa_traj_rel_l2"] or np.nan for h in hist_a]
    ax.semilogy(nk_t, np.maximum(d_t, 1e-16), "o-", label="trad ||d kappa||_L2 / ||P1||")
    ax.semilogy(nk_a, np.maximum(d_a, 1e-16), "s--", label="AI")
    ax.axhline(0.05, color="k", ls=":", lw=0.8)
    ax.set_xlabel("n_k")
    ax.set_ylabel("successive kappa trajectory error")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax = axes[1]
    ax.plot(nk_t, [h["A2"]["net"]["abs"] for h in hist_t], "o-", label="|A2| net trad")
    ax.plot(nk_t, [h["A2"]["gross"] for h in hist_t], "o--", label="A2 gross trad")
    ax.plot(nk_a, [h["A2"]["net"]["abs"] for h in hist_a], "s-", label="|A2| net AI")
    ax.plot(nk_a, [h["A2"]["gross"] for h in hist_a], "s--", label="A2 gross AI")
    ax.set_xlabel("n_k")
    ax.set_ylabel("A2")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "stage1i_kappa_err.png", dpi=140)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.4, 4.0))
    times = full_t[-1]["_times"]
    tp = float(times[-1])
    for rec, ls, lab in (
        (full_t[-1], "-", "trad final"),
        (full_a[-1], "--", "AI final"),
        (full_t[0], ":", "start 16k"),
    ):
        kap = rec["_kappa_t"]
        ax.plot(times / tp, np.linalg.norm(kap, axis=1), ls, label=lab, lw=1.2)
    ax.set_xlabel("t / Tp")
    ax.set_ylabel("|kappa_self(t)|")
    ax.set_title("full-pulse self-field trajectory")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "stage1i_kappa_t.png", dpi=140)
    plt.close(fig)


def save_round_arrays(name: str, rec: dict) -> None:
    np.savez(
        OUT / f"{name}.npz",
        kf=rec["kf"], w=rec["w"], a2=rec["a2"], b2=rec["b2"],
        kappa_t=rec["_kappa_t"], kappa_p1=rec["_kappa_p1"], times=rec["_times"],
    )


def run_one_method(name, use_ai, cache, geo, prot, gamma, pref, a_in, hk, mu, neel, ntpc, nl_cache):
    leaves = start_leaves(BUDGET["start_level"])
    ranker = ActiveRanker(score_fn=traditional_score_traj, keys=FEATURE_KEYS_TRAJ) if use_ai else None
    hist = []
    slim = []
    prev = None
    parents = None
    pick_feat = {}
    rank_mode = "start_4x4_midpoint"
    packed = None
    prev_kappa_t = None
    for rd in range(BUDGET["max_rounds"] + 1):
        t0 = time.perf_counter()
        rec, packed, parent_rows = evaluate_round(
            leaves, cache, geo, prot, gamma, pref, a_in, hk,
            mu=mu, neel=neel, ntpc=ntpc, prev=prev, parents=parents,
            nl_cache=nl_cache, prev_kappa_t=prev_kappa_t,
        )
        if use_ai and parent_rows:
            y_round = multi_objective_target(
                dA2=rec["delta"]["A2_successive_abs"] or 0.0,
                dB2=rec["delta"]["B2_successive_abs"] or 0.0,
                d_odd=rec["delta"]["I_odd_successive_abs"] or 0.0,
                kappa_rel_l2=rec["delta"]["kappa_traj_rel_l2"] or 0.0,
                d_ratio=rec["delta"]["kappa_ratio_successive_abs"] or 0.0,
            )
            weights = []
            for p, _ in parent_rows:
                kap = p.kappa_traj
                dt = p.traj_dt
                weights.append(p.area * max(traj_l2(kap, dt) if kap is not None else 0.0, 1e-30))
            wsum = float(sum(weights)) or 1.0
            for (p, _), wp in zip(parent_rows, weights):
                feat = pick_feat.get(p.key) or tile_features(p, leaves)
                ranker.observe(feat, y_round * (wp / wsum))
        wall = time.perf_counter() - t0
        rec["round_wall_s"] = wall
        rec["rank_mode"] = rank_mode
        hist.append(rec)
        slim.append(slim_round(rec, name, rd, rank_mode))
        d_a = rec["delta"]["A2_successive_abs"]
        d_k = rec["delta"]["kappa_traj_rel_l2"]
        d_s = "na" if d_a is None else f"{d_a:.4f}"
        k_s = "na" if d_k is None else f"{d_k:.4f}"
        print(
            f"[{name} r{rd}] nk={rec['n_k']} |A2|={rec['A2']['net']['abs']:.4f} "
            f"|B2|={rec['B2']['net']['abs']:.4f} odd={rec['I_odd_cycle1']['net']:.3e} "
            f"ks/kp={rec['kappa_self']['ratio_full']:.3f} dA2={d_s} dKapL2={k_s} "
            f"recon={rec['kappa_recon_err']:.1e} mode={rank_mode} wall={wall:.1f}s",
            flush=True,
        )
        if rec["n_k"] >= BUDGET["max_k"] or rd >= BUDGET["max_rounds"]:
            break
        feats = [tile_features(t, leaves) for t in leaves]
        if not use_ai:
            scores = np.array([traditional_score_traj(f) for f in feats], dtype=float)
            rank_mode = "traditional_traj_indicator"
        else:
            scores, rank_mode = ranker.scores(feats)
        parents = pick_tiles(leaves, scores, BUDGET["splits_per_round"], BUDGET["max_level"])
        if not parents:
            break
        pick_feat = {p.key: tile_features(p, leaves) for p in parents}
        prev_kappa_t = rec["_kappa_t"]
        leaves = split_leaves(leaves, parents)
        prev = rec
    return hist, slim, leaves, packed, ranker


def budget_compare(hist_t, hist_a) -> dict:
    k_t = k_points_to_abs_error(hist_t, "A2_successive_abs", 0.02)
    k_a = k_points_to_abs_error(hist_a, "A2_successive_abs", 0.02)
    k_kt = k_points_to_abs_error(hist_t, "kappa_traj_rel_l2", 0.05)
    k_ka = k_points_to_abs_error(hist_a, "kappa_traj_rel_l2", 0.05)
    saved_a2 = int(k_t - k_a) if (k_t is not None and k_a is not None) else None
    saved_k = int(k_kt - k_ka) if (k_kt is not None and k_ka is not None) else None
    return {
        "n_k_final_trad": hist_t[-1]["n_k"],
        "n_k_final_ai": hist_a[-1]["n_k"],
        "n_k_to_dA2_lt_0.02_trad": k_t,
        "n_k_to_dA2_lt_0.02_ai": k_a,
        "n_k_to_kappa_rel_l2_lt_0.05_trad": k_kt,
        "n_k_to_kappa_rel_l2_lt_0.05_ai": k_ka,
        "k_saved_by_ai_at_dA2": saved_a2,
        "k_saved_by_ai_at_kappa_traj": saved_k,
        "final_dA2_trad": hist_t[-1]["delta"]["A2_successive_abs"],
        "final_dA2_ai": hist_a[-1]["delta"]["A2_successive_abs"],
        "final_kappa_rel_l2_trad": hist_t[-1]["delta"]["kappa_traj_rel_l2"],
        "final_kappa_rel_l2_ai": hist_a[-1]["delta"]["kappa_traj_rel_l2"],
        "same_solver": True,
        "ai_predicts_A2": False,
        "ai_role": "next_tile_only_multiobjective",
        "traditional_is_baseline": True,
    }


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
    nt = prot.n_cycles * NTPC
    dt = prot.period / NTPC
    times = np.arange(nt + 1) * dt
    a_in, hk = h2_seed(times, prot, geo, EPS)
    cache = PerkCache()
    nl_cache = {}
    print("[stage1i] traditional trajectory-aware (baseline)", flush=True)
    hist_t, slim_t, leaves_t, _p, _r0 = run_one_method(
        "traditional", False, cache, geo, prot, gamma, pref, a_in, hk, MU, +1, NTPC, nl_cache,
    )
    print("[stage1i] AI multi-objective rank (same solver)", flush=True)
    hist_a, slim_a, leaves_a, _p2, ranker = run_one_method(
        "ai_rank", True, cache, geo, prot, gamma, pref, a_in, hk, MU, +1, NTPC, nl_cache,
    )
    verdict = traj_verdict(slim_t, slim_a)
    compare = budget_compare(slim_t, slim_a)
    if ranker is not None:
        compare["ai_n_ridge_picks"] = ranker.n_ai_picks
        compare["ai_n_cold_start"] = ranker.n_fallback
        compare["ai_n_train"] = len(ranker.y)
    make_figures(slim_t, slim_a, leaves_t, leaves_a, hist_t, hist_a, verdict)
    save_round_arrays("traditional_final", hist_t[-1])
    save_round_arrays("ai_rank_final", hist_a[-1])
    payload = {
        "g1i_id": G1I_ID,
        "adapt_id": ADAPT_ID,
        "spec": stage1_spec(),
        "budget": BUDGET,
        "frozen": {
            "no_T2": True,
            "no_DC_subtraction": True,
            "P1_local_classical": True,
            "no_48_96_192": True,
            "no_fock_expansion": True,
            "no_fno_for_A2": True,
            "production_V": "Born_reference_diagnostic_not_qualified",
        },
        "ranking": "time_trajectory_dI_and_kappa_k_traditional_baseline",
        "resummation_production_blocked": blocked,
        "nt_per_cycle": NTPC,
        "eps_seed_kappa": EPS,
        "traditional": slim_t,
        "ai_rank": slim_a,
        "budget_compare": compare,
        "verdict": verdict,
        "cjj_cache": {"n_k_evaluated": cache.n_cjj_k, "wall_s": cache.wall_cjj},
        "gain_claim": "not_evaluated",
        "squeezing_claim": "not_evaluated",
        "nk96": "not_run",
        "human_gate": verdict["human_gate"],
        "human_node": verdict["human_node"],
    }
    dump(OUT / "stage1i_traj.json", payload)
    print(json.dumps({
        "wrote": str(OUT / "stage1i_traj.json"),
        "born_restore_qualified": verdict["born_restore_qualified"],
        "probe6_A2_retired": verdict["probe6_A2_retired"],
        "kappa_strategies_agree": verdict["kappa_strategies_agree"],
        "human_gate": verdict["human_gate"],
        "human_node": verdict["human_node"],
        "A2_trad": verdict["A2_trad"],
        "A2_ai": verdict["A2_ai"],
        "kappa_ratio_trad": verdict["kappa_ratio_trad"],
        "kappa_ratio_ai": verdict["kappa_ratio_ai"],
        "budget_compare": compare,
        "cjj_k": cache.n_cjj_k,
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
