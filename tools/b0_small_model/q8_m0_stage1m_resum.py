#!/usr/bin/env python3
"""Stage-1M: H2-projected resummed BZ qualification, then n,m diagnostics.

Theory: linear feedback resummation is approved. Born is a benchmark.
No production V. No σ_max chase. No IR-hotspot adapt. No T2/DC/48/96/192.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from q8_m0_adapt_bz import (  # noqa: E402
    ActiveRanker,
    BUDGET,
    PerkCache,
    Tile,
    leaf_arrays,
    pick_tiles,
    split_leaves,
    start_leaves,
)
from q8_m0_cjj import dump  # noqa: E402
from q8_m0_classical import k_grid, p1_a2_protocol  # noqa: E402
from q8_m0_gsheet import gamma_self  # noqa: E402
from q8_m0_loop import ResummationNotChosenError, apply_resummation_to_output  # noqa: E402
from q8_m0_mode_io import eta2_ncell  # noqa: E402
from q8_m0_model import R0, build_geometry  # noqa: E402
from q8_m0_resummed_io import (  # noqa: E402
    RESUM_ID,
    covariant_residual,
    h2_channel,
    leave_one_out_scores,
    resummed_moments,
    vacuum_sheet_rms,
)
from q8_m0_s1_kernel import pack_complex  # noqa: E402
from q8_m0_stage1 import stage1_spec  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "results/q_line/stage1m_resum_20260923"
MU = -0.009010620372198276
NTPC = 400
G1M_ID = "h2_resummed_bz_nm_20260923"
THRESH = {
    "dA2_res": 0.15,
    "cross_A2_res": 0.20,
    "covariant_lt": 0.50,
}

FEATURE_KEYS = ("area", "abs_a2_res", "abs_a2_born", "loo_a2", "gross_a2_res", "level", "inherited_disc")


def slim_ch(ch: dict) -> dict:
    skip = {"a2_born_k", "b2_born_k", "a2_res_k", "b2_res_k", "A_sheet_re", "A_sheet_im", "I_res", "F", "e_perp", "k_a"}
    return {k: v for k, v in ch.items() if k not in skip}


def p1_peak(geo, prot) -> float:
    return float(prot.a0_dimless / geo.a_nn_mean_ang)


def eval_set(name, cache, geo, prot, kf, w, gamma, pref, *, want_cov: bool, want_loo: bool):
    t0 = time.perf_counter()
    print(f"[stage1m] {name} nk={len(kf)}", flush=True)
    cache.fill(geo, R0, prot, kf, mu=MU, neel=+1, ntpc=NTPC)
    packed = cache.assemble(kf, w)
    ch = h2_channel(packed, geo, prot, gamma, pref)
    loo = leave_one_out_scores(packed, geo, prot, gamma, pref, ch) if want_loo else None
    cov = None
    if want_cov:
        cov = covariant_residual(geo, R0, prot, cache, kf, w, mu=MU, neel=+1, ntpc=NTPC)
    wall = time.perf_counter() - t0
    rec = {
        "name": name,
        "n_k": int(len(kf)),
        "channel": slim_ch(ch),
        "covariant": None if cov is None else cov,
        "loo_max": None if loo is None else float(np.max(loo)),
        "wall_s": wall,
    }
    print(
        f"  |A2_res|={ch['A2_res']['abs']:.4f} |A2_born|={ch['A2_born']['abs']:.4f} "
        f"dph={ch['delta_born_h2_ph']:.4g} cov={None if cov is None else cov['compare']['rel_l2_plus']:.3f} "
        f"wall={wall:.1f}s",
        flush=True,
    )
    rec["_packed"] = packed
    rec["_ch"] = ch
    rec["_kf"] = np.asarray(kf)
    rec["_w"] = np.asarray(w)
    rec["_loo"] = loo
    return rec


def tile_feats(leaves, ch, loo):
    a2r = np.asarray(ch["a2_res_k"])
    a2b = np.asarray(ch["a2_born_k"])
    out = []
    for i, t in enumerate(leaves):
        t.a2 = complex(a2r[i])
        t.b2 = complex(ch["b2_res_k"][i])
        out.append({
            "area": float(t.area),
            "abs_a2_res": float(abs(a2r[i])),
            "abs_a2_born": float(abs(a2b[i])),
            "loo_a2": float(loo[i]),
            "gross_a2_res": float(abs(a2r[i]) * t.area),
            "level": float(t.level),
            "inherited_disc": float(t.inherited_disc),
        })
    return out


def res_score(feat: dict) -> float:
    return float(feat["loo_a2"] + 0.25 * feat["gross_a2_res"] + 0.5 * feat["inherited_disc"])


def run_adapt(name, cache, geo, prot, gamma, pref, *, use_ai: bool):
    leaves = start_leaves(BUDGET["start_level"])
    ranker = ActiveRanker(score_fn=res_score, keys=FEATURE_KEYS)
    hist = []
    prev = None
    for rd in range(BUDGET["max_rounds"] + 1):
        kf, w = leaf_arrays(leaves)
        rec = eval_set(
            f"{name}_r{rd}", cache, geo, prot, kf, w, gamma, pref,
            want_cov=True, want_loo=True,
        )
        rec["round"] = rd
        rec["method"] = name
        feats = tile_feats(leaves, rec["_ch"], rec["_loo"])
        if prev is not None:
            rec["dA2_res_successive"] = float(abs(
                complex(rec["channel"]["A2_res"]["re"], rec["channel"]["A2_res"]["im"])
                - complex(prev["channel"]["A2_res"]["re"], prev["channel"]["A2_res"]["im"])
            ))
            rec["d_cov_successive"] = float(abs(
                rec["covariant"]["compare"]["rel_l2_plus"]
                - prev["covariant"]["compare"]["rel_l2_plus"]
            ))
        else:
            rec["dA2_res_successive"] = None
            rec["d_cov_successive"] = None
        if use_ai:
            for f in feats:
                ranker.observe(f, f["loo_a2"])
        rec["_feats"] = feats
        if rec["n_k"] >= BUDGET["max_k"] or rd == BUDGET["max_rounds"]:
            rec["rank_mode"] = "stop"
            hist.append(rec)
            break
        scores, mode = ranker.scores(feats)
        rec["rank_mode"] = mode
        hist.append(rec)
        prev = rec
        picked = pick_tiles(leaves, scores, BUDGET["splits_per_round"], BUDGET["max_level"])
        if not picked:
            break
        for p, f in zip(leaves, feats):
            if p in picked:
                for c in p.children():
                    c.inherited_disc = float(f["loo_a2"])
        print(f"  split {[(p.level, p.ix, p.iy) for p in picked]} mode={mode}", flush=True)
        leaves = split_leaves(leaves, picked)
    return hist


def close_set(rec, geo, prot, gamma, pref, a_peak):
    packed, ch = rec["_packed"], rec["_ch"]
    mom = resummed_moments(packed, ch, gamma, pref)
    rms = vacuum_sheet_rms(packed, ch, gamma, pref, a_peak)
    return {"moments": mom, "rms": rms}


def bz_verdict(nested, hist_t, hist_a) -> dict:
    def a2(rec):
        return complex(rec["channel"]["A2_res"]["re"], rec["channel"]["A2_res"]["im"])

    t, a = hist_t[-1], hist_a[-1]
    d_t = t.get("dA2_res_successive")
    d_a = a.get("dA2_res_successive")
    cross = float(abs(a2(t) - a2(a)))
    cov_t = t["covariant"]["compare"]["rel_l2_plus"]
    cov_a = a["covariant"]["compare"]["rel_l2_plus"]
    cov0 = hist_t[0]["covariant"]["compare"]["rel_l2_plus"]
    nest = [r["channel"]["A2_res"]["abs"] for r in nested]
    stable = bool(
        d_t is not None and d_a is not None
        and d_t < THRESH["dA2_res"] and d_a < THRESH["dA2_res"]
        and cross < THRESH["cross_A2_res"]
    )
    cov_ok = bool(
        (cov_t < THRESH["covariant_lt"] and cov_a < THRESH["covariant_lt"])
        or (cov_t < 0.90 * cov0 and cov_a < 0.90 * cov0)
    )
    return {
        "A2_res_nested_abs": nest,
        "A2_res_trad": pack_complex(a2(t)),
        "A2_res_ai": pack_complex(a2(a)),
        "dA2_res_trad": d_t,
        "dA2_res_ai": d_a,
        "A2_res_cross": cross,
        "covariant_trad": cov_t,
        "covariant_ai": cov_a,
        "covariant_start": cov0,
        "h2_channel_stable": stable,
        "covariant_improving_or_small": cov_ok,
        "h2_resolvent_qualified": bool(stable and cov_ok),
        "production_V": "blocked",
    }


def make_figures(nested, hist_t, hist_a) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    ax.plot([r["n_k"] for r in nested], [r["channel"]["A2_res"]["abs"] for r in nested], "o-", label="|A2_res| nested")
    ax.plot([r["n_k"] for r in nested], [r["channel"]["A2_born"]["abs"] for r in nested], "s--", label="|A2_born|")
    ax.set_xlabel("n_k")
    ax.set_ylabel("|A2|")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "stage1m_nested_A2.png", dpi=140)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.8, 3.6))
    ax.plot([h["n_k"] for h in hist_t], [h["channel"]["A2_res"]["abs"] for h in hist_t], "o-", label="trad")
    ax.plot([h["n_k"] for h in hist_a], [h["channel"]["A2_res"]["abs"] for h in hist_a], "s-", label="AI")
    ax.set_xlabel("n_k")
    ax.set_ylabel("|A2_res|")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "stage1m_adapt_A2.png", dpi=140)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.8, 3.6))
    ax.plot([h["n_k"] for h in hist_t], [h["covariant"]["compare"]["rel_l2_plus"] for h in hist_t], "o-", label="trad")
    ax.plot([h["n_k"] for h in hist_a], [h["covariant"]["compare"]["rel_l2_plus"] for h in hist_a], "s-", label="AI")
    ax.set_xlabel("n_k")
    ax.set_ylabel("covariant rel L2")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "stage1m_covariant.png", dpi=140)
    plt.close(fig)


def slim_hist(h: dict) -> dict:
    return {k: v for k, v in h.items() if not k.startswith("_")}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    geo = build_geometry()
    prot = p1_a2_protocol(0.8)
    gamma = gamma_self(geo)
    pref = eta2_ncell(geo)
    a_peak = p1_peak(geo, prot)
    prod_blocked = False
    try:
        apply_resummation_to_output()
    except ResummationNotChosenError:
        prod_blocked = True
    cache = PerkCache()
    t0 = time.perf_counter()
    nested = []
    for nk, tag in ((4, "mp4"), (6, "mp6"), (8, "mp8")):
        kf, _, w = k_grid(geo, nk, shift=0.0)
        nested.append(eval_set(tag, cache, geo, prot, kf, w, gamma, pref, want_cov=True, want_loo=False))
    hist_t = run_adapt("trad", cache, geo, prot, gamma, pref, use_ai=False)
    hist_a = run_adapt("ai", cache, geo, prot, gamma, pref, use_ai=True)
    verdict = bz_verdict(nested, hist_t, hist_a)
    finals = {
        "mp8": nested[-1],
        "trad": hist_t[-1],
        "ai": hist_a[-1],
    }
    closed = {}
    lin_fail = False
    for key, rec in finals.items():
        print(f"[stage1m] n,m / RMS {key}", flush=True)
        closed[key] = close_set(rec, geo, prot, gamma, pref, a_peak)
        if closed[key]["rms"].get("linear_selfcons_fail"):
            lin_fail = True
        print(
            f"  comm={closed[key]['moments']['resummed']['commutator_abs_minus_1']:.4f} "
            f"pass={closed[key]['moments']['commutator_pass']} "
            f"rms={closed[key]['rms']['max_rms']:.3e} "
            f"vsP1={closed[key]['rms']['ratio_to_p1']:.3e}",
            flush=True,
        )
    make_figures(nested, hist_t, hist_a)
    payload = {
        "g1m_id": G1M_ID,
        "resum_id": RESUM_ID,
        "spec": stage1_spec(),
        "frozen": {
            "theory_main": "linear_feedback_resummation",
            "born_role": "perturbative_benchmark",
            "no_T2": True,
            "no_DC_subtraction": True,
            "no_48_96_192": True,
            "no_sigma_max_chase": True,
            "no_ir_hotspot_adapt": True,
            "production_V": "blocked",
        },
        "production_V_blocked": prod_blocked,
        "budget": BUDGET,
        "nested": [slim_hist(r) for r in nested],
        "adapt_trad": [slim_hist(r) for r in hist_t],
        "adapt_ai": [slim_hist(r) for r in hist_a],
        "bz_verdict": verdict,
        "closed": closed,
        "cjj_cache": {"n_k_evaluated": cache.n_cjj_k, "wall_s": cache.wall_cjj},
        "human_gate": bool(lin_fail),
        "human_node": "linear_R_out_of_range" if lin_fail else None,
        "squeezing_claim": "not_evaluated",
        "nk96": "not_run",
        "wall_s": time.perf_counter() - t0,
    }
    dump(OUT / "stage1m_resum.json", payload)
    print(json.dumps({
        "wrote": str(OUT / "stage1m_resum.json"),
        "A2_res": {r["name"]: r["channel"]["A2_res"]["abs"] for r in nested + [hist_t[-1], hist_a[-1]]},
        "bz": verdict,
        "comm": {k: v["moments"]["resummed"]["commutator_abs_minus_1"] for k, v in closed.items()},
        "rms": {k: v["rms"]["max_rms"] for k, v in closed.items()},
        "human_gate": payload["human_gate"],
        "cjj_k": cache.n_cjj_k,
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
