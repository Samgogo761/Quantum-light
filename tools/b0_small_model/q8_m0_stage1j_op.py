#!/usr/bin/env python3
"""Stage-1J: feedback-operator qualification. Nested A2/B2 reference + σ_max(L).

L = G_self R^A, open loop. κ_self/κ_P1 is kept only as a seed stress test.
Third A2/B2 baseline: nested Monkhorst-Pack 4 / 4-shift / 6 / 8, plus the
Stage-1I adaptive k-sets. No 48/96/192, no T2, no DC, no BSV, no production V.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from q8_m0_adapt_bz import PerkCache, net_gross_complex, perk_born_ab  # noqa: E402
from q8_m0_cjj import dump  # noqa: E402
from q8_m0_classical import k_grid, p1_a2_protocol  # noqa: E402
from q8_m0_feedback_op import (  # noqa: E402
    OP_ID,
    adjoint_residual,
    apply_L_perk,
    classify_sigma,
    leading_singular,
    rayleigh_perk,
    scale_residual,
)
from q8_m0_gsheet import gamma_self  # noqa: E402
from q8_m0_loop import ResummationNotChosenError, apply_resummation_to_output  # noqa: E402
from q8_m0_mode_io import eta2_ncell  # noqa: E402
from q8_m0_model import R0, build_geometry  # noqa: E402
from q8_m0_stage1 import stage1_spec  # noqa: E402
from q8_m0_stage1g_bzqual import h2_seed  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "results/q_line/stage1j_op_20260922"
I1I = REPO / "results/q_line/stage1i_traj_20260922"
MU = -0.009010620372198276
NTPC = 400
EPS = 3.0e-5
G1J_ID = "feedback_op_sigma_max_20260922"


def load_i1_set(name: str):
    path = I1I / f"{name}.npz"
    if not path.exists():
        return None
    z = np.load(path)
    return {"kf": np.asarray(z["kf"]), "w": np.asarray(z["w"])}


def one_set(name, kf, w, cache, geo, prot, gamma, pref, hk, *, want_map: bool):
    t0 = time.perf_counter()
    print(f"[stage1j] {name} nk={len(kf)}", flush=True)
    cache.fill(geo, R0, prot, kf, mu=MU, neel=+1, ntpc=NTPC)
    packed = cache.assemble(kf, w)
    a2, b2 = perk_born_ab(
        packed["X_full"], packed["perk_D_t"], hk["F"], hk["e_perp"], hk["k_a"],
        packed["dt"], pref,
    )
    qa = net_gross_complex(a2, packed["w_k"])
    qb = net_gross_complex(b2, packed["w_k"])
    f_h2 = np.real(hk["F"])[:, None] * np.asarray(hk["e_perp"])[None, :]
    print("  sigma_max subspace", flush=True)
    op = leading_singular(
        packed["X_full"], packed["w_k"], packed["mean_D_t"],
        packed["dt"], gamma, f_probe=f_h2, p=3, n_iter=5, rng=0,
    )
    alpha = None
    cancel_alpha = None
    if want_map:
        vk = apply_L_perk(
            packed["X_full"], packed["perk_D_t"], op["v_right"], packed["dt"], gamma,
        )
        alpha = rayleigh_perk(op["u_left"], vk, packed["dt"])
        net = float(np.dot(packed["w_k"], alpha))
        gross = float(np.dot(packed["w_k"], np.abs(alpha)))
        cancel_alpha = float(abs(net) / max(gross, 1e-30))
    ident = {}
    if name == "mp4":
        rng = np.random.default_rng(1)
        g_rand = rng.normal(size=f_h2.shape)
        ident = {
            "adjoint_rel": adjoint_residual(
                packed["X_full"], packed["w_k"], packed["mean_D_t"],
                f_h2, g_rand, packed["dt"], gamma,
            ),
            "scale_rel": scale_residual(
                packed["X_full"], packed["w_k"], packed["mean_D_t"],
                f_h2, packed["dt"], gamma,
            ),
        }
    wall = time.perf_counter() - t0
    rec = {
        "name": name,
        "n_k": int(len(kf)),
        "A2": qa,
        "B2": qb,
        "sigma_max": op["sigma_max"],
        "sigma_trial": op["sigma_trial"],
        "sigma_hist": op["sigma_hist"],
        "n_iter": op["n_iter"],
        "gain_h2_dir": op["gain_h2_dir"],
        "rayleigh_h2": op["rayleigh_h2"],
        "sigma_over_h2": op["sigma_over_h2"],
        "sigma_class": classify_sigma(op["sigma_max"]),
        "alpha_cancel": cancel_alpha,
        "alpha_net": None if alpha is None else float(np.dot(packed["w_k"], alpha)),
        "alpha_gross": None if alpha is None else float(np.dot(packed["w_k"], np.abs(alpha))),
        "identity": ident,
        "waveform": waveform_features(op["v_right"], op["u_left"], f_h2, packed["dt"]),
        "wall_s": wall,
    }
    print(
        f"  |A2|={qa['net']['abs']:.4f} |B2|={qb['net']['abs']:.4f} "
        f"smax={op['sigma_max']:.4f} h2gain={op['gain_h2_dir']} "
        f"class={rec['sigma_class']} wall={wall:.1f}s",
        flush=True,
    )
    slim_op = {k: v for k, v in op.items() if k not in ("v_right", "u_left")}
    rec["op_meta"] = slim_op
    rec["_kf"] = np.asarray(kf)
    rec["_w"] = packed["w_k"]
    rec["_a2"] = a2
    rec["_b2"] = b2
    rec["_alpha"] = alpha
    rec["_v"] = op["v_right"]
    rec["_u"] = op["u_left"]
    rec["_times"] = packed["times"]
    rec["_dt"] = packed["dt"]
    np.savez(
        OUT / f"{name}.npz",
        kf=rec["_kf"], w=packed["w_k"], a2=a2, b2=b2,
        v_right=op["v_right"], u_left=op["u_left"],
        times=packed["times"],
        **({"alpha": alpha} if alpha is not None else {}),
    )
    return rec


def op_verdict(rows: list) -> dict:
    uni = [r for r in rows if r["name"].startswith("mp")]
    sigs = np.array([r["sigma_max"] for r in uni], dtype=float)
    a2s = np.array([r["A2"]["net"]["abs"] for r in uni], dtype=float)
    spread = float(np.max(sigs) / max(float(np.min(sigs)), 1e-30))
    a2_spread = float(np.max(a2s) / max(float(np.min(a2s)), 1e-30))
    mean_s = float(np.mean(sigs))
    agree = bool(spread < 1.6 and float(np.max(sigs) - np.min(sigs)) < 0.25)
    a2_band = bool(np.all((a2s > 0.12) & (a2s < 0.28)) and a2_spread < 2.5)
    klass = classify_sigma(mean_s)
    gate = bool(agree and klass == "gtrsim_1_hidden_strong_direction")
    node = "hidden_strong_feedback_direction" if gate else None
    return {
        "sigma_uniform": sigs.tolist(),
        "sigma_mean_uniform": mean_s,
        "sigma_spread_max_over_min": spread,
        "sigma_agree": agree,
        "sigma_class_mean": klass,
        "A2_uniform": a2s.tolist(),
        "A2_spread_max_over_min": a2_spread,
        "A2_nested_in_teens_to_0p25": a2_band,
        "probe6_O1_retired": bool(float(np.max(a2s)) < 0.35),
        "all_sigma_gt1": bool(np.min(sigs) > 1.0),
        "all_h2_gain_gt1": bool(np.min([r["gain_h2_dir"] for r in rows]) > 1.0),
        "human_gate": gate,
        "human_node": node,
        "born_restore_qualified": False,
        "production_V": "Born_reference_diagnostic_not_qualified",
        "control_parameter": "sigma_max_L",
        "kappa_ratio": "stress_test_only",
        "note": (
            "Human gate only if nested uniform sigma_max agrees and is >=1. "
            "Qualitative sigma>>1 on every grid is reported, not a moved gate. "
            "A2~0.2 is not auto-promoted to production V."
        ),
    }


def waveform_features(v, u, f, dt: float) -> dict:
    from q8_m0_feedback_op import inner_dt, norm_dt

    v = np.asarray(v, dtype=float)
    u = np.asarray(u, dtype=float)
    f = np.asarray(f, dtype=float)
    nt = int(v.shape[0])

    def energy_frac(a, sl) -> float:
        e = np.sum(a * a, axis=1)
        tot = float(dt) * float(np.sum(e))
        return float(dt) * float(np.sum(e[sl])) / max(tot, 1e-30)

    nv = norm_dt(v, dt)
    nu = norm_dt(u, dt)
    nf = norm_dt(f, dt)
    dv = np.diff(v, axis=0)
    return {
        "overlap_v_H2": inner_dt(v, f, dt) / max(nv * nf, 1e-30),
        "overlap_u_H2": inner_dt(u, f, dt) / max(nu * nf, 1e-30),
        "overlap_vu": inner_dt(v, u, dt) / max(nv * nu, 1e-30),
        "v_energy_last20": energy_frac(v, slice(int(0.8 * nt), None)),
        "u_energy_last20": energy_frac(u, slice(int(0.8 * nt), None)),
        "v_energy_first_cycle": energy_frac(v, slice(0, max(nt // 8, 1))),
        "u_energy_first_cycle": energy_frac(u, slice(0, max(nt // 8, 1))),
        "v_adjacent_l2_over_norm": float(np.sqrt(float(dt) * np.sum(dv * dv))) / max(nv, 1e-30),
        "v_end_norm": float(np.linalg.norm(v[-1])),
        "u_end_norm": float(np.linalg.norm(u[-1])),
    }


def make_sv_figure(rows: list, hk) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    f = np.real(hk["F"])[:, None] * np.asarray(hk["e_perp"])[None, :]
    pick = [r for r in rows if r["name"] in ("mp4", "mp8", "i1_ai")]
    if not pick:
        pick = rows[: min(3, len(rows))]
    fig, axes = plt.subplots(len(pick), 1, figsize=(8.2, 2.4 * len(pick)), sharex=True, squeeze=False)
    for ax, rec in zip(axes[:, 0], pick):
        t = rec["_times"]
        tp = float(t[-1])
        x = t / max(tp, 1e-30)
        v = rec["_v"]
        u = rec["_u"]
        ax.plot(x, v[:, 0], label="v_0 right", lw=1.0)
        ax.plot(x, u[:, 0], label="u_0 left", lw=1.0)
        fn = f[:, 0] / max(float(np.max(np.abs(f))), 1e-30)
        ax.plot(x, 0.05 * fn, label="H2 shape (x0.05)", ls=":", color="k", lw=0.8)
        ax.set_ylabel(rec["name"])
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7, loc="upper left")
    axes[-1, 0].set_xlabel(r"$t/T_p$")
    fig.suptitle("leading singular pair vs H2 packet", fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT / "stage1j_singular_t.png", dpi=140)
    plt.close(fig)


def make_figures(rows: list, verdict: dict) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = [r["name"] for r in rows]
    fig, axes = plt.subplots(1, 2, figsize=(9.8, 3.8))
    ax = axes[0]
    ax.bar(names, [r["A2"]["net"]["abs"] for r in rows], color="#4C78A8")
    ax.axhline(0.2, color="k", ls=":", lw=0.8)
    ax.set_ylabel("|A2|")
    ax.tick_params(axis="x", rotation=25)
    ax.grid(True, axis="y", alpha=0.3)
    ax.set_title("nested A2 reference")
    ax = axes[1]
    ax.bar(names, [r["sigma_max"] for r in rows], color="#F58518")
    ax.axhline(1.0, color="k", ls=":", lw=0.8)
    ax.set_ylabel("sigma_max(L)")
    ax.tick_params(axis="x", rotation=25)
    ax.grid(True, axis="y", alpha=0.3)
    ax.set_title(f"L gain  agree={verdict['sigma_agree']}")
    fig.tight_layout()
    fig.savefig(OUT / "stage1j_a2_sigma.png", dpi=140)
    plt.close(fig)

    mapped = [r for r in rows if r["_alpha"] is not None]
    if mapped:
        fig, axes = plt.subplots(1, len(mapped), figsize=(4.2 * len(mapped), 3.8), squeeze=False)
        for ax, rec in zip(axes[0], mapped):
            kf = rec["_kf"]
            al = rec["_alpha"]
            sc = ax.scatter(kf[:, 0], kf[:, 1], c=al, cmap="coolwarm", s=36)
            ax.set_title(f"{rec['name']}  <u|L_k v>")
            ax.set_aspect("equal")
            ax.set_xlim(-0.05, 1.05)
            ax.set_ylim(-0.05, 1.05)
            fig.colorbar(sc, ax=ax, fraction=0.046)
        fig.tight_layout()
        fig.savefig(OUT / "stage1j_L_kmap.png", dpi=140)
        plt.close(fig)


def slim(rec: dict) -> dict:
    return {k: v for k, v in rec.items() if not k.startswith("_") and k != "op_meta"} | {
        "op_meta": {k: rec["op_meta"][k] for k in rec["op_meta"] if k not in ("v_right", "u_left")}
    }


def post_from_npz() -> None:
    """Attach waveform diagnostics and the singular-vector figure without re-CJJ."""
    payload = json.loads((OUT / "stage1j_op.json").read_text(encoding="utf-8"))
    geo = build_geometry()
    prot = p1_a2_protocol(0.8)
    nt = prot.n_cycles * NTPC
    times = np.arange(nt + 1) * (prot.period / NTPC)
    _a_in, hk = h2_seed(times, prot, geo, EPS)
    f_h2 = np.real(hk["F"])[:, None] * np.asarray(hk["e_perp"])[None, :]
    rows = []
    for g in payload["grids"]:
        z = np.load(OUT / f"{g['name']}.npz")
        rec = dict(g)
        rec["_kf"] = np.asarray(z["kf"])
        rec["_w"] = np.asarray(z["w"])
        rec["_v"] = np.asarray(z["v_right"])
        rec["_u"] = np.asarray(z["u_left"])
        rec["_times"] = np.asarray(z["times"])
        rec["_alpha"] = np.asarray(z["alpha"]) if "alpha" in z.files else None
        rec["_a2"] = np.asarray(z["a2"])
        dt = float(rec["_times"][1] - rec["_times"][0])
        rec["waveform"] = waveform_features(rec["_v"], rec["_u"], f_h2, dt)
        g["waveform"] = rec["waveform"]
        rows.append(rec)
    verdict = op_verdict(rows)
    payload["spec"] = stage1_spec()
    payload["verdict"] = verdict
    payload["human_gate"] = verdict["human_gate"]
    payload["human_node"] = verdict["human_node"]
    make_sv_figure(rows, hk)
    make_figures(rows, verdict)
    dump(OUT / "stage1j_op.json", payload)
    print(json.dumps({
        "post": True,
        "waveform": {r["name"]: r["waveform"] for r in rows},
        "verdict": verdict,
    }, indent=2), flush=True)


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
    times = np.arange(nt + 1) * (prot.period / NTPC)
    _a_in, hk = h2_seed(times, prot, geo, EPS)
    cache = PerkCache()
    jobs = []
    kf4, _, w4 = k_grid(geo, 4, shift=0.0)
    kf4s, _, w4s = k_grid(geo, 4, shift=0.5)
    kf6, _, w6 = k_grid(geo, 6, shift=0.0)
    kf8, _, w8 = k_grid(geo, 8, shift=0.0)
    jobs.append(("mp4", kf4, w4, False))
    jobs.append(("mp4_shift", kf4s, w4s, False))
    jobs.append(("mp6", kf6, w6, True))
    jobs.append(("mp8", kf8, w8, True))
    skipped_i1 = []
    for tag, fname in (("i1_trad", "traditional_final"), ("i1_ai", "ai_rank_final")):
        rec = load_i1_set(fname)
        if rec is None:
            skipped_i1.append(tag)
            continue
        jobs.append((tag, rec["kf"], rec["w"], True))
    rows = []
    for name, kf, w, want in jobs:
        rows.append(one_set(
            name, kf, w, cache, geo, prot, gamma, pref, hk, want_map=want,
        ))
    verdict = op_verdict(rows)
    make_sv_figure(rows, hk)
    make_figures(rows, verdict)
    payload = {
        "g1j_id": G1J_ID,
        "op_id": OP_ID,
        "spec": stage1_spec(),
        "frozen": {
            "no_T2": True,
            "no_DC_subtraction": True,
            "P1_local_classical": True,
            "no_48_96_192": True,
            "no_fock_expansion": True,
            "production_V": "Born_reference_diagnostic_not_qualified",
            "qualification_control": "sigma_max_L_not_kappa_ratio",
        },
        "resummation_production_blocked": blocked,
        "nt_per_cycle": NTPC,
        "skipped_i1_ksets": skipped_i1,
        "grids": [slim(r) for r in rows],
        "verdict": verdict,
        "cjj_cache": {"n_k_evaluated": cache.n_cjj_k, "wall_s": cache.wall_cjj},
        "gain_claim": "not_evaluated",
        "squeezing_claim": "not_evaluated",
        "nk96": "not_run",
        "human_gate": verdict["human_gate"],
        "human_node": verdict["human_node"],
    }
    dump(OUT / "stage1j_op.json", payload)
    print(json.dumps({
        "wrote": str(OUT / "stage1j_op.json"),
        "sigma": {r["name"]: r["sigma_max"] for r in rows},
        "A2": {r["name"]: r["A2"]["net"]["abs"] for r in rows},
        "h2gain": {r["name"]: r["gain_h2_dir"] for r in rows},
        "sigma_over_h2": {r["name"]: r["sigma_over_h2"] for r in rows},
        "verdict": verdict,
        "skipped_i1": skipped_i1,
        "cjj_k": cache.n_cjj_k,
    }, indent=2), flush=True)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "post":
        post_from_npz()
    else:
        main()
