#!/usr/bin/env python3
"""Stage-1K: photon-normalized feedback operator L_ph = Q L S.

Control parameter is σ_max(L_ph) in the positive-frequency packet inner
product. Unweighted time-L2 σ_max(L) is kept only as a stress diagnostic.
Gauge/contact cancellation is reported, not used as a DC cut.
No 48/96/192, no T2, no BSV, no production V, no k-adapt this round.
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
from q8_m0_gsheet import gamma_self  # noqa: E402
from q8_m0_loop import ResummationNotChosenError, apply_resummation_to_output  # noqa: E402
from q8_m0_mode_io import eta2_ncell  # noqa: E402
from q8_m0_model import R0, build_geometry  # noqa: E402
from q8_m0_photon_op import (  # noqa: E402
    PH_OP_ID,
    adjoint_residual_ph,
    apply_L_perk_from_A,
    classify_sigma,
    gauge_contact_cancel,
    h2_phi,
    h2_unweighted_gain,
    leading_singular_ph,
    lowest_bin_gains,
    omega_weights,
    pack_phi_spec,
    photon_omega_grid,
    qs_reconstruction,
    rayleigh_perk_ph,
)
from q8_m0_s1_kernel import parallel_perp_basis  # noqa: E402
from q8_m0_stage1 import stage1_spec  # noqa: E402
from q8_m0_stage1g_bzqual import h2_seed  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "results/q_line/stage1k_ph_20260922"
I1I = REPO / "results/q_line/stage1i_traj_20260922"
I1J = REPO / "results/q_line/stage1j_op_20260922"
MU = -0.009010620372198276
NTPC = 400
EPS = 3.0e-5
G1K_ID = "photon_norm_L_ph_20260922"


def load_kset(*cands):
    for path in cands:
        if path is None or not path.exists():
            continue
        z = np.load(path)
        if "kf" in z.files and "w" in z.files:
            return {"kf": np.asarray(z["kf"]), "w": np.asarray(z["w"])}
    return None


def load_unweighted_diag() -> dict:
    path = I1J / "stage1j_op.json"
    if not path.exists():
        return {"available": False}
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload.get("grids", [])
    return {
        "available": True,
        "label": "unweighted_A_time_L2_stress_diagnostic_not_control",
        "sigma": {r["name"]: r["sigma_max"] for r in rows},
        "gain_h2_dir": {r["name"]: r["gain_h2_dir"] for r in rows},
        "rayleigh_h2": {r["name"]: r.get("rayleigh_h2") for r in rows},
        "note": payload.get("verdict", {}).get("note"),
    }


def one_set(name, kf, w, cache, geo, prot, gamma, pref, hk, omega, dw, ph2, e_perp, *, want_map: bool):
    t0 = time.perf_counter()
    print(f"[stage1k] {name} nk={len(kf)}", flush=True)
    cache.fill(geo, R0, prot, kf, mu=MU, neel=+1, ntpc=NTPC)
    packed = cache.assemble(kf, w)
    times = packed["times"]
    dt = packed["dt"]
    a2, b2 = perk_born_ab(
        packed["X_full"], packed["perk_D_t"], hk["F"], hk["e_perp"], hk["k_a"],
        dt, pref,
    )
    qa = net_gross_complex(a2, packed["w_k"])
    qb = net_gross_complex(b2, packed["w_k"])
    print("  sigma_max L_ph", flush=True)
    op = leading_singular_ph(
        packed["X_full"], packed["w_k"], packed["mean_D_t"],
        omega, times, dt, dw, gamma, phi_probe=ph2, p=3, n_iter=5, rng=0,
    )
    uw = h2_unweighted_gain(
        packed["X_full"], packed["w_k"], packed["mean_D_t"],
        ph2, omega, times, dt, dw, gamma,
    )
    gauge = {
        "perp": gauge_contact_cancel(
            packed["X_full"], packed["w_k"], packed["mean_D_t"], dt, times, e_perp,
        ),
    }
    ident = {}
    low = None
    if name == "mp4":
        rng = np.random.default_rng(4)
        psi = rng.normal(size=ph2.shape) + 1j * rng.normal(size=ph2.shape)
        ident = {
            "adjoint_rel": adjoint_residual_ph(
                packed["X_full"], packed["w_k"], packed["mean_D_t"],
                ph2, psi, omega, times, dt, dw, gamma,
            ),
            "qs_h2": qs_reconstruction(ph2, omega, times, dt, dw),
        }
        low = lowest_bin_gains(
            packed["X_full"], packed["w_k"], packed["mean_D_t"],
            omega, times, dt, dw, gamma, e_perp,
        )
    alpha = None
    cancel_alpha = None
    if want_map:
        vk = apply_L_perk_from_A(
            packed["X_full"], packed["perk_D_t"], op["a_right"], dt, gamma,
        )
        alpha = rayleigh_perk_ph(op["u_left"], vk, omega, times, dt, dw)
        net = float(np.dot(packed["w_k"], alpha))
        gross = float(np.dot(packed["w_k"], np.abs(alpha)))
        cancel_alpha = float(abs(net) / max(gross, 1e-30))
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
        "unweighted_h2_dir_gain": uw["unweighted_dir_gain"],
        "sigma_class": classify_sigma(op["sigma_max"]),
        "v_spec": pack_phi_spec(op["v_right"], omega, dw, times, dt),
        "u_spec": pack_phi_spec(op["u_left"], omega, dw, times, dt),
        "gauge": gauge,
        "lowest_bin": low,
        "alpha_cancel": cancel_alpha,
        "alpha_net": None if alpha is None else float(np.dot(packed["w_k"], alpha)),
        "identity": ident,
        "wall_s": wall,
    }
    print(
        f"  |A2|={qa['net']['abs']:.4f} smax_ph={op['sigma_max']:.4f} "
        f"h2ph={op['gain_h2_dir']} uw={uw['unweighted_dir_gain']:.3f} "
        f"class={rec['sigma_class']} wall={wall:.1f}s",
        flush=True,
    )
    rec["op_meta"] = {k: v for k, v in op.items() if k not in ("v_right", "u_left", "a_right", "a_out")}
    rec["_kf"] = np.asarray(kf)
    rec["_w"] = packed["w_k"]
    rec["_alpha"] = alpha
    rec["_v"] = op["v_right"]
    rec["_u"] = op["u_left"]
    rec["_a_v"] = op["a_right"]
    rec["_a_out"] = op["a_out"]
    rec["_times"] = times
    rec["_omega"] = omega
    rec["_ph2"] = ph2
    np.savez(
        OUT / f"{name}.npz",
        kf=rec["_kf"], w=packed["w_k"], a2=a2, b2=b2,
        v_right=op["v_right"], u_left=op["u_left"],
        a_right=op["a_right"], a_out=op["a_out"],
        times=times, omega=omega,
        **({"alpha": alpha} if alpha is not None else {}),
    )
    return rec


def op_verdict(rows: list) -> dict:
    uni = [r for r in rows if r["name"].startswith("mp")]
    sigs = np.array([r["sigma_max"] for r in uni], dtype=float)
    a2s = np.array([r["A2"]["net"]["abs"] for r in uni], dtype=float)
    h2g = np.array([r["gain_h2_dir"] for r in uni], dtype=float)
    spread = float(np.max(sigs) / max(float(np.min(sigs)), 1e-30))
    mean_s = float(np.mean(sigs))
    agree = bool(spread < 1.6 and float(np.max(sigs) - np.min(sigs)) < 0.25)
    klass = classify_sigma(mean_s)
    gate = bool(agree and klass == "gtrsim_1_hidden_strong_direction")
    node = "photon_norm_hidden_strong_direction" if gate else None
    g_const = [r["gauge"]["perp"]["constant"]["late_cancel_ratio"] for r in uni]
    g_slow = [r["gauge"]["perp"]["slow_sin2_envelope"]["late_cancel_ratio"] for r in uni]
    return {
        "sigma_uniform": sigs.tolist(),
        "sigma_mean_uniform": mean_s,
        "sigma_spread_max_over_min": spread,
        "sigma_agree": agree,
        "sigma_class_mean": klass,
        "gain_h2_uniform": h2g.tolist(),
        "A2_uniform": a2s.tolist(),
        "A2_spread_max_over_min": float(np.max(a2s) / max(float(np.min(a2s)), 1e-30)),
        "probe6_O1_retired": bool(float(np.max(a2s)) < 0.35),
        "all_sigma_gt1": bool(np.min(sigs) > 1.0),
        "all_h2_gain_gt1": bool(np.min(h2g) > 1.0),
        "gauge_const_late_cancel": g_const,
        "gauge_slow_late_cancel": g_slow,
        "human_gate": gate,
        "human_node": node,
        "born_restore_qualified": False,
        "production_V": "Born_reference_diagnostic_not_qualified",
        "control_parameter": "sigma_max_L_ph_photon_inner_product",
        "unweighted_L2_A": "stress_diagnostic_only",
        "note": (
            "Human gate only if nested uniform sigma_max(L_ph) agrees and is >=1. "
            "Unweighted time-L2 sigma is not the control parameter."
        ),
    }


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
    ax.set_title("nested A2 (unchanged integrand)")
    ax = axes[1]
    ax.bar(names, [r["sigma_max"] for r in rows], color="#54A24B")
    ax.axhline(1.0, color="k", ls=":", lw=0.8)
    ax.set_ylabel("sigma_max(L_ph)")
    ax.tick_params(axis="x", rotation=25)
    ax.grid(True, axis="y", alpha=0.3)
    ax.set_title(f"photon-norm gain  agree={verdict['sigma_agree']}")
    fig.tight_layout()
    fig.savefig(OUT / "stage1k_a2_sigma_ph.png", dpi=140)
    plt.close(fig)

    pick = [r for r in rows if r["name"] in ("mp4", "mp8", "i1_ai")] or rows[: min(3, len(rows))]
    fig, axes = plt.subplots(len(pick), 2, figsize=(9.6, 2.5 * len(pick)), squeeze=False)
    for axrow, rec in zip(axes, pick):
        om = rec["_omega"]
        ax = axrow[0]
        ax.plot(om, np.sum(np.abs(rec["_v"]) ** 2, axis=1), label="|v_phi|^2")
        ax.plot(om, np.sum(np.abs(rec["_u"]) ** 2, axis=1), label="|u_phi|^2")
        ax.axvline(0.1, color="k", ls=":", lw=0.7)
        ax.set_ylabel(rec["name"])
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7)
        ax = axrow[1]
        t = rec["_times"]
        tp = float(t[-1])
        x = t / max(tp, 1e-30)
        ax.plot(x, rec["_a_v"][:, 0], label="S v", lw=1.0)
        ax.plot(x, rec["_a_out"][:, 0], label="L S v", lw=1.0)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7)
    axes[0, 0].set_title("photon spectrum")
    axes[0, 1].set_title("time: S v and L S v")
    axes[-1, 0].set_xlabel("omega")
    axes[-1, 1].set_xlabel(r"$t/T_p$")
    fig.tight_layout()
    fig.savefig(OUT / "stage1k_singular_phi.png", dpi=140)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    ax.plot(names, [r["gauge"]["perp"]["constant"]["late_cancel_ratio"] for r in rows], "o-", label="const A")
    ax.plot(names, [r["gauge"]["perp"]["slow_sin2_envelope"]["late_cancel_ratio"] for r in rows], "s-", label="slow sin2")
    ax.set_ylabel("late |I| / (|state|+|contact|)")
    ax.tick_params(axis="x", rotation=25)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    ax.set_title("gauge / contact cancellation (not a DC cut)")
    fig.tight_layout()
    fig.savefig(OUT / "stage1k_gauge.png", dpi=140)
    plt.close(fig)

    mapped = [r for r in rows if r["_alpha"] is not None]
    if mapped:
        fig, axes = plt.subplots(1, len(mapped), figsize=(4.2 * len(mapped), 3.8), squeeze=False)
        for ax, rec in zip(axes[0], mapped):
            sc = ax.scatter(rec["_kf"][:, 0], rec["_kf"][:, 1], c=rec["_alpha"], cmap="coolwarm", s=36)
            ax.set_title(f"{rec['name']}  ph-alpha_k")
            ax.set_aspect("equal")
            ax.set_xlim(-0.05, 1.05)
            ax.set_ylim(-0.05, 1.05)
            fig.colorbar(sc, ax=ax, fraction=0.046)
        fig.tight_layout()
        fig.savefig(OUT / "stage1k_Lph_kmap.png", dpi=140)
        plt.close(fig)


def slim(rec: dict) -> dict:
    return {k: v for k, v in rec.items() if not k.startswith("_") and k != "op_meta"} | {
        "op_meta": rec["op_meta"],
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    geo = build_geometry()
    prot = p1_a2_protocol(0.8)
    gamma = gamma_self(geo)
    pref = eta2_ncell(geo)
    _e_par, e_perp = parallel_perp_basis(geo)
    blocked = False
    try:
        apply_resummation_to_output()
    except ResummationNotChosenError:
        blocked = True
    nt = prot.n_cycles * NTPC
    times = np.arange(nt + 1) * (prot.period / NTPC)
    _a_in, hk = h2_seed(times, prot, geo, EPS)
    omega = photon_omega_grid(prot.t_pulse, prot.omega0, harm_max=12.0)
    dw = omega_weights(omega)
    ph2 = h2_phi(omega, prot.omega0, prot.t_pulse, e_perp)
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
    skipped = []
    for tag, cands in (
        ("i1_trad", (I1J / "i1_trad.npz", I1I / "traditional_final.npz")),
        ("i1_ai", (I1J / "i1_ai.npz", I1I / "ai_rank_final.npz")),
    ):
        rec = load_kset(*cands)
        if rec is None:
            skipped.append(tag)
            continue
        jobs.append((tag, rec["kf"], rec["w"], True))
    rows = []
    for name, kf, w, want in jobs:
        rows.append(one_set(
            name, kf, w, cache, geo, prot, gamma, pref, hk, omega, dw, ph2, e_perp,
            want_map=want,
        ))
    verdict = op_verdict(rows)
    make_figures(rows, verdict)
    payload = {
        "g1k_id": G1K_ID,
        "op_id": PH_OP_ID,
        "spec": stage1_spec(),
        "frozen": {
            "no_T2": True,
            "no_DC_subtraction": True,
            "P1_local_classical": True,
            "no_48_96_192": True,
            "no_fock_expansion": True,
            "no_k_adapt_this_round": True,
            "production_V": "Born_reference_diagnostic_not_qualified",
            "qualification_control": "sigma_max_L_ph_not_unweighted_L2_A",
        },
        "resummation_production_blocked": blocked,
        "nt_per_cycle": NTPC,
        "photon_grid": {
            "n_omega": int(omega.size),
            "omega_min": float(omega[0]),
            "omega_max": float(omega[-1]),
            "dw": dw,
            "excludes_omega_0": True,
            "harm_max": 12.0,
        },
        "unweighted_diagnostic": load_unweighted_diag(),
        "skipped_ksets": skipped,
        "grids": [slim(r) for r in rows],
        "verdict": verdict,
        "cjj_cache": {"n_k_evaluated": cache.n_cjj_k, "wall_s": cache.wall_cjj},
        "gain_claim": "not_evaluated",
        "squeezing_claim": "not_evaluated",
        "nk96": "not_run",
        "human_gate": verdict["human_gate"],
        "human_node": verdict["human_node"],
    }
    dump(OUT / "stage1k_ph.json", payload)
    print(json.dumps({
        "wrote": str(OUT / "stage1k_ph.json"),
        "sigma_ph": {r["name"]: r["sigma_max"] for r in rows},
        "A2": {r["name"]: r["A2"]["net"]["abs"] for r in rows},
        "h2ph": {r["name"]: r["gain_h2_dir"] for r in rows},
        "h2_unweighted": {r["name"]: r["unweighted_h2_dir_gain"] for r in rows},
        "verdict": verdict,
        "skipped": skipped,
        "cjj_k": cache.n_cjj_k,
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
