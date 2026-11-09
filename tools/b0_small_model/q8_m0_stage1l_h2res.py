#!/usr/bin/env python3
"""Stage-1L: covariant k-shift gauge + H2-projected resolvent + T_obs scan.

σ_max(L_ph) is only a global worst-case stress test. Born for the H2
output is not a production passport until (i) the covariant k-shift
leaves BZ-total J invariant up to quadrature, and (ii) P_H2 of
(1-L)^{-1}-(1+L) is small.

No k-adapt, no T2, no DC, no 48/96/192, no BSV, no production V.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from q8_m0_adapt_bz import PerkCache, net_gross_complex, perk_born_ab  # noqa: E402
from q8_m0_cjj import dump, propagate_cjj  # noqa: E402
from q8_m0_classical import k_grid, p1_a2_protocol  # noqa: E402
from q8_m0_gauge_shift import (  # noqa: E402
    GAUGE_ID,
    GENERIC_DK_FRAC,
    compare_currents,
    gauge_verdict,
    run_sudden_fixed_pi0,
    shift_kf,
    slim_gauge,
)
from q8_m0_gsheet import gamma_self  # noqa: E402
from q8_m0_h2_resolvent import (  # noqa: E402
    H2RES_ID,
    h2_a2_resolvent,
    h2_resolvent_ph,
    h2_return_ph,
    pack_h2_block,
)
from q8_m0_loop import ResummationNotChosenError, apply_resummation_to_output  # noqa: E402
from q8_m0_mode_io import eta2_ncell  # noqa: E402
from q8_m0_model import R0, build_geometry  # noqa: E402
from q8_m0_photon_op import (  # noqa: E402
    PH_OP_ID,
    h2_phi,
    leading_singular_ph,
    omega_weights,
    pack_phi_spec,
    photon_omega_grid,
)
from q8_m0_s1_kernel import parallel_perp_basis  # noqa: E402
from q8_m0_stage1 import stage1_spec  # noqa: E402
from q8_m0_volterra import h2_kernels  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "results/q_line/stage1l_h2res_20260923"
I1I = REPO / "results/q_line/stage1i_traj_20260922"
I1J = REPO / "results/q_line/stage1j_op_20260922"
I1K = REPO / "results/q_line/stage1k_ph_20260922"
MU = -0.009010620372198276
NTPC = 400
G1L_ID = "covariant_gauge_h2_resolvent_20260923"
HOLD_CYCLES = (0, 4, 8)


def load_kset(*cands):
    for path in cands:
        if path is None or not path.exists():
            continue
        z = np.load(path)
        if "kf" in z.files and "w" in z.files:
            return {"kf": np.asarray(z["kf"]), "w": np.asarray(z["w"])}
    return None


def photon_for_hold(prot, n_hold: int, e_perp):
    t_pulse = float(prot.t_pulse)
    t_obs = float(prot.n_cycles + n_hold) * float(prot.period)
    omega = photon_omega_grid(t_pulse, prot.omega0, harm_max=12.0, t_obs=t_obs)
    dw = omega_weights(omega)
    ph2 = h2_phi(omega, prot.omega0, t_pulse, e_perp)
    return {
        "t_pulse": t_pulse,
        "t_obs": t_obs,
        "omega_min": float(omega[0]),
        "omega_max": float(omega[-1]),
        "n_omega": int(omega.size),
        "dw": dw,
        "omega": omega,
        "ph2": ph2,
    }


def h2_on_packed(name, packed, geo, prot, gamma, pref, e_perp, ph, *, want_sigma: bool, want_resolvent: bool):
    t0 = time.perf_counter()
    times = packed["times"]
    dt = packed["dt"]
    hk = h2_kernels(times, prot.omega0, prot.t_pulse, geo)
    a2, b2 = perk_born_ab(
        packed["X_full"], packed["perk_D_t"], hk["F"], hk["e_perp"], hk["k_a"],
        dt, pref,
    )
    qa = net_gross_complex(a2, packed["w_k"])
    qb = net_gross_complex(b2, packed["w_k"])
    print(f"  H2 return {name}", flush=True)
    ret = h2_return_ph(
        packed["X_full"], packed["w_k"], packed["mean_D_t"],
        ph["ph2"], ph["omega"], times, dt, ph["dw"], gamma,
    )
    res_ph = None
    a2b = None
    if want_resolvent:
        print(f"  H2 resolvent {name}", flush=True)
        res_ph = h2_resolvent_ph(
            packed["X_full"], packed["w_k"], packed["mean_D_t"],
            ph["ph2"], ph["omega"], times, dt, ph["dw"], gamma,
        )
        a2b = h2_a2_resolvent(
            packed["X_full"], packed["w_k"], packed["mean_D_t"],
            hk["F"], hk["e_perp"], hk["k_a"], dt, gamma, pref,
        )
        block = pack_h2_block(ret, res_ph, a2b)
    else:
        block = {
            "return": ret,
            "resolvent_ph": None,
            "A2": None,
            "h2_class": None,
            "id": H2RES_ID,
        }
    op = None
    if want_sigma:
        print(f"  sigma_ph {name}", flush=True)
        op = leading_singular_ph(
            packed["X_full"], packed["w_k"], packed["mean_D_t"],
            ph["omega"], times, dt, ph["dw"], gamma, phi_probe=ph["ph2"],
            p=3, n_iter=5, rng=0,
        )
    wall = time.perf_counter() - t0
    rec = {
        "name": name,
        "n_k": int(packed["kf"].shape[0]),
        "A2_Ronly_net": qa,
        "B2_Ronly_net": qb,
        "h2": block,
        "sigma_max": None if op is None else float(op["sigma_max"]),
        "gain_h2_dir": None if op is None else op["gain_h2_dir"],
        "rayleigh_h2": None if op is None else op["rayleigh_h2"],
        "v_spec": None if op is None else pack_phi_spec(op["v_right"], ph["omega"], ph["dw"], times, dt),
        "t_obs": ph["t_obs"],
        "omega_min": ph["omega_min"],
        "wall_s": wall,
    }
    print(
        f"  {name} |A2|={qa['net']['abs']:.4f} L2={ret['h2_to_ir_to_h2']:.4g} "
        f"smax={rec['sigma_max']} class={block.get('h2_class')} wall={wall:.1f}s",
        flush=True,
    )
    extra = {}
    if op is not None:
        extra = {
            "v_right": op["v_right"],
            "u_left": op["u_left"],
            "a_right": op["a_right"],
            "a_out": op["a_out"],
        }
    np.savez(
        OUT / f"{name}.npz",
        kf=packed["kf"], w=packed["w_k"], a2=a2, b2=b2,
        times=times, omega=ph["omega"], **extra,
    )
    return rec


def _mu_from_cache(cache, kf, w):
    packed = cache.assemble(kf, w)
    return packed["mu_per_cell"], packed["times"], packed["dt"]


def _propagate_mu(geo, prot, kf, w):
    rec = propagate_cjj(
        geo, R0, prot, nt_per_cycle=NTPC, neel=+1, sign=+1, mu=MU,
        verbose=False, kf_frac=kf, w_k=w,
    )
    return rec["mu_per_cell"], rec["times"], float(rec["dt"]), float(rec["wall_seconds"])


def run_gauge_block(geo, prot, cache0):
    print("[stage1l] gauge permutation mp4 (exact set map)", flush=True)
    kf4, _, w4 = k_grid(geo, 4, shift=0.0)
    j0, times, dt = _mu_from_cache(cache0, kf4, w4)
    kf_perm = shift_kf(geo, kf4, delta_frac=np.array([0.25, 0.0]))
    jp, _, _ = _mu_from_cache(cache0, kf_perm, w4)
    perm = {
        "kind": "grid_permutation_1_over_nk",
        "exact_set_map": True,
        "delta_frac": [0.25, 0.0],
        "n_k": int(kf4.shape[0]),
        "compare": compare_currents(j0, jp, jp, dt),
        "wall_s": 0.0,
        "j0": j0,
        "jp": jp,
        "jm": jp,
        "times": times,
        "dt": dt,
    }
    generic = []
    for nk, tag in ((4, "mp4"), (6, "mp6"), (8, "mp8")):
        print(f"[stage1l] gauge generic {tag} dk={GENERIC_DK_FRAC}", flush=True)
        kf, _, w = k_grid(geo, nk, shift=0.0)
        j0g, tg, dtg = _mu_from_cache(cache0, kf, w)
        kf_p = shift_kf(geo, kf, delta_frac=GENERIC_DK_FRAC)
        kf_m = shift_kf(geo, kf, delta_frac=-GENERIC_DK_FRAC)
        jp, _, _, wp = _propagate_mu(geo, prot, kf_p, w)
        jm, _, _, wm = _propagate_mu(geo, prot, kf_m, w)
        rec = {
            "kind": "covariant_H_and_Pi0",
            "name": f"generic_{tag}",
            "delta_frac": GENERIC_DK_FRAC.tolist(),
            "n_k": int(kf.shape[0]),
            "compare": compare_currents(j0g, jp, jm, dtg),
            "wall_s": wp + wm,
            "j0": j0g,
            "jp": jp,
            "jm": jm,
            "times": tg,
            "dt": dtg,
        }
        generic.append(rec)
        np.savez(OUT / f"gauge_{tag}.npz", times=tg, j0=j0g, jp=jp, jm=jm)
    print("[stage1l] sudden-A contrast mp4 (not a gauge)", flush=True)
    sudden = run_sudden_fixed_pi0(
        geo, R0, prot, kf4, w4, mu=MU, neel=+1, ntpc=NTPC,
        delta_frac=GENERIC_DK_FRAC,
    )
    verdict = gauge_verdict(perm, generic)
    return {
        "id": GAUGE_ID,
        "permutation": slim_gauge(perm),
        "generic": [slim_gauge(r) | {"name": r["name"]} for r in generic],
        "sudden_fixed_Pi0": sudden,
        "verdict": verdict,
        "old_criterion": "withdrawn_insufficient_to_claim_gauge_failure",
    }


def fill_hold0_cache(geo, prot):
    cache = PerkCache()
    jobs = []
    for nk, tag in ((4, "mp4"), (6, "mp6"), (8, "mp8")):
        kf, _, w = k_grid(geo, nk, shift=0.0)
        jobs.append((tag, kf, w))
    skipped = []
    for tag, cands in (
        ("i1_trad", (I1K / "i1_trad.npz", I1J / "i1_trad.npz", I1I / "traditional_final.npz")),
        ("i1_ai", (I1K / "i1_ai.npz", I1J / "i1_ai.npz", I1I / "ai_rank_final.npz")),
    ):
        rec = load_kset(*cands)
        if rec is None:
            skipped.append(tag)
            continue
        jobs.append((tag, rec["kf"], rec["w"]))
    for tag, kf, w in jobs:
        print(f"[stage1l] fill hold0 {tag} nk={len(kf)}", flush=True)
        cache.fill(geo, R0, prot, kf, mu=MU, neel=+1, ntpc=NTPC, n_hold_cycles=0)
    return cache, jobs, skipped


def run_hold0_h2(geo, prot, gamma, pref, e_perp, cache0, jobs):
    ph = photon_for_hold(prot, 0, e_perp)
    rows = []
    for tag, kf, w in jobs:
        packed = cache0.assemble(kf, w)
        want_sigma = tag == "mp4"
        name = "mp4_hold0" if tag == "mp4" else tag
        print(f"[stage1l] H2 {name}", flush=True)
        rows.append(h2_on_packed(
            name, packed, geo, prot, gamma, pref, e_perp, ph,
            want_sigma=want_sigma, want_resolvent=True,
        ))
    return rows


def run_extended_holds(geo, prot, gamma, pref, e_perp):
    rows = []
    for n_hold in (4, 8):
        ph = photon_for_hold(prot, n_hold, e_perp)
        cache = PerkCache()
        print(f"[stage1l] T_obs hold={n_hold} T/Tp={ph['t_obs']/ph['t_pulse']:.2f} "
              f"wmin={ph['omega_min']:.6f}", flush=True)
        kf4, _, w4 = k_grid(geo, 4, shift=0.0)
        cache.fill(geo, R0, prot, kf4, mu=MU, neel=+1, ntpc=NTPC, n_hold_cycles=n_hold)
        rows.append(h2_on_packed(
            f"mp4_hold{n_hold}", cache.assemble(kf4, w4), geo, prot, gamma, pref, e_perp, ph,
            want_sigma=True, want_resolvent=False,
        ))
        kf8, _, w8 = k_grid(geo, 8, shift=0.0)
        cache.fill(geo, R0, prot, kf8, mu=MU, neel=+1, ntpc=NTPC, n_hold_cycles=n_hold)
        rows.append(h2_on_packed(
            f"mp8_hold{n_hold}", cache.assemble(kf8, w8), geo, prot, gamma, pref, e_perp, ph,
            want_sigma=False, want_resolvent=True,
        ))
    return rows


def window_scale(rows: list) -> dict:
    mp4 = [r for r in rows if r["name"].startswith("mp4_hold") and r["sigma_max"] is not None]
    if len(mp4) < 2:
        return {"available": False}
    wmin = np.array([r["omega_min"] for r in mp4], dtype=float)
    sig = np.array([r["sigma_max"] for r in mp4], dtype=float)
    l2 = np.array([r["h2"]["return"]["h2_to_ir_to_h2"] for r in mp4], dtype=float)
    prod = sig * wmin
    spread = float(np.max(prod) / max(float(np.min(prod)), 1e-30))
    return {
        "available": True,
        "hold": [r["name"] for r in mp4],
        "omega_min": wmin.tolist(),
        "sigma_max": sig.tolist(),
        "sigma_times_omega_min": prod.tolist(),
        "sigma_times_omega_min_spread": spread,
        "looks_like_1_over_omega_min": bool(spread < 1.6),
        "h2_to_ir_to_h2": l2.tolist(),
        "h2_return_stays_finite": bool(np.all(np.isfinite(l2)) and np.max(np.abs(l2)) < 50.0),
    }


def l_verdict(gauge, rows: list, wscale: dict) -> dict:
    gpass = bool(gauge["verdict"]["covariant_pass"])
    res_rows = [r for r in rows if r["h2"].get("resolvent_ph") is not None]
    l2s = [float(r["h2"]["return"]["h2_to_ir_to_h2"]) for r in rows]
    diffs = [float(r["h2"]["resolvent_ph"]["h2_component_diff"]) for r in res_rows]
    rel_a2 = [float(r["h2"]["A2"]["rel_A2_exact_vs_born_field"]) for r in res_rows]
    mag = 0.0
    if l2s:
        mag = max(mag, float(np.max(np.abs(np.array(l2s)))))
    if diffs:
        mag = max(mag, float(np.max(np.abs(np.array(diffs)))))
    if rel_a2:
        mag = max(mag, float(np.max(np.abs(np.array(rel_a2)))))
    o1 = bool(mag >= 0.7)
    nonneg = bool(mag >= 0.2)
    gate = bool(gpass and nonneg)
    return {
        "covariant_pass": gpass,
        "old_gauge_fail_withdrawn": True,
        "h2_to_ir_to_h2": l2s,
        "h2_resolvent_component": diffs,
        "rel_A2_exact_vs_born_field": rel_a2,
        "h2_return_maxabs": mag,
        "h2_return_nonnegligible": nonneg,
        "h2_return_O1": o1,
        "sigma_is_stress_only": True,
        "window": wscale,
        "human_gate": gate,
        "human_node": "upgrade_linear_feedback_resummation" if gate else None,
        "born_restore_qualified": False,
        "production_V": "Born_reference_diagnostic_not_qualified",
        "note": (
            "Human gate only if the covariant k-shift passes and the H2 "
            "multi-pass / projected resolvent error is non-negligible. "
            "σ_max(L_ph) alone does not open the gate."
        ),
    }


def make_figures(gauge, rows: list, verdict: dict) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    gen = gauge["generic"]
    fig, ax = plt.subplots(figsize=(6.2, 3.6))
    nks = [r["n_k"] for r in gen]
    ax.semilogy(nks, [r["compare"]["rel_l2_plus"] for r in gen], "o-", label="+dk")
    ax.semilogy(nks, [r["compare"]["rel_l2_minus"] for r in gen], "s-", label="-dk")
    ax.set_xlabel("n_k")
    ax.set_ylabel("||J_dk - J_0|| / ||J_0||")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    ax.set_title("covariant k-shift residual")
    fig.tight_layout()
    fig.savefig(OUT / "stage1l_gauge_residual.png", dpi=140)
    plt.close(fig)

    names = [r["name"] for r in rows]
    fig, ax = plt.subplots(figsize=(8.4, 3.8))
    ax.bar(names, [r["h2"]["return"]["h2_to_ir_to_h2"] for r in rows], color="#4C78A8")
    ax.axhline(0.0, color="k", lw=0.6)
    ax.axhline(0.2, color="k", ls=":", lw=0.8)
    ax.set_ylabel(r"$\langle H2|L_{ph}^2|H2\rangle$")
    ax.tick_params(axis="x", rotation=30)
    ax.grid(True, axis="y", alpha=0.3)
    ax.set_title("H2 to IR to H2")
    fig.tight_layout()
    fig.savefig(OUT / "stage1l_h2_L2.png", dpi=140)
    plt.close(fig)

    res = [r for r in rows if r["h2"].get("resolvent_ph") is not None]
    if res:
        fig, ax = plt.subplots(figsize=(7.6, 3.8))
        ax.bar([r["name"] for r in res],
               [r["h2"]["resolvent_ph"]["h2_component_diff"] for r in res],
               color="#54A24B")
        ax.axhline(0.2, color="k", ls=":", lw=0.8)
        ax.set_ylabel("H2 · [(1-L)^{-1}-(1+L)]")
        ax.tick_params(axis="x", rotation=30)
        ax.grid(True, axis="y", alpha=0.3)
        fig.tight_layout()
        fig.savefig(OUT / "stage1l_h2_resolvent.png", dpi=140)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(7.6, 3.8))
        ax.bar([r["name"] for r in res],
               [r["h2"]["A2"]["rel_A2_exact_vs_born_field"] for r in res],
               color="#E45756")
        ax.set_ylabel("|A2_exact - A2_Born-field| / |A2_exact|")
        ax.tick_params(axis="x", rotation=30)
        ax.grid(True, axis="y", alpha=0.3)
        fig.tight_layout()
        fig.savefig(OUT / "stage1l_A2_resolvent.png", dpi=140)
        plt.close(fig)

    mp4 = [r for r in rows if r["name"].startswith("mp4_hold") and r["sigma_max"] is not None]
    if len(mp4) >= 2:
        fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.6))
        ax = axes[0]
        ax.loglog([r["omega_min"] for r in mp4], [r["sigma_max"] for r in mp4], "o-")
        w = np.array([r["omega_min"] for r in mp4], dtype=float)
        s0 = mp4[0]["sigma_max"] * mp4[0]["omega_min"]
        ax.loglog(w, s0 / w, "k:", label="~1/omega_min")
        ax.set_xlabel("omega_min")
        ax.set_ylabel("sigma_max(L_ph)")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(fontsize=8)
        ax = axes[1]
        ax.plot([r["omega_min"] for r in mp4],
                [r["h2"]["return"]["h2_to_ir_to_h2"] for r in mp4], "s-")
        ax.set_xlabel("omega_min")
        ax.set_ylabel("H2 to IR to H2")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(OUT / "stage1l_window_scale.png", dpi=140)
        plt.close(fig)


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
    t0 = time.perf_counter()
    cache0, jobs, skipped = fill_hold0_cache(geo, prot)
    gauge = run_gauge_block(geo, prot, cache0)
    rows = run_hold0_h2(geo, prot, gamma, pref, e_perp, cache0, jobs)
    rows.extend(run_extended_holds(geo, prot, gamma, pref, e_perp))
    wscale = window_scale(rows)
    verdict = l_verdict(gauge, rows, wscale)
    make_figures(gauge, rows, verdict)
    payload = {
        "g1l_id": G1L_ID,
        "op_id": PH_OP_ID,
        "h2res_id": H2RES_ID,
        "gauge_id": GAUGE_ID,
        "spec": stage1_spec(),
        "frozen": {
            "no_T2": True,
            "no_DC_subtraction": True,
            "P1_local_classical": True,
            "no_48_96_192": True,
            "no_fock_expansion": True,
            "no_k_adapt_this_round": True,
            "production_V": "Born_reference_diagnostic_not_qualified",
            "sigma_max_L_ph": "worst_case_stress_not_born_passport",
            "old_gauge_I_state_plus_contact": "withdrawn",
        },
        "resummation_production_blocked": blocked,
        "nt_per_cycle": NTPC,
        "hold_cycles": list(HOLD_CYCLES),
        "generic_dk_frac": GENERIC_DK_FRAC.tolist(),
        "skipped_ksets": skipped,
        "cjj_cache_hold0": {"n_k_evaluated": cache0.n_cjj_k, "wall_s": cache0.wall_cjj},
        "gauge": gauge,
        "grids": rows,
        "verdict": verdict,
        "gain_claim": "not_evaluated",
        "squeezing_claim": "not_evaluated",
        "nk96": "not_run",
        "human_gate": verdict["human_gate"],
        "human_node": verdict["human_node"],
        "wall_s": time.perf_counter() - t0,
    }
    dump(OUT / "stage1l_h2res.json", payload)
    print(json.dumps({
        "wrote": str(OUT / "stage1l_h2res.json"),
        "gauge": gauge["verdict"],
        "h2_L2": {r["name"]: r["h2"]["return"]["h2_to_ir_to_h2"] for r in rows},
        "sigma": {r["name"]: r["sigma_max"] for r in rows if r["sigma_max"] is not None},
        "verdict": verdict,
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
