#!/usr/bin/env python3
"""Stage-1O: canonical-defect order in γ and required-noise structure.

Diagnostics only. No ΔC patch, no T2, no nonlinear, no BZ adapt, no production n,m.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from q8_m0_adapt_bz import PerkCache  # noqa: E402
from q8_m0_cjj import dump  # noqa: E402
from q8_m0_classical import k_grid, p1_a2_protocol  # noqa: E402
from q8_m0_gsheet import gamma_self  # noqa: E402
from q8_m0_io_audit import (  # noqa: E402
    audit_grid,
    k_two_mode_comparison,
    kubo_born_current,
    packet_basis,
    prepare_drives,
)
from q8_m0_loop import ResummationNotChosenError, apply_resummation_to_output  # noqa: E402
from q8_m0_mode_io import eta2_ncell  # noqa: E402
from q8_m0_model import R0, build_geometry  # noqa: E402
from q8_m0_order_audit import (  # noqa: E402
    DELTA_C_NOTE,
    LAMBDAS,
    ORDER_ID,
    cel_matrix,
    h2_identity_at_gamma,
    leading_power,
    nambu_deficit,
    pack_cmatrix,
    packet_omega_centroid,
    scale_table,
    slim_h2,
    st_matrices,
)
from q8_m0_s1_kernel import parallel_perp_basis  # noqa: E402
from q8_m0_stage1 import stage1_spec  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "results/q_line/stage1o_order_20260923"
MU = -0.009010620372198276
NTPC = 400
G1O_ID = "canonical_defect_order_20260923"
N_DEFICIT = 8
NEUMANN_NS = (1, 2, 3, 4)
NEUMANN_LAMBDAS = (0.125, 1.0)


def log(msg: str) -> None:
    print(f"[stage1o] {msg}", flush=True)


def make_figures(scan, powers, deficit_rows, neu_rows) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6.8, 3.9))
    for key, lab, fmt in (
        ("res_n1", "resummed n=1", "o-"),
        ("res_n4", "resummed n=4", "s-"),
        ("born_n1", "Born n=1", "^-"),
        ("k_two", "K two-mode", "d--"),
    ):
        xs = [r["lambda"] for r in scan[key] if r["lambda"] > 0]
        ys = [abs(r["delta"]) for r in scan[key] if r["lambda"] > 0]
        ax.loglog(xs, ys, fmt, label=lab)
    ax.set_xlabel(r"$\lambda$ in $\gamma\to\lambda\gamma$")
    ax.set_ylabel(r"$|\Delta_{\mathrm{can}}|$")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "stage1o_delta_vs_lambda.png", dpi=140)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.8, 3.9))
    rows = scan["res_n1"]
    xs = [r["lambda"] for r in rows if r["lambda"] > 0]
    ax.semilogx(xs, [r["over_l2"] for r in rows if r["lambda"] > 0], "o-", label=r"$\Delta/\lambda^2$")
    ax.semilogx(xs, [r["over_l3"] for r in rows if r["lambda"] > 0], "s-", label=r"$\Delta/\lambda^3$")
    ax.semilogx(xs, [r["over_l4"] for r in rows if r["lambda"] > 0], "^-", label=r"$\Delta/\lambda^4$")
    ax.set_xlabel(r"$\lambda$")
    ax.set_ylabel(r"resummed n=1 scaled $\Delta$")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "stage1o_scaled_delta.png", dpi=140)
    plt.close(fig)

    if deficit_rows:
        last = deficit_rows[-1]
        d = np.asarray(last["delta_C"]["re"])
        fig, ax = plt.subplots(figsize=(5.2, 4.4))
        im = ax.imshow(d, origin="lower", cmap="coolwarm")
        fig.colorbar(im, ax=ax, fraction=0.046)
        labs = last["labels"]
        ax.set_xticks(range(len(labs)))
        ax.set_yticks(range(len(labs)))
        ax.set_xticklabels(labs, rotation=75, fontsize=7)
        ax.set_yticklabels(labs, fontsize=7)
        ax.set_title(r"$\mathrm{Re}\,\Delta C$, $\lambda=" + f"{last['lambda']:.4g}$ (not a patch)")
        fig.tight_layout()
        fig.savefig(OUT / "stage1o_deltaC_heatmap.png", dpi=140)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(6.4, 3.6))
        for rec, fmt in zip(deficit_rows, ("o-", "s-", "^-")):
            ax.plot(rec["omega_c"], rec["diag_delta"], fmt, label=rf"$\lambda={rec['lambda']:.4g}$")
        ax.axvline(0.1, color="k", lw=0.7, alpha=0.4)
        ax.set_xlabel(r"packet $\omega$ centroid")
        ax.set_ylabel(r"$\mathrm{Re}\,\Delta C_{\mu\mu}$")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(OUT / "stage1o_deltaC_diag.png", dpi=140)
        plt.close(fig)

    if neu_rows:
        fig, ax = plt.subplots(figsize=(6.2, 3.6))
        by = {}
        for r in neu_rows:
            by.setdefault(r["lambda"], []).append(r)
        for lam, rs in by.items():
            ax.plot([r["n"] for r in rs], [r["delta_can"] for r in rs], "o-", label=rf"$\lambda={lam:.4g}$")
        ax.set_xlabel("Neumann order n")
        ax.set_ylabel(r"$\Delta_{\mathrm{can}}$")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(OUT / "stage1o_neumann.png", dpi=140)
        plt.close(fig)


def verdict_of(powers, scan, kubo) -> dict:
    p_res = powers["res_n1"]["p"]
    p_born = powers["born_n1"]["p"]
    d1 = [r["delta"] for r in scan["res_n1"] if abs(r["lambda"] - 1.0) < 1e-15][0]
    db = [r["delta"] for r in scan["born_n1"] if abs(r["lambda"] - 1.0) < 1e-15][0]
    born_small = bool(p_born is None or (p_res is not None and p_res + 0.4 < p_born) or abs(db) < 0.25 * abs(d1))
    after_born = bool(
        p_res is not None
        and p_res > 1.4
        and kubo["residual_state_abs"] < 1e-6
        and abs(scan["res_n1"][0]["delta"]) < 1e-8
    )
    born_order_leak = bool(p_res is not None and p_res < 1.35)
    return {
        "p_gamma_resummed_n1": p_res,
        "p_eta_resummed_n1": powers["res_n1"]["p_eta_bookkeeping"],
        "p_gamma_born_n1": p_born,
        "r2_resummed": powers["res_n1"]["r2"],
        "delta_res_lambda1": d1,
        "delta_born_lambda1": db,
        "born_kubo_abs": kubo["residual_state_abs"],
        "defect_starts_after_born": bool(after_born and not born_order_leak),
        "defect_at_born_order": born_order_leak,
        "born_delta_much_smaller": born_small,
        "required_noise_is_not_a_patch": True,
        "rpa_joint_linear": "not_adopted",
        "nonlinear_vs_T2": "still_deferred",
        "production_n_m": "not_written",
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    geo = build_geometry()
    prot = p1_a2_protocol(0.8)
    gamma0 = gamma_self(geo)
    pref = eta2_ncell(geo)
    e_par, e_perp = parallel_perp_basis(geo)
    omega, dw = audit_grid(prot, harm_max=12.0)
    t0 = time.perf_counter()

    prod_blocked = False
    try:
        apply_resummation_to_output()
    except ResummationNotChosenError:
        prod_blocked = True

    log("fill MP8")
    cache = PerkCache()
    kf, _, w = k_grid(geo, 8, shift=0.0)
    cache.fill(geo, R0, prot, kf, mu=MU, neel=+1, ntpc=NTPC)
    packed = cache.assemble(kf, w)
    x, wk, md = packed["X_full"], packed["w_k"], packed["mean_D_t"]
    times, dt = packed["times"], packed["dt"]
    phis, labs = packet_basis(omega, dw, prot.omega0, prot.t_pulse, e_perp, e_par, include_par=False)
    phi_h2 = phis[0]
    log(f"MP8 nk={len(kf)} nω={omega.size} γ={gamma0:.6g} n_basis={len(phis)}")

    kubo = kubo_born_current(packed, geo, prot)
    log(f"Born Kubo |res|={kubo['residual_state_abs']:.4g}")

    raw = {"res_n1": [], "res_n4": [], "born_n1": [], "k_two": []}
    for lam in LAMBDAS:
        g = float(lam) * gamma0
        log(f"λ={lam:g} γ={g:.6g}")
        r1 = h2_identity_at_gamma(
            x, wk, md, times, dt, g, omega, dw, phis[:1], phi_h2, kind="resummed",
        )
        r4 = h2_identity_at_gamma(
            x, wk, md, times, dt, g, omega, dw, phis[:4], phi_h2, kind="resummed",
        )
        b1 = h2_identity_at_gamma(
            x, wk, md, times, dt, g, omega, dw, phis[:1], phi_h2, kind="born",
        )
        kcmp = k_two_mode_comparison(packed, geo, prot, g, pref)
        raw["res_n1"].append(slim_h2(r1) | {"lambda": float(lam)})
        raw["res_n4"].append(slim_h2(r4) | {"lambda": float(lam)})
        raw["born_n1"].append(slim_h2(b1) | {"lambda": float(lam)})
        raw["k_two"].append({
            "lambda": float(lam),
            "delta": float(kcmp["commutator"]["re"] - 1.0),
            "commutator": kcmp["commutator"],
            "comm_s": kcmp["comm_s"],
            "A2": kcmp["A2"],
        })
        log(
            f"  Δ_res1={r1['delta_can']:.6g} Δ_res4={r4['delta_can']:.6g} "
            f"Δ_born={b1['delta_can']:.6g} Δ_K={kcmp['commutator']['re']-1:.6g} "
            f"C_el={r1['C_el_real']:.6g} Δ_lin={r1['delta_lin_probe']:.6g}"
        )

    scan = {
        "res_n1": scale_table(LAMBDAS, [r["delta_can"] for r in raw["res_n1"]]),
        "res_n4": scale_table(LAMBDAS, [r["delta_can"] for r in raw["res_n4"]]),
        "born_n1": scale_table(LAMBDAS, [r["delta_can"] for r in raw["born_n1"]]),
        "k_two": scale_table(LAMBDAS, [r["delta"] for r in raw["k_two"]]),
        "res_n1_lin": scale_table(LAMBDAS, [r["delta_lin_probe"] for r in raw["res_n1"]]),
        "res_n1_cel": scale_table(LAMBDAS, [r["C_el_real"] for r in raw["res_n1"]]),
    }
    powers = {
        "res_n1": leading_power(LAMBDAS, [r["delta_can"] for r in raw["res_n1"]]),
        "res_n4": leading_power(LAMBDAS, [r["delta_can"] for r in raw["res_n4"]]),
        "born_n1": leading_power(LAMBDAS, [r["delta_can"] for r in raw["born_n1"]]),
        "k_two": leading_power(LAMBDAS, [r["delta"] for r in raw["k_two"]]),
        "res_n1_lin": leading_power(LAMBDAS, [r["delta_lin_probe"] for r in raw["res_n1"]]),
        "res_n1_cel": leading_power(LAMBDAS, [r["C_el_real"] for r in raw["res_n1"]]),
    }
    log(f"powers p_γ res1={powers['res_n1']['p']} born={powers['born_n1']['p']} K={powers['k_two']['p']}")

    neu_rows = []
    for lam in NEUMANN_LAMBDAS:
        g = float(lam) * gamma0
        for n in NEUMANN_NS:
            rec = h2_identity_at_gamma(
                x, wk, md, times, dt, g, omega, dw, phis[:1], phi_h2,
                kind="neumann", neumann_n=n,
            )
            neu_rows.append({
                "lambda": float(lam),
                "n": int(n),
                "delta_can": rec["delta_can"],
                "C_el_real": rec["C_el_real"],
                "vac_piece_real": rec["vac_piece_real"],
                "delta_lin_probe": rec["delta_lin_probe"],
            })
            log(f"  Neumann λ={lam:g} n={n} Δ={rec['delta_can']:.6g}")

    deficit_rows = []
    ph_d = phis[:N_DEFICIT]
    lab_d = labs[:N_DEFICIT]
    om_c = [packet_omega_centroid(p, omega, dw) for p in ph_d]
    for lam in (1.0 / 16.0, 0.25, 1.0):
        g = float(lam) * gamma0
        log(f"ΔC matrix λ={lam:g} n={N_DEFICIT}")
        drives = prepare_drives(x, wk, md, dt, g, ph_d, omega, times, dw)
        s, t = st_matrices(ph_d, drives, omega, times, dt, dw)
        c = cel_matrix(x, wk, md, dt, g, ph_d, omega, times, dw, kind="resummed")
        dfn = nambu_deficit(s, t, c)
        deficit_rows.append({
            "lambda": float(lam),
            "labels": lab_d,
            "omega_c": om_c,
            "fro_delta": dfn["fro_delta"],
            "diag_delta": dfn["diag_delta"].tolist(),
            "herm_delta": dfn["herm_delta"],
            "ident_00": {
                "re": float(np.real(dfn["ident"][0, 0])),
                "im": float(np.imag(dfn["ident"][0, 0])),
            },
            "delta_C": pack_cmatrix(dfn["delta_C"]),
            "C_req": pack_cmatrix(dfn["C_req"]),
            "C_el": pack_cmatrix(c),
            "note": DELTA_C_NOTE,
        })
        log(
            f"  fro ΔC={dfn['fro_delta']:.4g} diag0={dfn['diag_delta'][0]:.4g} "
            f"ident00={dfn['ident'][0,0].real:.4g}"
        )

    make_figures(scan, powers, deficit_rows, neu_rows)
    verdict = verdict_of(powers, scan, kubo)
    log(f"verdict {verdict}")

    payload = {
        "g1o_id": G1O_ID,
        "order_id": ORDER_ID,
        "spec": stage1_spec(),
        "frozen": {
            "theory_main_expectation_resolvent": "mathematically_ok",
            "quantum_IO_from_resolvent": "not_automatic",
            "missing_orthogonal_modes": "excluded_stage1n",
            "born_xi_R_mismatch": "excluded_stage1n_kubo",
            "no_T2": True,
            "no_nonlinear_feedback": True,
            "no_bz_adaptive": True,
            "no_production_n_m": True,
            "required_noise": DELTA_C_NOTE,
            "rpa_joint_linear": "not_adopted",
        },
        "production_V_blocked": prod_blocked,
        "gamma0": float(gamma0),
        "lambdas": list(LAMBDAS),
        "raw": raw,
        "scan": scan,
        "powers": powers,
        "neumann": neu_rows,
        "deficit": deficit_rows,
        "kubo_born_current_K": kubo,
        "verdict": verdict,
        "cjj_cache": {"n_k_evaluated": cache.n_cjj_k, "wall_s": cache.wall_cjj},
        "human_gate": False,
        "human_node": None,
        "squeezing_claim": "not_evaluated",
        "nk96": "not_run",
        "wall_s": time.perf_counter() - t0,
    }
    dump(OUT / "stage1o_order.json", payload)
    print(json.dumps({
        "wrote": str(OUT / "stage1o_order.json"),
        "powers": {k: v.get("p") for k, v in powers.items()},
        "verdict": verdict,
        "delta_res_n1": [r["delta"] for r in scan["res_n1"]],
        "delta_born_n1": [r["delta"] for r in scan["born_n1"]],
        "delta_k": [r["delta"] for r in scan["k_two"]],
        "neumann": neu_rows,
        "fro_deltaC": [r["fro_delta"] for r in deficit_rows],
        "wall_s": payload["wall_s"],
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
