#!/usr/bin/env python3
"""Stage-1N: canonical linear quantum I/O audit. No new physics.

No BZ adaptive, no T2, no nonlinear feedback, no production n,m.
Fixed MP8 only. Expand the positive-frequency input basis and ask
whether the H2 field commutator approaches 1.
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
    AUDIT_ID,
    CANONICAL_GATE,
    audit_grid,
    basis_gram,
    empty_sheet_identity,
    gamma0_field_identity,
    identity_scan,
    k_two_mode_comparison,
    kubo_born_current,
    packet_basis,
    slim_cel,
    slim_row,
)
from q8_m0_loop import ResummationNotChosenError, apply_resummation_to_output  # noqa: E402
from q8_m0_mode_io import eta2_ncell  # noqa: E402
from q8_m0_model import R0, build_geometry  # noqa: E402
from q8_m0_photon_op import h2_phi, qs_reconstruction  # noqa: E402
from q8_m0_s1_kernel import parallel_perp_basis  # noqa: E402
from q8_m0_stage1 import stage1_spec  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "results/q_line/stage1n_qio_20260923"
MU = -0.009010620372198276
NTPC = 400
G1N_ID = "canonical_io_audit_20260923"
N_LIST = (1, 2, 4, 8, 16, 32)
N_LIST_PAR = (1, 4, 16)
PASS_ABS = 0.08


def log(msg: str) -> None:
    print(f"[stage1n] {msg}", flush=True)


def make_figures(empty_rows, g0_rows, matter_rows, matter_par_rows) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6.8, 3.8))
    for rows, lab, fmt in (
        (empty_rows, "empty sheet", "o-"),
        (g0_rows, "matter γ=0", "s--"),
        (matter_rows, "MP8 e_perp", "^-"),
        (matter_par_rows, "MP8 +e_par", "d-"),
    ):
        if not rows:
            continue
        ax.plot(
            [r["n_inputs"] for r in rows],
            [r["commutator_real"] for r in rows],
            fmt,
            label=lab,
        )
    ax.axhline(1.0, color="k", lw=0.8, alpha=0.5)
    ax.set_xlabel("n inputs (packet × 2 ports)")
    ax.set_ylabel(r"$[b_{H2},b_{H2}^\dagger]$ field I/O")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "stage1n_h2_commutator.png", dpi=140)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.8, 3.8))
    ax.plot([r["n_inputs"] for r in matter_rows], [r["sum_absS2"] for r in matter_rows], "o-", label=r"$\sum|S|^2$")
    ax.plot([r["n_inputs"] for r in matter_rows], [r["sum_absT2"] for r in matter_rows], "s-", label=r"$\sum|T|^2$")
    ax.plot([r["n_inputs"] for r in matter_rows], [r["vac_piece_real"] for r in matter_rows], "^-", label="vac piece")
    cel = matter_rows[-1]["C_el"]["re"]
    ax.axhline(cel, color="C3", ls="--", label=r"$C_{\mathrm{el}}$")
    ax.set_xlabel("n inputs")
    ax.set_ylabel("H2 R_out pieces")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "stage1n_ST_Cel_breakdown.png", dpi=140)
    plt.close(fig)


def verdict_from(empty_rows, matter_rows, kubo, kcmp) -> dict:
    e1 = empty_rows[0]
    e_last = empty_rows[-1]
    m1 = matter_rows[0]
    m_last = matter_rows[-1]
    qs_floor = float(e_last["abs_minus_1"])
    approached = bool(
        m_last["abs_minus_1"] < m1["abs_minus_1"]
        and m_last["abs_minus_1"] < 0.5 * max(m1["abs_minus_1"], 1.0)
    )
    passed = bool(m_last["abs_minus_1"] < max(PASS_ABS, 2.0 * qs_floor + 0.02))
    empty_ok = bool(e_last["abs_minus_1"] < 0.35)
    return {
        "canonical_gate": CANONICAL_GATE,
        "empty_n1_comm": e1["commutator_real"],
        "empty_last_comm": e_last["commutator_real"],
        "empty_qs_floor": qs_floor,
        "empty_algebra_ok": empty_ok,
        "matter_n1_comm": m1["commutator_real"],
        "matter_last_comm": m_last["commutator_real"],
        "matter_last_abs_minus_1": m_last["abs_minus_1"],
        "C_el": m_last["C_el"],
        "approaches_1": approached,
        "canonical_identity_pass": bool(passed and empty_ok),
        "k_two_mode_abs_minus_1": kcmp["commutator_abs_minus_1"],
        "kubo_state_abs": kubo["residual_state_abs"],
        "kubo_full_abs": kubo["residual_full_abs"],
        "production_n_m": "not_written_identity_failed" if not passed else "still_blocked_until_BZ",
        "rms_is_not_this_gate": True,
        "nonlinear_vs_dissipation": "still_deferred",
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    geo = build_geometry()
    prot = p1_a2_protocol(0.8)
    gamma = gamma_self(geo)
    pref = eta2_ncell(geo)
    e_par, e_perp = parallel_perp_basis(geo)
    omega, dw = audit_grid(prot, harm_max=12.0)
    t0 = time.perf_counter()

    prod_blocked = False
    try:
        apply_resummation_to_output()
    except ResummationNotChosenError:
        prod_blocked = True

    times_empty = None
    # Empty-sheet control on the production time grid after CJJ fill (same nt).
    log("fill MP8 CJJ")
    cache = PerkCache()
    kf, _, w = k_grid(geo, 8, shift=0.0)
    cache.fill(geo, R0, prot, kf, mu=MU, neel=+1, ntpc=NTPC)
    packed = cache.assemble(kf, w)
    times = packed["times"]
    dt = packed["dt"]
    times_empty = times
    log(f"MP8 nk={len(kf)} nt={len(times)} nω={omega.size} γ={gamma:.6g}")

    qs = qs_reconstruction(h2_phi(omega, prot.omega0, prot.t_pulse, e_perp), omega, times, dt, dw)
    log(f"QS H2 rel_l2={qs['rel_l2']:.4g} overlap={qs['overlap']:.4g}")

    log("empty sheet")
    empty = empty_sheet_identity(
        times_empty, dt, omega, dw, prot.omega0, prot.t_pulse, e_perp, e_par, N_LIST, gamma=gamma,
    )
    log("  empty comm=" + ",".join(f"{r['commutator_real']:.4g}" for r in empty["scan"]["rows"]))
    log("gamma=0 matter")
    g0 = gamma0_field_identity(
        packed["X_full"], packed["w_k"], packed["mean_D_t"],
        times, dt, omega, dw, prot.omega0, prot.t_pulse, e_perp, e_par, N_LIST,
    )
    log("  g0 comm=" + ",".join(f"{r['commutator_real']:.4g}" for r in g0["scan"]["rows"]))

    phis, labs = packet_basis(omega, dw, prot.omega0, prot.t_pulse, e_perp, e_par, include_par=False)
    gram = basis_gram(phis, dw)
    log(f"basis e_perp n={gram['n']} ortho={gram['orthonormal']}")
    log("matter γ full, e_perp")
    matter = identity_scan(
        packed["X_full"], packed["w_k"], packed["mean_D_t"],
        times, dt, gamma, omega, dw, phis, labs, phis[0], N_LIST, log=log,
    )

    log("matter γ full, e_perp+e_par (H2+harmonics+bins)")
    phis_p, labs_p = packet_basis(omega, dw, prot.omega0, prot.t_pulse, e_perp, e_par, include_par=True)
    matter_par = identity_scan(
        packed["X_full"], packed["w_k"], packed["mean_D_t"],
        times, dt, gamma, omega, dw, phis_p, labs_p, phis_p[0], N_LIST_PAR, log=log,
    )

    log("Born Kubo current-K")
    kubo = kubo_born_current(packed, geo, prot)
    log(
        f"  state+comm={kubo['residual_state_abs']:.4g} "
        f"full+comm={kubo['residual_full_abs']:.4g} "
        f"contact={kubo['two_Re_A2_contact']:.4g}"
    )
    log("K two-mode comparison (not canonical)")
    kcmp = k_two_mode_comparison(packed, geo, prot, gamma, pref)
    log(f"  K-comm |·-1|={kcmp['commutator_abs_minus_1']:.4g}")

    empty_rows = [slim_row(r) for r in empty["scan"]["rows"]]
    g0_rows = [slim_row(r) for r in g0["scan"]["rows"]]
    matter_rows = [slim_row(r) for r in matter["rows"]]
    matter_par_rows = [slim_row(r) for r in matter_par["rows"]]
    make_figures(empty_rows, g0_rows, matter_rows, matter_par_rows)

    verdict = verdict_from(empty_rows, matter_rows, kubo, kcmp)
    log(
        f"verdict pass={verdict['canonical_identity_pass']} "
        f"empty_last={verdict['empty_last_comm']:.4g} "
        f"matter_last={verdict['matter_last_comm']:.4g} "
        f"C_el={verdict['C_el']['re']:.4g}"
    )

    payload = {
        "g1n_id": G1N_ID,
        "audit_id": AUDIT_ID,
        "spec": stage1_spec(),
        "frozen": {
            "theory_main": "linear_feedback_resummation_candidate",
            "born_role": "perturbative_benchmark",
            "canonical_gate": CANONICAL_GATE,
            "no_T2": True,
            "no_nonlinear_feedback": True,
            "no_bz_adaptive": True,
            "no_production_n_m": True,
            "vacuum_rms": "provisional_self_consistency_stress_diagnostic",
            "production_V": "blocked",
        },
        "production_V_blocked": prod_blocked,
        "grid": {
            "n_k": int(len(kf)),
            "n_t": int(len(times)),
            "n_omega": int(omega.size),
            "harm_max": 12.0,
            "n_basis_perp": gram["n"],
            "n_basis_par": len(phis_p),
            "gamma": float(gamma),
            "qs_h2": qs,
            "gram_perp": gram,
        },
        "empty_sheet": {"qs": empty["qs"], "rows": empty_rows, "C_el": slim_cel(empty["scan"]["C_el"])},
        "gamma0_matter": {"rows": g0_rows, "C_el": slim_cel(g0["scan"]["C_el"])},
        "matter_perp": {"rows": matter_rows, "C_el": slim_cel(matter["C_el"])},
        "matter_par": {"rows": matter_par_rows, "C_el": slim_cel(matter_par["C_el"])},
        "kubo_born_current_K": kubo,
        "k_two_mode_comparison": kcmp,
        "verdict": verdict,
        "cjj_cache": {"n_k_evaluated": cache.n_cjj_k, "wall_s": cache.wall_cjj},
        "human_gate": False,
        "human_node": None,
        "squeezing_claim": "not_evaluated",
        "nk96": "not_run",
        "wall_s": time.perf_counter() - t0,
    }
    dump(OUT / "stage1n_qio.json", payload)
    print(json.dumps({
        "wrote": str(OUT / "stage1n_qio.json"),
        "verdict": verdict,
        "empty_comm": [r["commutator_real"] for r in empty_rows],
        "g0_comm": [r["commutator_real"] for r in g0_rows],
        "matter_comm": [r["commutator_real"] for r in matter_rows],
        "matter_par_comm": [r["commutator_real"] for r in matter_par_rows],
        "C_el": matter["C_el"]["C_el_real"],
        "K_two_mode_abs_minus_1": kcmp["commutator_abs_minus_1"],
        "kubo_state": kubo["residual_state_abs"],
        "wall_s": payload["wall_s"],
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
