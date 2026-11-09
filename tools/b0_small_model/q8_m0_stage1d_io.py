#!/usr/bin/env python3
"""Stage-1D: frozen two-port G_self + causal Volterra + 1-k Fock cut.

Local CPU, six k, +N. Does not replace production Born V(θ).
Does not run 48/96/192 or g2.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from q8_m0_cjj import choose_probe_k, dump, propagate_cjj  # noqa: E402
from q8_m0_classical import p1_a2_protocol  # noqa: E402
from q8_m0_fock_1k import run_fock_1k  # noqa: E402
from q8_m0_gsheet import gsheet_spec, gamma_self  # noqa: E402
from q8_m0_k1_kernel import trapz_weight  # noqa: E402
from q8_m0_k1_packet import packet_on_grid  # noqa: E402
from q8_m0_loop import ResummationNotChosenError, apply_resummation_to_output, shape_leftover  # noqa: E402
from q8_m0_mode_io import close_output_moments, eta2_ncell  # noqa: E402
from q8_m0_model import R0, build_geometry  # noqa: E402
from q8_m0_s1_kernel import pack_complex  # noqa: E402
from q8_m0_stage1 import stage1_spec  # noqa: E402
from q8_m0_volterra import (  # noqa: E402
    VOLTERRA_ID,
    collect_A2_B2,
    continuum_incoming_from_adjoint,
    filtered_source_moments,
    gamma0_recovers_born,
    h2_kernels,
    two_port_output_moments,
    volterra_adjoint,
    volterra_response_to_complex_F,
)

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "results/q_line/stage1d_twoport_20260921"
MU = -0.009010620372198276
NTPC = 400


def _json_arr(a):
    return np.asarray(a).tolist()


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    geo = build_geometry()
    prot = p1_a2_protocol(0.8)
    kf = choose_probe_k(geo, R0, MU)["kf"]
    blocked = False
    try:
        apply_resummation_to_output()
    except ResummationNotChosenError:
        blocked = True

    print("[stage1d] unperturbed +N 400", flush=True)
    r0 = propagate_cjj(
        geo, R0, prot, nt_per_cycle=NTPC, neel=+1, sign=+1, mu=MU,
        verbose=True, kf_frac=kf, save_full_x=True, save_mean_D=True,
    )
    times = np.asarray(r0["times"])
    dt = float(r0["dt"])
    x = r0["X_full"]
    w = r0["w_k"]
    mean_d = r0["mean_D_t"]
    closed = close_output_moments(x, w, r0["mu_per_cell"], mean_d, times, dt, geo, prot)
    pref = eta2_ncell(geo)
    gamma = gamma_self(geo)
    hk = h2_kernels(times, prot.omega0, prot.t_pulse, geo)
    F = hk["F"]
    e_perp = hk["e_perp"]
    k_a = hk["k_a"]

    print("[stage1d] gamma=0 recovers Born", flush=True)
    rec0 = gamma0_recovers_born(x, w, mean_d, dt, F, e_perp, k_a, pref)

    print("[stage1d] coupling-scale scan (includes opposite-sign diagnostic)", flush=True)
    scales = (0.0, 0.01, 0.03, 0.1, 0.3, 1.0, -0.1)
    scan = []
    born_A2 = complex(closed["A2"]["re"], closed["A2"]["im"])
    born_B2 = complex(closed["B2"]["re"], closed["B2"]["im"])
    last_usable = None
    i_ref = None
    for sc in scales:
        vol = volterra_response_to_complex_F(x, w, mean_d, dt, sc * gamma, F, e_perp)
        finite = bool(vol["finite"])
        if sc == 0.0:
            i_ref = float(vol["maxabs_I"])
        coeff = collect_A2_B2(k_a, vol["total"], dt, pref) if finite else {
            "A2": complex("nan"), "B2": complex("nan"),
        }
        rec = {
            "gamma_scale": sc,
            "finite": finite,
            "maxabs_I": vol["maxabs_I"],
            "maxabs_A_self": vol["maxabs_A_self"],
            "A2": pack_complex(coeff["A2"]) if finite else {"re": None, "im": None, "abs": None},
            "B2": pack_complex(coeff["B2"]) if finite else {"re": None, "im": None, "abs": None},
        }
        usable = False
        if finite:
            rec["dA2_from_born"] = pack_complex(coeff["A2"] - born_A2)
            rec["dB2_from_born"] = pack_complex(coeff["B2"] - born_B2)
            usable = (
                sc >= 0.0
                and i_ref is not None
                and vol["maxabs_I"] <= 3.0 * i_ref
                and abs(coeff["A2"]) < 3.0
            )
        rec["usable_for_moments"] = bool(usable)
        if usable:
            last_usable = (vol, coeff, sc)
        scan.append(rec)

    if last_usable is None:
        raise RuntimeError("no usable Volterra scale")
    vol1, coeff1, used_scale = last_usable
    print(f"[stage1d] moments at largest usable gamma_scale={used_scale}", flush=True)
    leftover = shape_leftover(k_a, vol1["total"], dt, ehat=e_perp)
    g_used = used_scale * gamma
    noise = filtered_source_moments(x, w, mean_d, dt, g_used, k_a)
    n_src = pref * noise["n_F"]
    m_src = -pref * noise["m_F"]
    comm_s = pref * noise["commutator_F"]
    moments = two_port_output_moments(coeff1["A2"], coeff1["B2"], n_src, m_src, comm_s)

    wt = trapz_weight(k_a.shape[0], dt)
    adj_r = volterra_adjoint(x, w, mean_d, dt, g_used, np.real(wt[:, None] * k_a))
    adj_i = volterra_adjoint(x, w, mean_d, dt, g_used, np.imag(wt[:, None] * k_a))
    g_Ain = adj_r["g_Ain"] + 1j * adj_i["g_Ain"]
    pkt = packet_on_grid(prot.omega0, prot.t_pulse, n_harm=2)
    cont = continuum_incoming_from_adjoint(np.real(adj_r["g_Ain"]), times, pkt["omega"], e_perp, pref)
    phi = np.asarray(pkt["phi"])
    recon = complex(np.trapezoid(np.conj(phi) * cont["alpha_omega_unscaled"], pkt["omega"]))
    g_real = adj_r["g_Ain"]
    par = g_real @ e_perp
    fr, fi = np.real(F), np.imag(F)
    gram = np.array([
        [float(np.dot(fr, fr)), float(np.dot(fr, fi))],
        [float(np.dot(fr, fi)), float(np.dot(fi, fi))],
    ], dtype=float)
    rhs = np.array([float(np.dot(fr, par)), float(np.dot(fi, par))])
    try:
        c = np.linalg.solve(gram + 1e-18 * np.eye(2), rhs)
    except np.linalg.LinAlgError:
        c = np.zeros(2)
    proj = c[0] * fr + c[1] * fi
    adj_left = float(np.linalg.norm(par - proj) / max(np.linalg.norm(par), 1e-30))

    print("[stage1d] 1-k Fock cut n_max=3, ntpc=8", flush=True)
    fock = run_fock_1k(
        geo, n_max=3, nt_per_cycle=8, neel=+1, kfrac=(0.0, 0.0),
        ehrenfest_self=True, incoming_scale=1.0, verbose=True,
    )
    i_fock = np.asarray(fock.pop("I"))
    t_fock = np.asarray(fock.pop("times"))
    nph_t = np.asarray(fock.pop("n_photon_t"))
    di_fock = np.asarray(fock.pop("dI"))
    i_p1_f = np.asarray(fock.pop("I_p1"))
    np.savez(
        OUT / "fock_1k_gamma_ntpc8.npz",
        I=i_fock, times=t_fock, n_photon_t=nph_t, dI=di_fock, I_p1=i_p1_f,
    )
    fock["I_maxabs"] = float(np.max(np.abs(i_fock)))
    fock["dI_maxabs"] = float(np.max(np.abs(di_fock)))
    fock["n_photon_t_max"] = float(np.max(nph_t))

    out = {
        "volterra_id": VOLTERRA_ID,
        "spec": stage1_spec(),
        "gsheet": gsheet_spec(geo),
        "resummation_production_blocked": blocked,
        "n_k": 6,
        "neel": "+1_method_only",
        "gamma_self": gamma,
        "eta2_Ncell": pref,
        "born_frozen": {
            "A2": closed["A2"],
            "B2": closed["B2"],
            "n_source": closed["term_ledger"]["n_source_O_lambda2"],
            "m_source": closed["term_ledger"]["m_source_O_lambda2"],
            "V_min_born": closed["V_theta"]["min"],
            "O2_identity_residual": closed["commutator_check"]["residual_abs"],
            "label": "Born_one_pass_production_ledger_unchanged",
        },
        "gamma0_recovers_born": rec0,
        "gamma_scale_scan": scan,
        "volterra_full": {
            "moments_at_gamma_scale": used_scale,
            "full_gamma_scale_1_divergent": bool(used_scale < 1.0),
            "A2": pack_complex(coeff1["A2"]),
            "B2": pack_complex(coeff1["B2"]),
            "abs_A2": float(abs(coeff1["A2"])),
            "abs_B2": float(abs(coeff1["B2"])),
            "dA2_from_born_abs": float(abs(coeff1["A2"] - born_A2)),
            "dB2_from_born_abs": float(abs(coeff1["B2"] - born_B2)),
            "maxabs_I": vol1["maxabs_I"],
            "maxabs_A_self": vol1["maxabs_A_self"],
            "shape_leftover_after_feedback": leftover,
            "n_source_filtered": pack_complex(n_src),
            "m_source_filtered": pack_complex(m_src),
            "s_commutator": pack_complex(comm_s),
            "same_mode_two_port": moments,
            "adjoint_drive_leftover_vs_H2_Re": adj_left,
            "continuum_alpha_l2": cont["l2"],
            "packet_recon_overlap_abs": float(abs(recon)),
            "note": (
                "Feedback is on the full two-component I(t). H2 is detection. "
                "Same-mode two-port incoming is a packet truncation. "
                "If gamma_scale=1 diverges, moments are at the largest finite scale; "
                "that is a linear-closure diagnostic, not a patched G_self."
            ),
        },
        "fock_1k": fock,
        "production_V_still_born": True,
        "nk96": "not_run",
        "g2": "not_run",
        "squeezing_claim": "not_evaluated",
        "physics_fork_q_parallel": False,
        "physics_fork_classical_self_consistency": False,
        "physics_fork_added_dissipation": False,
        "linear_Gself_at_physical_gamma": "divergent_on_sixk_P1_no_dissipation_added",
    }
    dump(OUT / "stage1d_twoport.json", out)
    print(json.dumps({
        "wrote": str(OUT / "stage1d_twoport.json"),
        "gamma0_pass": rec0["pass"],
        "gamma0_rel": rec0["rel_maxabs"],
        "born_abs_A2": float(abs(born_A2)),
        "vol_scale_used": used_scale,
        "vol_abs_A2": float(abs(coeff1["A2"])),
        "dA2": float(abs(coeff1["A2"] - born_A2)),
        "finite_full_gamma": bool(used_scale >= 1.0),
        "scan_finite": [(s["gamma_scale"], s["finite"], s["maxabs_I"]) for s in scan],
        "comm_minus_1": moments["commutator_abs_minus_1"],
        "V_min_vol": moments["V_min"],
        "V_optical_valid": moments["V_optical_valid"],
        "V_min_born": closed["V_theta"]["min"],
        "adj_leftover": adj_left,
        "fock_nph": fock["n_photon_final"],
        "fock_dim": fock["dim"],
        "production_blocked": blocked,
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
