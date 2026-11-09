#!/usr/bin/env python3
"""Identity checks on stored a2 sixk X(t). No new physical trajectory.

Compares: commutator vs C^> difference; prefix vs two-time state response;
analytic D vs finite difference; pulse-matched φ vs |α|² Q9 (narrowband).
Does not evaluate photons or fill chi_rad / R^A for output.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from q8_m0_cjj import dump  # noqa: E402
from q8_m0_cjj_validate import pack_z  # noqa: E402
from q8_m0_classical import p1_a2_protocol  # noqa: E402
from q8_m0_k1_packet import packet_on_grid, scale_psi_over_q9  # noqa: E402
from q8_m0_model import R0, build_geometry, sha256_file  # noqa: E402
from q8_m0_response import (  # noqa: E402
    ResponseNotChosenError,
    apply_response_to_output,
    c_greater_from_x,
    commutator_from_c_greater,
    contact_dmu,
    d_ops,
    d_ops_finite_difference,
    mean_d_from_bases,
    prefix_state_dmu,
    response_spec,
    two_time_state_dmu,
)
from q8_m0_k1_chi_interface import ChiNotChosenError, apply_chi_to_shape_kernel  # noqa: E402
from q8_m0_s1_kernel import s1_d_mu_nu  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
SIXK = REPO / "results/q_line/cjj_m0_p1_a2_s1_20260917/B0_sixk/B0_sixk_N+1_A+1.npz"
OUT = REPO / "results/q_line/response_prep_20260921"
SUBSTEP = 40


def _trapz_weight(nt: int, dt: float) -> np.ndarray:
    w = np.full(nt, dt, dtype=float)
    if nt >= 2:
        w[0] *= 0.5
        w[-1] *= 0.5
    return w


def psi_on_times(times: np.ndarray, omega0: float, tp: float, n_harm: int = 2) -> np.ndarray:
    rec = packet_on_grid(omega0, tp, n_harm=n_harm)
    omega = np.asarray(rec["omega"], dtype=float)
    phi = np.asarray(rec["phi"])
    phase = np.exp(1j * times[:, None] * omega[None, :])
    return np.trapezoid(phi[None, :] * phase, omega, axis=1)


def check_d_ops(geo, kf) -> dict:
    kc = geo.k_cart(np.mod(kf, 1.0))
    ana = d_ops(geo, R0, kc, +1)
    fd = d_ops_finite_difference(geo, R0, kc, +1, eps=1e-4)
    err = float(np.max(np.abs(ana - fd)))
    scale = float(np.max(np.abs(ana)))
    return {"max_abs_ana_minus_fd": err, "rel_to_max": err / max(scale, 1e-30), "n_k": int(kc.shape[0])}


def check_commutator_snapshots(x_full, w, snap_idx) -> dict:
    rows = []
    ids = [1, len(snap_idx) // 2, max(2, len(snap_idx) - 3)]
    for si, sj in ((ids[0], ids[1]), (ids[1], ids[2]), (ids[1], ids[1])):
        ti = int(snap_idx[si])
        tj = int(snap_idx[sj])
        c = c_greater_from_x(x_full[ti], x_full[tj], w)
        comm = commutator_from_c_greater(c)
        expect = 2j * np.imag(c)
        rows.append({
            "si": int(si),
            "sj": int(sj),
            "ti": ti,
            "tj": tj,
            "max_abs_comm_minus_2iImC": float(np.max(np.abs(comm - expect))),
        })
    return {"pairs": rows}


def check_prefix_vs_twotime(x_full, w, times, prot, geo) -> dict:
    from q8_m0_s1_kernel import parallel_perp_basis

    e_par, _e_perp = parallel_perp_basis(geo)
    nt = times.size
    idx = np.arange(0, nt, SUBSTEP)
    if idx[-1] != nt - 1:
        idx = np.concatenate([idx, np.array([nt - 1])])
    xs = x_full[idx]
    ts = times[idx]
    dt_s = float(ts[1] - ts[0])
    h = np.outer(_sin2(ts, prot.t_pulse), e_par)
    d_pre = prefix_state_dmu(xs, w, h, dt_s)
    d_tt = two_time_state_dmu(xs, w, h, dt_s)
    diff = float(np.max(np.abs(d_pre - d_tt)))
    scale = float(np.max(np.abs(d_tt)))
    return {
        "n_sub": int(idx.size),
        "dt_sub": dt_s,
        "h": "W(t) * e_parallel, real",
        "max_abs_prefix_minus_two_time": diff,
        "rel_to_max": diff / max(scale, 1e-30),
        "dmu_prefix_abs_max": scale,
        "note": "state piece only; contact -<D>h not included; subsampled stored X, no new trajectory",
    }


def _sin2(t, tp):
    u = np.asarray(t, dtype=float) / tp
    w = np.sin(np.pi * u) ** 2
    w[(u < 0.0) | (u > 1.0)] = 0.0
    return w


def check_contact_at_snapshots(geo, r) -> dict:
    kf = np.asarray(r["kf"], dtype=float)
    w = np.asarray(r["w_k"], dtype=float)
    snap_t = np.asarray(r["snapshot_t"], dtype=float)
    snap_vv = np.asarray(r["snapshot_Vv"])
    from q8_m0_classical import kappa_shift, p1_a2_protocol
    from q8_m0_s1_kernel import parallel_perp_basis

    prot = p1_a2_protocol(0.8)
    e_par, _ = parallel_perp_basis(geo)
    kc0 = geo.k_cart(np.mod(kf, 1.0))
    rows = []
    for i, t in enumerate(snap_t):
        kap = kc0 + kappa_shift(float(t), geo, prot, +1)
        d_ab = d_ops(geo, R0, kap, +1)
        mean_k = mean_d_from_bases(d_ab, snap_vv[i])
        mean = np.einsum("k,kab->ab", w, mean_k)
        h = _sin2(np.array([t]), prot.t_pulse)[0] * e_par
        ctc = contact_dmu(mean, h[None, :])[0]
        rows.append({
            "t": float(t),
            "mean_D_xx": float(mean[0, 0]),
            "mean_D_xy": float(mean[0, 1]),
            "contact_abs": float(np.linalg.norm(ctc)),
        })
    return {
        "n_snapshots": len(rows),
        "full_time_Vv": "missing_on_disk_contact_only_at_snapshots",
        "samples": rows[::4],
        "contact_abs_max_over_snapshots": max(x["contact_abs"] for x in rows),
    }


def check_narrowband(r, geo, prot) -> dict:
    times = np.asarray(r["times"], dtype=float)
    dt = float(r["dt"])
    x = np.asarray(r["X_full"])
    y = np.asarray(r["Y"])
    w = np.asarray(r["w_k"], dtype=float)
    wt = _trapz_weight(times.size, dt)
    psi = psi_on_times(times, prot.omega0, prot.t_pulse, n_harm=2)
    y_phi = np.einsum("t,tka...->ka...", wt * psi, x)
    # pack as (nk, 1, 2, 4, 4) fake harmonic slot 0, then rotate via s1 on a synthetic Y
    y_fake = np.zeros_like(y)
    y_fake[:, 1] = y_phi  # H2 cartesian slot, matching Q9 order index 1
    d_direct = complex(s1_d_mu_nu(y_fake, w, geo)[1, 1])
    d_q9 = complex(s1_d_mu_nu(y, w, geo)[1, 1])
    sc = scale_psi_over_q9(prot.omega0, prot.t_pulse)
    alpha = complex(sc["alpha_from_t_mid"]["re"], sc["alpha_from_t_mid"]["im"])
    d_scaled = (alpha * np.conj(alpha)) * d_q9
    rel = abs(d_direct - d_scaled) / max(abs(d_scaled), abs(d_direct), 1e-30)
    return {
        "kind": "narrowband_approximation_not_strict_identity",
        "sixk_only_not_full_BZ_certificate": True,
        "d_phi_direct": pack_z(d_direct),
        "d_phi_const_scale": pack_z(d_scaled),
        "d_Q9": pack_z(d_q9),
        "rel_direct_vs_const_scale": float(rel),
        "alpha_abs2": float(abs(alpha) ** 2),
        "note": "constants still multiply V48/V48S by the same factor; 2.58% is unchanged by construction",
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    if not SIXK.exists():
        dump(OUT / "sixk_response_identity.json", {"missing": str(SIXK), "ran": False})
        print(json.dumps({"missing": str(SIXK)}, indent=2), flush=True)
        return
    geo = build_geometry()
    prot = p1_a2_protocol(0.8)
    raw = np.load(SIXK, allow_pickle=True)
    r = {k: raw[k] for k in raw.files}
    x = np.asarray(r["X_full"])
    w = np.asarray(r["w_k"], dtype=float)
    times = np.asarray(r["times"], dtype=float)
    snap_idx = np.asarray(r["snapshot_index"], dtype=int)
    chi_blocked = False
    try:
        apply_chi_to_shape_kernel(None, None)
    except ChiNotChosenError:
        chi_blocked = True
    r_blocked = False
    try:
        apply_response_to_output()
    except ResponseNotChosenError:
        r_blocked = True
    out = {
        "source_npz": str(SIXK.relative_to(REPO)).replace("\\", "/"),
        "source_sha256": sha256_file(SIXK),
        "new_trajectory": False,
        "photon_output": "not_run",
        "chi_rad_apply_blocked": chi_blocked,
        "R_A_apply_blocked": r_blocked,
        "response_spec": response_spec(),
        "D_analytic_vs_fd": check_d_ops(geo, np.asarray(r["kf"], dtype=float)),
        "commutator_from_C": check_commutator_snapshots(x, w, snap_idx),
        "prefix_vs_two_time_state": check_prefix_vs_twotime(x, w, times, prot, geo),
        "contact_snapshots_only": check_contact_at_snapshots(geo, r),
        "narrowband_phi_vs_Q9_scale": check_narrowband(r, geo, prot),
        "missing_on_large_grids": {
            "V48_V48S_V96": "Y_plus_25_snapshots_no_X_full",
            "cannot": "invert_arbitrary_causal_R_from_Y",
        },
    }
    dump(OUT / "sixk_response_identity.json", out)
    print(json.dumps({
        "wrote": str(OUT / "sixk_response_identity.json"),
        "D_rel": out["D_analytic_vs_fd"]["rel_to_max"],
        "prefix_rel": out["prefix_vs_two_time_state"]["rel_to_max"],
        "narrowband_rel": out["narrowband_phi_vs_Q9_scale"]["rel_direct_vs_const_scale"],
        "chi_blocked": chi_blocked,
        "R_blocked": r_blocked,
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
