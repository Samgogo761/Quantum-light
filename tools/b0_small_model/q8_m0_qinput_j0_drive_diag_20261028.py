#!/usr/bin/env python3
"""Single-k J0 light-field displacement diagnostic. Not HHG, not a BSV pulse.

Independent archive. Does not rewrite trial.json or preflight.json.
Does not run G1/H2, shared-mode k/-k dynamics, full BZ, SV, or spectra.
Does not install a J0-subtraction term as a physical Hamiltonian.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

import q8_m0_qinput_dynamics_20261026 as dyn
import q8_m0_qinput_minimal_20261020 as qm
from q8_m0_model import R0, build_geometry, current_ops, h8

G_DIAG = 0.005
NMAX = 12
KIND_VAC = "vacuum"
WALL_LIMIT = 120.0
PREFLIGHT = Path("results/q_line/qinput_a_truncation_preflight_20261027/preflight.json")
TRIAL = Path("results/q_line/qinput_a_dynamics_trial_20261026/trial.json")


def polarization_record(geo, e):
    a2 = np.asarray(geo.a2_cart, float)[:2]
    assumed = np.array([-0.5, 0.5 * np.sqrt(3.0)])
    return {
        "e": e.tolist(),
        "a2_cart_inplane": a2.tolist(),
        "a2_norm": float(np.linalg.norm(a2)),
        "assumed_minus_half_sqrt3_over_2": assumed.tolist(),
        "max_abs_e_minus_assumed": float(np.max(np.abs(e - assumed))),
        "source": "a2_cart / |a2_cart|, not an assumed 120-degree vector",
    }


def electronic_energy_classical_x(model, el, x):
    onsite = model.diag.reshape(qm.ELECTRON_DIM, model.d1)[:, 0]
    val = np.vdot(el, onsite * el)
    for F, q in zip(model.f_group, model.q_units):
        val = val + np.vdot(el, F @ el) * np.exp(1j * float(q) * float(x))
    return complex(val)


def fk_from_groups(model, el):
    acc = 0.0j
    for F, q in zip(model.f_group, model.q_units):
        acc = acc + 1j * float(q) * np.vdot(el, F @ el)
    return complex(acc)


def fk_from_current(g, e, mean_j):
    return -float(g) * float(np.dot(e, np.asarray(mean_j, float)))


def eight_band_slater_current(geo, p, k, neel):
    _ev, vecs = np.linalg.eigh(h8(geo, p, k, neel))
    occ = vecs[:, :4]
    jops = current_ops(geo, p, k, neel)
    out = np.zeros(2, float)
    for a in range(2):
        out[a] = float(np.einsum("in,ij,jn->", occ.conj(), jops[a], occ).real)
    return out, occ


def four_fermion_current(k, neel, g=G_DIAG, n_max=3):
    geo = build_geometry()
    model = dyn.build_joint(geo, R0, k, neel, n_max=n_max, omega0=dyn.TRIAL_OMEGA, g=g)
    occ = dyn.electron_ground_state(geo, R0, k, neel)
    el = qm.slater_state(occ)
    el = el / np.linalg.norm(el)
    coeff = np.zeros(n_max + 1, complex)
    coeff[0] = 1.0
    psi = dyn.product_state(model, occ, coeff)
    snap = dyn.snapshot(model, psi, 0.0)
    return np.array(snap["mean_J_re"], float), model, el, snap


def static_force(g=G_DIAG, n_max=NMAX):
    geo, k, model, occupied = dyn.build_trial_model(n_max, g=g)
    el = qm.slater_state(occupied)
    el = el / np.linalg.norm(el)
    coeff, _meta = dyn.photon_input("coherent", n_max, dyn.TRIAL_NBAR)
    psi = dyn.product_state(model, occupied, coeff)
    snap0 = dyn.snapshot(model, psi, 0.0)
    mean_j = np.array(snap0["mean_J_re"], float)
    e = np.asarray(model.e, float)
    eps = 1e-6
    e_plus = electronic_energy_classical_x(model, el, eps)
    e_minus = electronic_energy_classical_x(model, el, -eps)
    fk_fd = (e_plus - e_minus) / (2.0 * eps)
    fk_grp = fk_from_groups(model, el)
    fk_j = fk_from_current(g, e, mean_j)
    j8, _occ8 = eight_band_slater_current(geo, R0, k, 1)
    qmax = float(np.max(np.abs(model.q_units)))
    return {
        "g": g,
        "omega0": model.omega0,
        "k_cart": k.tolist(),
        "k_fractional": dyn.K_FRAC,
        "polarization": polarization_record(geo, e),
        "mean_J_s0": mean_j.tolist(),
        "mean_J_im_s0": snap0["mean_J_im"],
        "J_parallel_e_dot_J": float(np.dot(e, mean_j)),
        "J_parallel_abs": float(abs(np.dot(e, mean_j))),
        "eight_band_slater_J": j8.tolist(),
        "max_abs_four_fermion_minus_eight_band_J": float(np.max(np.abs(mean_j - j8))),
        "F_k_finite_diff_dE_dx": {"re": float(fk_fd.real), "im": float(fk_fd.imag)},
        "F_k_from_groups_i_q_F": {"re": float(fk_grp.real), "im": float(fk_grp.imag)},
        "F_k_from_minus_g_e_dot_J": float(fk_j),
        "max_abs_Fk_fd_minus_groups": float(abs(fk_fd - fk_grp)),
        "max_abs_Fk_groups_minus_minus_g_eJ": float(abs(fk_grp.real - fk_j)),
        "F_k_definition": (
            "F_k is the coefficient of s X in the small-q expansion "
            "H_e = sum F_g ⊗ exp(i q_g s X) ≈ H_e(sX=0) + s X F_k, "
            "with X=a+a^dagger. It is not <∂H/∂X> at s=0, which vanishes."
        ),
        "equivalence_note": (
            "q = g e·delta and J_a = -i delta_a F on the same bonds imply "
            "dE/dx|_{x=0,s=1 classical} = sum i q <F> = -g e·<J>. "
            "This is a code identity, checked numerically, not an extra assumption. "
            "The outside estimate F_k = +g <J_parallel> used the opposite current "
            "sign convention; the photon-number scale |F_k|/ω0 is the same."
        ),
        "static_alpha_eq": float(-fk_j / model.omega0),
        "static_n_eq": float((fk_j / model.omega0) ** 2),
        "q_abs_max": qmax,
        "qX_scale_if_X_eq_2": float(2.0 * qmax),
        "units": {
            "hbar_e": 1,
            "energy": "t_h",
            "length": "Angstrom",
            "g": "Angstrom^-1 abstract, not experimental V_eff",
            "X": "a + a^dagger, not (a+a^dagger)/sqrt(2)",
        },
        "F_k": float(fk_j),
        "n_ph_initial_coherent": snap0["n_ph"],
        "e_total_s0": snap0["energy"],
    }


def oscillator_series(alpha0, F_of_t, env, times, omega, nsub=8001):
    t1 = float(max(times))
    t_grid = np.linspace(0.0, t1, nsub)
    dt = t_grid[1] - t_grid[0]
    alpha = np.zeros(nsub, complex)
    alpha[0] = complex(alpha0)
    def rhs(tt, aa):
        return -1j * omega * aa - 1j * F_of_t(float(tt)) * env.s(float(tt))

    for i in range(nsub - 1):
        t = t_grid[i]
        k1 = rhs(t, alpha[i])
        k2 = rhs(t + 0.5 * dt, alpha[i] + 0.5 * dt * k1)
        k3 = rhs(t + 0.5 * dt, alpha[i] + 0.5 * dt * k2)
        k4 = rhs(t + dt, alpha[i] + dt * k3)
        alpha[i + 1] = alpha[i] + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
    n_ph = np.abs(alpha) ** 2
    out_n, out_a = [], []
    for t in times:
        j = int(np.argmin(np.abs(t_grid - t)))
        out_n.append(float(n_ph[j]))
        out_a.append({"re": float(alpha[j].real), "im": float(alpha[j].imag)})
    return out_n, out_a


def interp_J_force(g, e, t_src, j_src):
    t_src = np.asarray(t_src, float)
    j_src = np.asarray(j_src, float)
    par = j_src @ np.asarray(e, float)

    def F_of_t(t):
        return -float(g) * float(np.interp(t, t_src, par))
    return F_of_t


def compare_n(name, n_ref, n_osc):
    a = np.asarray(n_ref, float)
    b = np.asarray(n_osc, float)
    dn = a - b
    return {
        "name": name,
        "n_ref_final": float(a[-1]),
        "n_osc_final": float(b[-1]),
        "delta_n_final": float(dn[-1]),
        "max_abs_delta_n": float(np.max(np.abs(dn))),
        "delta_n": dn.tolist(),
        "growth_ref": float(a[-1] - a[0]),
        "growth_osc": float(b[-1] - b[0]),
        "fraction_of_growth_matched_at_end": (
            None if abs(a[-1] - a[0]) < 1e-12
            else float(1.0 - abs(dn[-1]) / abs(a[-1] - a[0]))
        ),
    }


def dHds_expect(model, psi, s, eps=1e-6):
    sp = min(1.0, s + eps)
    sm = max(0.0, s - eps)
    if sp == sm:
        return 0.0
    vp = dyn.apply_h(model, psi, sp)
    vm = dyn.apply_h(model, psi, sm)
    return complex(np.vdot(psi, (vp - vm) / (sp - sm)))


def propagate_vacuum(model, psi, env, t1, n_steps, clock):
    dt = t1 / n_steps
    rec = sorted(set([0] + [min(n_steps, round(i * n_steps / 16)) for i in range(17)]))
    rec_set = set(rec)
    s0 = dyn.snapshot(model, psi, 0.0)
    samples = [dict(s0, t=0.0, external_work_delta_E=0.0,
                    dHds_re=0.0, work_integral=0.0)]
    work = 0.0
    e0 = s0["energy"]
    worst = 0.0
    for j in range(n_steps):
        clock.check()
        tm = (j + 0.5) * dt
        s = env.s(tm)
        Ds = dyn.peierls_at(model.q_units, s, model.n_max)
        dH = dHds_expect(model, psi, s)
        work += dt * env.ds(tm) * float(dH.real)
        psi, hdt, last = dyn.taylor_step(model, psi, s, dt, order=16, Ds=Ds)
        worst = max(worst, hdt)
        if not np.isfinite(psi).all():
            raise RuntimeError(f"non-finite at step {j} last_term={last}")
        if (j + 1) in rec_set:
            t = (j + 1) * dt
            obs = dyn.snapshot(model, psi, env.s(t))
            obs["t"] = t
            obs["external_work_delta_E"] = obs["energy"] - e0
            obs["dHds_re"] = float(dH.real)
            obs["work_integral"] = work
            samples.append(obs)
    return psi, samples, {"steps": n_steps, "dt": dt, "max_abs_Hshift_dt": worst,
                          "work_integral": work, "delta_E": samples[-1]["energy"] - e0}


def k_minus_k_static():
    geo = build_geometry()
    k = geo.k_cart(dyn.K_FRAC)
    rows = {}
    for tag, kk, neel in (
        ("k_plusN", k, 1),
        ("minusk_plusN", -k, 1),
        ("k_minusN", k, -1),
        ("minusk_minusN", -k, -1),
    ):
        j4, _, _, _ = four_fermion_current(kk, neel, g=0.0, n_max=2)
        j8, _ = eight_band_slater_current(geo, R0, kk, neel)
        rows[tag] = {"J4": j4.tolist(), "J8": j8.tolist(),
                     "neel": neel, "k_cart": kk.tolist()}
    e = qm.a2_polarization(geo)
    j_k = np.array(rows["k_plusN"]["J4"], float)
    j_mk = np.array(rows["minusk_plusN"]["J4"], float)
    j_mk_flip = np.array(rows["minusk_minusN"]["J4"], float)
    return {
        "rows": rows,
        "same_domain_J_k_plus_J_minusk": (j_k + j_mk).tolist(),
        "same_domain_J_parallel_sum": float(np.dot(e, j_k + j_mk)),
        "P_partner_J_k_plus_J_minusk_flipN": (j_k + j_mk_flip).tolist(),
        "P_partner_J_parallel_sum": float(np.dot(e, j_k + j_mk_flip)),
        "identity_used": "P H_N(k) P = H_{-N}(-k); same-N at -k is not the P image",
        "not_a_full_BZ_sum": True,
        "not_shared_mode_dynamics": True,
        "do_not_assume_pointwise_k_minusk_cancel_in_one_domain": True,
    }


def pack_samples(samples):
    return {
        "t": [s["t"] for s in samples],
        "s": [s["s"] for s in samples],
        "n_ph": [s["n_ph"] for s in samples],
        "energy": [s["energy"] for s in samples],
        "mean_J_re": [s["mean_J_re"] for s in samples],
        "top_two_mass": [s["top_two_mass"] for s in samples],
        "external_work_delta_E": [s["external_work_delta_E"] for s in samples],
        "work_integral": [s["work_integral"] for s in samples],
        "norm": [s["norm"] for s in samples],
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", type=Path,
                    default=Path("results/q_line/qinput_a_j0_drive_diag_20261028/diag.json"))
    ap.add_argument("--wall-seconds", type=float, default=WALL_LIMIT)
    ap.add_argument("--skip-vacuum", action="store_true")
    args = ap.parse_args(argv)
    if args.report.exists():
        print(f"report already exists: {args.report}", file=sys.stderr)
        return 2
    if not PREFLIGHT.exists():
        print(f"missing preflight {PREFLIGHT}", file=sys.stderr)
        return 2

    t_wall0 = time.perf_counter()
    pre = json.loads(PREFLIGHT.read_text(encoding="utf-8"))
    trial = json.loads(TRIAL.read_text(encoding="utf-8")) if TRIAL.exists() else {}
    env = dyn.Envelope(dyn.TRIAL_OMEGA, ramp_cycles=1.0, flat_cycles=2.0)
    force = static_force(G_DIAG, NMAX)
    Fk = force["F_k"]
    omega = force["omega0"]
    e = np.array(force["polarization"]["e"], float)

    job12 = pre["jobs"]["12"]
    t_ser = np.array(job12["time_series"]["t"], float)
    n_coh = np.array(job12["time_series"]["n_ph"], float)
    j_coh = np.array(job12["time_series"]["mean_J_re"], float)
    alpha0_coh = 0.5 + 0.0j

    n_static_coh, _a = oscillator_series(
        alpha0_coh, lambda t: Fk, env, t_ser, omega)
    n_jt_coh, _b = oscillator_series(
        alpha0_coh, interp_J_force(G_DIAG, e, t_ser, j_coh), env, t_ser, omega)
    n_static_vac, _c = oscillator_series(
        0.0, lambda t: Fk, env, t_ser, omega)

    cmp_coh_static = compare_n("coherent_vs_static_J0_oscillator", n_coh, n_static_coh)
    cmp_coh_jt = compare_n("coherent_vs_frozen_meanJ_t_oscillator", n_coh, n_jt_coh)

    km = k_minus_k_static()

    vacuum = None
    stopped_reason = "completed_static"
    clock = dyn.WallClock(args.wall_seconds)
    if not args.skip_vacuum:
        try:
            _, _, model, occupied = dyn.build_trial_model(NMAX, g=G_DIAG)
            coeff, meta = dyn.photon_input(KIND_VAC, NMAX, 0.0)
            psi0 = dyn.product_state(model, occupied, coeff)
            rho = dyn.spectral_radius(model, 1.0)
            t1 = env.t_ramp
            n_steps = int(job12["ledger"]["steps"])
            _psi, samples, led = propagate_vacuum(model, psi0, env, t1, n_steps, clock)
            series = pack_samples(samples)
            n_vac = np.array(series["n_ph"], float)
            cmp_vac = compare_n("vacuum_vs_static_J0_oscillator", n_vac, n_static_vac)
            vacuum = {
                "job": "vacuum_input_nonzero_s_envelope",
                "not_the_same_as_zero_coupling": True,
                "g": G_DIAG,
                "n_max": NMAX,
                "dim": model.dim,
                "input": meta,
                "rho": rho,
                "ledger": led,
                "time_series": series,
                "final_n_ph": float(n_vac[-1]),
                "growth": float(n_vac[-1] - n_vac[0]),
                "max_top_two_mass": float(max(series["top_two_mass"])),
                "work_check": {
                    "delta_E": led["delta_E"],
                    "integral_dsdt_dHds": led["work_integral"],
                    "abs_diff": abs(led["delta_E"] - led["work_integral"]),
                    "archived_external_work_definition": (
                        "trial.json and preflight.json set external_work = E(t)-E(0). "
                        "That is an energy-change record, not an independent "
                        "quadrature of (ds/dt)<dH/ds>."
                    ),
                },
                "comparison_to_static_oscillator": cmp_vac,
            }
            stopped_reason = "completed_static_and_vacuum_ramp"
        except TimeoutError as exc:
            stopped_reason = f"wall_timeout: {exc}"

    zero_n = None
    if trial:
        z = trial.get("jobs", {}).get("zero", {})
        zero_n = {
            "g": z.get("g"),
            "n_ph_initial": (z.get("initial") or {}).get("n_ph"),
            "n_ph_final": (z.get("exact_final") or {}).get("n_ph"),
            "delta_n_ph_exact": z.get("delta_n_ph_exact"),
            "note": "s is irrelevant at g=0; this is not vacuum-input with s(t) on",
        }

    growth_coh = float(n_coh[-1] - n_coh[0])
    matched = cmp_coh_static["growth_osc"]
    vac_growth = None if vacuum is None else vacuum["growth"]
    report = {
        "record_date": "2026-10-28",
        "model_id": "M0_single_k_j0_drive_diag_v1",
        "claim_level": "displacement_mechanism_diagnostic_not_magnetic_HHG",
        "not_free_space_BSV_pulse": True,
        "g": G_DIAG,
        "n_max_vacuum": NMAX,
        "k_fractional": dyn.K_FRAC,
        "omega0": dyn.TRIAL_OMEGA,
        "did_not": [
            "rewrite trial.json", "rewrite preflight.json", "G1", "H2",
            "HHG spectra", "SV scan", "semiclassical trajectories",
            "k/-k shared-mode dynamics", "full BZ", "AI inverse design",
            "install J0 subtraction as physical model",
        ],
        "static_force": force,
        "zero_coupling_archive": zero_n,
        "preflight_coherent_used": {
            "path": str(PREFLIGHT),
            "n_max": 12,
            "n_ph": n_coh.tolist(),
            "t": t_ser.tolist(),
            "s": job12["time_series"]["s"],
            "mean_J_re": j_coh.tolist(),
        },
        "oscillator_reference": {
            "hamiltonian": "omega0 a^dagger a + F_k s(t) (a + a^dagger)",
            "F_k": Fk,
            "static_n_eq": force["static_n_eq"],
            "n_static_J0_coherent": n_static_coh,
            "n_meanJ_t_coherent": n_jt_coh,
            "n_static_J0_vacuum": n_static_vac,
            "comparison_coherent_static": cmp_coh_static,
            "comparison_coherent_meanJ_t": cmp_coh_jt,
        },
        "vacuum_ramp": vacuum,
        "k_minus_k_static_currents": km,
        "centered_coupling_not_installed": {
            "label": "diagnostic_only_if_ever_used",
            "would_be": "H - s(t) F_k X  or linear replacement J -> J - <J>_0",
            "not_the_original_Peierls_Hamiltonian": True,
            "not_run": True,
        },
        "external_work_in_archives": {
            "preflight": "E(t)-E(0) only",
            "trial": "external_work_Delta_energy = E_end - E_0 only",
            "independent_integral_exists_in_archives": False,
        },
        "headline_numbers": {
            "coherent_n_ph_0": float(n_coh[0]),
            "coherent_n_ph_ramp_end": float(n_coh[-1]),
            "coherent_growth": growth_coh,
            "static_oscillator_growth_from_same_alpha0": float(matched),
            "static_n_eq_from_J0": force["static_n_eq"],
            "vacuum_growth": vac_growth,
            "residual_coherent_minus_static_osc_at_end": cmp_coh_static["delta_n_final"],
            "same_domain_J_parallel_k_plus_minusk": km["same_domain_J_parallel_sum"],
        },
        "files": {
            Path(p).name: hashlib.sha256(Path(p).read_bytes()).hexdigest()
            for p in (dyn.__file__, qm.__file__, __file__, PREFLIGHT)
        },
        "wall_seconds_total": time.perf_counter() - t_wall0,
        "stopped_reason": stopped_reason,
        "stopped": True,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    with args.report.open("x", encoding="utf-8") as fh:
        json.dump(report, fh, ensure_ascii=False, indent=2, allow_nan=False)
        fh.write("\n")
    print(json.dumps({
        "stopped_reason": stopped_reason,
        "wall": report["wall_seconds_total"],
        "headline": report["headline_numbers"],
        "Fk": Fk,
        "e": force["polarization"]["e"],
        "work": None if vacuum is None else vacuum["work_check"],
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
