#!/usr/bin/env python3
"""Artificial finite-time electronic Ward fixture; never an M0 production run.

Use hbar=1, H(k,t)=d(k+A(t)).sigma, j=-partial_k H and normalized
periodic BZ measure dk/(2*pi). The current sign is a declared convention.
A constant shift kappa also shifts the initial occupied projector P0(k+kappa).
Consequently the integrated J is invariant by a periodic change of variable.
The tangent identity includes BOTH the initial-projector derivative and contact
partial_k j. This is a necessary electronic mean-current Ward control only:
it does not prove a Gaussian photon continuum, incoming/outgoing completeness,
strong-output existence, covariance cancellation, or any infrared statement.
The toy Hamiltonian, drive, and initial state are explicitly artificial and
have no claim to represent P1, M0, or a production material/state/gauge choice.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import numpy as np
from scipy.integrate import solve_ivp

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = ROOT / "results/q_line/continuum_ground_bridge_20261003/driven_ward_fixture.json"


def maxabs(x):
    return float(np.max(np.abs(x)))


def dvec(k, order=0):
    """Periodic, smooth and gapped: d_z >= 1.45 - .32 = 1.13."""
    if order == 0:
        return np.stack((.75*np.sin(k)+.18*np.sin(2*k),
                         .60*np.cos(k)-.12*np.cos(2*k),
                         1.45+.32*np.cos(k+.3)), axis=-1)
    if order == 1:
        return np.stack((.75*np.cos(k)+.36*np.cos(2*k),
                         -.60*np.sin(k)+.24*np.sin(2*k),
                         -.32*np.sin(k+.3)), axis=-1)
    if order == 2:
        return np.stack((-.75*np.sin(k)-.72*np.sin(2*k),
                         -.60*np.cos(k)+.48*np.cos(2*k),
                         -.32*np.cos(k+.3)), axis=-1)
    raise ValueError("Only derivatives of order zero, one, and two are used")


def drive(t, duration):
    return .9*np.sin(np.pi*t/duration)**2*np.sin(1.3*t+.2)


def sigma_matrix(d):
    out = np.empty(d.shape[:-1]+(2, 2), dtype=complex)
    out[..., 0, 0] = d[..., 2]
    out[..., 0, 1] = d[..., 0]-1j*d[..., 1]
    out[..., 1, 0] = d[..., 0]+1j*d[..., 1]
    out[..., 1, 1] = -d[..., 2]
    return out


def tangent_ode(k, duration, times):
    """Bloch form of rho'=-i[H,rho], S'=-i[H,S]-i[dH,rho]."""
    d0, dk0 = dvec(k), dvec(k, 1)
    norm = np.linalg.norm(d0, axis=-1, keepdims=True)
    r0 = -d0/norm
    s0 = -dk0/norm+d0*np.sum(d0*dk0, axis=-1, keepdims=True)/norm**3
    initial = np.stack((r0, s0, np.zeros_like(s0)))

    def rhs(t, flat):
        r, s, s_no_initial = flat.reshape(initial.shape)
        d = dvec(k+drive(t, duration))
        dk = dvec(k+drive(t, duration), 1)
        source = 2*np.cross(dk, r)
        return np.stack((2*np.cross(d, r),
                         2*np.cross(d, s)+source,
                         2*np.cross(d, s_no_initial)+source)).ravel()

    solution = solve_ivp(rhs, (0., duration), initial.ravel(), t_eval=times,
                         method="DOP853", rtol=2e-11, atol=2e-13)
    if not solution.success:
        raise RuntimeError(solution.message)
    states = solution.y.T.reshape((len(times),)+initial.shape)
    return states, int(solution.nfev)


def midpoint_spinors(k, duration, nsteps, delta, finite_shift):
    """Independent finite differences: eigh(P0), unitary SU(2) midpoint steps.

    Finite-difference trials use neither tangent ODE nor projector derivatives.
    A separate band-coherence check integrates b with the analytic initial column.
    Fixed-initial trials deliberately break the initial state transformation.
    """
    shifts = np.array([0., finite_shift, delta, -delta, delta, -delta])
    initial_shifts = np.array([0., finite_shift, delta, -delta, 0., 0.])
    momenta = k[None, :]+shifts[:, None]
    energies, vectors = np.linalg.eigh(sigma_matrix(dvec(k[None, :]+initial_shifts[:, None])))
    psi = vectors[..., :, 0].copy()
    empty = vectors[0, ..., :, 1].copy()
    j0 = sigma_matrix(-dvec(k, 1))
    x0 = np.einsum("...i,...ij,...j->...", empty.conj(), j0, psi[0])
    gap = energies[0, :, 1]-energies[0, :, 0]
    soft_b = x0/gap
    snapshots = []
    dt = duration/nsteps
    save_steps = {0, nsteps//4, nsteps//2, 3*nsteps//4, nsteps}
    norm_error = 0.
    for step in range(nsteps+1):
        if step in save_steps:
            rho = psi[..., :, None]*psi[..., None, :].conj()
            r = np.stack((2*rho[..., 0, 1].real, -2*rho[..., 0, 1].imag,
                          (rho[..., 0, 0]-rho[..., 1, 1]).real), axis=-1)
            j = -dvec(momenta+drive(step*dt, duration), 1)
            currents = np.mean(np.sum(r*j, axis=-1), axis=-1)
            cv = empty[..., :, None]*psi[0, ..., None, :].conj()
            soft_rho = soft_b[..., None, None]*cv
            soft_rho += np.swapaxes(soft_rho.conj(), -1, -2)
            soft_s = np.stack((2*soft_rho[..., 0, 1].real, -2*soft_rho[..., 0, 1].imag,
                               (soft_rho[..., 0, 0]-soft_rho[..., 1, 1]).real), axis=-1)
            xt = np.einsum("...i,...ij,...j->...", empty.conj(), sigma_matrix(j[0]), psi[0])
            soft_paramagnetic = float(np.mean(2*np.real(soft_b*xt.conj())))
            soft_contact = float(np.mean(np.sum(r[0]*(-dvec(k+drive(step*dt, duration), 2)), axis=-1)))
            snapshots.append((r.copy(), currents, soft_s, soft_paramagnetic, soft_contact))
            norm_error = max(norm_error, maxabs(np.sum(abs(psi)**2, axis=-1)-1))
        if step == nsteps:
            break
        d = dvec(momenta+drive((step+.5)*dt, duration))
        norm = np.linalg.norm(d, axis=-1)
        hpsi = np.einsum("...ij,...j->...i", sigma_matrix(d), psi)
        # Independent interaction-picture band coherence: bdot=i<c|j|v>.
        # Midpoint states use a half step of the same unitary exponential.
        hempty = np.einsum("...ij,...j->...i", sigma_matrix(d[0]), empty)
        cos_half = np.cos(norm[0]*dt/2)[..., None]
        sin_half = (np.sin(norm[0]*dt/2)/norm[0])[..., None]
        v_mid = cos_half*psi[0]-1j*sin_half*hpsi[0]
        c_mid = cos_half*empty-1j*sin_half*hempty
        j_mid = sigma_matrix(-dvec(k+drive((step+.5)*dt, duration), 1))
        x_mid = np.einsum("...i,...ij,...j->...", c_mid.conj(), j_mid, v_mid)
        soft_b += 1j*x_mid*dt
        empty = np.cos(norm[0]*dt)[..., None]*empty-1j*(np.sin(norm[0]*dt)/norm[0])[..., None]*hempty
        psi = np.cos(norm*dt)[..., None]*psi-1j*(np.sin(norm*dt)/norm)[..., None]*hpsi
    return snapshots, norm_error


def analyze_spinors(snapshots, states, k, duration, times, delta):
    local_r_errors, local_s_errors, local_no_initial_errors = [], [], []
    shift_residuals, fd_currents, fd_bad_currents = [], [], []
    baseline_current_errors = []
    soft_tangent_errors, soft_current_errors, soft_ward = [], [], []
    for (r, currents, soft_s, soft_paramagnetic, soft_contact), tangent_state, t in zip(snapshots, states, times):
        rt, st, snt = tangent_state
        dk = -dvec(k+drive(t, duration), 1)
        soft_tangent_errors.append(maxabs(soft_s-st))
        soft_current_errors.append(float(soft_paramagnetic-np.mean(np.sum(st*dk, axis=-1))))
        soft_ward.append(float(soft_paramagnetic+soft_contact))
        local_r_errors.append(maxabs(r[0]-rt))
        local_s_errors.append(maxabs((r[2]-r[3])/(2*delta)-st))
        local_no_initial_errors.append(maxabs((r[4]-r[5])/(2*delta)-snt))
        shift_residuals.append(float(currents[1]-currents[0]))
        fd_currents.append(float((currents[2]-currents[3])/(2*delta)))
        fd_bad_currents.append(float((currents[4]-currents[5])/(2*delta)))
        baseline_current_errors.append(float(currents[0]-np.mean(np.sum(rt*dk, axis=-1))))
    return {
        "band_coherence_tangent_vs_ODE_max_errors": soft_tangent_errors,
        "band_coherence_current_vs_ODE_errors": soft_current_errors,
        "band_coherence_full_Ward_sums": soft_ward,
        "local_bloch_vs_ODE_max_errors": local_r_errors,
        "local_full_tangent_vs_FD_max_errors": local_s_errors,
        "local_omitted_initial_tangent_vs_FD_max_errors": local_no_initial_errors,
        "finite_kappa_current_differences": shift_residuals,
        "full_shift_FD_current_derivatives": fd_currents,
        "fixed_initial_FD_current_derivatives": fd_bad_currents,
        "baseline_current_vs_ODE_errors": baseline_current_errors,
    }


def fixture_report():
    nk, duration, delta, finite_shift = 192, 3.2, 1e-4, .237
    k = -np.pi+2*np.pi*np.arange(nk)/nk
    times = np.linspace(0., duration, 5)
    states, nfev = tangent_ode(k, duration, times)
    rows = []
    for t, (r, s, s_no_initial) in zip(times, states):
        dk, dkk = -dvec(k+drive(t, duration), 1), -dvec(k+drive(t, duration), 2)
        paramagnetic = float(np.mean(np.sum(s*dk, axis=-1)))
        contact = float(np.mean(np.sum(r*dkk, axis=-1)))
        omitted_initial = float(np.mean(np.sum(s_no_initial*dk, axis=-1))+contact)
        rows.append({"t": float(t), "A": float(drive(t, duration)),
                     "current": float(np.mean(np.sum(r*dk, axis=-1))),
                     "tangent_term": paramagnetic, "contact_term": contact,
                     "full_Ward_sum": paramagnetic+contact,
                     "omit_initial_projector_derivative": omitted_initial,
                     "omit_contact_derivative": paramagnetic})
    independent = {}
    for name, steps in (("coarse", 800), ("fine", 1600)):
        snapshots, norm_error = midpoint_spinors(k, duration, steps, delta, finite_shift)
        independent[name] = analyze_spinors(snapshots, states, k, duration, times, delta)
        independent[name].update({"steps": steps, "spinor_norm_max_error": norm_error})
    coarse, fine = independent["coarse"], independent["fine"]
    tangent_max = maxabs(fine["local_full_tangent_vs_FD_max_errors"])
    coarse_tangent_max = maxabs(coarse["local_full_tangent_vs_FD_max_errors"])
    band_ratio = maxabs(fine["band_coherence_tangent_vs_ODE_max_errors"])/maxabs(coarse["band_coherence_tangent_vs_ODE_max_errors"])
    omission = np.array([row["omit_initial_projector_derivative"] for row in rows])
    contact_omission = np.array([row["omit_contact_derivative"] for row in rows])
    noncommuting = maxabs(np.cross(dvec(k+drive(.4, duration)),
                                  dvec(k+drive(1.4, duration))))
    checks = {
        "band_coherence_midpoint_refinement": band_ratio < .35,
        "band_coherence_reconstructs_full_tangent": maxabs(fine["band_coherence_tangent_vs_ODE_max_errors"]) < 8e-6,
        "band_coherence_reconstructs_current_term": maxabs(fine["band_coherence_current_vs_ODE_errors"]) < 5e-6,
        "band_coherence_plus_contact_Ward": maxabs(fine["band_coherence_full_Ward_sums"]) < 5e-6,
        "tangent_full_BZ_cancellation": maxabs([row["full_Ward_sum"] for row in rows]) < 2e-9,
        "independent_finite_shift_BZ_invariance": maxabs(fine["finite_kappa_current_differences"]) < 2e-10,
        "independent_FD_integrated_Ward": maxabs(fine["full_shift_FD_current_derivatives"]) < 2e-8,
        "independent_local_tangent_agreement": tangent_max < 8e-6,
        "independent_local_density_agreement": maxabs(fine["local_bloch_vs_ODE_max_errors"]) < 5e-6,
        "midpoint_tangent_refinement": tangent_max < .35*coarse_tangent_max,
        "unitary_norm_preserved": fine["spinor_norm_max_error"] < 3e-12,
        "fixed_initial_FD_matches_omitted_initial_tangent": maxabs(np.array(fine["fixed_initial_FD_current_derivatives"])-omission) < 5e-6,
        "omitted_initial_local_FD_agrees": maxabs(fine["local_omitted_initial_tangent_vs_FD_max_errors"]) < 8e-6,
        "omit_initial_is_detectably_nonzero_during_drive": maxabs(omission[1:4]) > .02,
        "omit_contact_is_detectably_nonzero_during_drive": maxabs(contact_omission[1:4]) > .02,
        "noncommuting_drive": noncommuting > .02,
        "driven_mean_current_is_nontrivial": maxabs([row["current"] for row in rows][1:]) > .02,
    }
    return {
        "schema": "artificial_finite_time_electronic_ward_fixture_v1",
        "scope": "Independent artificial periodic two-band electronic mean-current Ward control only; not M0 production and not a P1 model.",
        "not_proven": ["Gaussian photon continuum construction", "incoming/outgoing scattering completeness",
                       "strong-output existence or infrared convergence", "photon covariance Ward identities"],
        "identity": "J[A+kappa; P0(k+kappa)] = J[A; P0(k)] after full periodic BZ integration",
        "tangent_identity": "integral dk/(2*pi) Tr[(partial_k rho) j + rho (partial_k j)] = 0",
        "dynamics": "d_t rho = -i[H,rho]; d_t S = -i[H,S]-i[partial_k H,rho]; S(0)=partial_k P0",
        "model": {"H": "d(k+A(t)).sigma", "d": ["0.75 sin(k)+0.18 sin(2k)",
                  "0.60 cos(k)-0.12 cos(2k)", "1.45+0.32 cos(k+0.3)"],
                  "j": "-partial_k H (repository charge convention)", "P0": "occupied lower-band projector of H(k,0)",
                  "A": "0.9 sin(pi*t/T)^2 sin(1.3*t+0.2)", "T": duration,
                  "direct_gap_lower_bound": 2.26, "noncommuting_cross_product_max_component": noncommuting},
        "numerics": {"BZ": "[-pi,pi), normalized periodic uniform quadrature", "Nk": nk,
                     "sample_times": times.tolist(), "finite_kappa": finite_shift, "FD_delta": delta,
                     "tangent_method": "DOP853 Bloch/tangent ODE, rtol=2e-11, atol=2e-13", "tangent_nfev": nfev,
                     "independent_method": "eigh initial occupied spinor; exact SU(2) exponential at time midpoints; central finite differences",
                     "midpoint_tangent_error_ratio_fine_over_coarse": tangent_max/coarse_tangent_max,
                     "band_coherence_error_ratio_fine_over_coarse": band_ratio,
                     "finite_BZ_quadrature_note": "Analytic identity follows from periodicity; finite quadrature and time refinement are numerical diagnostics only."},
        "band_coherence_mapping": {
            "basis": "Occupied v(t) and empty c(t) each solve i*d_t psi=H psi; eigenbasis comes from independent eigh.",
            "x": "<c(t)|j(t)|v(t)> with j=-partial_k H", "gap": "Delta(k)=2|d(k)|",
            "initial_column": "b(0)=x(0)/Delta(k)", "column_dynamics": "d_t b=i*x(t)",
            "reconstruction": "partial_k rho=c*b*v_dagger+v*conjugate(b)*c_dagger",
            "current_tangent": "2 Re[b*conjugate(x)]+Tr[rho*partial_k j]",
            "scope": "Finite-time electronic static-shift column only; no finite-frequency soft expansion or infinite-time tail proved."
        },
        "tangent_samples": rows, "independent_propagation": independent,
        "checks": checks, "passed": all(checks.values()),
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(f"Refusing to overwrite existing fixture result: {args.out}")
    report = fixture_report()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, allow_nan=False)
        handle.write("\n")
    print(json.dumps({"out": str(args.out), "passed": report["passed"], "checks": report["checks"]}, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
