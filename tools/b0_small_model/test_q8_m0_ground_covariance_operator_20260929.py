#!/usr/bin/env python3
"""Finite manufactured checks; does not run M0/P1 or choose photon cutoffs."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

from q8_m0_boundary_math_fixture_20260929 import stationary_matching, jmat
from q8_m0_ground_covariance_operator_20260929 import StationaryGroundOperator
from q8_m0_phgauss import linear_K

def maxabs(a):
    return float(np.max(np.abs(a)))

def make_fixture():
    rng = np.random.default_rng(20260929)
    # Degenerate opposite ports for each polarization; both A components commute.
    # These are manufactured frequencies, not a chosen continuum discretization.
    pf = np.array([.47, .47, 1.19, 1.19])
    gap = np.linspace(.7, 2.8, 16)
    pv = np.array([[.06, .06, 0., 0.], [0., 0., .05, .05]], dtype=complex)
    pv[0] *= np.exp(.23j)
    pv[1] *= np.exp(-.17j)
    ev = .10 * (rng.normal(size=(2, 16)) + 1j*rng.normal(size=(2, 16)))
    ev[:, -1] = 0  # Exact invariant dark oscillator, NOT arbitrary vertex kernel.
    param = 2*np.real((ev/gap) @ ev.conj().T)
    param = (param + param.T)/2
    contact = param + np.array([[.08, .015], [.015, .11]])
    op = StationaryGroundOperator.from_vertices(
        pf, gap, pv, ev, contact, provenance="manufactured finite positive two-component model")
    return op, pf, gap, pv, ev, contact

def report():
    op, pf, gap, pv, ev, contact = make_fixture()
    n = len(pf) + len(gap)
    eye = np.eye(2*n)
    jj = jmat(n)
    diagonal = np.diag(np.r_[pf, gap, pf, gap])
    dense = (diagonal - op.ca.T@op.ci - op.ci.T@op.ca
             + op.ca.T@contact@op.ca)
    # Independent comparison to the actual pre-existing p-h generator convention.
    x = ev.reshape(2, 4, 4)[None, ...]
    legacy_k = linear_K(len(pf), np.ones(1), x, pv.T.real, pv.T.imag, contact, 1.)
    legacy_stationary = legacy_k + jj@diagonal
    rng = np.random.default_rng(8129)
    rhs = rng.normal(size=(2*n, 5))
    nu_samples = [0., .03, .31, 1.7, 12., 70.]
    resolvent_errors = {
        str(nu): maxabs(op.resolvent_action(nu, rhs)
                       - np.linalg.solve(dense + nu*jj, rhs))
        for nu in nu_samples
    }
    gamma64 = op.projected_ground_covariance(eye, order=64)
    gamma128 = op.projected_ground_covariance(eye, order=128)
    _, gamma_normal, _, gamma_sqrt = stationary_matching(dense)
    rows = rng.normal(size=(4, 2*n)) / np.sqrt(2*n)
    projected = op.projected_ground_covariance(rows, order=96)
    dark_indices = [n-1, 2*n-1]
    dark_rhs = eye[:, dark_indices]
    dark_nu = .31
    dark_diff = maxabs(op.resolvent_action(dark_nu, dark_rhs)
                      - op.free_resolvent_action(dark_nu, dark_rhs))
    # Opposite-port antisymmetric combinations are also retained as free modes.
    odd = np.column_stack([(eye[:,0]-eye[:,1])/np.sqrt(2),
                           (eye[:,n]-eye[:,n+1])/np.sqrt(2),
                           (eye[:,2]-eye[:,3])/np.sqrt(2),
                           (eye[:,n+2]-eye[:,n+3])/np.sqrt(2)])
    odd_residual = maxabs(op.resolvent_action(.31, odd)
                         - op.free_resolvent_action(.31, odd))
    certificate = op.positivity_certificate()
    # Negative-contact manufactured counterexample: rejected, never repaired.
    bad = StationaryGroundOperator.from_vertices(
        pf, gap, pv, ev, -1000*np.eye(2), provenance="negative finite control")
    negative_rejected = False
    try:
        bad.projected_ground_covariance(rows, order=16)
    except ValueError:
        negative_rejected = True
    zero_rejected = False
    try:
        StationaryGroundOperator.from_vertices(
            [0., *pf[1:]], gap, pv, ev, contact, provenance="zero mode control")
    except ValueError:
        zero_rejected = True
    checks = {
        "two_vector_potential_components_commute": maxabs(op.ca@jj@op.ca.T),
        "odd_port_resolvent_vs_free": odd_residual,
        "matrix_free_h_vs_independent_dense": maxabs(op.h_action(rhs)-dense@rhs),
        "generator_vs_existing_phgauss_plus_free_once":
            maxabs(op.generator_action(eye)-legacy_stationary),
        "woodbury_vs_dense_max": max(resolvent_errors.values()),
        "covariance_vs_normal_modes": maxabs(gamma128-gamma_normal),
        "covariance_vs_independent_sqrt": maxabs(gamma128-gamma_sqrt),
        "covariance_64_vs_128": maxabs(gamma64-gamma128),
        "four_projected_rows_vs_full_covariance": maxabs(projected-rows@gamma_sqrt@rows.T),
        "stationary_covariance_equation": maxabs(legacy_stationary@gamma128
                                              + gamma128@legacy_stationary.T),
        "pure_covariance_identity": maxabs(gamma128@jj@gamma128-jj/4),
        "invariant_dark_resolvent_vs_free": dark_diff,
        "dark_q_variance_vs_half": abs(float(gamma128[n-1,n-1])-.5),
        "dark_p_variance_vs_half": abs(float(gamma128[-1,-1])-.5),
        "original_contact_unchanged": maxabs(op.contact-contact),
    }
    uncertainty_min = float(np.linalg.eigvalsh(gamma128 + .5j*jj).min())
    passed = (max(checks.values()) < 2e-10 and negative_rejected and zero_rejected
              and uncertainty_min > -2e-10
              and certificate["status"] == "finite_positive_certified")
    return {
        "status": "finite_full_channel_ground_interface_verified" if passed else "FAILED",
        "all_checks_pass": bool(passed),
        "residuals": checks,
        "resolvent_errors_by_virtual_frequency": resolvent_errors,
        "finite_positivity_certificate": certificate,
        "dense_h_min_eigenvalue_independent": float(np.linalg.eigvalsh(dense).min()),
        "uncertainty_matrix_min_eigenvalue": uncertainty_min,
        "negative_example_rejected_without_repair": negative_rejected,
        "zero_frequency_rejected": zero_rejected,
        "quadrature_orders": [64, 96, 128],
        "dimensions": {"photon": len(pf), "particle_hole": len(gap),
                       "real_coordinates": 2*n, "woodbury_solve": 4},
        "M0_propagation": False,
        "physical_photon_cutoff_selected": False,
        "continuum_ground_existence_proven": False,
        "quantum_domain_added": False,
        "production_optical_moments_computed": False,
        "note": "Preserves all supplied channels and original D; finite manufactured input only.",
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    record = report()
    here = Path(__file__).resolve().parent
    record["source_sha256"] = {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest()
        for p in [Path(__file__).resolve(), here/"q8_m0_ground_covariance_operator_20260929.py"]
    }
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(record, ensure_ascii=False, indent=2))
    if not record["all_checks_pass"]:
        raise SystemExit("Ground interface failed.")

if __name__ == "__main__":
    main()
