#!/usr/bin/env python3
"""Independent finite oscillator checks for photon functional rows.

No M0 production, P1 propagation, continuum limit, far-field photon count,
or replacement of the approved initial state is performed here.
All output rows are finite-time bare-photon coordinates of the coupled
stationary diagnostic Hamiltonian. Its ground state need not be bare vacuum.
"""
from __future__ import annotations

import json
import numpy as np
from scipy.linalg import expm

from q8_m0_ground_covariance_operator_20260929 import StationaryGroundOperator
from q8_m0_finite_photon_rows_20261002 import (
    finite_time_photon_rows, photon_packet_rows, duhamel_photon_rows,
    quadrature_rows, mode_commutators,
)


def maxabs(a):
    a = np.asarray(a)
    return float(np.max(np.abs(a))) if a.size else 0.0


def fixture(*, coupled=True):
    """Two frequencies x two sides x two polarizations, plus two p-h modes.

    Order (polarization, side, frequency); each side has the same source
    coupling. Packets select one side and distinct polarizations.
    Frequency weights enter field vertices, separately from packet weights.
    """
    omega = np.array([0.17, 0.29])
    weights = np.array([0.09, 0.13])
    pf = np.tile(omega, 4)
    pv = np.zeros((2, 8), complex)
    eta = 0.03 if coupled else 0.0
    for pol in range(2):
        for side in range(2):
            lo = 4*pol + 2*side
            pv[pol, lo:lo+2] = eta*np.sqrt(weights/omega)
    ev = np.array([[0.12, 0.03+0.07j], [0.04j, 0.09]], complex)
    contact = np.array([[0.2, 0.01], [0.01, 0.15]])
    op = StationaryGroundOperator.from_vertices(
        pf, np.array([0.8, 1.1]), pv, ev, contact,
        provenance="Independent finite row fixture; no M0 or continuum claim")
    coeff = np.zeros((2, 8), complex)
    packet = np.array([np.cos(0.47), np.sin(0.47)*np.exp(0.3j)])
    coeff[0, :2] = packet
    coeff[1, 4:6] = packet
    return op, coeff, pv


def independent_dense(op):
    """Assemble the quadratic Hamiltonian without calling h_action()."""
    n = len(op.frequencies)
    j = np.block([[np.zeros((n, n)), np.eye(n)],
                  [-np.eye(n), np.zeros((n, n))]])
    h = (np.diag(np.tile(op.frequencies, 2))
         - op.ca.T@op.ci - op.ci.T@op.ca
         + op.ca.T@op.contact@op.ca)
    return h, j, j@h


def independent_annihilation_rows(op):
    n = len(op.frequencies)
    r = np.zeros((op.n_photon, 2*n), complex)
    r[np.arange(op.n_photon), np.arange(op.n_photon)] = 1/np.sqrt(2)
    r[np.arange(op.n_photon), n+np.arange(op.n_photon)] = 1j/np.sqrt(2)
    return r


def independent_ground(h, j):
    """Dense positive-Hamiltonian ground covariance, unrelated to resolvent quadrature."""
    val, vec = np.linalg.eigh(h)
    if val.min() <= 0:
        raise AssertionError("Fixture Hamiltonian must be strictly positive.")
    sqrt_h = (vec*np.sqrt(val))@vec.T
    invsqrt_h = (vec/np.sqrt(val))@vec.T
    aa = sqrt_h@j@sqrt_h
    target = -aa@aa
    target = (target+target.T)/2
    symval, symvec = np.linalg.eigh(target)
    if symval.min() <= 0:
        raise AssertionError("Fixture frequencies must be strictly positive.")
    abs_a = (symvec*np.sqrt(symval))@symvec.T
    gamma = 0.5*invsqrt_h@abs_a@invsqrt_h
    return (gamma+gamma.T)/2


def report():
    op, coeff, pv = fixture()
    h, j, k = independent_dense(op)
    time = 3.7
    phi = expm(k*time)
    r = independent_annihilation_rows(op)
    direct = (coeff*np.exp(1j*op.frequencies[:op.n_photon]*time))@r@phi
    result = finite_time_photon_rows(op, coeff, time)
    source = duhamel_photon_rows(op, coeff, time, samples=257, include_contact=True)
    missing_contact = duhamel_photon_rows(
        op, coeff, time, samples=257, include_contact=False)
    comm = mode_commutators(op, result)
    qrows = quadrature_rows(result)
    expected_qrows = np.empty((4, op.dimension))
    expected_qrows[0::2] = np.sqrt(2)*result.real
    expected_qrows[1::2] = np.sqrt(2)*result.imag
    mode_j = np.kron(np.eye(2), np.array([[0., 1.], [-1., 0.]]))
    gamma = independent_ground(h, j)

    # Test generator/source identity separately from any time quadrature.
    omega = op.frequencies[:op.n_photon]
    total_current = op.ci-op.contact@op.ca
    source_identity = r@k + 1j*omega[:, None]*r - 1j*pv.T@total_current

    # These contain a finite coupled ground-state dressing contribution.
    # No assertion of zero bare occupation is made or implied.
    projected = qrows@gamma@qrows.T
    uncertainty_eigen = np.linalg.eigvalsh(
        projected.astype(complex)+0.5j*mode_j).min()

    free, free_coeff, _ = fixture(coupled=False)
    free_result = finite_time_photon_rows(free, free_coeff, time)
    free_r = independent_annihilation_rows(free)
    free_q = quadrature_rows(free_result)
    free_cov = free_q@(0.5*np.eye(free.dimension))@free_q.T
    checks = {
        "hamiltonian_generator_identity": maxabs(k@j+j@k.T),
        "symplectic_propagation": maxabs(phi@j@phi.T-j),
        "generator_current_source_identity": maxabs(source_identity),
        "packet_t0_against_independent_rows": maxabs(photon_packet_rows(op, coeff)-coeff@r),
        "propagated_rows_against_dense_expm": maxabs(result-direct),
        "duhamel_against_dense_expm": maxabs(source-direct),
        "normal_commutator_against_identity": maxabs(comm["normal"]-np.eye(2)),
        "anomalous_commutator": maxabs(comm["anomalous"]),
        "independent_normal_commutator": maxabs(1j*result@j@result.conj().T-np.eye(2)),
        "independent_anomalous_commutator": maxabs(1j*result@j@result.T),
        "quadrature_sqrt2_and_order": maxabs(qrows-expected_qrows),
        "quadrature_canonical_pairing": maxabs(qrows@j@qrows.T-mode_j),
        "coupled_ground_stationarity": maxabs(k@gamma+gamma@k.T),
        "coupled_ground_purity": maxabs(gamma@j@gamma-j/4),
        "coupled_ground_finite_time_invariance": maxabs(phi@gamma@phi.T-gamma),
        "free_demodulation_identity": maxabs(free_result-free_coeff@free_r),
        "free_two_polarization_vacuum": maxabs(free_cov-0.5*np.eye(4)),
    }
    limits = {name: 2e-10 for name in checks}
    limits["duhamel_against_dense_expm"] = 2e-8
    negative = maxabs(missing_contact-direct)
    negative_ok = negative > max(1e-7, 100*checks["duhamel_against_dense_expm"])
    flags = {name: bool(value < limits[name]) for name, value in checks.items()}
    flags["contact_omission_detected"] = bool(negative_ok)
    flags["ground_positive"] = bool(
        op.positivity_certificate()["status"] == "finite_positive_certified")
    flags["projected_uncertainty"] = bool(uncertainty_eigen > -2e-10)
    return {
        "all_checks_pass": bool(all(flags.values())),
        "checks": checks,
        "limits": limits,
        "check_pass": flags,
        "missing_contact_negative_control_error": negative,
        "projected_ground_uncertainty_min_eigenvalue": float(uncertainty_eigen),
        "dimension": op.dimension,
        "time": time,
        "diagnostic_scope": "finite_time_coupled_oscillator_rows",
        "zero_bare_occupation_in_interacting_ground_required": False,
        "farfield_or_continuum_claim": False,
        "P1_run": False,
    }


def test_report():
    r = report()
    assert r["all_checks_pass"], json.dumps(r, indent=2)


if __name__ == "__main__":
    output = report()
    print(json.dumps(output, indent=2))
    raise SystemExit(0 if output["all_checks_pass"] else 1)

