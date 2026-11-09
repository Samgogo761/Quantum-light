#!/usr/bin/env python3
"""Finite-time photon functionals, NOT asymptotic input/output channels.

Coordinates are all q then all p, [q,p]=i. The full time evolution already
includes its current source: do not add Duhamel's source a second time.
"""
from __future__ import annotations
import numpy as np
from scipy.integrate import simpson
from scipy.sparse.linalg import LinearOperator, expm_multiply
from q8_m0_ground_covariance_operator_20260929 import j_action


def photon_packet_rows(op, coefficients):
    c = np.asarray(coefficients, complex)
    if c.ndim != 2 or c.shape[1] != op.n_photon or not np.isfinite(c).all():
        raise ValueError("Supply mode-by-photon finite packet coefficients.")
    n = op.frequencies.size
    rows = np.zeros((c.shape[0], op.dimension), complex)
    rows[:, :op.n_photon] = c / np.sqrt(2)
    rows[:, n:n+op.n_photon] = 1j * c / np.sqrt(2)
    return rows


def _transpose_generator(op):
    # K=J h is real, K.T=-h J. rmatvec is the adjoint, i.e. K.
    return LinearOperator(
        (op.dimension, op.dimension),
        matvec=lambda x: -op.h_action(j_action(x)),
        matmat=lambda x: -op.h_action(j_action(x)),
        rmatvec=op.generator_action,
        rmatmat=op.generator_action,
        dtype=np.dtype(float),
    )


def finite_time_photon_rows(op, coefficients, time):
    """Rows for sum_j c_j exp(i omega_j t) a_j(t) on Xi(0).

    These remain internal finite-oscillator observables when the coupling
    is always on. Sending time to infinity is NOT an out-mode construction.
    """
    time = float(time)
    if not np.isfinite(time) or time < 0:
        raise ValueError("Finite nonnegative time required.")
    c = np.asarray(coefficients, complex)
    row = photon_packet_rows(op, c * np.exp(1j * op.frequencies[:op.n_photon] * time))
    if time == 0:
        return row
    return expm_multiply(time * _transpose_generator(op), row.T, traceA=0.).T


def duhamel_photon_rows(op, coefficients, time, samples=257, include_contact=True):
    """Independent sampled source integral, intended for small fixtures.

    da_j/dt = -i omega_j a_j + i v_aj I_total,a.
    I_total = (ci-D ca) Xi, with the ORIGINAL contact retained.
    """
    time = float(time)
    if not np.isfinite(time) or time < 0:
        raise ValueError("Finite nonnegative time required.")
    if not isinstance(samples, int) or samples < 3 or samples % 2 == 0:
        raise ValueError("Odd Simpson sample count >=3 required.")
    c = np.asarray(coefficients, complex)
    initial = photon_packet_rows(op, c)
    if time == 0:
        return initial
    times = np.linspace(0., time, samples)
    current = op.ci - op.contact @ op.ca if include_contact else op.ci
    evolved = expm_multiply(_transpose_generator(op), current.T,
                           start=0., stop=time, num=samples, traceA=0.)
    n = op.frequencies.size
    v = (op.ca[:, :op.n_photon] + 1j * op.ca[:, n:n+op.n_photon]) / np.sqrt(2)
    kernels = 1j * np.einsum("mj,tj,aj->tma", c,
                np.exp(1j * times[:, None] * op.frequencies[None, :op.n_photon]), v)
    integral = simpson(np.einsum("tma,tda->tmd", kernels, evolved), x=times, axis=0)
    return initial + integral


def quadrature_rows(ell):
    ell = np.asarray(ell, complex)
    if ell.ndim != 2:
        raise ValueError("Mode rows must be two dimensional.")
    return np.stack((np.sqrt(2)*ell.real, np.sqrt(2)*ell.imag), axis=1).reshape(
        2*ell.shape[0], ell.shape[1])


def mode_commutators(op, ell):
    ell = np.asarray(ell, complex)
    if ell.ndim != 2 or ell.shape[1] != op.dimension:
        raise ValueError("Invalid mode rows.")
    return {"normal": 1j * ell @ j_action(ell.conj().T),
            "anomalous": 1j * ell @ j_action(ell.T)}


def asymptotic_output_rows(*args, **kwargs):
    raise NotImplementedError(
        "Finite photon rows do not define asymptotic out channels. Supply a "
        "continuous scattering embedding and an approved outgoing projection.")
