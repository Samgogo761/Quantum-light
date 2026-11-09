"""B1-dev closed-system Wick interface for a future M88-c20 backend.

This module is deliberately narrower than a material calculation.  It evaluates
the connected two-time current correlator for independent single-particle
characteristic blocks whose step propagators are explicitly unitary:

    J_A(t) = sum_k w_k c_k^dagger A_k(t) c_k

    C^>_AB(t,t') = sum_k w_k^2 Tr[
        A_k(t) U_k(t,t') (I-rho_k(t')) B_k(t') rho_k(t') U_k^dagger(t,t')
    ].

The squared weights follow algebraically from the *declared discrete operator*
above.  They are not yet a validated continuum/BZ normalization for CrI3:
canonical k-state normalization can move a cell-volume factor into each block
correlator.  A future material backend must therefore provide explicit
``k_normalization_provenance`` before this convention is used physically.

The current production ``lg_cov`` solver evolves a finite-difference k grid and
does not yet export the independent-characteristic propagators required here.
Consequently, this file is an interface/reference implementation, not a claim
that a CrI3 B1 result has been produced.  In particular:

* the production instantaneous-eigenbasis T2 map is rejected;
* QRT is not used or claimed;
* no optical output g2, N_nm, or M_nm is constructed;
* every result is labeled as a development candidate, not a physics result.

The algebra reuses ``wick.two_time_wick_greater`` (complete-theory Eq. 7.10).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np

try:  # package import
    from .wick import equal_time_wick, mean_current, two_time_wick_greater
except ImportError:  # direct script/test import
    from wick import equal_time_wick, mean_current, two_time_wick_greater


Array = np.ndarray

SCHEMA = "b1_dev_closed_wick_v1"
LABEL = "B1_DEV_CLOSED_WICK_CANDIDATE"
STATUS = "development_only / material_backend_not_validated"
ALLOWED_LAYOUT = "independent_k_characteristics"
TARGET_MODEL = "M88-c20"
TARGET_N_BANDS = 78
TARGET_N_VALENCE = 58


@dataclass(frozen=True)
class ClosedWickPacket:
    """Single-particle data needed by the B1-dev reference contraction.

    Shapes are:

    * ``times``: ``(nt,)``;
    * ``k_weights``: ``(nk,)``, nonnegative and normalized to one;
    * ``rho0``: ``(nk, nb, nb)``;
    * ``step_unitaries``: ``(nt-1, nk, nb, nb)``;
    * ``current_a`` and ``current_b``: ``(nt, nk, nb, nb)``.

    ``step_unitaries[s, k]`` maps the characteristic block from time ``s`` to
    ``s+1``.  A/B are unweighted current matrices; k weights are applied here.
    """

    times: Array
    k_weights: Array
    rho0: Array
    step_unitaries: Array
    current_a: Array
    current_b: Array
    reference_indices: Sequence[int]


@dataclass(frozen=True)
class ClosedWickResult:
    """Development-only result; ``c_greater[:, r]`` uses reference ``refs[r]``.

    Entries with ``time_index < refs[r]`` are zero and marked false by
    ``valid_mask``.  They must not be interpreted as calculated correlators.
    """

    metadata: Mapping[str, object]
    times: Array
    refs: Array
    mean_a: Array
    mean_b: Array
    c_greater: Array
    valid_mask: Array
    rho_trace: Array


def _max_hermitian_residual(mats: Array) -> float:
    return float(np.max(np.abs(mats - np.swapaxes(mats.conj(), -1, -2))))


def _validate_packet(packet: ClosedWickPacket, *, tol: float) -> tuple[int, int, int]:
    times = np.asarray(packet.times, dtype=float)
    weights = np.asarray(packet.k_weights, dtype=float)
    rho0 = np.asarray(packet.rho0, dtype=complex)
    steps = np.asarray(packet.step_unitaries, dtype=complex)
    cur_a = np.asarray(packet.current_a, dtype=complex)
    cur_b = np.asarray(packet.current_b, dtype=complex)

    if times.ndim != 1 or times.size < 1 or not np.all(np.isfinite(times)):
        raise ValueError("times must be a finite one-dimensional array")
    if times.size > 1 and not np.all(np.diff(times) > 0.0):
        raise ValueError("times must be strictly increasing")
    if rho0.ndim != 3 or rho0.shape[-1] != rho0.shape[-2]:
        raise ValueError("rho0 must have shape (nk, nb, nb)")
    nk, nb, _ = rho0.shape
    nt = times.size
    if weights.shape != (nk,) or not np.all(np.isfinite(weights)):
        raise ValueError("k_weights must have finite shape (nk,)")
    if np.any(weights < 0.0) or abs(float(weights.sum()) - 1.0) > tol:
        raise ValueError("k_weights must be nonnegative and sum to one")
    if steps.shape != (max(nt - 1, 0), nk, nb, nb):
        raise ValueError("step_unitaries has inconsistent shape")
    if cur_a.shape != (nt, nk, nb, nb) or cur_b.shape != cur_a.shape:
        raise ValueError("current_a/current_b must have shape (nt, nk, nb, nb)")
    if not all(np.all(np.isfinite(x)) for x in (rho0, steps, cur_a, cur_b)):
        raise ValueError("packet arrays must contain only finite values")
    if _max_hermitian_residual(rho0) > tol:
        raise ValueError("rho0 is not Hermitian")
    if _max_hermitian_residual(cur_a) > tol or _max_hermitian_residual(cur_b) > tol:
        raise ValueError("current matrices must be Hermitian")

    eigs = np.linalg.eigvalsh(rho0)
    if float(eigs.min()) < -tol or float(eigs.max()) > 1.0 + tol:
        raise ValueError("rho0 violates the fermionic 0 <= rho <= I bound")

    eye = np.eye(nb, dtype=complex)
    if steps.size:
        gram = np.matmul(np.swapaxes(steps.conj(), -1, -2), steps)
        if float(np.max(np.abs(gram - eye))) > tol:
            raise ValueError("step_unitaries are not unitary within tolerance")

    refs = np.asarray(tuple(packet.reference_indices), dtype=int)
    if refs.ndim != 1 or refs.size == 0:
        raise ValueError("reference_indices must contain at least one index")
    if len(set(refs.tolist())) != refs.size:
        raise ValueError("reference_indices must be unique")
    if np.any(refs < 0) or np.any(refs >= nt):
        raise ValueError("reference index is outside the time grid")
    return nt, nk, nb


def _rho_trajectory(packet: ClosedWickPacket) -> Array:
    rho0 = np.asarray(packet.rho0, dtype=complex)
    steps = np.asarray(packet.step_unitaries, dtype=complex)
    nt = np.asarray(packet.times).size
    rho_t = np.empty((nt,) + rho0.shape, dtype=complex)
    rho_t[0] = rho0
    for it in range(1, nt):
        u = steps[it - 1]
        rho_t[it] = u @ rho_t[it - 1] @ np.swapaxes(u.conj(), -1, -2)
    return rho_t


def closed_wick_reference(
    packet: ClosedWickPacket,
    *,
    validation_tolerance: float = 1.0e-10,
    metadata: Mapping[str, object] | None = None,
) -> ClosedWickResult:
    """Evaluate selected rows of the closed-system greater correlator.

    This dense/block reference routine is intended for fixtures and a future
    validated characteristic backend.  It is not the production-scale lg_cov
    implementation.
    """

    nt, nk, nb = _validate_packet(packet, tol=validation_tolerance)
    weights = np.asarray(packet.k_weights, dtype=float)
    cur_a = np.asarray(packet.current_a, dtype=complex)
    cur_b = np.asarray(packet.current_b, dtype=complex)
    steps = np.asarray(packet.step_unitaries, dtype=complex)
    refs = np.asarray(tuple(packet.reference_indices), dtype=int)
    rho_t = _rho_trajectory(packet)

    mean_a = np.zeros(nt, dtype=complex)
    mean_b = np.zeros(nt, dtype=complex)
    for it in range(nt):
        for ik in range(nk):
            mean_a[it] += weights[ik] * mean_current(cur_a[it, ik], rho_t[it, ik])
            mean_b[it] += weights[ik] * mean_current(cur_b[it, ik], rho_t[it, ik])

    corr = np.zeros((nt, refs.size), dtype=complex)
    valid = np.zeros((nt, refs.size), dtype=bool)
    equal_time_residual = 0.0
    for iref, jt in enumerate(refs):
        u_from_ref = np.broadcast_to(np.eye(nb, dtype=complex), (nk, nb, nb)).copy()
        for it in range(jt, nt):
            if it > jt:
                u_from_ref = steps[it - 1] @ u_from_ref
            val = 0.0j
            for ik in range(nk):
                val += weights[ik] ** 2 * two_time_wick_greater(
                    cur_a[it, ik],
                    cur_b[jt, ik],
                    rho_t[jt, ik],
                    u_from_ref[ik],
                )
            corr[it, iref] = val
            valid[it, iref] = True

        expected_eq = 0.0j
        for ik in range(nk):
            expected_eq += weights[ik] ** 2 * equal_time_wick(
                cur_a[jt, ik], cur_b[jt, ik], rho_t[jt, ik]
            )
        equal_time_residual = max(equal_time_residual, abs(corr[jt, iref] - expected_eq))

    trace = np.trace(rho_t, axis1=-2, axis2=-1)
    base_metadata: dict[str, object] = {
        "schema": SCHEMA,
        "label": LABEL,
        "status": STATUS,
        "not_a_physics_result": True,
        "not_quantum_output_g2": True,
        "dynamics": "closed_unitary",
        "basis_layout": ALLOWED_LAYOUT,
        "dephasing_enabled": False,
        "production_t2_used": False,
        "qrt_used": False,
        "weight_convention": "J=sum_k(w_k*j_k); connected_C=sum_k(w_k^2*C_k)",
        "k_normalization_status": "conditional_discrete_operator_convention",
        "absolute_k_normalization_validated": False,
        "n_times": nt,
        "n_k_blocks": nk,
        "n_bands": nb,
        "equal_time_residual": float(equal_time_residual),
        "max_trace_drift": float(np.max(np.abs(trace - trace[0]))),
    }
    if metadata:
        protected = set(base_metadata).intersection(metadata)
        if protected:
            raise ValueError(f"caller metadata attempts to replace protected keys: {sorted(protected)}")
        base_metadata.update(dict(metadata))

    return ClosedWickResult(
        metadata=base_metadata,
        times=np.asarray(packet.times, dtype=float).copy(),
        refs=refs.copy(),
        mean_a=mean_a,
        mean_b=mean_b,
        c_greater=corr,
        valid_mask=valid,
        rho_trace=trace,
    )


def run_m88_c20_interface(
    packet: ClosedWickPacket,
    *,
    contract: Mapping[str, object],
    validation_tolerance: float = 1.0e-10,
) -> ClosedWickResult:
    """Apply the strict M88-c20 B1-dev contract, then run the reference core.

    Required contract fields make the scientific limitations machine-readable.
    The current production ``lg_cov_grid`` layout is intentionally rejected;
    an exporter/backend must first establish equivalent independent
    characteristics and provide their unitary step maps.
    """

    required = {
        "target_model": TARGET_MODEL,
        "n_bands": TARGET_N_BANDS,
        "n_valence": TARGET_N_VALENCE,
        "basis_layout": ALLOWED_LAYOUT,
        "dynamics": "closed_unitary",
        "dephasing_enabled": False,
        "production_t2_used": False,
        "qrt_used": False,
    }
    missing = sorted(set(required).difference(contract))
    if missing:
        raise ValueError(f"M88-c20 contract missing fields: {missing}")
    bad = {key: (contract[key], expected) for key, expected in required.items() if contract[key] != expected}
    if bad:
        raise ValueError(f"M88-c20 contract mismatch: {bad}")
    for key in (
        "operator_provenance",
        "propagator_provenance",
        "k_normalization_provenance",
    ):
        if not isinstance(contract.get(key), str) or not str(contract[key]).strip():
            raise ValueError(f"M88-c20 contract requires nonempty {key}")

    rho0 = np.asarray(packet.rho0)
    if rho0.ndim != 3 or rho0.shape[-1] != TARGET_N_BANDS:
        raise ValueError(f"M88-c20 packet must contain {TARGET_N_BANDS} active bands")
    traces = np.trace(rho0, axis1=-2, axis2=-1)
    if float(np.max(np.abs(traces - TARGET_N_VALENCE))) > validation_tolerance:
        raise ValueError(f"M88-c20 initial trace must be {TARGET_N_VALENCE} at every k block")

    result = closed_wick_reference(
        packet,
        validation_tolerance=validation_tolerance,
        metadata={
            "target_model": TARGET_MODEL,
            "n_valence": TARGET_N_VALENCE,
            "operator_provenance": contract["operator_provenance"],
            "propagator_provenance": contract["propagator_provenance"],
            "k_normalization_provenance": contract["k_normalization_provenance"],
            "material_backend_status": "interface_only_not_connected_to_production_lg_cov",
        },
    )
    return result
