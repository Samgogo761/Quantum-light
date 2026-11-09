#!/usr/bin/env python3
"""Incoming-channel spectral completeness for the joint quadratic ground.

Finite discrete electrons + analytic photon continuum. This is a scattering
decomposition, not an M0 production ground and not a full-BZ absorption
spectrum. Bright discrete electrons mix into the photon continuum; they are
not extra real incoming channels. Dark/point modes are.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from q8_m0_continuum_columns_20261004 import (
    ColumnState,
    PhotonReference,
    current,
    midpoint_step,
    real_frequency_kernel,
    stationary_photon_columns,
)
from q8_m0_continuum_ground_bridge_20261003 import ContinuumGroundBridge, photon_resolvent
from q8_m0_relative_output_20261003 import dress_continuous_channels


def maxabs(x) -> float:
    x = np.asarray(x)
    return float(np.max(np.abs(x))) if x.size else 0.0


def spectral_jump(q_retarded, q_advanced):
    return (np.asarray(q_retarded) - np.asarray(q_advanced)) / (2j * np.pi)


def axis_near_zero_mask(vertices, *, relative=1e-12):
    """Basis-dependent column strengths. Not the dark-subspace dimension."""
    vertices = np.asarray(vertices, complex)
    strength = np.sqrt(np.sum(np.abs(vertices) ** 2, axis=0))
    scale = max(float(np.max(strength)), 1.0)
    return strength <= relative * scale


dark_mask = axis_near_zero_mask


def equal_gap_clusters(gaps, *, atol=1e-12):
    gaps = np.asarray(gaps, float)
    if gaps.ndim != 1 or not len(gaps):
        raise ValueError("Need a 1-D gap vector.")
    order = np.argsort(gaps, kind="stable")
    clusters = []
    cur = [int(order[0])]
    for i in order[1:]:
        i = int(i)
        if abs(gaps[i] - gaps[cur[0]]) <= atol:
            cur.append(i)
        else:
            clusters.append(np.array(cur, int))
            cur = [i]
    clusters.append(np.array(cur, int))
    return clusters


def near_degenerate_pairs(gaps, *, exact_atol=1e-12, window=1e-6):
    """Report close gaps that are NOT treated as exact degeneracy."""
    gaps = np.asarray(gaps, float)
    pairs = []
    for i in range(len(gaps)):
        for j in range(i + 1, len(gaps)):
            d = abs(gaps[i] - gaps[j])
            if exact_atol < d <= window:
                pairs.append((int(i), int(j), float(d)))
    return pairs


def _phase_fix(v):
    v = np.asarray(v, complex).copy()
    loc = int(np.argmax(np.abs(v)))
    phase = v[loc]
    if abs(phase) > 0:
        v *= np.abs(phase) / phase
    return v


def dark_projector(basis):
    """Invariant projector onto a dark basis in original electronic labels."""
    basis = np.asarray(basis, complex)
    if basis.ndim != 2:
        raise ValueError("Dark basis must be a 2-D array.")
    if basis.size == 0:
        return np.zeros((basis.shape[0], basis.shape[0]), complex)
    q, _ = np.linalg.qr(basis, mode="reduced")
    return q @ q.conj().T


def embed_block_unitary(q, n, idx):
    """Unitary that acts as q on original labels idx and as identity elsewhere."""
    q = np.asarray(q, complex)
    idx = np.asarray(idx, int)
    if q.shape != (len(idx), len(idx)):
        raise ValueError("Block unitary shape does not match the index set.")
    out = np.eye(n, dtype=complex)
    out[np.ix_(idx, idx)] = q
    return out


def annihilator_current(vertices, b_columns):
    """I from b-only incoming columns: I = u* b. A = d = 0."""
    vertices = np.asarray(vertices, complex)
    b_columns = np.asarray(b_columns, complex)
    if b_columns.size == 0:
        return np.zeros((2, 0), complex)
    return vertices.conj() @ b_columns


def coupling_singular_values(vertices):
    """Singular values of the physical map b → I = u* b."""
    vertices = np.asarray(vertices, complex)
    if vertices.size == 0:
        return np.zeros(0)
    return np.linalg.svd(vertices.conj(), compute_uv=False)


def dark_subspace(gaps, vertices, *, gap_atol=1e-12, svd_rel=1e-12, svd_abs=1e-12):
    """Kernel of I = u* b inside each exact-gap invariant subspace.

    Incoming annihilators live in b, so the physical kernel is ker(u*),
    not ker(u). SVD is on the coupling matrix itself, not on u†u.
    Different gaps are never merged. Near-degeneracy is diagnostic only.
    Weak but nonzero singular values stay bright; they are never forced
    into the kernel. Inconclusive values are reported, not zeroed.
    """
    vertices = np.asarray(vertices, complex)
    gaps = np.asarray(gaps, float)
    if vertices.ndim != 2 or vertices.shape[0] != 2 or vertices.shape[1] != len(gaps):
        raise ValueError("vertices must be shape (2, n_el).")
    n = len(gaps)
    cols = []
    info = []
    eps = float(np.finfo(float).eps)
    for idx in equal_gap_clusters(gaps, atol=gap_atol):
        cluster = vertices[:, idx]
        n_cl = int(len(idx))
        coupling = cluster.conj()
        u_mat, sval, vh = np.linalg.svd(coupling, full_matrices=True)
        s_full = np.zeros(n_cl)
        s_full[:len(sval)] = sval
        scale = max(float(s_full[0]) if n_cl else 0.0, 1.0)
        machine = 64.0 * eps * scale * max(n_cl, 2)
        floor = max(float(svd_rel) * scale, float(svd_abs), machine)
        kernel = s_full <= floor
        unresolved = (s_full > floor) & (s_full <= 1e3 * floor)
        for j in np.flatnonzero(kernel):
            v = np.zeros(n, complex)
            v[idx] = vh[j].conj()
            cols.append(_phase_fix(v))
        info.append({
            "indices": idx.tolist(),
            "gap": float(gaps[idx[0]]),
            "size": n_cl,
            "coupling_singular_values": s_full.tolist(),
            "kernel_floor": float(floor),
            "coupling_rank": int(np.count_nonzero(~kernel)),
            "nullity": int(np.count_nonzero(kernel)),
            "unresolved_count": int(np.count_nonzero(unresolved)),
        })
    basis = np.column_stack(cols) if cols else np.zeros((n, 0), complex)
    return basis, info, near_degenerate_pairs(gaps)


def dark_column_gaps(info):
    """One gap per dark column, in the same order as dark_subspace columns."""
    gaps = []
    for item in info:
        gaps.extend([float(item["gap"])] * int(item["nullity"]))
    return np.asarray(gaps, float)


def classify_dark_columns(info, *, k_of_index=None):
    """Finite-sample type of each dark combination. Not a BZ theorem.

    A strictly decoupled oscillator of one isolated gap is a point mass
    inside a finite electronic model. If the same current-null direction
    rides on a k-dependent gap, densifying the BZ turns it into part of
    the electronic continuum measure; it must not be appended as an extra
    discrete mode. Near-degeneracy is never promoted here.
    """
    rows = []
    for item in info:
        idx = list(item["indices"])
        k_ids = None
        if k_of_index is not None:
            k_ids = sorted({int(k_of_index[i]) for i in idx})
        n_k = None if k_ids is None else len(k_ids)
        if int(item["nullity"]) == 0:
            kind = "no_dark_in_cluster"
        elif k_of_index is None:
            kind = "finite_isolated_oscillator_point"
        elif n_k == 1:
            kind = "finite_sample_point_unknown_dispersion"
        elif n_k > 1:
            kind = "exact_flat_multi_k_in_this_sample"
        else:
            kind = "finite_point_unknown_k"
        automatically_discrete = bool(k_of_index is None and kind == "finite_isolated_oscillator_point")
        rows.append({
            "gap": float(item["gap"]),
            "size": int(item["size"]),
            "nullity": int(item["nullity"]),
            "k_ids": k_ids,
            "spectral_type": kind,
            "not_automatically_a_discrete_mode": (not automatically_discrete),
        })
    return rows


def electronic_spectral_measure_spec():
    """Code-backed continuum representation. No full-BZ quadrature here."""
    return {
        "representation": "original_kcv_quadrature",
        "measure": "sum_kcv w_k delta(omega-Delta_kcv) |kcv><kcv|",
        "do_not_square_delta_arrays": True,
        "principal_value": "same K(z) as stage1ai, Im z>=0, real axis only below min gap",
        "jump": "same K(omega+i eta)-K(omega-i eta)=-i(S^>-S^<) as stage1ak",
        "bright_finite_oscillators": "already mixed into photon continuum; no extra vacuum",
        "finite_isolated_dark": "atomic point mass inside that finite model only",
        "k_dependent_dark_gap": "stays inside the electronic continuum measure",
        "flat_band_or_critical_point": "needs Jacobian or separate atomic-mass analysis",
        "eta_is_not_material_dissipation": True,
        "not_a_full_BZ_implementation": True,
    }


@dataclass(frozen=True)
class IncomingBundle:
    """Photon continuum columns plus discrete point columns in the same (b,bdag,A)."""

    photon: PhotonReference
    point: ColumnState
    point_gaps: np.ndarray
    point_labels: tuple
    dark_indices: np.ndarray
    dark_basis: np.ndarray
    dark_info: tuple = ()

    @property
    def n_el(self):
        return int(self.photon.initial.b.shape[0])

    @property
    def n_photon_columns(self):
        return int(self.photon.initial.b.shape[1])

    @property
    def n_point_columns(self):
        return int(self.point.b.shape[1])


def aa_resolvent(kernel, omega, gamma, *, retarded: bool):
    """Real-axis photon q is the documented off-axis limit iγ/ω or -iγ/ω."""
    photon_resolvent((omega + 1j) if retarded else (omega - 1j), gamma)
    q = (1j if retarded else -1j) * float(gamma) / float(omega)
    return np.linalg.solve(np.eye(2) + q * kernel, q * np.eye(2))


def photon_column_density(reference: PhotonReference, omega: float):
    """A and I spectral densities of the four columns at one selected omega."""
    omega = float(omega)
    idx = [i for i, lab in enumerate(reference.labels) if abs(lab[0] - omega) < 1e-14]
    if len(idx) != 4:
        raise ValueError("Need two sides x two polarizations at this frequency.")
    a = reference.initial.A[:, idx]
    i = current(ColumnState(reference.initial.b[:, idx],
                            reference.initial.bdag[:, idx], a),
                reference.u0, reference.D0)
    return a @ a.conj().T, i @ i.conj().T, a, i


def electron_column_density(reference: PhotonReference, omega: float):
    """Original-label electronic density of the four photon columns at omega."""
    omega = float(omega)
    idx = [i for i, lab in enumerate(reference.labels) if abs(lab[0] - omega) < 1e-14]
    if len(idx) != 4:
        raise ValueError("Need two sides x two polarizations at this frequency.")
    b = reference.initial.b[:, idx]
    return b @ b.conj().T, b


def stationary_point_electron_columns(gaps, u0, D0, gamma, indices):
    """Interaction-picture incoming annihilators of uncoupled oscillators.

    u already carries free electronic phases, so a dark coefficient is
    constant when A=0. This is the same picture as the photon columns.
    """
    gaps = np.asarray(gaps, float)
    u0 = np.asarray(u0, complex)
    indices = np.asarray(indices, int)
    n = len(gaps)
    if indices.ndim != 1:
        raise ValueError("Dark/point indices must be a 1-D integer vector.")
    if np.any(indices < 0) or np.any(indices >= n):
        raise ValueError("Point index outside the original electronic labels.")
    leftover = np.max(np.abs(u0[:, indices])) if indices.size else 0.0
    if leftover > 1e-12:
        raise ValueError("Point columns are only for originally dark vertices.")
    state = ColumnState.zeros(n, len(indices))
    labels = []
    for j, p in enumerate(indices):
        state.b[p, j] = 1.0
        labels.append(("dark_electron", int(p), float(gaps[p])))
    return state, tuple(labels)


def stationary_dark_basis_columns(gaps, u0, basis, info=None, *, leftover=1e-12):
    """Incoming annihilators for dark combinations in original labels.

    Columns occupy b with d=A=0, so leftover is measured by I = u* b.
    """
    gaps = np.asarray(gaps, float)
    u0 = np.asarray(u0, complex)
    basis = np.asarray(basis, complex)
    n = len(gaps)
    if basis.ndim != 2 or basis.shape[0] != n:
        raise ValueError("Dark basis must have one original-label row per electron.")
    coupling = annihilator_current(u0, basis)
    if maxabs(coupling) > leftover:
        raise ValueError("Dark basis is not in the annihilator current kernel.")
    state = ColumnState.zeros(n, basis.shape[1])
    if basis.shape[1]:
        state.b[:, :] = basis
    if info is None:
        point_gaps = np.array(
            [float(gaps[np.argmax(np.abs(basis[:, j]))]) for j in range(basis.shape[1])],
            float,
        )
        origins = [None] * basis.shape[1]
    else:
        point_gaps = dark_column_gaps(info)
        origins = []
        for item in info:
            origins.extend([tuple(item["indices"])] * int(item["nullity"]))
        if len(point_gaps) != basis.shape[1]:
            raise ValueError("Dark column count disagrees with gap-cluster metadata.")
    labels = tuple(
        ("dark_combination", int(j), float(point_gaps[j]), origins[j])
        for j in range(basis.shape[1])
    )
    return state, labels, point_gaps


def assemble_incoming_bundle(gaps, u0, D0, gamma, frequencies, polarizations=None):
    photon = stationary_photon_columns(gaps, u0, D0, gamma, frequencies, polarizations)
    basis, info, near = dark_subspace(gaps, u0)
    point, labels, point_gaps = stationary_dark_basis_columns(gaps, u0, basis, info)
    axis = np.nonzero(axis_near_zero_mask(u0))[0]
    if len(labels) != point.b.shape[1] or len(point_gaps) != point.b.shape[1]:
        raise ValueError("Point columns, gaps and labels must have the same length.")
    return IncomingBundle(
        photon, point, point_gaps, labels, axis, basis, tuple(info),
    )


def aa_ii_jumps_from_kernel(kernel, omega, gamma):
    rr = aa_resolvent(kernel, omega, gamma, retarded=True)
    ra = aa_resolvent(kernel, omega, gamma, retarded=False)
    jump_aa = spectral_jump(rr, ra)
    # I = -K A for photon incoming. K Hermitian off poles.
    jump_ii = kernel @ jump_aa @ kernel
    jump_ai = -jump_aa @ kernel
    jump_ia = -kernel @ jump_aa
    return jump_aa, jump_ii, jump_ai, jump_ia


def bright_bare_electron_current_jump(gaps, u0, omega):
    """Illegal extra vacuum: on-shell current jump of UNDRESSED bright electrons.

    Discrete poles are not a real-frequency density. This formula is the
    Lorentzian-free principal sum that people might add by mistake; it is
    Hermitian and does not represent an extra continuum. Used only as a
    finite off-pole 'paramagnetic' overcount diagnostic.
    """
    gaps = np.asarray(gaps, float)
    u0 = np.asarray(u0, complex)
    mask = ~dark_mask(u0)
    u = u0[:, mask]
    g = gaps[mask]
    return (u.conj() / (g - omega)) @ u.T + (u / (g + omega)) @ u.conj().T


def two_band_incoming_fixture(*, photon_band=(0.4, 1.2), electron_band=(2.0, 2.8)):
    """Analytic two continuous bands. NOT an M0 bath or physical cutoff."""
    a_ph, b_ph = map(float, photon_band)
    a_el, b_el = map(float, electron_band)
    w_ph, w_el = b_ph - a_ph, b_el - a_el
    if w_ph <= 0 or w_el <= 0:
        raise ValueError("Each band must have positive width.")
    projector = 0.5 * np.array([[1, 1j], [-1j, 1]])
    mix = 0.22 * np.eye(2)
    interaction = np.block([
        [np.diag([0.35, 0.0]), mix],
        [mix, np.diag([0.40, 0.0])],
    ])

    def free_q(omega, lo, hi, width, inside):
        plus = (np.log(abs((hi - omega) / (lo - omega))) + (1j * np.pi if inside else 0.0)) / width
        minus = np.log((hi + omega) / (lo + omega)) / width
        return projector * plus + projector.conj() * minus

    def block_free(omega, *, photon_inside=False, electron_inside=False):
        qph = free_q(omega, a_ph, b_ph, w_ph, photon_inside)
        qel = free_q(omega, a_el, b_el, w_el, electron_inside)
        return np.block([[qph, np.zeros((2, 2))], [np.zeros((2, 2)), qel]])

    def inside_photon(omega):
        return a_ph < float(omega) < b_ph

    def inside_electron(omega):
        return a_el < float(omega) < b_el

    def incoming_columns(omega, *, photon="support", electron="support"):
        """photon/electron: 'support' | False | 'force'."""
        cols = []
        use_ph = inside_photon(omega) if photon == "support" else bool(photon)
        use_el = inside_electron(omega) if electron == "support" else bool(electron)
        if use_ph:
            src = np.array([[1], [-1j]]) / np.sqrt(2 * w_ph)
            cols.append(np.vstack((src, np.zeros((2, 1)))))
        if use_el:
            src = np.array([[1], [-1j]]) / np.sqrt(2 * w_el)
            cols.append(np.vstack((np.zeros((2, 1)), src)))
        return np.hstack(cols) if cols else np.zeros((4, 0), complex)

    return {
        "photon_band": (a_ph, b_ph),
        "electron_band": (a_el, b_el),
        "overlap": (max(a_ph, a_el), min(b_ph, b_el)),
        "interaction": interaction,
        "block_free": block_free,
        "inside_photon": inside_photon,
        "inside_electron": inside_electron,
        "incoming_columns": incoming_columns,
    }


def dress_two_band(spec, omega, *, photon="support", electron="support"):
    inside_ph = spec["inside_photon"](omega)
    inside_el = spec["inside_electron"](omega)
    qf = spec["block_free"](omega, photon_inside=inside_ph, electron_inside=inside_el)
    free = spec["incoming_columns"](omega, photon=photon, electron=electron)
    if free.shape[1] == 0:
        q = np.linalg.solve(np.eye(qf.shape[0]) + qf @ spec["interaction"], qf)
        return q, free
    return dress_continuous_channels(qf, spec["interaction"], free)


def duplicate_column_density(jump, columns):
    """Spectral-weight error from repeating already-counted incoming columns."""
    columns = np.asarray(columns, complex)
    jump = np.asarray(jump, complex)
    if columns.size == 0:
        return maxabs(jump), jump
    doubled = np.hstack((columns, columns))
    extra = doubled @ doubled.conj().T
    return maxabs(extra - jump), extra


def propagate_bundle(bundle: IncomingBundle, times, u_of_t, D_of_t):
    """Reuse the verified implicit midpoint. No new integrator."""
    times = np.asarray(times, float)
    if times.ndim != 1 or len(times) < 2 or np.any(np.diff(times) <= 0):
        raise ValueError("Strictly increasing times required.")
    gamma = bundle.photon.gamma
    n_ph = bundle.n_photon_columns
    packed = []
    photon = bundle.photon.initial.copy()
    point = bundle.point.copy()
    packed.append(np.concatenate((photon.packed(), point.packed()), axis=1))
    for t0, t1 in zip(times[:-1], times[1:]):
        dt = float(t1 - t0)
        tmid = 0.5 * (t0 + t1)
        u, D = u_of_t(tmid), D_of_t(tmid)
        fa = bundle.photon.free_A(t1) - bundle.photon.free_A(t0)
        photon = midpoint_step(photon, u, D, dt, gamma, free_A_increment=fa)
        point = midpoint_step(point, u, D, dt, gamma, free_A_increment=np.zeros_like(point.A))
        packed.append(np.concatenate((photon.packed(), point.packed()), axis=1))
    return np.array(packed)
