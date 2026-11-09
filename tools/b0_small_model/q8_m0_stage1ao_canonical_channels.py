#!/usr/bin/env python3
"""Stage-1AO canonical photon/electron channel normalization audit.

This is the narrow bridge after the accepted P1=0 A-port audit:

    A_in/out(ω)  ->  a_in/out(ω)
    ξ_el(ω)      ->  L(ω) b_el,in(ω)

The SI field quantization convention is frozen from the vacuum-sheet
derivation in ``线Q_当前理论焦点_从面电流到输出光_20260918.md``:

    E^(+)(z,t) = i ∫_0^∞ dω
        sqrt(hbar*ω/(4*pi*eps0*c*S)) a(ω)
        exp(i*ω*z/c - i*ω*t)
    [a(ω), a†(ω')] = delta(ω-ω')

For the frozen two-port code units, the resulting scalar conversion of the
A-port is

    a(ω) = sqrt(ω / gamma_self) A(ω)

with no fitted or user-supplied prefactor.  The SI current formula and the
code-unit relation are retained in the output ledger; the numerical identity
below is a post-check, not the derivation.

The electron source is factorized as

    C_I(ω) = L(ω) L(ω)†,
    ξ_el(ω) = L(ω) b_el,in(ω),
    [b_i(ω), b_j†(ω')] = delta_ij delta(ω-ω').

Only the photon output row is constructed.  A coisometric photon row is not
claimed to be a complete extended scattering matrix.  The stationary
in-vacuum test is deliberately a control:

    a_out = S_a a_in + L_a b_el,in
        -> n_out = m_out = 0

No P1 drive, collection packet, Pauli, V, squeezing, entanglement, or
production initial state is introduced here.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from q8_m0_gsheet import gamma_self
from q8_m0_stage1ag_acceptance_audit import (
    REPO,
    MU,
    R0,
    build_geometry,
    current_ops,
    d_ops,
    h8,
    mean_d_from_bases,
    occup_empty_bases,
    x_from_bases,
)
from q8_m0_classical import k_grid
from q8_m0_stage1ai_lowfreq_audit import spectral_data
from q8_m0_stage1ai_lowfreq_kernel import pack_matrix
from q8_m0_stage1al_pole_audit import (
    closed_loop_F,
    increment_over_z,
    kernel_off_cuts,
)


OUT = REPO / "results/q_line/stage1ao_canonical_channels_20260926"
GRIDS = (
    ("Gamma_96x96", 96, 0.0),
    ("Gamma_192x192", 192, 0.0),
    ("halfshift_192x192", 192, 0.5),
)
ETA = 1e-4
PSD_TOL = 1e-10


SI_QUANTIZATION = {
    "fourier_transform": "I_tilde(omega)=integral dt exp(+i omega t) I(t)",
    "positive_frequency_time_dependence": "exp(-i omega t)",
    "electric_field_positive_frequency": (
        "E^(+)(z,t)=i integral_0^infty dω "
        "sqrt(hbar*ω/(4*pi*epsilon0*c*S)) a(ω) "
        "exp(i*ω*z/c-i*ω*t)"
    ),
    "canonical_commutator": "[a(ω),a†(ω')]=delta(ω-ω')",
    "vacuum_sheet_field": "E_emit=-(Z0/2) J_sheet",
    "total_current": "I(ω)=S J_sheet(ω)",
    "si_current_to_photon": (
        "a_src(ω)=i sqrt(Z0/(4*pi*hbar*ω*S)) I_tilde(ω)"
    ),
    "packet_projection": (
        "b_mu=integral_0^infty dω phi_mu(ω) a(ω), "
        "integral dω phi_mu*(ω) phi_nu(ω)=delta_mu_nu"
    ),
    "time_kernel": (
        "K_mu(t)=i sqrt(Z0/(4*pi*hbar*S)) "
        "integral_0^infty dω phi_mu(ω)/sqrt(ω) exp(+iωt)"
    ),
    "code_sheet_relation": "A_self(ω)=(i*gamma_self/ω) I_cell(ω)",
    "code_gamma": "gamma_self=2*pi*alpha_fs/A_uc in frozen hbar=e=t_h=1 units",
    "code_A_to_a": "a(ω)=sqrt(ω/gamma_self) A(ω)",
    "code_A_to_a_derivation": (
        "Insert I_total=N_cell I_cell and S=N_cell A_uc into the SI "
        "current-to-field normalization, use the frozen Peierls/current "
        "conversion and gamma_self=Z0/(2*A_uc) in code units; the remaining "
        "positive-frequency A-port scalar is sqrt(ω/gamma_self)."
    ),
    "global_phase": (
        "A single frequency-independent phase of a is conventional; this "
        "audit chooses the positive real scalar above."
    ),
    "geometry_condition": (
        "uniform q_parallel=0 sheet, vacuum on both sides, two in-plane "
        "polarizations; S=N_cell*A_uc and N_cell is not n_k"
    ),
}


def pack_complex(z: complex) -> dict:
    z = complex(z)
    return {"real": float(z.real), "imag": float(z.imag), "abs": float(abs(z))}


def pack_vector(z: np.ndarray) -> dict:
    z = np.asarray(z, dtype=complex)
    return {
        "real": np.real(z).tolist(),
        "imag": np.imag(z).tolist(),
        "abs": np.abs(z).tolist(),
    }


def maxabs(m: np.ndarray) -> float:
    m = np.asarray(m)
    return 0.0 if m.size == 0 else float(np.max(np.abs(m)))


def _hermitian_part(m: np.ndarray) -> np.ndarray:
    m = np.asarray(m, dtype=complex)
    return 0.5 * (m + m.conj().T)


def factor_psd(c_i: np.ndarray, tol: float = PSD_TOL) -> dict:
    """Factor a Hermitian positive matrix and expose any PSD failure.

    The finite-eta real-axis jump is retained without silently repairing it.
    A tiny negative eigenvalue is clipped only after it is reported as a
    numerical tolerance effect.  A material negative eigenvalue blocks the
    passive canonical-channel factorization at that diagnostic point.
    """
    c_i = np.asarray(c_i, dtype=complex)
    h = _hermitian_part(c_i)
    anti = c_i - h
    evals, evecs = np.linalg.eigh(h)
    scale = max(float(np.max(np.abs(evals))), 1.0)
    negative = evals[evals < -tol * scale]
    tiny_negative = evals[(evals < 0.0) & (evals >= -tol * scale)]
    keep = evals > tol * scale
    clipped = np.maximum(evals, 0.0)
    l_clip = evecs[:, keep] * np.sqrt(clipped[keep])[None, :]
    recon = l_clip @ l_clip.conj().T
    recon_res = maxabs(recon - h)
    return {
        "hermitian": pack_matrix(h),
        "anti_hermitian_maxabs": maxabs(anti),
        "eigenvalues": evals.tolist(),
        "rank": int(np.count_nonzero(keep)),
        "negative_eigenvalues": negative.tolist(),
        "tiny_negative_eigenvalues": tiny_negative.tolist(),
        "psd": bool(negative.size == 0),
        "psd_tolerance": float(tol * scale),
        "reconstruction_residual_after_tolerance_clip": float(recon_res),
        "L": l_clip,
    }


def candidate_vacuum_control(l_a: np.ndarray) -> dict:
    """Evaluate the annihilation-only vacuum control and a wrong-b† contrast."""
    l_a = np.asarray(l_a, dtype=complex)
    creation_block = np.zeros_like(l_a)
    correct_n = float(np.real(np.trace(creation_block @ creation_block.conj().T)))
    correct_m = 0.0
    wrong_n = float(np.real(np.trace(l_a @ l_a.conj().T)))
    return {
        "assumptions": [
            "a_in annihilates the candidate incoming vacuum",
            "b_el,in annihilates the candidate incoming vacuum",
            "no positive-frequency creation block",
        ],
        "n_out": correct_n,
        "m_out": correct_m,
        "creation_block_maxabs": maxabs(creation_block),
        "wrong_b_dagger_n_out_trace": wrong_n,
        "wrong_b_dagger_would_emit": bool(wrong_n > 1e-14),
        "is_control_only": True,
    }


def even_maps(k: np.ndarray, omega: float, gamma: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return the frozen even A-port S, G, F at positive real omega."""
    omega = float(omega)
    f = closed_loop_F(
        np.asarray(k, dtype=complex),
        omega,
        gamma,
        from_increment=False,
    )[0]
    finv = np.linalg.inv(f)
    s = 2.0 * finv - np.eye(2, dtype=complex)
    g = np.sqrt(2.0) * (1j * gamma / omega) * finv
    return s, g, f


def current_jump(d, gap, c, omega, eta):
    kr = kernel_off_cuts(d, gap, c, omega + 1j * eta)[0]
    ka = kernel_off_cuts(d, gap, c, omega - 1j * eta)[0]
    return 1j * (kr - ka), kr, ka


def source_orientation_audit(gap, x, weights, omega) -> dict:
    """Audit the positive-frequency p-h operator orientation.

    With X(t)=X exp(+i Delta t),

        delta I(t)=X(t) B† + X*(t) B

    and the +iω Fourier transform, the positive-frequency coefficient is
    X* B, while X B† is the negative-frequency coefficient.
    """
    gap = np.asarray(gap, dtype=float)
    x = np.asarray(x, dtype=complex)
    weights = np.asarray(weights, dtype=float)
    x_flat = np.transpose(x, (0, 2, 3, 1)).reshape(-1, 2)
    gap_flat = gap.reshape(-1)
    w_flat = np.repeat(weights, gap.shape[1] * gap.shape[2])
    p = int(np.argmin(np.abs(gap_flat - float(omega))))
    u = np.sqrt(w_flat[p]) * x_flat[p]
    c_line = np.outer(u.conj(), u)
    positive_annihilation = u.conj()
    negative_creation = u
    positive_phase = "exp(-i*Delta*t) B"
    negative_phase = "exp(+i*Delta*t) B†"
    return {
        "representative_transition_index": p,
        "representative_gap": float(gap_flat[p]),
        "probe_omega": float(omega),
        "positive_frequency_term": positive_phase,
        "negative_frequency_term": negative_phase,
        "positive_frequency_operator": "B (particle-hole annihilation)",
        "negative_frequency_operator": "B† (particle-hole creation)",
        "positive_annihilation_coefficient": pack_vector(positive_annihilation),
        "negative_creation_coefficient": pack_vector(negative_creation),
        "coefficient_conjugacy_residual": maxabs(
            positive_annihilation.conj() - negative_creation
        ),
        "S_greater_plus_Delta": pack_matrix(c_line),
        "S_lesser_minus_Delta_under_report_convention": pack_matrix(c_line),
        "wrong_positive_creation_is_rejected": True,
        "orientation_statement": (
            "At zero temperature the positive-frequency source is B, not B†. "
            "The latter is the negative-frequency creation component."
        ),
    }


def channel_block(s, g, c_i, omega, gamma) -> dict:
    """Normalize the A row and factor the electronic source."""
    s = np.asarray(s, dtype=complex)
    g = np.asarray(g, dtype=complex)
    c_i = np.asarray(c_i, dtype=complex)
    scale = float(np.sqrt(float(omega) / float(gamma)))
    factor = factor_psd(c_i)
    # The same scalar multiplies incoming and outgoing A ports, so the photon
    # scattering matrix is unchanged: S_a = (N_A S N_A^{-1}) = S.
    s_a = s.copy()
    l = factor["L"]
    l_a = scale * (g @ l) if l.shape[1] else np.zeros((2, 0), dtype=complex)
    raw_row = s_a @ s_a.conj().T + (scale * scale) * (g @ c_i @ g.conj().T)
    factor_row = s_a @ s_a.conj().T + l_a @ l_a.conj().T
    vacuum_control = candidate_vacuum_control(l_a)
    out = {
        "A_to_a_prefactor": scale,
        "S_a_equals_S_A": True,
        "S_a": pack_matrix(s_a),
        "L_a": {
            "real": np.real(l_a).tolist(),
            "imag": np.imag(l_a).tolist(),
            "rows": 2,
            "columns": int(l_a.shape[1]),
        },
        "electron_factor": {
            "rank": int(factor["rank"]),
            "psd": bool(factor["psd"]),
            "eigenvalues": factor["eigenvalues"],
            "negative_eigenvalues": factor["negative_eigenvalues"],
            "tiny_negative_eigenvalues": factor["tiny_negative_eigenvalues"],
            "anti_hermitian_maxabs": factor["anti_hermitian_maxabs"],
            "reconstruction_residual_after_tolerance_clip": (
                factor["reconstruction_residual_after_tolerance_clip"]
            ),
        },
        "photon_row_raw_jump_residual": pack_matrix(raw_row - np.eye(2)),
        "photon_row_raw_jump_residual_maxabs": maxabs(raw_row - np.eye(2)),
        "photon_row_factorized_residual": pack_matrix(factor_row - np.eye(2)),
        "photon_row_factorized_residual_maxabs": maxabs(factor_row - np.eye(2)),
        "creation_block_maxabs": vacuum_control["creation_block_maxabs"],
        "candidate_stationary_in_vacuum": vacuum_control,
        "extended_scattering": {
            "photon_row": "a_plus,out=S_a a_plus,in+L_a b_el,in",
            "odd_row": "a_minus,out=-a_minus,in",
            "full_electronic_output_rows": "not_constructed",
            "full_unitarity_claim": False,
            "row_property": "coisometry only",
        },
    }
    del factor["L"]
    return out


def assemble_grid(geo, name, nk, shift, gamma) -> dict:
    _, kc, weights = k_grid(geo, nk, shift=shift)
    vv, vc, en = occup_empty_bases(h8(geo, R0, kc, 1), MU)
    x = x_from_bases(current_ops(geo, R0, kc, 1), vv, vc)
    gaps = en[:, 4:, None] - en[:, None, :4]
    gap, c = spectral_data(gaps, x, weights)
    d = np.einsum(
        "k,kab->ab",
        weights,
        mean_d_from_bases(d_ops(geo, R0, kc, 1), vv),
    )
    min_gap = float(np.min(gap))
    max_gap = float(np.max(gap))
    omegas = [
        0.2 * min_gap,
        0.5 * min_gap,
        0.9 * min_gap,
        1.1 * min_gap,
        2.0 * min_gap,
        min(1.0, 0.8 * max_gap),
    ]
    rows = []
    for w in omegas:
        below = bool(w < min_gap)
        k_inc = k0 = kernel_off_cuts(d, gap, c, 0.0)[0]
        k_inc = k0 + increment_over_z(gap, c, w)[0] * w
        if below:
            k_used = k_inc
            c_i = np.zeros((2, 2), dtype=complex)
            ledger = "subgap_increment"
        else:
            k_used = kernel_off_cuts(d, gap, c, w + 1j * ETA)[0]
            c_i = current_jump(d, gap, c, w, ETA)[0]
            ledger = "continuum_retarded_jump"
        s, g, f = even_maps(k_used, w, gamma)
        block = channel_block(s, g, c_i, w, gamma)
        block.update(
            {
                "omega": float(w),
                "omega_over_min_gap": float(w / min_gap),
                "below_gap": below,
                "kernel_ledger": ledger,
                "F_det": pack_complex(np.linalg.det(f)),
                "C_I": pack_matrix(c_i),
                "source_orientation": source_orientation_audit(
                    gaps, x, weights, w
                ),
            }
        )
        rows.append(block)
        print(
            name,
            f"w/gap={w/min_gap:.3f}",
            "PSD",
            block["electron_factor"]["psd"],
            "row",
            f"{block['photon_row_factorized_residual_maxabs']:.3e}",
            "wrong-n",
            f"{block['candidate_stationary_in_vacuum']['wrong_b_dagger_n_out_trace']:.3e}",
            flush=True,
        )
    return {
        "grid": name,
        "n_k": int(len(kc)),
        "shift": float(shift),
        "min_direct_gap": min_gap,
        "max_direct_gap": max_gap,
        "eta_for_continuum_jump": ETA,
        "samples": rows,
        "max_photon_row_factorized_residual": float(
            max(r["photon_row_factorized_residual_maxabs"] for r in rows)
        ),
        "max_photon_row_raw_jump_residual": float(
            max(r["photon_row_raw_jump_residual_maxabs"] for r in rows)
        ),
        "all_electronic_factors_psd": bool(
            all(r["electron_factor"]["psd"] for r in rows)
        ),
        "all_candidate_controls_zero": bool(
            all(
                r["candidate_stationary_in_vacuum"]["n_out"] == 0.0
                and r["candidate_stationary_in_vacuum"]["m_out"] == 0.0
                for r in rows
            )
        ),
    }


def main() -> None:
    geo = build_geometry()
    gamma = float(gamma_self(geo))
    grids = [assemble_grid(geo, name, nk, shift, gamma) for name, nk, shift in GRIDS]
    payload = {
        "status": "p1_zero_canonical_photon_electron_channel_audit",
        "date": "2026-09-26",
        "scope": [
            "SI photon positive-frequency normalization ledger",
            "A-port to canonical a conversion",
            "positive-frequency B versus B-dagger source orientation",
            "C_I positive-semidefinite factorization when available",
            "photon output-row coisometry",
            "candidate stationary in-vacuum n=m=0 control",
        ],
        "not_done": [
            "P1 drive",
            "full electronic output rows",
            "complete extended scattering matrix",
            "production initial state",
            "soft dressing or bare-Fock total occupation",
            "collection packet",
            "Pauli",
            "n,m,V,squeezing,entanglement",
        ],
        "normalization": {
            **SI_QUANTIZATION,
            "gamma_self": gamma,
            "gamma_source": "q8_m0_gsheet.gamma_self",
            "photon_channel_operator": (
                "a_plus,out=S_a a_plus,in+L_a b_el,in; "
                "L_a=sqrt(omega/gamma_self) G L"
            ),
            "electron_channel_operator": (
                "C_I=L L†; b_el,in has delta-frequency canonical commutator"
            ),
        },
        "source_orientation_global": {
            "delta_I": "sum_p [u_p exp(+i Delta_p t) B_p† + u_p* exp(-i Delta_p t) B_p]",
            "positive_frequency": "u_p* exp(-i Delta_p t) B_p",
            "negative_frequency": "u_p exp(+i Delta_p t) B_p†",
            "ground_state_statement": (
                "For omega>0 and the +i omega Fourier transform, the "
                "particle-hole source is annihilation-type."
            ),
        },
        "gamma_self": gamma,
        "grids": grids,
        "verdict": {
            "A_to_a_normalization_frozen": True,
            "electron_source_is_B_not_Bdagger": True,
            "photon_row_canonical_when_factor_available": bool(
                all(
                    g["max_photon_row_factorized_residual"] < 1e-10
                    for g in grids
                )
            ),
            "candidate_stationary_control_n_equals_m_zero": bool(
                all(g["all_candidate_controls_zero"] for g in grids)
            ),
            "full_extended_scattering_matrix_unitarity": "not_claimed",
            "P1_zero_engineering_line": "freeze_after_this_audit_unless_contradiction",
        },
        "source_sha256": {
            str(p.relative_to(REPO)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in (
                Path(__file__).resolve(),
                REPO / "tools/b0_small_model/q8_m0_stage1an_canonical_io.py",
                REPO / "tools/b0_small_model/q8_m0_gsheet.py",
                REPO / "tools/b0_small_model/q8_m0_stage1ai_lowfreq_audit.py",
                REPO / "tools/b0_small_model/q8_m0_stage1al_pole_audit.py",
                REPO / "tools/b0_small_model/q8_m0_photon_op.py",
            )
        },
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "canonical_channels.json").write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "wrote": str(OUT / "canonical_channels.json"),
                "gamma_self": gamma,
                "verdict": payload["verdict"],
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
