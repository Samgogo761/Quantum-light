"""Read-only algebra review of saved G1 output rows; no propagation.

Separates a stationary-output reference from the genuinely free incoming
packet. Also demonstrates the existing two_mode_quad_check diagnostic bug
using a canonical, pure two-mode Gaussian counterexample. No frozen module
or saved row is modified. Each output artifact is created exclusively.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

import q8_m0_p1_h2_finite_kcv_input_close_20261005 as q
from q8_m0_classical import p1_a2_protocol
from q8_m0_continuum_columns_20261004 import stationary_photon_columns

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "results/q_line/gamma12_output_transpose_input_refine_20261008/endpoint_low.npz"
ARRAYS = ROOT / "results/q_line/p1_h2_finite_kcv_gamma12_local_plusN_20261007/arrays"
OUT = ROOT / "results/q_line/g1_io_crosscheck_review_20261014/cross_commutator_and_covariance_audit.json"


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for part in iter(lambda: f.read(1024 * 1024), b""):
            h.update(part)
    return h.hexdigest()


def norm(x):
    return float(np.max(np.abs(x))) if np.asarray(x).size else 0.0


def pack(x):
    x = np.asarray(x)
    return {"real": x.real.tolist(), "imag": x.imag.tolist(), "maxabs": norm(x)}


def decompose(reference, u, v):
    f = u - reference
    ref = reference @ reference.conj().T - np.eye(u.shape[0])
    cross = reference @ f.conj().T + f @ reference.conj().T
    ff = f @ f.conj().T
    vv = v @ v.conj().T
    source = ff - vv
    across = reference @ v.T - v @ reference.T
    asource = f @ v.T - v @ f.T
    rn = u @ u.conj().T - v @ v.conj().T - np.eye(u.shape[0])
    ra = u @ v.T - v @ u.T
    return {
        "reference_defect": pack(ref),
        "normal_cross": pack(cross),
        "normal_source_FFdag": pack(ff),
        "normal_source_VVdag": pack(vv),
        "normal_source_difference": pack(source),
        "anomalous_cross": pack(across),
        "anomalous_source": pack(asource),
        "normal_direct": pack(rn), "anomalous_direct": pack(ra),
        "normal_reconstruction_error": pack(ref + cross + source - rn),
        "anomalous_reconstruction_error": pack(across + asource - ra),
    }


def covariance_from_nm(n, m):
    """For n_ij=<a_i^dagger a_j>, m_ij=<a_i a_j>; no clipping/projection."""
    n, m = np.asarray(n, complex), np.asarray(m, complex)
    if norm(n - n.conj().T) > 1e-12 or norm(m - m.T) > 1e-12:
        raise ValueError("This independent formula expects Hermitian n and symmetric m")
    out = np.empty((4, 4), float)
    out[::2, ::2] = np.eye(2) / 2 + (n + m).real
    out[1::2, 1::2] = np.eye(2) / 2 + (n - m).real
    out[::2, 1::2] = (n + m).imag
    out[1::2, ::2] = (n + m).imag.T
    return out


def covariance_counterexample():
    # Passive unitary mixing of two independent real squeezed vacua.
    w = np.array([[1, 1j], [1j, 1]], complex) / np.sqrt(2)
    r = np.array([0.7, 0.2])
    u, v = w @ np.diag(np.cosh(r)), w @ np.diag(np.sinh(r))
    n, m = v.conj() @ v.T, u @ v.T
    coeff = np.empty((4, 2), complex)
    coeff[::2] = (u + v.conj()) / np.sqrt(2)
    coeff[1::2] = (u - v.conj()) / (1j * np.sqrt(2))
    direct = (coeff @ coeff.conj().T).real
    by_moments = covariance_from_nm(n, m)
    omega = np.kron(np.eye(2), [[0.0, 1.0], [-1.0, 0.0]])
    correct_min = float(np.linalg.eigvalsh(direct + 1j * omega / 2)[0])
    old = q.two_mode_quad_check(n, m)
    return {
        "construction": "U=W diag(cosh r), V=W diag(sinh r); W=[[1,i],[i,1]]/sqrt(2); r=(0.7,0.2)",
        "U": pack(u), "V": pack(v), "n": pack(n), "m": pack(m),
        "normal_ccr": norm(u @ u.conj().T - v @ v.conj().T - np.eye(2)),
        "anomalous_ccr": norm(u @ v.T - v @ u.T),
        "covariance_direct_from_quadrature_coefficients": pack(direct),
        "covariance_from_correct_nm_formula": pack(by_moments),
        "two_independent_covariance_formulas_difference": norm(direct - by_moments),
        "correct_uncertainty_min_eigenvalue": correct_min,
        "existing_function_result": old,
        "bug_confirmed": bool(correct_min > -1e-12 and old["min_eig_cov_plus_i_omega_over_2"] < -0.1),
        "scope": "Diagnostic bug only; does not cause or repair the stored U,V commutator residual",
        "frozen_module_modified": False,
    }


def main():
    if OUT.exists():
        raise FileExistsError(OUT)
    tracked = [SOURCE, Path(__file__), Path(q.__file__),
        ROOT / "tools/b0_small_model/q8_m0_continuum_columns_20261004.py",
        ROOT / "tools/b0_small_model/q8_m0_gamma12_local_prepare_20261007.py",
        ROOT / "tools/b0_small_model/q8_m0_gamma12_residual_source_20261014.py",
        ROOT / "tools/b0_small_model/q8_m0_relative_output_20261003.py"]
    tracked += [ARRAYS / (name + ".npy") for name in ("pol", "gaps", "u0", "D0", "gamma")]
    before = {str(p.relative_to(ROOT)): sha(p) for p in tracked}
    with np.load(SOURCE, allow_pickle=False) as z:
        vals = {name: z[name].copy() for name in z.files}
    u, v, u0, du = (vals[name] for name in ("U", "V", "u0", "delta_u"))
    oms, wts = vals["oms"], vals["wts"]
    if np.any(wts <= 0) or np.any(np.diff(oms) <= 0):
        raise ValueError("Invalid saved positive-frequency quadrature")
    if u.shape != (2, 4 * len(oms) + 1976):
        raise ValueError("Unexpected column ledger; do not infer a new ordering")
    pol = np.load(ARRAYS / "pol.npy")
    if norm(pol.imag) > 1e-12 or norm(pol @ pol.T - np.eye(2)) > 1e-12:
        raise ValueError("Current direct row assumes the frozen two real polarizations")
    prot = p1_a2_protocol(q.A_STAR)
    op = q.omega_grid(prot.omega0, prot.t_pulse)
    op = op[op > 0]
    _, normalizer = q.normalize_phi(op, q.phi_unnormalized(op, 2 * prot.omega0, prot.t_pulse))
    amp = q.phi_at(oms, prot, normalizer) * np.sqrt(wts)
    direct = np.zeros_like(u)
    direct[:, :4 * len(oms):4] = pol[:, 0, None] * amp
    direct[:, 1:4 * len(oms):4] = pol[:, 1, None] * amp
    gamma = float(np.load(ARRAYS / "gamma.npy"))
    gaps, vertex, d0 = (np.load(ARRAYS / (n + ".npy")) for n in ("gaps", "u0", "D0"))
    ph = stationary_photon_columns(gaps, vertex, d0, gamma, [0.05, 0.2, 0.5, 1.5])
    eq_errors = []
    for i, omega in enumerate((0.05, 0.2, 0.5, 1.5)):
        sl = slice(4 * i, 4 * i + 4)
        expected = np.linalg.solve(np.eye(2) + 1j * gamma * ph.kernels[i] / omega, ph.free_A0[:, sl])
        eq_errors.append(norm(ph.initial.A[:, sl] - expected))
    report = {
        "status": "saved_output_algebra_and_diagnostic_review_not_production_qualification",
        "date": "2026-10-14", "source_sha256": before[str(SOURCE.relative_to(ROOT))],
        "code_and_input_sha256": before,
        "n_columns": int(u.shape[1]), "n_frequency_nodes": int(len(oms)),
        "column_order": "4 per frequency: Lx,Ly,Rx,Ry; then 1976 original dark columns",
        "assembly_errors": {
            "U_equals_U0_plus_deltaU": norm(u - u0 - du),
            "deltaU_equals_raw_driven_minus_raw_undriven": norm(du - vals["driven_raw_u"] + vals["undriven_raw_u"]),
            "V_equals_raw_driven_minus_raw_undriven": norm(v - vals["driven_raw_v"] + vals["undriven_raw_v"]),
            "m_U0VT": norm(vals["m_U0VT"] - u0 @ v.T),
        },
        "stationary_output_reference_split": decompose(u0, u, v),
        "genuine_free_incident_split": decompose(direct, u, v),
        "U0_minus_direct_incoming_row_maxabs": norm(u0 - direct),
        "initial_condition_evidence": {
            "production_total_A0": "solve(I+i gamma K(omega)/omega, Af0), not zero",
            "initial_electron_coefficients": "b0=(u0.T@A0)/(gap-omega); bdag0=(u0.conj().T@A0)/(gap+omega)",
            "source": "q8_m0_continuum_columns_20261004.py:139-165; local_prepare copies these arrays",
            "probe_omega": [0.05, 0.2, 0.5, 1.5],
            "linear_system_recheck_errors": eq_errors,
            "A0_maxabs": norm(ph.initial.A), "Af0_maxabs": norm(ph.free_A0),
            "not_an_independent_scattering_completeness_proof": True,
        },
        "quadrature_covariance_counterexample": covariance_counterexample(),
        "limitations": [
            "Both cross-commutator splits are algebraic reorganizations of the same stored U,V; they are not independent continuum-elimination tests.",
            "The source self-commutator still subtracts large FFdag and VVdag matrices; this does not remove cancellation conditioning.",
            "F_U=U-P includes the stationary material response as well as the driven increment; F is not isolated intrinsic electronic noise.",
            "No propagation, no changes to gamma, Hamiltonian, initial state, packet or stored output rows.",
        ],
    }
    after = {str(p.relative_to(ROOT)): sha(p) for p in tracked}
    report["all_tracked_inputs_unchanged"] = before == after
    if not report["all_tracked_inputs_unchanged"]:
        raise AssertionError("Inputs changed during read-only review")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("x", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False, allow_nan=False)
        f.write("\n")
    print(json.dumps({"output": str(OUT), "normal": report["genuine_free_incident_split"]["normal_direct"]["maxabs"],
        "anomalous": report["genuine_free_incident_split"]["anomalous_direct"]["maxabs"],
        "covariance_bug_confirmed": report["quadrature_covariance_counterexample"]["bug_confirmed"],
        "inputs_unchanged": report["all_tracked_inputs_unchanged"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
