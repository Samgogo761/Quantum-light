"""Sensitivity of R_N and R_A to the undriven boundary mismatch.

epsilon_u = raw_undriven - radiated sheet field, on the final nine panels
plus the infrared tail, with production weights. The radiated field is
phi*(sqrt(omega)/eta)*(A-A_f) on every port. Replacing the subtracted
undriven integral by that field would change production U by +epsilon_u.
That replacement is not applied and is not an approved correction.

The infinite-time undriven anomalous target used for the sensitivity is 0,
because the undriven generator is time-independent and the packet integrates
positive frequencies. The code stores no stationary V0. Matrices that do not
use delta_V are reported separately.
"""
from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np

import q8_m0_gamma12_input_wing_20261008 as wing
import q8_m0_gamma12_transpose_input_refine_20261008 as base
from q8_m0_audit_artifacts_20261014 import provenance, write_new_json
from q8_m0_classical import p1_a2_protocol
from q8_m0_continuum_columns_20261004 import current
from q8_m0_g1_dynamic_map_audit_20261015 import require_cache
from q8_m0_g1_gamma12_ir_tail_20261015 import POLE, load_adjoint, log_gauss
from q8_m0_g1_gamma12_richardson_2048_20261015 import final_frequencies
from q8_m0_g1_gamma12_richardson_corrected_20261015 import select_panels
from q8_m0_g1_gamma12_richardson_adjoint_20261015 import contract_initial
from q8_m0_g1_gamma12_tail_columns_20261014 import BASELINE, load_rows

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results/q_line/g1_gamma12_time_order_20261015/boundary_sensitivity.json"
PACKET_PHI = 1.0e-3
KNOWN_WEIGHT1_GAP = 6.013361711968586e-6


def pack(m):
    m = np.asarray(m, complex)
    return {"real": np.real(m).tolist(), "imag": np.imag(m).tolist(),
            "maxabs": float(np.max(np.abs(m)))}


def zeros():
    return np.zeros((2, 2), complex)


class Bucket:
    def __init__(self):
        self.n_columns = 0
        self.weight = 0.0
        self.max_abs_eps_u = 0.0
        self.max_abs_eps_v = 0.0
        self.uu = zeros()
        self.vv = zeros()
        self.uv = zeros()
        self.vu = zeros()
        self.du_du = zeros()
        self.dv_dv = zeros()
        self.u_du = zeros()
        self.du_u = zeros()
        self.v_dv = zeros()
        self.dv_v = zeros()
        self.du_v = zeros()
        self.v_du = zeros()
        self.u_dv = zeros()
        self.dv_u = zeros()
        self.du_dv = zeros()
        self.dv_du = zeros()

    def add(self, u, v, du, dv, weight):
        self.n_columns += int(u.shape[1])
        self.weight += float(np.sum(weight))
        self.max_abs_eps_u = max(self.max_abs_eps_u, float(np.max(np.abs(du))) if du.size else 0.0)
        self.max_abs_eps_v = max(self.max_abs_eps_v, float(np.max(np.abs(dv))) if dv.size else 0.0)
        self.uu += u @ u.conj().T
        self.vv += v @ v.conj().T
        self.uv += u @ v.T
        self.vu += v @ u.T
        self.du_du += du @ du.conj().T
        self.dv_dv += dv @ dv.conj().T
        self.u_du += u @ du.conj().T
        self.du_u += du @ u.conj().T
        self.v_dv += v @ dv.conj().T
        self.dv_v += dv @ v.conj().T
        self.du_v += du @ v.T
        self.v_du += v @ du.T
        self.u_dv += u @ dv.T
        self.dv_u += dv @ u.T
        self.du_dv += du @ dv.T
        self.dv_du += dv @ du.T

    def report(self):
        cross_u = self.u_du + self.du_u
        cross_v = self.v_dv + self.dv_v
        delta_n_u = cross_u + self.du_du
        delta_n_v = -(cross_v + self.dv_dv)
        delta_n = delta_n_u + delta_n_v
        delta_a = (self.du_v + self.u_dv + self.du_dv
                   - self.v_du - self.dv_u - self.dv_du)
        residual_n = self.uu - self.vv - np.eye(2)
        residual_a = self.uv - self.vu
        return {
            "n_columns": self.n_columns,
            "weight_sum": self.weight,
            "max_abs_delta_U": self.max_abs_eps_u,
            "max_abs_delta_V": self.max_abs_eps_v,
            "self_deltaU": pack(self.du_du),
            "self_deltaV": pack(self.dv_dv),
            "cross_U_deltaU": pack(cross_u),
            "cross_V_deltaV": pack(cross_v),
            "delta_RN_from_deltaU_only": pack(delta_n_u),
            "delta_RN_from_deltaV_only": pack(delta_n_v),
            "delta_RN_joint": pack(delta_n),
            "delta_RA_joint": pack(delta_a),
            "photon_RN_before_correction": pack(residual_n),
            "photon_RA_before_correction": pack(residual_a),
            "photon_RN_after_hypothesis": pack(residual_n + delta_n),
            "photon_RA_after_hypothesis": pack(residual_a + delta_a),
            "hermitian_defect_delta_RN": float(np.max(np.abs(delta_n - delta_n.conj().T))),
        }


def radiated_from(ph, phi_of, eta, pol):
    field = ph.initial.A - ph.free_A0
    amplitude = (np.sqrt(ph.omega) / eta)[None, :] * field
    phi = np.asarray(phi_of(ph.omega), complex)
    return phi[None, :] * (pol @ amplitude)


def sheet_gap(cache, gamma, ph):
    state_current = current(ph.initial, cache["u0"], np.asarray(cache["D0"], float))
    predicted = 1j * gamma * state_current / ph.omega[None, :]
    return float(np.max(np.abs(ph.initial.A - ph.free_A0 - predicted)))


def right_columns(nfreq):
    return np.array([4 * i + k for i in range(nfreq) for k in (2, 3)], int)


def consume(bucket_all, bucket_in, bucket_out, ph, u0, driven, undriven, phi_abs, eta, pol, weight):
    radiated = radiated_from(ph, phi_abs, eta, pol)
    right = right_columns(ph.omega.size // 4)
    port_gap = float(np.max(np.abs(u0[:, right] - radiated[:, right])))
    same = float(np.max(np.abs(radiated[:, right] - radiated[:, right - 2])))
    u = u0 + driven[0] - undriven[0]
    v = driven[1] - undriven[1]
    delta_u = undriven[0] - radiated
    delta_v = undriven[1]
    mask = np.repeat(np.abs(phi_abs(ph.omega[0::4])) >= PACKET_PHI, 4)
    for bucket, cols, wsel in (
        (bucket_all, slice(None), weight),
        (bucket_in, mask, np.asarray(weight)[mask[0::4]]),
        (bucket_out, ~mask, np.asarray(weight)[~mask[0::4]]),
    ):
        if u[:, cols].shape[1] == 0:
            continue
        bucket.add(u[:, cols], v[:, cols], delta_u[:, cols], delta_v[:, cols], wsel)
    return port_gap, same


def frequencies(cache, gamma):
    with np.load(BASELINE, allow_pickle=False) as saved:
        saved_oms, saved_wts = np.array(saved["oms"]), np.array(saved["wts"])
    tails, wing_panel, adjacent, core, big, identities = select_panels(
        cache, gamma, saved_oms, saved_wts)
    pieces = final_frequencies(
        saved_oms, saved_wts, tails, wing_panel, adjacent, core, big, identities)
    panels = []
    wing.wing_quadrature(cache["gaps"], cache["u0"], gamma, n_panel=160, wing_factor=2,
                         omega_lo=1.0e-4, omega_hi=12.0, panels=panels)
    panel = next(item for item in panels if item["kind"] == "sinh_tail"
                 and abs(item["pole"] - POLE) < 1e-10 and item["omega_hi"] < 0.05)
    ir_om, ir_wt = log_gauss(1.0e-8, panel["omega_lo"], 64)
    pieces.append(("infrared_1e-8_log64", ir_om, ir_wt))
    return pieces, float(panel["omega_lo"])


def stack_forces(forces):
    grouped = {}
    for force, t, h in forces:
        grouped.setdefault(h, []).append((force, t))
    stacked = []
    for h, pairs in grouped.items():
        stacked.append((
            float(h),
            np.stack([force for force, _t in pairs]),
            np.array([t for _force, t in pairs], float),
        ))
    return stacked


def contract_stacked(row, stacked, b, bdag, a, free, omega):
    raw_u, raw_v = contract_initial(row, b, bdag, a)
    for h, forces, times in stacked:
        phase = np.exp(-1j * omega * times[:, None]) * np.expm1(-1j * omega * h)
        raw = np.einsum("sab,bc,sc->ac", forces, free, phase, optimize=True)
        raw_u = raw_u + raw[:2]
        raw_v = raw_v + np.conjugate(raw[2:])
    return raw_u, raw_v


def weight1_regression(stacked_u, row_u, force_u, cache, gamma, pol, phi_of, eta):
    from q8_m0_g1_gamma12_richardson_full_20261015 import accumulate_arrays
    ph, _u0 = base.generated_photon_block(
        cache, gamma, np.array([0.11]), np.array([1.0]), pol, phi_of, eta)
    raw_u, raw_v = contract_stacked(
        row_u, stacked_u, ph.initial.b, ph.initial.bdag, ph.initial.A, ph.free_A0, ph.omega)
    ref_u, ref_v = accumulate_arrays(
        row_u, force_u, ph.initial.b, ph.initial.bdag, ph.initial.A, ph.free_A0, ph.omega)
    if max(float(np.max(np.abs(raw_u - ref_u))), float(np.max(np.abs(raw_v - ref_v)))) > 1e-10:
        raise AssertionError("Stacked contraction disagrees with the explicit accumulator")
    gap = float(np.max(np.abs(raw_u - radiated_from(ph, phi_of, eta, pol))))
    if abs(gap - KNOWN_WEIGHT1_GAP) > 1e-12:
        raise AssertionError(f"Weight-1 radiated gap changed: {gap}")
    return gap


def run_macro(macro, pieces, cache, gamma, pol, phi_of, eta):
    path, digest = require_cache(macro)
    row_d, force_d, row_u, force_u = load_adjoint(path)
    stacked_d = stack_forces(force_d)
    stacked_u = stack_forces(force_u)
    if macro == 2048:
        weight1_regression(stacked_u, row_u, force_u, cache, gamma, pol, phi_of, eta)
    buckets = {"all": Bucket(), "packet_phi": Bucket(), "outside_packet": Bucket()}
    sheet = 0.0
    port = 0.0
    copy = 0.0
    for name, om, wt in pieces:
        print(macro, name, len(om), flush=True)
        for start in range(0, len(om), 200):
            stop = min(start + 200, len(om))
            if start % 4000 == 0:
                print(macro, name, stop, flush=True)
            ph, u0 = base.generated_photon_block(
                cache, gamma, om[start:stop], wt[start:stop], pol, phi_of, eta)
            driven = contract_stacked(
                row_d, stacked_d, ph.initial.b, ph.initial.bdag, ph.initial.A, ph.free_A0, ph.omega)
            undriven = contract_stacked(
                row_u, stacked_u, ph.initial.b, ph.initial.bdag, ph.initial.A, ph.free_A0, ph.omega)
            sheet = max(sheet, sheet_gap(cache, gamma, ph))
            one_port, one_copy = consume(
                buckets["all"], buckets["packet_phi"], buckets["outside_packet"],
                ph, u0, driven, undriven, phi_of, eta, pol, wt[start:stop])
            port = max(port, one_port)
            copy = max(copy, one_copy)
    if sheet > 1e-8 or port > 1e-8:
        raise AssertionError(f"Sheet or right-port target drifted: {sheet}, {port}")
    return {
        "cache_sha256": digest,
        "sheet_A_minus_Af_minus_i_gamma_I_over_omega": sheet,
        "right_port_U0_minus_radiated": port,
        "left_radiated_minus_right_radiated": copy,
        "buckets": {name: bucket.report() for name, bucket in buckets.items()},
    }


def main():
    if OUT.exists():
        raise FileExistsError(OUT)
    _saved, cache, t0, _meta = load_rows()
    if abs(float(t0)) > 0:
        raise AssertionError(t0)
    gamma = float(np.asarray(cache["gamma"]))
    pol = np.array(cache["pol"], copy=True)
    prot = p1_a2_protocol(0.8)
    grid = base.q.omega_grid(prot.omega0, prot.t_pulse)
    grid = grid[grid > 0.0]
    _phi, norm = base.q.normalize_phi(grid, base.q.phi_unnormalized(grid, 2.0 * prot.omega0, prot.t_pulse))
    phi_of = lambda nu: base.q.phi_at(nu, prot, norm)
    eta = base.q.eta_of(gamma)
    pieces, edge = frequencies(cache, gamma)
    levels = {}
    for macro in (2048, 1024):
        levels[str(macro)] = run_macro(macro, pieces, cache, gamma, pol, phi_of, eta)
    write_new_json(OUT, {
        "hypothesis_not_an_approved_replacement": True,
        "delta_U": "raw_undriven_u - phi*(sqrt(omega)/eta)*(A-A_f)",
        "delta_V": "raw_undriven_v - 0; infinite-time target, no stationary V0 is stored",
        "production_map": "U = U0 + raw_driven_u - raw_undriven_u; V = raw_driven_v - raw_undriven_v",
        "expansion_RN": "U dU† + dU U† + dU dU† - V dV† - dV V† - dV dV†",
        "packet_phi_threshold": PACKET_PHI,
        "infrared_upper_edge": edge,
        "dark_columns_not_in_this_sum": True,
        "levels": levels,
        "production_G1_pass": False,
        "provenance_sha256": provenance([__file__]),
    })
    joint = levels["2048"]["buckets"]["all"]["delta_RN_joint"]["maxabs"]
    print("delta RN", joint, "sheet", levels["2048"]["sheet_A_minus_Af_minus_i_gamma_I_over_omega"], flush=True)


if __name__ == "__main__":
    main()
