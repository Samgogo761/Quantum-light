"""Match the production sheet law to the input and output normalizations.

(q, p, A) in the previous audit are not assumed to be a closed canonical
system. This checks whether gamma, eta, the free-field amplitude, the
stationary radiation response, and the H2 current kernel are the same
electromagnetic constants. The prescribed-current DFT-box test does not
by itself qualify elimination of the quantum continuum.
"""
from __future__ import annotations

import json
import argparse
import os
from pathlib import Path

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
from q8_m0_audit_artifacts_20261014 import provenance, write_new_json

import q8_m0_p1_h2_finite_kcv_input_close_20261005 as q
from q8_m0_classical import p1_a2_protocol
from q8_m0_continuum_columns_20261004 import ColumnState, current, stationary_photon_columns
from q8_m0_gself_emerge import emerge_A_from_I, emerge_on_basis, long_window_bins
from q8_m0_gsheet import gamma_self
from q8_m0_model import build_geometry
from q8_m0_p1_h2_finite_kcv_gamma12_backend_short_20261006 import kernel_at_times
from q8_m0_phgauss import h2_two_port_packets
from q8_m0_s1_kernel import parallel_perp_basis

ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT / "results/q_line/p1_h2_finite_kcv_gamma12_local_plusN_20261007"
OUT = ROOT / "results/q_line/gamma12_output_transpose_input_refine_20261008"
PROBE = (0.05, 0.2, 0.5, 1.5)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=OUT / "reduction_match_20261014.json")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Use --output with a new path: {args.output}")
    geo = build_geometry()
    gamma_geo = float(gamma_self(geo))
    gamma = float(np.load(RUN / "arrays/gamma.npy"))
    eta = q.eta_of(gamma)
    gaps = np.load(RUN / "arrays/gaps.npy")
    u0 = np.load(RUN / "arrays/u0.npy")
    d0 = np.load(RUN / "arrays/D0.npy")
    k_saved = np.load(RUN / "arrays/k.npy")
    h = float(np.load(RUN / "arrays/h.npy"))
    if abs(gamma - gamma_geo) > 1e-15:
        raise AssertionError(f"Stored gamma {gamma} differs from gamma_self {gamma_geo}")
    if abs(gamma - 2.0 * np.pi * eta * eta) > 1e-15:
        raise AssertionError("eta is not sqrt(gamma/2pi)")

    probes = []
    for omega in PROBE:
        if np.min(np.abs(gaps - omega)) < 1e-6:
            raise AssertionError(f"Probe {omega} sits on an electronic gap")
        ref = stationary_photon_columns(gaps, u0, d0, gamma, [omega])
        eye = np.eye(2)
        for col, side, pol in ((0, "L", 0), (1, "L", 1), (2, "R", 0), (3, "R", 1)):
            state = ColumnState(ref.initial.b[:, col:col + 1],
                                ref.initial.bdag[:, col:col + 1],
                                ref.initial.A[:, col:col + 1])
            ii = current(state, u0, d0)[:, 0]
            a = ref.initial.A[:, col]
            af = ref.free_A0[:, col]
            self_field = a - af
            maxwell = 1j * gamma * ii / omega
            free_round = np.sqrt(omega) / eta * af
            out_self = np.sqrt(omega) / eta * (a if side == "L" else self_field)
            kernel_self = 1j * 2.0 * np.pi * eta / np.sqrt(omega) * ii
            transmitted = eye[:, pol]
            expected = transmitted + kernel_self if side == "L" else kernel_self
            probes.append({
                "omega": omega, "side": side, "polarization": pol,
                "maxwell_A_minus_Af": float(np.max(np.abs(self_field - maxwell))),
                "free_amplitude_roundtrip": float(np.max(np.abs(free_round - transmitted))),
                "output_map_versus_current_kernel": float(np.max(np.abs(out_self - expected))),
            })

    prot = p1_a2_protocol(q.A_STAR)
    sample_times = (np.array([10, 100, 1000, 4000, 8000]) + 0.5) * h
    k_new, nrm = kernel_at_times(gamma, prot, sample_times)
    k_old = k_saved[np.array([10, 100, 1000, 4000, 8000])]
    kernel_gap = float(np.max(np.abs(k_new - k_old)))

    e_par, e_perp = parallel_perp_basis(geo)
    windows = {}
    for ratio in (1.0, 2.0):
        pack = long_window_bins(prot, e_perp, t_obs_over_tp=ratio, ntpc=8, harm_max=4.0)
        rec = emerge_on_basis(
            pack["phis"], pack["omega"], pack["times"], pack["dw"],
            pack["i_cell"], pack["dt"], geo, 1.0, n_pulse=pack["n_pulse"])
        windows[f"Tobs_over_Tp_{ratio:g}"] = {
            "n_positive_frequencies": int(pack["n_omega"]),
            "n_oscillators_two_ports": int(rec["n_ph"]),
            "A0": rec["compare"]["A0"],
            "rel_maxabs_A": rec["compare"]["rel_maxabs_A"],
            "l2_rel_A": rec["compare"]["l2_rel_A"],
            "maxabs_dA_minus_gamma_I_past": rec["compare"]["maxabs_dA_minus_gamma_I_past"],
        }
        print(ratio, windows[f"Tobs_over_Tp_{ratio:g}"], flush=True)
    pack = long_window_bins(prot, e_perp, t_obs_over_tp=2.0, ntpc=8, harm_max=4.0)
    flist, _labs = h2_two_port_packets(
        pack["times"], prot.omega0, prot.t_pulse, e_perp, pack["dw"], pack["omega"])
    a_two = emerge_A_from_I(flist, pack["i_cell"], pack["dt"], eta, eta)
    a_mx = emerge_on_basis(
        pack["phis"], pack["omega"], pack["times"], pack["dw"],
        pack["i_cell"], pack["dt"], geo, 1.0, n_pulse=pack["n_pulse"])["A_mx"]
    sl = slice(0, pack["n_pulse"])
    two = {
        "n_oscillators": 2,
        "rel_maxabs_A_on_source_window": float(
            np.max(np.abs(a_two[sl] - a_mx[sl])) / max(np.max(np.abs(a_mx[sl])), 1e-30)),
    }
    report = {
        "provenance_sha256": provenance([__file__, *[RUN / "arrays" / name
            for name in ("gamma.npy", "gaps.npy", "u0.npy", "D0.npy", "k.npy", "h.npy")]]),
        "variables": {
            "b_bdag": "Coefficients of one incoming annihilator in the particle-hole annihilation and creation coordinates. bdag is stored independently and is not set to conj(b).",
            "A": "Coefficient of the two in-plane sheet vector potentials for that same incoming annihilator.",
            "A_f": "Free incoming field at the sheet, amplitude eta/sqrt(omega) on each port and polarization.",
            "q_p_previous_audit": "Real and imaginary parts used only while searching for a symplectic form. They are not extra stored canonical partners of the photon continuum.",
            "photon_Q_P": "Not stored. The production step keeps the sheet field and the incoming free field after the continuum has been eliminated.",
        },
        "gamma": gamma,
        "gamma_self_geometry": gamma_geo,
        "eta": eta,
        "gamma_equals_2pi_eta_squared": True,
        "stationary_columns": probes,
        "stationary_maxwell_maxabs": float(max(row["maxwell_A_minus_Af"] for row in probes)),
        "free_roundtrip_maxabs": float(max(row["free_amplitude_roundtrip"] for row in probes)),
        "output_versus_current_kernel_maxabs": float(max(
            row["output_map_versus_current_kernel"] for row in probes)),
        "detector_kernel_versus_saved_maxabs": kernel_gap,
        "detector_kernel_normalizer": nrm,
        "detector_prefactor": "i*eta*integral d omega phi/sqrt(omega) exp(+i omega t)",
        "propagation_law": "dA/dt = dA_f/dt + gamma*I, I = u.conj()@b + u@bdag - D@A",
        "output_assembly": "Stationary U0 uses sqrt(omega)/eta times the right-going sheet field. The driven rows add the time integral of the same-eta kernel against the current, then subtract the undriven current.",
        "continuum_elimination": windows,
        "two_h2_oscillators": two,
        "production_qualification": False,
    }
    path = args.output
    write_new_json(path, report)
    print("maxwell", report["stationary_maxwell_maxabs"])
    print("roundtrip", report["free_roundtrip_maxabs"])
    print("kernel map", report["output_versus_current_kernel_maxabs"])
    print("saved k", kernel_gap)
    print("two oscillators", two["rel_maxabs_A_on_source_window"])
    print("wrote", path)


if __name__ == "__main__":
    main()
