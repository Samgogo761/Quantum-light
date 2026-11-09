#!/usr/bin/env python3
"""Small static tests only; no dynamics, BZ scan or physical BSV claims.

python this_file.py runs unittest without writing artifacts.
python this_file.py --report results/q_line/qinput_minimal_preflight_20261020/report.json
runs the same tests and writes a new exclusive-create JSON; never overwrites.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from itertools import combinations
from pathlib import Path
import sys
import time
import unittest

import numpy as np
from scipy.linalg import expm

import q8_m0_qinput_minimal_20261020 as qm
from q8_m0_model import R0, build_geometry, current_ops, h8, op_P, op_theta_unitary


EVIDENCE = {}


def maxabs(x):
    return float(np.max(np.abs(x)))


class QuantumInputStaticTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.geo = build_geometry()
        cls.k = cls.geo.k_cart([0.173, 0.271])
        cls.n_max = 8
        cls.omega = 0.05
        cls.g = 0.02  # abstract method-test parameter, not experimental normalization

    def test_01_scalar_model_and_current(self):
        errors = []
        for neel in (-1, 1):
            for sign in (-1, 1):
                k = sign * self.k
                for x in (0.0, 0.37):
                    kap = k + self.g * x * qm.a2_polarization(self.geo)
                    hs, js = qm.scalar_h_and_current(self.geo, R0, kap, neel)
                    errors += [maxabs(hs - h8(self.geo, R0, kap, neel)),
                               maxabs(js - current_ops(self.geo, R0, kap, neel))]
        EVIDENCE["scalar_model_and_current_max_abs"] = max(errors)
        self.assertLess(max(errors), 2e-13)

    def test_02_fermion_lift_and_slater(self):
        h = h8(self.geo, R0, self.k, 1)
        energies, vectors = np.linalg.eigh(h)
        h70 = qm.lift_one_body(h)
        target = sorted(sum(energies[list(det)]) for det in combinations(range(8), 4))
        error = maxabs(np.linalg.eigvalsh(h70) - target)
        psi = qm.slater_state(vectors[:, :4])
        residual = maxabs(h70 @ psi - sum(energies[:4]) * psi)
        number = sum(qm.fermion_bilinear(i, i) for i in range(8))
        number_error = maxabs(number - 4 * np.eye(70))
        EVIDENCE["fermion_lift"] = {"eigensum_max_abs": error, "slater_residual": residual,
                                    "slater_mass_error": abs(float(np.vdot(psi, psi).real) - 1),
                                    "fixed_number_error": number_error}
        self.assertLess(error, 5e-13)
        self.assertLess(residual, 5e-13)
        self.assertLess(abs(np.vdot(psi, psi) - 1), 5e-13)
        self.assertEqual(number_error, 0.0)

    def test_03_joint_hermiticity_spin_and_inversion(self):
        kw = dict(n_max=self.n_max, omega=self.omega, g=self.g)
        h, j = qm.assemble_joint(self.geo, R0, self.k, 1, **kw)
        hm, jm = qm.assemble_joint(self.geo, R0, -self.k, -1, **kw)
        gp = qm.wedge_unitary(op_P())
        perm_e = np.argmax(np.abs(gp), axis=0)
        phase_e = gp[perm_e, np.arange(70)]
        d = self.n_max + 1
        perm = np.array([i * d + n for i in perm_e for n in range(d)])
        phase = np.array([s * (-1) ** n for s in phase_e for n in range(d)])
        mapped_h = phase[:, None] * h * phase.conj()[None, :]
        mapped_j = phase[None, :, None] * j * phase.conj()[None, None, :]
        inversion_h = maxabs(mapped_h - hm[np.ix_(perm, perm)])
        inversion_j = maxabs(mapped_j + jm[:, perm][:, :, perm])
        antiunitary_errors = {}
        # Global complex conjugation uses the fixed orbital and real number
        # bases. X is a time-odd vector potential: T includes photon parity.
        # These are static operator identities, not driven-protocol symmetries.
        for name, onebody, photon_parity, target_h, target_j, jsign in (
            ("T", op_theta_unitary(), True, hm, jm, -1),
            ("PT", op_P() @ op_theta_unitary(), False, h, j, +1),
        ):
            ge = qm.wedge_unitary(onebody)
            pe = np.argmax(np.abs(ge), axis=0)
            se = ge[pe, np.arange(70)]
            pi = np.array([i * d + n for i in pe for n in range(d)])
            si = np.array([s * ((-1) ** n if photon_parity else 1)
                           for s in se for n in range(d)])
            th = si[:, None] * h.conj() * si.conj()[None, :]
            tj = si[None, :, None] * j.conj() * si.conj()[None, None, :]
            antiunitary_errors[name] = {
                "H_max_abs": maxabs(th - target_h[np.ix_(pi, pi)]),
                "J_max_abs": maxabs(tj - jsign * target_j[:, pi][:, :, pi]),
            }
        sz_e = np.array([sum(1 if i % 2 == 0 else -1 for i in det) for det in qm.DETS])
        sz = np.repeat(sz_e, d)
        spin_error = maxabs((sz[:, None] - sz[None, :]) * h)
        hermiticity = max(maxabs(h - h.conj().T), maxabs(j - j.conj().transpose(0, 2, 1)))
        EVIDENCE["joint_static"] = {"dimension": len(h), "g_Angstrom_inverse": self.g,
                                     "hermiticity_max_abs": hermiticity,
                                     "spin_commutator_max_abs": spin_error,
                                     "inversion_H_max_abs": inversion_h,
                                     "inversion_J_max_abs": inversion_j,
                                     "static_antiunitary_identities": antiunitary_errors,
                                     "nup2_ndown2_electron_dimension": int(np.count_nonzero(sz_e == 0))}
        self.assertLess(hermiticity, 2e-13)
        self.assertLess(inversion_h, 2e-12)
        self.assertLess(inversion_j, 2e-12)
        for errors in antiunitary_errors.values():
            self.assertLess(errors["H_max_abs"], 2e-12)
            self.assertLess(errors["J_max_abs"], 2e-12)
        self.assertEqual(spin_error, 0.0)
        self.assertEqual(np.count_nonzero(sz_e == 0), 36)

    def test_04_zero_coupling(self):
        h, j = qm.assemble_joint(self.geo, R0, self.k, 1,
                                 n_max=3, omega=self.omega, g=0.0)
        hp = np.diag(self.omega * np.arange(4))
        he = qm.lift_one_body(h8(self.geo, R0, self.k, 1))
        exact = np.kron(he, np.eye(4)) + np.kron(np.eye(70), hp)
        je = current_ops(self.geo, R0, self.k, 1)
        jref = np.array([np.kron(qm.lift_one_body(a), np.eye(4)) for a in je])
        error = max(maxabs(h - exact), maxabs(j - jref))
        EVIDENCE["zero_coupling_max_abs"] = error
        self.assertLess(error, 3e-13)

    def test_05_projected_displacement_and_cutoff_algebra(self):
        q, cutoff, enlarged = 0.27, 8, 48
        d = cutoff + 1
        a, ad, _ = qm.boson_ops(cutoff)
        commutator_target = np.eye(d)
        commutator_target[-1, -1] -= d
        algebra = maxabs(a @ ad - ad @ a - commutator_target)
        ae, ade, _ = qm.boson_ops(enlarged)
        reference = expm(1j * q * (ae + ade))[:d, :d]
        projected = qm.projected_peierls(q, cutoff)
        independent_error = maxabs(projected - reference)
        compression = maxabs(projected - qm.projected_peierls(q, cutoff + 5)[:d, :d])
        wrong_order_difference = maxabs(projected - expm(1j * q * (a + ad)))
        adjoint = maxabs(projected.conj().T - qm.projected_peierls(-q, cutoff))
        EVIDENCE["photon_projection"] = {
            "cutoff_CCR_boundary_max_abs": algebra,
            "analytic_vs_enlarged_exponential_max_abs": independent_error,
            "nested_compression_max_abs": compression,
            "projection_vs_exponentiation_differ_max_abs": wrong_order_difference,
            "adjoint_max_abs": adjoint,
        }
        self.assertLess(algebra, 5e-14)
        self.assertLess(independent_error, 5e-13)
        self.assertLess(compression, 5e-14)
        self.assertGreater(wrong_order_difference, 1e-3)
        self.assertLess(adjoint, 5e-14)

    def test_06_low_occupation_input_moments(self):
        alpha, r, theta = 0.6 * np.exp(0.31j), 0.3, 0.43
        nsv = np.sinh(r) ** 2
        references = {
            "coherent": (abs(alpha) ** 2, abs(alpha) ** 4 + abs(alpha) ** 2, alpha ** 2),
            "squeezed_vacuum": (nsv, 3 * nsv ** 2 + 2 * nsv,
                                -np.exp(1j * theta) * np.sinh(r) * np.cosh(r)),
        }
        evidence = {}
        for name in references:
            ladder = []
            mean, second, aa = references[name]
            for cutoff in (8, 16, 24):
                c = (qm.coherent_coefficients(alpha, cutoff) if name == "coherent"
                     else qm.squeezed_vacuum_coefficients(r, theta, cutoff))
                item = qm.state_diagnostics(c)
                actual_aa = complex(item["raw_aa"]["real"], item["raw_aa"]["imag"])
                item.update(n_max=cutoff, mean_abs_error=abs(item["raw_mean_n"] - mean),
                            second_abs_error=abs(item["raw_second_n"] - second),
                            aa_abs_error=abs(actual_aa - aa))
                ladder.append(item)
            evidence[name] = {"reference_mean": mean, "reference_second_n": second,
                              "reference_aa": {"real": float(aa.real), "imag": float(aa.imag)},
                              "cutoffs": ladder}
            self.assertLess(ladder[-1]["omitted_mass"], 1e-12)
            self.assertLess(ladder[-1]["mean_abs_error"], 2e-12)
            self.assertLess(ladder[-1]["second_abs_error"], 2e-11)
            self.assertLess(ladder[-1]["aa_abs_error"], 2e-12)
            self.assertGreaterEqual(ladder[0]["omitted_mass"], ladder[1]["omitted_mass"] - 2e-15)
            self.assertGreaterEqual(ladder[1]["omitted_mass"], ladder[2]["omitted_mass"] - 2e-15)
        EVIDENCE["input_state_moment_ladders"] = evidence

    def test_07_joint_current_and_nested_compression(self):
        kw = dict(omega=self.omega, g=self.g)
        h, j = qm.assemble_joint(self.geo, R0, self.k, 1, n_max=2, **kw)
        larger_h, larger_j = qm.assemble_joint(self.geo, R0, self.k, 1, n_max=4, **kw)
        keep = np.array([i * 5 + n for i in range(70) for n in range(3)])
        nested = max(maxabs(h - larger_h[np.ix_(keep, keep)]),
                     maxabs(j - larger_j[:, keep][:, :, keep]))
        step, errors = 1e-6, []
        for axis in range(2):
            delta = np.zeros(2)
            delta[axis] = step
            hp, _ = qm.assemble_joint(self.geo, R0, self.k + delta, 1, n_max=2, **kw)
            hm, _ = qm.assemble_joint(self.geo, R0, self.k - delta, 1, n_max=2, **kw)
            errors.append(maxabs(j[axis] + (hp - hm) / (2 * step)))
        EVIDENCE["joint_current_and_compression"] = {
            "nested_joint_compression_max_abs": nested,
            "probe_derivative_step_Angstrom_inverse": step,
            "probe_derivative_max_abs": max(errors),
        }
        self.assertLess(nested, 3e-13)
        self.assertLess(max(errors), 3e-8)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    if args.report is not None and args.report.exists():
        parser.error("report already exists; refusing to overwrite")
    start = time.perf_counter()
    result = unittest.TextTestRunner(verbosity=2).run(
        unittest.defaultTestLoader.loadTestsFromTestCase(QuantumInputStaticTests))
    report = {
        "model_id": qm.MODEL_ID, "all_checks_pass": result.wasSuccessful(),
        "record_date": "2026-10-20",
        "tests_run": result.testsRun, "wall_seconds": time.perf_counter() - start,
        "k_fractional": [0.173, 0.271], "electron_dimension": 70,
        "scope": "static_single_k_method_preflight_only",
        "not_claimed": ["time_propagation", "dynamical_Fock_convergence", "BSV_experimental_normalization",
                        "joint_ground_state_at_nonzero_coupling", "physical_pulse_mode",
                        "crystal_HHG_selection_rules", "G1_pass", "production_changes"],
        "files": {str(Path(p).name): hashlib.sha256(Path(p).read_bytes()).hexdigest()
                  for p in (qm.__file__, __file__)},
        "frozen_model_and_geometry_sources": getattr(
            getattr(QuantumInputStaticTests, "geo", None), "source_hashes", {}),
        "evidence": EVIDENCE,
        "failures": [{"test": str(test), "traceback": trace} for test, trace in result.failures],
        "errors": [{"test": str(test), "traceback": trace} for test, trace in result.errors],
    }
    if args.report is not None:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        with args.report.open("x", encoding="utf-8") as fh:
            json.dump(report, fh, ensure_ascii=False, indent=2, allow_nan=False)
            fh.write("\n")
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    sys.exit(main())
