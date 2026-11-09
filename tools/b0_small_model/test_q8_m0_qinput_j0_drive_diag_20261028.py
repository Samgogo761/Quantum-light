#!/usr/bin/env python3
"""Static checks for the J0-displacement diagnostic. No long propagation."""
from __future__ import annotations

import json
import unittest
from pathlib import Path

import numpy as np

import q8_m0_qinput_dynamics_20261026 as dyn
import q8_m0_qinput_j0_drive_diag_20261028 as diag
import q8_m0_qinput_minimal_20261020 as qm
from q8_m0_model import R0, build_geometry

ROOT = Path(__file__).resolve().parents[2]


class J0DriveDiagTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.geo = build_geometry()
        cls.k = cls.geo.k_cart(dyn.K_FRAC)
        cls.env = dyn.Envelope(dyn.TRIAL_OMEGA, ramp_cycles=1.0, flat_cycles=2.0)
        cls.model = dyn.build_joint(
            cls.geo, R0, cls.k, 1, n_max=3, omega0=dyn.TRIAL_OMEGA, g=diag.G_DIAG)

    def test_polarization_is_a2_not_an_assumed_vector(self):
        e = qm.a2_polarization(self.geo)
        rec = diag.polarization_record(self.geo, e)
        self.assertLess(abs(np.linalg.norm(e) - 1.0), 1e-15)
        self.assertEqual(rec["source"].split(",")[0], "a2_cart / |a2_cart|")
        a2 = np.asarray(self.geo.a2_cart, float)[:2]
        self.assertLess(np.max(np.abs(e - a2 / np.linalg.norm(a2))), 1e-15)

    def test_fk_three_paths_agree(self):
        occ = dyn.electron_ground_state(self.geo, R0, self.k, 1)
        el = qm.slater_state(occ)
        el = el / np.linalg.norm(el)
        psi = dyn.product_state(self.model, occ, np.array([1.0, 0, 0, 0], complex))
        mean_j = np.array(dyn.snapshot(self.model, psi, 0.0)["mean_J_re"], float)
        fk_j = diag.fk_from_current(diag.G_DIAG, self.model.e, mean_j)
        fk_g = diag.fk_from_groups(self.model, el)
        eps = 1e-6
        fk_fd = (diag.electronic_energy_classical_x(self.model, el, eps)
                 - diag.electronic_energy_classical_x(self.model, el, -eps)) / (2 * eps)
        self.assertLess(abs(fk_g.imag), 1e-10)
        self.assertLess(abs(fk_fd.imag), 1e-8)
        self.assertLess(abs(fk_g.real - fk_j), 1e-10)
        self.assertLess(abs(fk_fd.real - fk_j), 1e-8)

    def test_fk_is_coefficient_of_sX_not_force_at_s0(self):
        occ = dyn.electron_ground_state(self.geo, R0, self.k, 1)
        el = qm.slater_state(occ)
        el = el / np.linalg.norm(el)
        x = 1e-4
        e_s0_plus = 0.0j
        onsite = self.model.diag.reshape(qm.ELECTRON_DIM, self.model.d1)[:, 0]
        e0 = np.vdot(el, onsite * el)
        for F in self.model.f_group:
            e0 = e0 + np.vdot(el, F @ el)
        self.assertLess(abs(diag.electronic_energy_classical_x(self.model, el, x) - e0), 1.0)
        self.assertGreater(abs(diag.fk_from_groups(self.model, el)), 1e-4)

    def test_oscillator_constant_s_matches_closed_form(self):
        class Flat:
            def s(self, t):
                return 1.0
        F, omega, alpha0, t = 0.07, 0.05, 0.5 + 0.0j, 20.0
        n_num, a_num = diag.oscillator_series(alpha0, lambda _t: F, Flat(), [t], omega)
        alpha_eq = -F / omega
        alpha = alpha_eq + (alpha0 - alpha_eq) * np.exp(-1j * omega * t)
        self.assertLess(abs(n_num[0] - abs(alpha) ** 2), 1e-8)
        self.assertLess(abs(a_num[0]["re"] + 1j * a_num[0]["im"] - alpha), 1e-8)

    def test_archives_external_work_is_energy_difference(self):
        pre = json.loads((ROOT / "results/q_line/qinput_a_truncation_preflight_20261027/preflight.json").read_text(encoding="utf-8"))
        series = pre["jobs"]["12"]["time_series"]
        e0 = series["e_total"][0]
        for e, w in zip(series["e_total"], series["external_work"]):
            self.assertLess(abs((e - e0) - w), 1e-14)
        trial = json.loads((ROOT / "results/q_line/qinput_a_dynamics_trial_20261026/trial.json").read_text(encoding="utf-8"))
        job = trial["jobs"]["coherent"]
        self.assertIn("external_work_Delta_energy", job)
        self.assertAlmostEqual(job["external_work_Delta_energy"],
                               job["after_fall"]["energy"] - job["initial"]["energy"],
                               places=12)

    def test_zero_coupling_archive_is_not_vacuum_input(self):
        trial = json.loads((ROOT / "results/q_line/qinput_a_dynamics_trial_20261026/trial.json").read_text(encoding="utf-8"))
        z = trial["jobs"]["zero"]
        self.assertEqual(z["g"], 0.0)
        self.assertEqual(z["input"]["kind"], "coherent")
        self.assertLess(abs(z["delta_n_ph_exact"]), 1e-12)


if __name__ == "__main__":
    unittest.main()
