#!/usr/bin/env python3
"""Fast dynamics-module checks. Not a BSV/HHG physics claim and not a long run."""
from __future__ import annotations

import unittest

import numpy as np
from scipy.linalg import expm

import q8_m0_qinput_dynamics_20261026 as dyn
import q8_m0_qinput_minimal_20261020 as qm
from q8_m0_model import R0, build_geometry


class DynamicsModuleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.geo = build_geometry()
        cls.k = cls.geo.k_cart([0.173, 0.271])
        cls.env = dyn.Envelope(0.05, ramp_cycles=1.0, flat_cycles=2.0)
        cls.model = dyn.build_joint(cls.geo, R0, cls.k, 1, n_max=3, omega0=0.05, g=0.02)

    def test_envelope_starts_and_ends_at_zero(self):
        e = self.env
        self.assertEqual(e.s(0.0), 0.0)
        self.assertEqual(e.s(e.t_total), 0.0)
        self.assertEqual(e.s(-1.0), 0.0)
        self.assertAlmostEqual(e.s(e.t_ramp + 0.5 * e.t_flat), 1.0)
        self.assertEqual(e.ds(e.t_ramp + 0.1 * e.t_flat), 0.0)
        self.assertGreater(e.ds(0.3 * e.t_ramp), 0.0)
        self.assertLess(e.ds(e.t_total - 0.3 * e.t_ramp), 0.0)

    def test_hann_closed_form(self):
        T = 12.0
        self.assertAlmostEqual(dyn.hann_window_transform(0.0, T).real, 0.5 * T, places=12)
        self.assertAlmostEqual(abs(dyn.hann_window_transform(0.0, T).imag), 0.0, places=12)
        pole = 2.0 * np.pi / T
        self.assertAlmostEqual(dyn.hann_window_transform(pole, T).real, -0.25 * T, places=10)
        t = np.linspace(0.0, T, 4001)
        w = np.sin(np.pi * t / T) ** 2
        om = 0.37
        dt = t[1] - t[0]
        num = np.sum(w * np.exp(1j * om * t)) * dt
        self.assertLess(abs(dyn.hann_window_transform(om, T) - num), 5e-4)

    def test_apply_h_matches_dense_including_s0_hopping(self):
        rng = np.random.default_rng(20261026)
        v = rng.normal(size=self.model.dim) + 1j * rng.normal(size=self.model.dim)
        v = v / np.linalg.norm(v)
        for s in (0.0, 0.4, 1.0):
            Hv = dyn.apply_h(self.model, v, s)
            Hd = dyn.dense_h(self.model, s) @ v
            self.assertLess(np.max(np.abs(Hv - Hd)), 2e-12)
        H0 = dyn.dense_h(self.model, 0.0)
        self.assertGreater(np.max(np.abs(H0 - np.diag(np.diag(H0)))), 1e-8)

    def test_taylor_matches_expm_small_dt(self):
        rng = np.random.default_rng(7)
        v = rng.normal(size=self.model.dim) + 1j * rng.normal(size=self.model.dim)
        v = v / np.linalg.norm(v)
        dt = 0.02
        s = 0.35
        vt, _, _ = dyn.taylor_step(self.model, v, s, dt, order=16)
        ve = expm(-1j * dyn.dense_h(self.model, s) * dt) @ v
        self.assertLess(np.max(np.abs(vt - ve)), 1e-10)

    def test_g0_is_s_independent(self):
        m = dyn.build_joint(self.geo, R0, self.k, 1, n_max=2, omega0=0.05, g=0.0)
        rng = np.random.default_rng(1)
        v = rng.normal(size=m.dim) + 1j * rng.normal(size=m.dim)
        self.assertLess(np.max(np.abs(dyn.apply_h(m, v, 0.0) - dyn.apply_h(m, v, 1.0))), 1e-12)

    def test_displaced_squeezed_interface_exists_and_is_not_normalized(self):
        c = dyn.displaced_squeezed_coefficients(0.4 + 0.2j, 0.7, 0.3, 8)
        mass = float(np.vdot(c, c).real)
        self.assertGreater(mass, 0.7)
        self.assertLess(mass, 1.0 + 1e-12)
        vac = dyn.galerkin_displacement(0.0, 4)
        self.assertLess(np.max(np.abs(vac - np.eye(5))), 1e-12)

    def test_source_spectra_uses_requested_s_flat(self):
        rng = np.random.default_rng(26)
        v = rng.normal(size=self.model.dim) + 1j * rng.normal(size=self.model.dim)
        v = v / np.linalg.norm(v)
        sp0 = dyn.source_spectra(self.model, v, dyn.dense_currents(self.model, 0.0),
                                 [0.0], 1.0, s_flat=0.0)
        sp1 = dyn.source_spectra(self.model, v, dyn.dense_currents(self.model, 1.0),
                                 [0.0], 1.0, s_flat=1.0)
        self.assertEqual(sp0.diagnostics["s_flat"], 0.0)
        self.assertEqual(sp1.diagnostics["s_flat"], 1.0)
        self.assertGreater(abs(sp0.diagnostics["eig_max"] - sp1.diagnostics["eig_max"]), 1e-8)


if __name__ == "__main__":
    unittest.main()
