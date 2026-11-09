#!/usr/bin/env python3
"""Quadrature diagnostic and archive-rebuild checks. No long propagation."""
from __future__ import annotations

import json
import unittest
from pathlib import Path

import numpy as np

import q8_m0_qinput_shared_kk_dyn_preflight_20261030 as pf
import q8_m0_qinput_shared_kk_squeeze_diag_20261031 as diag

ROOT = Path(__file__).resolve().parents[2]
OLD = ROOT / "results/q_line/qinput_a_shared_kk_dyn_preflight_20261030/preflight.json"


class SqueezeDiagTests(unittest.TestCase):
    def test_vacuum_variance_is_quarter(self):
        q = pf.quadrature_from_moments(0.0, 0.0, n_max=8, p_top=0.0)
        self.assertAlmostEqual(q["var_min"], 0.25, places=15)
        self.assertAlmostEqual(q["var_max"], 0.25, places=15)
        self.assertAlmostEqual(q["W_sq"], 0.0, places=15)
        self.assertFalse(q["classical_P_bound_violated"])
        self.assertAlmostEqual(q["commutator_expect"], 1.0, places=15)

    def test_squeezed_vacuum_matches_exp_minus_2r_over_4(self):
        r = 0.4
        n = float(np.sinh(r) ** 2)
        m = -float(np.sinh(r) * np.cosh(r))
        q = pf.quadrature_from_moments(n, m, n_max=12, p_top=0.0)
        self.assertAlmostEqual(q["var_min_if_commutator_1"], np.exp(-2.0 * r) / 4.0, places=12)
        self.assertAlmostEqual(q["var_max_if_commutator_1"], np.exp(2.0 * r) / 4.0, places=12)
        self.assertTrue(q["classical_P_bound_violated"])
        self.assertLess(q["W_sq"], 0.0)
        self.assertAlmostEqual(q["theta_min"], 0.0, places=12)

    def test_finite_cutoff_commutator(self):
        q = pf.quadrature_from_moments(0.0, 0.0, n_max=2, p_top=0.1)
        self.assertAlmostEqual(q["commutator_deficit"], 0.3, places=15)
        self.assertAlmostEqual(q["var_min"], 0.25 * 0.7, places=15)

    def test_archive_moments_rebuild(self):
        data = json.loads(OLD.read_text(encoding="utf-8"))
        snap = data["dynamics"]["driven"]["after_ramp"]
        mom = diag.moments_from_snap(snap)
        self.assertLess(mom["max_abs_n_ph_minus_sum_n_pn"], 1e-15)
        self.assertLess(mom["max_abs_n_c_minus_rebuild"], 1e-18)
        self.assertLess(mom["max_abs_m_c_minus_rebuild"], 1e-18)
        self.assertAlmostEqual(mom["p_n_sum"], 1.0, places=12)
        q = mom["quadrature"]
        self.assertGreater(q["abs_m"], q["n_c"])
        self.assertLess(q["var_min"], 0.25)
        self.assertAlmostEqual(q["W_sq"], q["n_c"] - q["abs_m"], places=15)
        self.assertLess(q["commutator_deficit"], 1e-14)
        self.assertAlmostEqual(mom["purity_archive"], 0.9991916380066825, places=12)

    def test_work_audit_flags_one_percent_gap(self):
        data = json.loads(OLD.read_text(encoding="utf-8"))
        w = diag.work_audit(data["dynamics"]["driven"])
        self.assertGreater(w["relative_gap"], 0.005)
        self.assertLess(w["relative_gap"], 0.02)
        self.assertGreater(w["n_samples"], 8)
        self.assertGreater(w["n_steps"] / w["n_samples"], 30.0)

    def test_old_archives_present(self):
        for rel in diag.OLD_ARCHIVES:
            self.assertTrue((ROOT / rel).is_file(), rel)

    def test_snapshot_gains_quadrature_on_tiny_space(self):
        import q8_m0_qinput_shared_kk_static_20261029 as sh
        from q8_m0_model import R0, build_geometry
        geo = build_geometry()
        k = geo.k_cart(sh.K_FRAC)
        model = sh.build_shared(
            geo, R0, k, sh.NEEL, n_max=2, omega0=sh.OMEGA0, g=sh.G_ABS)
        ek, emk, _ = pf.ground_pair(geo, k)
        psi, _ = pf.vacuum_product(model, ek, emk)
        snap = pf.snapshot(model, psi, 0.0)
        q = snap["quadrature"]
        self.assertAlmostEqual(q["var_min"], 0.25, places=12)
        self.assertAlmostEqual(q["W_sq"], 0.0, places=12)


if __name__ == "__main__":
    unittest.main()
