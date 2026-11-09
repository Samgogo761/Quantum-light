#!/usr/bin/env python3
"""Static checks for the shared-mode k,-k construction. No propagation."""
from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np

import q8_m0_qinput_dynamics_20261026 as dyn
import q8_m0_qinput_minimal_20261020 as qm
import q8_m0_qinput_shared_kk_static_20261029 as sh
from q8_m0_model import R0, build_geometry

ROOT = Path(__file__).resolve().parents[2]


class SharedKKStaticTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.geo = build_geometry()
        cls.k = cls.geo.k_cart(sh.K_FRAC)
        cls.model = sh.build_shared(
            cls.geo, R0, cls.k, sh.NEEL, n_max=sh.NMAX_APPLY, omega0=sh.OMEGA0, g=sh.G_ABS)
        occ_k = dyn.electron_ground_state(cls.geo, R0, cls.k, sh.NEEL)
        occ_mk = dyn.electron_ground_state(cls.geo, R0, -cls.k, sh.NEEL)
        cls.ek, cls.mass_k, cls.leak_k = sh.slater36(occ_k)
        cls.emk, cls.mass_mk, cls.leak_mk = sh.slater36(occ_mk)
        vac = np.zeros(cls.model.d1, complex)
        vac[0] = 1.0
        cls.vac = vac
        cls.psi = sh.product_state(cls.ek, cls.emk, vac)

    def test_sector_is_36_and_closed(self):
        self.assertEqual(sh.EL36, 36)
        self.assertEqual(qm.ELECTRON_DIM, 70)
        closure = sh.spin_closure_report(self.geo, R0)
        self.assertEqual(closure["n_spin_flip_bonds"], 0)
        self.assertLess(closure["max_abs_H_sector_to_complement"], 1e-14)
        self.assertLess(closure["max_abs_J_sector_to_complement"], 1e-14)
        self.assertTrue(closure["closed_under_onsite_and_hopping"])
        self.assertGreater(self.mass_k, 1.0 - 1e-12)
        self.assertGreater(self.mass_mk, 1.0 - 1e-12)
        self.assertLess(self.leak_k, 1e-12)
        self.assertLess(self.leak_mk, 1e-12)

    def test_36_block_matches_70(self):
        single_k = dyn.build_joint(
            self.geo, R0, self.k, sh.NEEL, n_max=1, omega0=sh.OMEGA0, g=sh.G_ABS)
        single_mk = dyn.build_joint(
            self.geo, R0, -self.k, sh.NEEL, n_max=1, omega0=sh.OMEGA0, g=sh.G_ABS)
        self.assertLess(np.max(np.abs(sh.restrict36(sum(single_k.f_group)) - sum(self.model.f_k))), 1e-14)
        self.assertLess(np.max(np.abs(sh.restrict36(sum(single_mk.f_group)) - sum(self.model.f_mk))), 1e-14)
        self.assertLess(
            np.max(np.abs(sh.restrict36(sum(single_k.j_group)) - sum(self.model.j_k))), 1e-14)

    def test_shared_dim_is_not_two_independent_photon_modes(self):
        n12 = sh.NMAX_INTERFACE + 1
        self.assertEqual(36 * 36 * n12, 16848)
        self.assertNotEqual(36 * 36 * n12, 2 * 36 * n12)
        self.assertNotEqual(36 * 36 * n12, (36 * n12) ** 2)
        self.assertEqual(self.model.dim, 36 * 36 * (sh.NMAX_APPLY + 1))

    def test_apply_h_product_and_hermiticity(self):
        rng = np.random.default_rng(1)
        v = rng.normal(size=self.model.dim) + 1j * rng.normal(size=self.model.dim)
        v = v / np.linalg.norm(v)
        for s in (0.0, 0.4, 1.0):
            hv = sh.apply_h(self.model, self.psi, s)
            href = sh.apply_h_product_reference(self.model, self.ek, self.emk, self.vac, s)
            self.assertLess(np.max(np.abs(hv - href)), 1e-12)
            hr = sh.apply_h(self.model, v, s)
            self.assertLess(abs(np.vdot(v, hr) - np.vdot(hr, v).conjugate()), 1e-10)

    def test_zero_field_current_and_F_identity(self):
        mean, conn, jpar = sh.connected_current_cov(self.model, self.ek, self.emk)
        Fk = sh.fk_from_groups(self.model.f_k, self.model.q_units, self.ek)
        Fmk = sh.fk_from_groups(self.model.f_mk, self.model.q_units, self.emk)
        Fj = -sh.G_ABS * np.dot(self.model.e, mean.real)
        self.assertLess(np.max(np.abs(mean)), 1e-10)
        self.assertLess(abs((Fk + Fmk).real - Fj), 1e-12)
        eps = 1e-6
        fd = (sh.classical_x_energy(self.model, self.ek, self.emk, eps)
              - sh.classical_x_energy(self.model, self.ek, self.emk, -eps)) / (2 * eps)
        self.assertLess(abs((Fk + Fmk) - fd), 1e-8)
        self.assertGreater(abs(jpar), 1e-8)
        Jv = sh.apply_current(self.model, self.psi, 0.0)
        mean_apply = np.array([np.vdot(self.psi, Jv[a]) for a in range(2)])
        self.assertLess(np.max(np.abs(mean_apply - mean)), 1e-12)

    def test_photon_parity_maps_g_not_N(self):
        model_gneg = sh.build_shared(
            self.geo, R0, self.k, sh.NEEL, n_max=self.model.n_max, omega0=sh.OMEGA0, g=-sh.G_ABS)
        psi_p = sh.apply_photon_parity(self.psi, sh.EL36, self.model.d1)
        lhs = sh.apply_photon_parity(sh.apply_h(self.model, psi_p, 1.0), sh.EL36, self.model.d1)
        rhs = sh.apply_h(model_gneg, self.psi, 1.0)
        self.assertLess(np.max(np.abs(lhs - rhs)), 1e-12)

    def test_photon_interface_vacuum(self):
        ph = sh.photon_observables(self.psi, sh.EL36, self.model.d1)
        self.assertLess(abs(ph["n_ph"]), 1e-15)
        self.assertLess(abs(ph["mean_a"]["re"]) + abs(ph["mean_a"]["im"]), 1e-15)
        self.assertAlmostEqual(ph["purity"], 1.0, places=12)
        self.assertAlmostEqual(ph["n_incoherent"], ph["n_ph"] - abs(ph["mean_a"]["re"] + 1j * ph["mean_a"]["im"]) ** 2)

    def test_poxp_maps_to_factor_swapped_H_minusN(self):
        wp = sh.wedge_p36()
        model0 = sh.build_shared(
            self.geo, R0, self.k, sh.NEEL, n_max=0, omega0=sh.OMEGA0, g=sh.G_ABS)
        model_mN = sh.build_shared(
            self.geo, R0, self.k, -sh.NEEL, n_max=0, omega0=sh.OMEGA0, g=sh.G_ABS)
        H_N = sh.dense_electronic_pair(model0)
        H_mN = sh.dense_electronic_pair(model_mN)
        mapped = np.kron(wp, wp) @ H_N @ np.kron(wp, wp).conj().T
        self.assertGreater(np.max(np.abs(mapped - H_mN)), 1.0)
        self.assertLess(np.max(np.abs(mapped - sh.swap_pair_matrix(H_mN))), 1e-12)

    def test_relabel_k_minusk_after_factor_swap(self):
        model_swap = sh.SharedModel(
            self.model.k, self.model.neel, self.model.n_max, self.model.omega0,
            self.model.g, self.model.e, self.model.onsite_mk, self.model.onsite_k,
            self.model.q_units, self.model.f_mk, self.model.f_k, self.model.j_mk, self.model.j_k)
        h_orig = sh.apply_h(self.model, sh.product_state(self.ek, self.emk, self.vac), 0.0)
        h_swap = sh.apply_h(model_swap, sh.product_state(self.emk, self.ek, self.vac), 0.0)
        self.assertLess(
            np.max(np.abs(sh.swap_electronic_factors(h_orig, sh.EL36, self.model.d1) - h_swap)),
            1e-12)

    def test_coherent_interface_is_normalized(self):
        coh = np.zeros(self.model.d1, complex)
        coh[0] = 1.0
        alpha = 0.5
        for n in range(1, self.model.d1):
            coh[n] = alpha * coh[n - 1] / np.sqrt(n)
        coh = coh / np.linalg.norm(coh)
        ph = sh.photon_observables(
            sh.product_state(self.ek, self.emk, coh), sh.EL36, self.model.d1)
        self.assertAlmostEqual(ph["trace"], 1.0, places=12)
        self.assertGreaterEqual(ph["n_incoherent"], -1e-12)
        self.assertAlmostEqual(ph["purity"], 1.0, places=12)

    def test_old_archives_still_there(self):
        for rel in (
            "results/q_line/qinput_a_dynamics_trial_20261026/trial.json",
            "results/q_line/qinput_a_truncation_preflight_20261027/preflight.json",
            "results/q_line/qinput_a_j0_drive_diag_20261028/diag.json",
        ):
            self.assertTrue((ROOT / rel).is_file(), rel)


if __name__ == "__main__":
    unittest.main()
