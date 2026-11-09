"""Independent sign/ordering checks for the zero-pump electronic kernel."""
import unittest
import numpy as np
from scipy.integrate import quad_vec

from q8_m0_stage1ai_lowfreq_audit import spectral_data, kernel, increment, expansion
from q8_m0_response import c_greater_from_x, commutator_from_c_greater
from q8_m0_stage1ag_acceptance_audit import (
    MU, R0, build_geometry, h8, current_ops, occup_empty_bases,
    x_from_bases, d_ops, mean_d_from_bases,
)


class LowFrequencyTests(unittest.TestCase):
    def test_actual_M0_full_density_matrix_resolvent(self):
        """Solve all 64 density-matrix components, not the c-v spectral formula."""
        geo = build_geometry()
        kc = geo.k_cart(np.array([[.17, .41], [.32, .11]]))
        hs, js = h8(geo, R0, kc, 1), current_ops(geo, R0, kc, 1)
        vv, vc, en = occup_empty_bases(hs, MU)
        x = x_from_bases(js, vv, vc)
        weights = np.array([.4, .6])
        d_k = mean_d_from_bases(d_ops(geo, R0, kc, 1), vv)
        d = np.einsum("k,kab->ab", weights, d_k)
        gap, c = spectral_data(en[:, 4:, None]-en[:, None, :4], x, weights)
        for z in (.021+.013j, -.017+.008j):
            response = -d.astype(complex)
            for ik, (h, j) in enumerate(zip(hs, js)):
                p = vv[ik] @ vv[ik].conj().T
                liouville = np.kron(np.eye(8), h) - np.kron(h.T, np.eye(8))
                for b in range(2):
                    rhs = -(j[b] @ p - p @ j[b])
                    rho = np.linalg.solve(z*np.eye(64)-liouville, rhs.reshape(-1, order="F")).reshape((8, 8), order="F")
                    for a in range(2):
                        response[a, b] += weights[ik]*np.trace(j[a] @ rho)
            np.testing.assert_allclose(response, -kernel(d, gap, c, z), rtol=2e-11, atol=2e-11)

    def test_existing_time_commutator_transform_and_odd_tensor(self):
        """A complex-polarization fixture catches transposes and retarded signs."""
        gap0 = .7
        x0 = np.array([.4+.1j, .2-.3j]).reshape(1, 2, 1, 1)
        d = np.array([[.6, .05], [.05, .5]])
        gap, c = spectral_data(np.array([[[gap0]]]), x0, np.ones(1))
        z = .09+.12j  # Laplace convergence parameter in a fixture, not dephasing.
        def integrand(t):
            ct = c_greater_from_x(x0*np.exp(1j*gap0*t), x0, np.ones(1))
            return 1j*commutator_from_c_greater(ct)*np.exp(1j*z*t)
        response, _ = quad_vec(integrand, 0., 40/z.imag, epsabs=1e-11, epsrel=1e-11)
        np.testing.assert_allclose(response-d, -kernel(d, gap, c, z), atol=2e-11, rtol=2e-11)
        linear, _ = expansion(gap, c)
        self.assertGreater(np.max(abs(linear)), .1)
        w = .001
        delta, _, _ = increment(gap, c, w)
        np.testing.assert_allclose(delta, kernel(d, gap, c, w)-kernel(d, gap, c, 0), atol=1e-14)
        np.testing.assert_allclose(kernel(d, gap, c, -w), kernel(d, gap, c, w).conj(), atol=1e-14)


class IntrinsicNoiseIdentityTests(unittest.TestCase):
    def test_spectral_commutator_matches_K_retarded_minus_advanced(self):
        from q8_m0_stage1ak_intrinsic_noise import kernel_any, lorentzian_S
        gap0 = .7
        x0 = np.array([.4+.1j, .2-.3j]).reshape(1, 2, 1, 1)
        d = np.array([[.6, .05], [.05, .5]])
        gap, c = spectral_data(np.array([[[gap0]]]), x0, np.ones(1))
        eta = 1e-3
        for w in (-1.1, -0.2, 0.0, 0.2, 0.9):
            sg = lorentzian_S(gap, c, w, eta, greater=True)
            sl = lorentzian_S(gap, c, w, eta, greater=False)
            lhs = kernel_any(d, gap, c, w + 1j * eta) - kernel_any(d, gap, c, w - 1j * eta)
            rhs = -1j * (sg - sl)
            np.testing.assert_allclose(lhs, rhs, atol=1e-12, rtol=1e-12)


class ClosedLoopPoleTests(unittest.TestCase):
    def test_fourier_d_dt_and_Gself_give_i_gamma_over_z(self):
        """∫ e^{+izt} κ̇ = -i z κ when κ(0)=0, hence κ=(iγ/z)I from κ̇=γI."""
        from q8_m0_stage1al_pole_audit import closed_loop_F
        gamma = 0.017
        z = 0.31 + 0.11j
        i0 = np.array([0.4, -0.2])
        kappa = (1j * gamma / z) * i0
        np.testing.assert_allclose(-1j * z * kappa, gamma * i0)
        k = np.array([[0.05, 0.01], [0.01, 0.04]])
        current = -k @ kappa
        f = closed_loop_F(k, z, gamma, from_increment=False)[0]
        np.testing.assert_allclose(f @ current, current + (1j * gamma / z) * (k @ current), atol=1e-14)

    def test_increment_F_is_identity_at_origin_and_raw_has_soft_pole(self):
        from q8_m0_stage1ai_lowfreq_audit import expansion
        from q8_m0_stage1al_pole_audit import (
            closed_loop_F, increment_over_z, kernel_off_cuts, det2,
            predicted_raw_soft_poles,
        )
        gap0 = 0.7
        x0 = np.array([0.4 + 0.1j, 0.2 - 0.3j]).reshape(1, 2, 1, 1)
        d = np.array([[0.6, 0.05], [0.05, 0.5]])
        gap, c = spectral_data(np.array([[[gap0]]]), x0, np.ones(1))
        k0 = kernel_off_cuts(d, gap, c, 0.0)[0]
        gamma = 0.02
        linear, _ = expansion(gap, c)
        f0 = closed_loop_F(increment_over_z(gap, c, 0.0), 0.0, gamma, from_increment=True)[0]
        np.testing.assert_allclose(f0, np.eye(2) + 1j * gamma * linear, atol=1e-12)
        z = 0.05 + 0.03j
        dkz = increment_over_z(gap, c, z)[0]
        k = kernel_off_cuts(d, gap, c, z)[0]
        np.testing.assert_allclose(dkz, (k - k0) / z, atol=1e-10)
        poles = predicted_raw_soft_poles(k0, gamma)
        for p in poles:
            zp = p["z_real"] + 1j * p["z_imag"]
            fraw = closed_loop_F(k0, zp, gamma, from_increment=False)[0]
            self.assertLess(abs(det2(fraw)), 1e-10)
        # Real C ⇒ L=0 ⇒ F_Δ(0)=I, matching the continuous-BZ IR prediction.
        x_real = np.array([0.4, 0.2]).reshape(1, 2, 1, 1)
        gap_r, c_r = spectral_data(np.array([[[gap0]]]), x_real, np.ones(1))
        f0r = closed_loop_F(increment_over_z(gap_r, c_r, 0.0), 0.0, gamma, from_increment=True)[0]
        np.testing.assert_allclose(f0r, np.eye(2), atol=1e-12)

    def test_argument_principle_counts_a_known_uhp_zero(self):
        from q8_m0_stage1al_pole_audit import (
            closed_loop_F, kernel_off_cuts, adaptive_winding, rectangle_polygon,
        )
        # Constant K = M < 0 ⇒ raw F has a UHP zero at z = -i γ M.
        d = np.array([[-0.2, 0.0], [0.0, -0.05]])
        gap = np.array([10.0])
        c = np.zeros((2, 2, 1))
        gamma = 0.03
        k0 = kernel_off_cuts(d, gap, c, 0.0)[0]
        poly = rectangle_polygon(-0.05, 0.05, 1e-6, 0.02, 41, 21)
        rec = adaptive_winding(d, gap, c, k0, gamma, "raw", poly)
        self.assertEqual(rec["winding_nearest_int"], 2)
        self.assertLess(rec["winding_integer_residual"], 0.05)
        rec_i = adaptive_winding(d, gap, c, k0, gamma, "increment", poly)
        self.assertEqual(rec_i["winding_nearest_int"], 0)
        from q8_m0_stage1al_pole_audit import raw_soft_box_spec, run_halfplane
        box = raw_soft_box_spec(k0, gamma)
        rec_box = run_halfplane(d, gap, c, k0, gamma, "raw", box, "UHP")
        self.assertEqual(rec_box["winding_nearest_int"], 2)


class PassivityTests(unittest.TestCase):
    def test_laplace_map_and_full_tensor_quadratic_form(self):
        from q8_m0_stage1am_passivity import analytic_re_form, y_delta, herm_part
        gap0 = 0.7
        x0 = np.array([0.4 + 0.1j, 0.2 - 0.3j]).reshape(1, 2, 1, 1)
        gap, c = spectral_data(np.array([[[gap0]]]), x0, np.ones(1))
        v = np.array([0.8 - 0.1j, 0.2 + 0.3j])
        v = v / np.linalg.norm(v)
        for s in (0.11 + 0.4j, 0.03 - 1.2j, 1e-3 + 0.7j, 0.2 + 0.0j):
            y = y_delta(gap, c, s)
            form = complex(v.conj() @ y @ v)
            analytic = analytic_re_form(gap, c, s, v)
            self.assertAlmostEqual(form.real, analytic, places=12)
            self.assertGreaterEqual(analytic, -1e-14)
            eigs = np.linalg.eigvalsh(herm_part(y))
            self.assertGreaterEqual(float(np.min(eigs)), -1e-12)

    def test_raw_Y_is_not_positive_real_when_M_negative(self):
        from q8_m0_stage1am_passivity import y_raw, herm_part, y_delta
        d = np.array([[-0.2, 0.0], [0.0, -0.05]])
        gap = np.array([10.0])
        c = np.zeros((2, 2, 1))
        s = 0.01 + 0.0j
        eigs_r = np.linalg.eigvalsh(herm_part(y_raw(d, gap, c, s)))
        eigs_d = np.linalg.eigvalsh(herm_part(y_delta(gap, c, s)))
        self.assertLess(float(np.min(eigs_r)), 0.0)
        self.assertGreaterEqual(float(np.min(eigs_d)), -1e-12)


class CanonicalIOTests(unittest.TestCase):
    def test_odd_channel_decouples_and_even_cayley_identity(self):
        from q8_m0_stage1ak_intrinsic_noise import kernel_any
        from q8_m0_stage1an_canonical_io import even_maps, identity_matrices
        gap0 = 0.7
        x0 = np.array([0.4 + 0.1j, 0.2 - 0.3j]).reshape(1, 2, 1, 1)
        d = np.array([[0.6, 0.05], [0.05, 0.5]])
        gap, c = spectral_data(np.array([[[gap0]]]), x0, np.ones(1))
        gamma = 0.02
        eta = 1e-3
        for w in (0.2, 0.9, 1.3):
            k = kernel_any(d, gap, c, w + 1j * eta)
            s, g, _f = even_maps(k, w, gamma)
            jump = 1j * (kernel_any(d, gap, c, w + 1j * eta)
                         - kernel_any(d, gap, c, w - 1j * eta))
            rec = identity_matrices(s, g, jump, w, gamma)
            self.assertLess(rec["canonical_ss_residual"], 1e-10)
            self.assertEqual(rec["odd_channel"]["A_minus_out_plus_A_minus_in"], 0.0)
            if w < gap0:
                self.assertLess(rec["unitary_residual"], 5e-3)

    def test_subgap_increment_is_unitary_without_jump(self):
        from q8_m0_stage1al_pole_audit import increment_over_z, kernel_off_cuts
        from q8_m0_stage1an_canonical_io import even_maps
        gap0 = 0.7
        x0 = np.array([0.4, 0.2]).reshape(1, 2, 1, 1)
        d = np.array([[0.6, 0.05], [0.05, 0.5]])
        gap, c = spectral_data(np.array([[[gap0]]]), x0, np.ones(1))
        k0 = kernel_off_cuts(d, gap, c, 0.0)[0]
        w = 0.2
        k = k0 + increment_over_z(gap, c, w)[0] * w
        s, _g, _f = even_maps(k, w, 0.02)
        np.testing.assert_allclose(s.conj().T @ s, np.eye(2), atol=1e-12)


if __name__ == "__main__":
    unittest.main()
