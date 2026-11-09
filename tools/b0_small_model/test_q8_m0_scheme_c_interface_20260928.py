"""Physical algebra fixtures only; no M0 time evolution or production state choice."""
import unittest
import numpy as np
from q8_m0_scheme_c_interface_20260928 import (
    canonical_residuals, checks, fixture_bogoliubov, fixture_input,
    input_moment_matrix, nm_from_symmetrized_covariance, output_moments,
    real_map, rows_from_existing_phi, transform_moments,
)

class SchemeCInterfaceTests(unittest.TestCase):
    def test_frozen_protocol_ports_and_packet_algebra(self):
        rec=checks()
        self.assertTrue(rec["all_within_2e_minus10"],rec["max_residuals"])
        self.assertEqual(rec["protocol"]["polarization"],"a2")
        self.assertEqual(rec["protocol"]["a0_dimless"],.8)

    def test_nonvacuum_correlated_input_against_real_covariance(self):
        f,g=fixture_bogoliubov()
        v,n,m=fixture_input()
        _,nout,mout=output_moments(f,g,np.zeros(4),n_in=n,m_in=m,input_state_id="fixture")
        p=real_map(f,g)
        nr,mr=nm_from_symmetrized_covariance(p@v@p.T)
        np.testing.assert_allclose(nout,nr,atol=1e-12)
        np.testing.assert_allclose(mout,mr,atol=1e-12)

    def test_both_canonical_relations_are_needed(self):
        # First CCR alone passes, but the antisymmetric creation block breaks [b,b]=0.
        r=.2
        f=np.cosh(r)*np.eye(2)
        g=np.sinh(r)*np.array([[0.,1.],[-1.,0.]])
        err=canonical_residuals(f,g)
        self.assertLess(err["b_bdag"],1e-12)
        self.assertGreater(err["b_b"],.1)
        with self.assertRaises(ValueError):
            output_moments(f,g,np.zeros(2),n_in=np.zeros((2,2)),m_in=np.zeros((2,2)),input_state_id="invalid")

    def test_no_implicit_vacuum_or_unphysical_covariance(self):
        with self.assertRaises(ValueError):
            input_moment_matrix(None,None)
        with self.assertRaises(ValueError):
            input_moment_matrix(np.zeros((2,2)),np.eye(2))
        with self.assertRaises(ValueError):
            output_moments(np.eye(2),np.zeros((2,2)),np.zeros(2),
                           n_in=np.zeros((2,2)),m_in=np.zeros((2,2)),input_state_id="")

    def test_overlapping_packets_not_silently_orthogonalized(self):
        f,g=fixture_bogoliubov()
        with self.assertRaises(ValueError):
            rows_from_existing_phi(real_map(f,g),2,2,np.ones((2,2))/np.sqrt(2))

    def test_matched_antiunitary_moments_not_same_pulse_rule(self):
        v,n,m=fixture_input()
        sign=np.diag(np.r_[np.ones(4),-np.ones(4)])
        nt,mt=nm_from_symmetrized_covariance(sign@v@sign)
        d=np.diag(np.exp(1j*np.array([.2,-.4,.6,-.7])))
        beta=np.array([.1j,.2,.3-.1j,-.4j])
        bt,n2,m2=transform_moments(d,beta,n,m,antiunitary=True)
        np.testing.assert_allclose(bt,d@beta.conj(),atol=1e-12)
        np.testing.assert_allclose(n2,d.conj()@nt@d.T,atol=1e-12)
        np.testing.assert_allclose(m2,d@mt@d.T,atol=1e-12)


    def test_electronic_input_basis_rotation_leaves_lab_output_unchanged(self):
        f,g=fixture_bogoliubov()
        _,n,m=fixture_input()
        beta=np.array([.1j,.2,.3-.1j,-.4j])
        _,nr,mr=output_moments(f,g,beta,n_in=n,m_in=m,input_state_id="fixture_original_basis")
        u=np.eye(4,dtype=complex)
        u[2:,2:]=np.array([[1.,1j],[1j,1.]])/np.sqrt(2)
        _,nt,mt=transform_moments(u,np.zeros(4),n,m)
        _,no,mo=output_moments(f@u.conj().T,g@u.T,beta,n_in=nt,m_in=mt,
                              input_state_id="same_fixture_rotated_electronic_basis")
        np.testing.assert_allclose(no,nr,atol=1e-12)
        np.testing.assert_allclose(mo,mr,atol=1e-12)

    def test_vacuum_control_and_pair_creation_fixture(self):
        # Known two-mode transformation, not a prediction that M0 creates pairs.
        r=.2
        f=np.cosh(r)*np.eye(2)
        g=np.sinh(r)*np.array([[0.,1.],[1.,0.]])
        z=np.zeros((2,2))
        _,n,m=output_moments(f,g,np.zeros(2),n_in=z,m_in=z,input_state_id="explicit_fixture_vacuum")
        np.testing.assert_allclose(n,np.sinh(r)**2*np.eye(2),atol=1e-12)
        np.testing.assert_allclose(m,np.cosh(r)*g,atol=1e-12)
        _,nv,mv=output_moments(np.eye(2),z,np.zeros(2),n_in=z,m_in=z,input_state_id="passive_fixture_vacuum")
        np.testing.assert_allclose(nv,0.,atol=1e-12)
        np.testing.assert_allclose(mv,0.,atol=1e-12)

if __name__=="__main__":
    unittest.main()

