"""Independent ODE test for the autonomous-frame reference transformation."""
import unittest
import numpy as np
from scipy.integrate import solve_ivp
from scipy.linalg import expm
from q8_m0_stage1ag_acceptance_audit import rotation
from q8_m0_phgauss import symplectic_J

class AutonomousReferenceTest(unittest.TestCase):
    def test_interaction_picture_against_adaptive_ode(self):
        freq=np.array([.7,1.3]);j=symplectic_J(2)
        h=np.zeros((4,4));h[0,1]=h[1,0]=.2
        k0=j@h;free=j@np.diag(np.r_[freq,freq]);t=2.3
        def rhs(s,y):
            r=rotation(freq,s)
            return (r.T@k0@r@y.reshape(4,4)).ravel()
        sol=solve_ivp(rhs,(0.,t),np.eye(4).ravel(),method='DOP853',rtol=1e-12,atol=1e-14)
        self.assertTrue(sol.success)
        expected=rotation(freq,-t)@expm(t*(free+k0))
        np.testing.assert_allclose(sol.y[:,-1].reshape(4,4),expected,atol=2e-12,rtol=2e-12)
        np.testing.assert_allclose(expected.T@j@expected,j,atol=2e-14)

if __name__=='__main__':unittest.main()
