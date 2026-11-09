"""Analytic tests, including DC leakage and forbidden window extrapolation."""
import unittest
import numpy as np
from qinput_spectrum_reanalysis_20261028 import spectrum, PERIOD, OMEGA, domain_stats


class FourierTests(unittest.TestCase):
    def test_constant_on_real_saved_grid(self):
        t = np.arange(350)*32*(6*PERIOD/11154)
        qs = np.arange(1,7)
        old = spectrum(t,np.ones_like(t),2*PERIOD,4*PERIOD,qs,"legacy")
        new = spectrum(t,np.ones_like(t),2*PERIOD,4*PERIOD,qs)
        self.assertGreater(abs(old[1,0]),.009)
        self.assertLess(np.max(abs(new)),2e-14)

    def test_linear_current_exact(self):
        t = np.linspace(0,6*PERIOD,101)
        lo,hi = 2*PERIOD,4*PERIOD
        for q in (0,1,2,6):
            k=q*OMEGA
            target=(lo+hi)/2 if q==0 else (np.exp(1j*k*hi)*(hi/(1j*k)+1/k**2)-np.exp(1j*k*lo)*(lo/(1j*k)+1/k**2))/(hi-lo)
            got=spectrum(t,t,lo,hi,[q])[0,0]
            self.assertLess(abs(got-target),1e-11)

    def test_known_single_and_double_complex_frequency(self):
        t=np.linspace(0,6*PERIOD,24001)
        phase=.73
        y=2*np.exp(1j*phase)*np.exp(-2j*OMEGA*t)+(.3-.2j)*np.exp(-5j*OMEGA*t)
        vals=spectrum(t,y,2*PERIOD,4*PERIOD,[1,2,5,6])[:,0]
        self.assertLess(abs(vals[1]-2*np.exp(1j*phase)),2e-6)
        self.assertLess(abs(vals[2]-(.3-.2j)),2e-6)
        self.assertLess(max(abs(vals[[0,3]])),1e-11)

    def test_quadratic_exact(self):
        t=np.linspace(0,6*PERIOD,81)
        lo,hi=2*PERIOD,4*PERIOD
        val=spectrum(t,t*t,lo,hi,[0],"quadratic_exact")[0,0]
        self.assertLess(abs(val-(hi**3-lo**3)/(3*(hi-lo))),1e-8)

    def test_missing_window_rejected(self):
        t=np.linspace(0,4*PERIOD,31)
        with self.assertRaisesRegex(ValueError,"unavailable"):
            spectrum(t,np.ones_like(t),3*PERIOD,5*PERIOD,[2])

    def test_domain_interference_identity(self):
        rec=domain_stats(np.array([1+.3j,.2+.7j]),np.array([.9+.4j,.1-.1j]))
        for a in rec["analyzers"].values():
            self.assertAlmostEqual(a["difference"],a["interference_4Re"],places=14)


if __name__=="__main__":
    unittest.main()
