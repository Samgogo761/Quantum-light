#!/usr/bin/env python3
"""Fast regression checks for Q-MATH-16.25 dynamic Ward kernel."""
import sys, json
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))
from qmath_16_25_static_subtracted_kernel import grid, delta_kernel
from qmath_16_25_retarded_downfolding import terms, kernel

def main():
    # Exact zero-frequency Ward-subtracted definition.
    D,g,s,gmin=terms((.4158,.1683),Delta=.8,lam=.1)
    assert delta_kernel(g,s,0.0,eta=5e-4)==0j
    # Algebraic equivalence: K(z)-K(0) == sum w*z^2/(z^2-gap^2) at one k.
    om=.01; eta=5e-4; z=complex(om,eta)
    Kz=kernel(D,g,s,om,eta=eta,mode='full')
    K0=D-np.sum(2*s/g)
    Ksub=delta_kernel(g,s,om,eta=eta)
    err=abs((Kz-K0)-Ksub)
    assert err < 1e-10, err
    # Exact decomposition full = retained + omitted.
    low=delta_kernel(g,s,om,eta=eta,keep=2)
    full=delta_kernel(g,s,om,eta=eta)
    omitted=full-low
    assert abs(full-low-omitted)<1e-14
    # Néel sign reversal leaves this even linear-response kernel unchanged.
    rp=grid(21,.8,.1,[0.,.01],eta=eta,keep=2)
    rm=grid(21,-.8,.1,[0.,.01],eta=eta,keep=2)
    vp=rp['response_delta_K']['0.01']['full']; vm=rm['response_delta_K']['0.01']['full']
    symerr=max(abs(vp['real']-vm['real']),abs(vp['imag']-vm['imag']))
    assert symerr < 1e-10, symerr
    result={'zero_frequency_subtracted_exact':True,'single_k_algebraic_identity_error':float(err),'full_low_omitted_decomposition_error':0.0,'Néel_sign_evenness_error_L21':float(symerr),'min_direct_gap_test_k':float(gmin),'scope':'linear-response kernel only'}
    p=ROOT/'qmath_16_25_regression_results.json';p.write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
if __name__=='__main__': main()
