"""Independent Gauss integration and archive immutability checks; no propagation."""
import json
import time
from pathlib import Path
import numpy as np
from qinput_spectrum_reanalysis_20261028 import ROOT, BASE, OUT, PERIOD, OMEGA, cases, spectrum, sha, dump


def gauss_stored_linear(t,j,lo,hi,qs):
    # Independent integration: no exponential moment formulas from the analyzer.
    knots=np.r_[lo,t[(t>lo)&(t<hi)],hi]
    node,weight=np.polynomial.legendre.leggauss(12)
    h=np.diff(knots)
    xx=knots[:-1,None]+h[:,None]*(node[None,:]+1)/2
    yy=np.stack([np.interp(xx,t,c.real)+1j*np.interp(xx,t,c.imag) for c in j.T],axis=-1)
    return np.array([np.sum(h[:,None,None]/2*weight[None,:,None]*np.exp(1j*q*OMEGA*xx)[:,:,None]*yy,axis=(0,1))/(hi-lo) for q in qs])


def main():
    start=time.perf_counter();checks=[]
    for c in cases():
        for n in ('p','m'):
            p=BASE/c['folder']/('series_'+c['prefix'].format(n=n)+'.npz')
            with np.load(p,allow_pickle=False) as f:
                t=f['t'];j=np.column_stack([f['Jx'],f['Jy']])
            a=spectrum(t,j,2*PERIOD,4*PERIOD,np.arange(1,7))
            b=gauss_stored_linear(t,j,2*PERIOD,4*PERIOD,np.arange(1,7))
            checks.append({'file':str(p.relative_to(ROOT)), 'max_abs_difference':float(np.max(abs(a-b)))})
    assert max(c['max_abs_difference'] for c in checks)<2e-14
    man=json.loads((OUT/'input_manifest.json').read_text(encoding='utf-8'))
    same={p:sha(ROOT/p)==h for p,h in man['before'].items()}
    assert all(same.values())
    source=ROOT/'tools/b0_small_model'
    scripts=['qinput_spectrum_reanalysis_20261028.py','test_qinput_spectrum_reanalysis_20261028.py',
             'qinput_reanalysis_report_20261028.py',Path(__file__).name]
    dump(OUT/'verification.json',{'independent_quadrature':'12-point Gauss per original linear segment',
         'checks':checks,'max_abs_difference':max(c['max_abs_difference'] for c in checks),
         'input_unchanged':same,'all_inputs_unchanged':all(same.values()),
         'scripts_sha256':{x:sha(source/x) for x in scripts},'wall_s':time.perf_counter()-start,
         'scope':'integrates reconstructed stored data; not a dynamics/Fock/BZ convergence certificate'})
    print(json.dumps({'series':len(checks),'independent_max_abs_difference':max(c['max_abs_difference'] for c in checks),
                      'all_inputs_unchanged':all(same.values()),'wall_s':time.perf_counter()-start}))


if __name__=='__main__':main()
