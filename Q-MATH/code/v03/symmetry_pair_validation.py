#!/usr/bin/env python3
"""Test the exact parity-covariant M0 pump+H2 photon-number equality at N,k vs -N,-k."""
import json, sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parent))
import qmath03_exact_unitary as q, q8_m0_model as m
BASE=Path(__file__).resolve().parent
geo=m.build_geometry(BASE/'CrI3_scf_soc_a1h_clean.in');basis,G=q.make_ops();k=geo.k_cart([.4158,.1683])
param=dict(npump=25,nout=5,mean=.7,azp=.05,azout=.005,omega=.18,ncycles=2,nsteps=160,det_polar='parallel')
a=q.simulate(geo,G,basis,k,1,'squeezed',**param)
b=q.simulate(geo,G,basis,-k,-1,'squeezed',**param)
err=max(abs(x-y) for x,y in zip(a['P_H2'],b['P_H2']))
print('BSV parity inversion test: max photon population difference',err)
print('number',a['n_H2'],b['n_H2'],'g2',a['g2_H2'],b['g2_H2'])
(BASE/'SYMMETRY_PAIR_RESULT.json').write_text(json.dumps({'Nplus_k':a,'Nminus_minus_k':b,'max_distribution_difference':err},indent=2))
assert err<1e-10
print('PARITY_PAIRED_H2_PASSED')
