#!/usr/bin/env python3
"""Check Peierls second derivative (diamagnetic/contact vertex) against M0."""
import pathlib,sys,json
import numpy as np
sys.path.insert(0,str(pathlib.Path(__file__).parent))
import q8_m0_model as m
geo=m.build_geometry(pathlib.Path(__file__).resolve().with_name('CrI3_scf_soc_a1h_clean.in'))
N=1

def h2_analytic(k, a,b):
    ff=0j; gg=0.; tt=np.zeros((2,2),complex)
    for delta in geo.nn_deltas_cart:
        delta=np.array(delta); ff+=-delta[a]*delta[b]*np.exp(1j*np.dot(k,delta[:2]))
    for nu, delta in geo.nnn_terms:
        delta=np.array(delta); gg+=-2*nu*delta[a]*delta[b]*np.sin(np.dot(k,delta[:2]))
    for bond in geo.interlayer_bonds:
        delta=np.array(bond['delta_cart']);i=0 if bond['from'][1]=='A' else 1; j=0 if bond['to'][1]=='A' else 1
        amp=getattr(m.R0,bond['amplitude'])
        tt[i,j]+=-delta[a]*delta[b]*amp*np.exp(1j*np.dot(k,delta[:2]))
    hh=m._layer_block(ff,gg,0.,m.R0.t_h,m.R0.lam_so)
    return m.assemble(hh,hh,m._interlayer_block(tt))
res=[]
for kfrac in [[0.,0.],[.14,.28],[.2,.61],[1/3,1/3]]:
    k=geo.k_cart(kfrac)
    for a in range(2):
        for b in range(2):
            eps=2e-5
            ek=np.zeros(2);ek[b]=eps
            numerical=(m.dh8(geo,m.R0,k+ek,N)[a]-m.dh8(geo,m.R0,k-ek,N)[a])/(2*eps)
            analytical=h2_analytic(k,a,b)
            res.append(float(np.linalg.norm(numerical-analytical)))
print('max contact tensor abs F residual=',max(res))
pathlib.Path(__file__).resolve().with_name('M0_contact_derivative_results.json').write_text(json.dumps({'max_contact_F':max(res),'eps_k_Ainv':2e-5,'diagnostic':'analytic contact matrix = second k derivative, including NN, NNN, interlayer bonds'},indent=2))
