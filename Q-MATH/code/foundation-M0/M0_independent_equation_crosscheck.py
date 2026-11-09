#!/usr/bin/env python3
"""A SECOND implementation from geometry JSON (no model code for constructing h)."""
import json,sys,pathlib
import numpy as np
base=pathlib.Path(__file__).resolve().parent
sys.path.insert(0,str(base))
import q8_m0_model as src
geodata=json.loads((base/'M0_geometry_reference_uploaded.json').read_text())
g=geodata['geometry'];pr=geodata['parameters']
I=np.eye(2,dtype=complex);sx=np.array([[0,1],[1,0]]);sy=np.array([[0,-1j],[1j,0]]);sz=np.diag([1,-1]);

def evaluate(k,N):
    f=sum(np.exp(1j*np.dot(k,d[:2])) for d in g['nn_deltas_cart'])
    gg=sum(2*nu*np.sin(np.dot(k,d[:2])) for nu,d in g['nnn_terms'])
    t0=np.zeros((2,2),complex)
    for bond in g['interlayer_bonds']:
        ii=0 if bond['from'].endswith('A') else 1
        jj=0 if bond['to'].endswith('A') else 1
        t0[ii,jj]+=pr[bond['amplitude']]*np.exp(1j*np.dot(k,bond['delta_cart'][:2]))
    def hl(eta):
        return pr['t_h']*np.kron(f.real*sx-f.imag*sy,I)+pr['lam_so']*gg*np.kron(sz,sz)+N*eta*pr['delta']*np.kron(I,sz)
    it=np.kron(t0,I)
    return np.block([[hl(+1),it],[it.conj().T,hl(-1)]])

geo=src.build_geometry(base/'CrI3_scf_soc_a1h_clean.in')
ks=[[0,0],[0.11,0.32],[1/3,1/3],[0.26,0.59],[-0.17,0.4]]
diffs=[]
for N in [-1,1]:
    for kf in ks:
        k=geo.k_cart(kf)
        diffs.append(float(np.linalg.norm(evaluate(k,N)-src.h8(geo,src.R0,k,N))))
res={'max_second_implementation_h8_abs_F':max(diffs),'k_samples':len(ks),'neel_samples':2}
(base/'M0_independent_equation_results.json').write_text(json.dumps(res,indent=2))
print(json.dumps(res,indent=2))
if max(diffs)>1e-12: raise SystemExit(2)
