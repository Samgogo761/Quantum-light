#!/usr/bin/env python3
"""M0 noninteracting-electron influence: genuine 70-dimensional wedge vs 8x8 determinant.
No photons or output optical state are assumed. Uses the user-provided M0 module.
"""
import itertools, json, sys
from pathlib import Path
import numpy as np
from scipy.linalg import expm, expm_frechet
sys.path.insert(0,str(Path(__file__).resolve().parent))
from q8_m0_model import build_geometry, h8, dh8, R0

GEO = build_geometry(Path(__file__).with_name('CrI3_scf_soc_a1h_clean.in'))
I8=np.eye(8,dtype=complex)
N=1
k=GEO.k_cart([0.21, 0.32]); ePol=np.array(GEO.a2_cart[:2]);ePol/=np.linalg.norm(ePol)
# Peierls kappa=k+A, with e=hbar=1; A in inverse-Angstrom units
Abase=np.array([0.0, 0.037, -0.025, 0.041]);
Aminus=Abase.copy(); Aplus=Abase.copy(); Aplus[1]+=0.011; Aplus[3]-=0.008
steps=np.array([0.18,0.15,0.20,0.12])  # times hbar / t_h

def propagator(amp):
    U=I8.copy()
    for aj,dt in zip(amp,steps):
        hh=h8(GEO,R0,k+aj*ePol,N)
        U=expm(-1j*dt*hh)@U
    return U

h0=h8(GEO,R0,k,N)
val,vec=np.linalg.eigh(h0)
V0=vec[:,:4]
P0=V0@V0.conj().T
assert np.linalg.norm(P0@P0-P0)<1e-12
Up=propagator(Aplus); Um=propagator(Aminus)
W=Um.conj().T@Up
F4=np.linalg.det(V0.conj().T@W@V0)
F8=np.linalg.det(I8-P0+P0@W)
# Exterior fourth-power amplitude for 4-electron Slater determinants:
rows=list(itertools.combinations(range(8),4))
# components of the wedge state = determinants of four occupied columns
vPlus=Up@V0; vMinus=Um@V0
wPlus=np.array([np.linalg.det(vPlus[list(r),:]) for r in rows])
wMinus=np.array([np.linalg.det(vMinus[list(r),:]) for r in rows])
Fwedge=np.vdot(wMinus,wPlus)
# diagonal and opposite branch identities
Fdiag=np.linalg.det(V0.conj().T@Up.conj().T@Up@V0)
Fswap=np.linalg.det(V0.conj().T@Up.conj().T@Um@V0)
# derivative with respect to one A+ slice, evaluated at identical histories
idx=2
AA=Abase.copy(); U0=propagator(AA)
# analytic Frechet derivative through one local exponential
pieces=[]
for aj,dt in zip(AA,steps):
    h=h8(GEO,R0,k+aj*ePol,N)
    pieces.append(expm(-1j*dt*h))
h_j=h8(GEO,R0,k+AA[idx]*ePol,N)
dh=np.einsum('a,aij->ij',ePol,dh8(GEO,R0,k+AA[idx]*ePol,N))
dEj=expm_frechet(-1j*steps[idx]*h_j,-1j*steps[idx]*dh,compute_expm=False)
Ub=I8.copy()
for z in pieces[:idx]:Ub=z@Ub
Ua=I8.copy()
for z in pieces[idx+1:]:Ua=z@Ua
dU=Ua@dEj@Ub
dlogF=np.trace(V0.conj().T@U0.conj().T@dU@V0)
def ff(Aup):
    W=U0.conj().T@propagator(Aup)
    return np.linalg.det(V0.conj().T@W@V0)
eps=1e-5
Ad=AA.copy();Ad[idx]+=eps
Am=AA.copy();Am[idx]-=eps
dlogF_num=(ff(Ad)-ff(Am))/(2*eps)  # F=1 at zero difference

# e=hbar=1: derivative of influence is i * dt * average current in the slice;
# the latter has its appropriate internal time average, not just midpoint.
# We only validate the exact derivative vs finite difference here.
result={
 'model':'Q_M0_eightband_v1', 'k_cart_Ainv':k.tolist(),'n_electrons':4,
 'exterior_dim':len(rows),'field_Aplus':Aplus.tolist(),'field_Aminus':Aminus.tolist(),
 'branch_overlap':{'det4': [float(F4.real),float(F4.imag)], 'det8':[float(F8.real),float(F8.imag)],'wedge70':[float(Fwedge.real),float(Fwedge.imag)]},
 'max_det_wedge_error':float(max(abs(F4-F8),abs(F4-Fwedge))),
 'normalization_error':float(abs(Fdiag-1)),
 'hermiticity_swap_error':float(abs(Fswap-F4.conjugate())),
 'physical_overlap_bound_error':float(max(0,abs(F4)-1)),
 'branch_variation_derivative':{'frechet':[float(dlogF.real),float(dlogF.imag)], 'finite_difference':[float(dlogF_num.real),float(dlogF_num.imag)],'difference':float(abs(dlogF-dlogF_num))},
 'scope':'Exact free-fermion field-history influence at this k, not photon emission or full output field.'
}
# The exact inversion theorem holds for the complete two-branch influence:
# F_{N,k}[A+,A-] = F_{-N,-k}[-A+,-A-].
def propagator_at(kvec, nval, amps):
    U=I8.copy()
    for aj,dt in zip(amps,steps):
        U=expm(-1j*dt*h8(GEO,R0,kvec+aj*ePol,nval))@U
    return U

values_m, vecs_m=np.linalg.eigh(h8(GEO,R0,-k,-N))
V0_m=vecs_m[:,:4]
Up_m=propagator_at(-k,-N,-Aplus)
Um_m=propagator_at(-k,-N,-Aminus)
Finverted=np.linalg.det(V0_m.conj().T @ Um_m.conj().T @ Up_m @ V0_m)
result['inversion_on_both_histories']={
    'F_inv_real_imag':[float(Finverted.real),float(Finverted.imag)],
    'error':float(abs(Finverted-F4)),
    'meaning':'Compare (N,k,Aplus,Aminus) against (-N,-k,-Aplus,-Aminus); does NOT compare same applied optical field.'
}
print(json.dumps(result,ensure_ascii=False,indent=2))
Path(__file__).with_name('M0_influence_exterior_results.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
if max(result['max_det_wedge_error'],result['normalization_error'],result['hermiticity_swap_error'])>1e-10 or result['branch_variation_derivative']['difference']>1e-7:
    raise SystemExit('VALIDATION FAILED')
