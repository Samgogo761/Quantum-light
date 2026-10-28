import os
os.environ.update(OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')
import sys,dataclasses,json,numpy as np
from pathlib import Path
B=Path(__file__).resolve().parent
sys.path.insert(0,str(B/'sources/QMATH_16_14/Q16_13_original/src_0'))
import q167_m0_three_photon_T as q
Sx=np.kron(np.eye(4),np.array([[0,1],[1,0]],complex));P=q.m.op_P()
orig,oldb=q.m.R0,q.BONDS
cases=[]
for kf in [(.173,.271),(.4158,.1683),(-.19,.27)]:
 for lam,delta in [(.0,.8),(.02,.8),(.1,.8),(-.1,.8),(.1,0.)]:
  k=q.geo.k_cart(kf)
  for order in range(8):
   dirs=[q.p]*(order//2)+[q.h]*(order-order//2)
   q.m.R0=dataclasses.replace(orig,lam_so=lam,delta=delta);q.BONDS=q.bonds()
   O=q.exact_vertex(k,1,dirs)
   q.m.R0=dataclasses.replace(orig,lam_so=-lam,delta=delta);q.BONDS=q.bonds()
   spin_target=q.exact_vertex(k,-1,dirs)
   q.m.R0=dataclasses.replace(orig,lam_so=lam,delta=delta);q.BONDS=q.bonds()
   inv_target=q.exact_vertex(-k,-1,dirs)
   spin_err=float(np.linalg.norm(Sx@O@Sx.conj().T-spin_target))
   P_err=float(np.linalg.norm(P@O@P.conj().T-(-1)**order*inv_target))
   cases.append(dict(kf=list(kf),lam=lam,delta=delta,order=order,spin_err=spin_err,P_err=P_err))
q.m.R0=orig;q.BONDS=oldb
result=dict(description='exact Fourier Peierls derivatives 0..7 M0 spin π and spatial inversion with original source code',num_cases=len(cases),max_spin_err=max(c['spin_err'] for c in cases),max_P_err=max(c['P_err'] for c in cases),cases=cases)
(B/'qmath_16_15_vertices_symmetry.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
print('tested',len(cases),'max spin',result['max_spin_err'],'max parity',result['max_P_err'])
