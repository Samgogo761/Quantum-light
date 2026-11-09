import numpy as np, sys, pathlib, json
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from q8_m0_model import *
BASE=pathlib.Path(__file__).resolve().parent
SCF=BASE / 'CrI3_scf_soc_a1h_clean.in'
geo=build_geometry(SCF)
print('Lattice',np.array(geo.lattice_ang).round(8)); print('paired',geo.sites_paired_frac);print('d',geo.d_cart,'q',geo.q_cart); print('bonds',geo.interlayer_bonds)
print('Area',geo.a_cell_ang2,'NN mean',geo.a_nn_mean_ang)
P=op_P(); Ut=op_theta_unitary(); Upt=P@Ut; sZ=op_sz(); I8=np.eye(8)
res={'hermitian':0,'P':0,'T':0,'PT':0,'PT2':float(np.linalg.norm(Upt@Upt.conj()+I8)),'sz':0,'currentP':0,'currentT':0,'sewingH':0,'sewingj':0,'derivative':0,'kramers':0,'current_h8':0}
kfs=[[0,0],[1/3,1/3],[.21,.13],[.44,.37],[.1,.87],[.5,.5],[-.33,.72]]
for N in [-1,1]:
 for kf in kfs:
  k=np.array(geo.k_cart(kf))
  h=h8(geo,R0,k,N); hneg=h8(geo,R0,-k,-N); j=current_ops(geo,R0,k,N); jneg=current_ops(geo,R0,-k,-N)
  res['hermitian']=max(res['hermitian'],np.linalg.norm(h-h.conj().T))
  res['P']=max(res['P'],np.linalg.norm(P@h@P-hneg))
  res['T']=max(res['T'],np.linalg.norm(Ut@h.conj()@Ut.conj().T-hneg))
  res['PT']=max(res['PT'],np.linalg.norm(Upt@h.conj()@Upt.conj().T-h))
  res['sz']=max(res['sz'],np.linalg.norm(h@sZ-sZ@h))
  res['currentP']=max(res['currentP'],max(np.linalg.norm(P@j[a]@P+jneg[a]) for a in [0,1]))
  res['currentT']=max(res['currentT'],max(np.linalg.norm(Ut@j[a].conj()@Ut.conj().T+jneg[a]) for a in [0,1]))
  h2,j2=h8_and_current(geo,R0,k,N)
  res['current_h8']=max(res['current_h8'],np.linalg.norm(h2-h),np.linalg.norm(j2-j))
  ee=np.linalg.eigvalsh(h);res['kramers']=max(res['kramers'],max(abs(ee[::2]-ee[1::2])))
  for G in [geo.k_cart([1,0]),geo.k_cart([0,1])]:
   Ug=op_sewing(geo,G)
   hg=h8(geo,R0,k+G,N); jg=current_ops(geo,R0,k+G,N)
   res['sewingH']=max(res['sewingH'],np.linalg.norm(hg-Ug.conj().T@h@Ug))
   res['sewingj']=max(res['sewingj'],max(np.linalg.norm(jg[a]-Ug.conj().T@j[a]@Ug) for a in [0,1]))
  for a in [0,1]:
   eps=1e-5; dA=np.zeros(2);dA[a]=eps
   numerical=(h8(geo,R0,k+dA,N)-h8(geo,R0,k-dA,N))/(2*eps)
   analytic=dh8(geo,R0,k,N)[a]
   res['derivative']=max(res['derivative'],np.linalg.norm(numerical-analytic))
print('results',json.dumps(res,indent=2))
print('Gamma eigenvalues',np.linalg.eigvalsh(h8(geo,R0,np.zeros(2),1)))
print('gamma ref energies',gamma_energies(R0))
# gap in a coarse grid to ensure insulating half-filled rank-4 projector
kfs=np.stack(np.meshgrid(np.arange(40)/40,np.arange(40)/40,indexing='ij'),axis=-1).reshape(-1,2)
hs=h8(geo,R0,geo.k_cart(kfs),1)
e=np.linalg.eigvalsh(hs)
print('grid 40 gap dir',min(e[:,4]-e[:,3]),'ind',min(e[:,4])-max(e[:,3]))
P0=np.linalg.eigh(hs[413])[1][:,:4]; P0=P0@P0.conj().T
print('projector diagnostics',np.linalg.norm(P0@P0-P0),np.linalg.norm(P0-P0.conj().T))
out={'geometry':{'lattice_ang':geo.lattice_ang,'sites_orig_frac':geo.sites_orig_frac,'sites_paired_frac':geo.sites_paired_frac,'inversion_center_frac':geo.inversion_center_frac,'d_cart':geo.d_cart,'q_cart':geo.q_cart,'a1_cart':geo.a1_cart,'a2_cart':geo.a2_cart,'nn_deltas_cart':geo.nn_deltas_cart,'nnn_terms':geo.nnn_terms,'interlayer_bonds':geo.interlayer_bonds,'reciprocal_2d':geo.reciprocal_2d,'a_cell_ang2':geo.a_cell_ang2},'parameters':R0.as_dict(),'test_residuals':res,'gap_grid_40':{'direct':float(min(e[:,4]-e[:,3])),'indirect':float(min(e[:,4])-max(e[:,3]))}}
(BASE / 'M0_geometry_reference_uploaded.json').write_text(json.dumps(out,ensure_ascii=False,indent=2))
