#!/usr/bin/env python3
"""Q-MATH-16.30: production Q16.7 elastic 3-photon T (all Peierls contacts)
versus Q16.29 classical causal chi2 modal output maps. No claim of LSZ/Maxwell.
Also audit frequency-flat calibration and finite-output-bin orthonormality.
"""
import os
for x in ('OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS'):os.environ[x]='1'
import sys,json,time,itertools
from pathlib import Path
import numpy as np
root=Path(__file__).resolve().parent
sys.path.insert(0,str(root))
sys.path.insert(0,str(root/'q16_27_reference/source/reference_q167'))
from qmath_16_29_same_state_readout import modes,overlap_grid,row_project,state
import q167_m0_three_photon_T as q
bins=np.array([.32,.34,.36,.38,.40])

def pc(z):return [float(np.real(z)),float(np.imag(z))]
def dc(z):return complex(*z)
def matrix_from_vec(c):return np.array([[c[0],c[2]/2],[c[2]/2,c[1]]],complex)

def classical_ratio(Q,y,W,U):
    e,v=np.linalg.eigh(W)
    wmh=(v*(1/np.sqrt(e)))@v.T
    from scipy.optimize import minimize
    def eval_lam(l):
        k=np.conj(l)@Q
        bn=U*np.linalg.svd(wmh@matrix_from_vec(k)@wmh,compute_uv=False)[0]
        return float(abs(np.vdot(l,y))/bn) if bn>0 else 0.
    if Q.shape[0]==1: return {'ratio':eval_lam(np.ones(1,complex))}
    # normalized complex 2-vector projective slope
    best=[]
    for a in [(0,0),(1,0),(0,1),(-1,1),(2,-2),(-2,4)]:
        def fun(x):return -eval_lam(np.array([1,x[0]+1j*x[1]]))
        r=minimize(fun,np.array(a,dtype=float),method='Nelder-Mead',options={'maxiter':1400,'xatol':1e-10,'fatol':1e-10})
        best.append(r)
    op=min(best,key=lambda t:t.fun)
    l=np.array([1,op.x[0]+1j*op.x[1]])
    return {'ratio':eval_lam(l),'dual_vector':[pc(x) for x in l],'optimizer_success':bool(op.success),'spread':float(max(r.fun for r in best)-op.fun)}

def raw_data():
    d=json.loads((root/'qmath_16_29_material_dual_results.json').read_text())
    r=json.loads((root/'qmath_16_29_results.json').read_text())
    m=np.array([dc(z) for z in r['input_m']]);Uin=float(r['U']);W=np.array(r['input_energy_W'])
    return d,r,m,Uin,W

def audit_previous():
    doc,r,m,U,W=raw_data()
    rows=[]
    for case in doc['cases']:
        if case['L']!=201 or abs(case['Delta']-.8)>1e-8: continue
        K=np.array([[dc(z) for z in row['material_row']] for row in case['rows']]);F=np.array([[dc(z) for z in row['frequency_flat_control_row']] for row in case['rows']]);
        UF,s,V=np.linalg.svd(F,full_matrices=True);P=UF[:,3:]@UF[:,3:].conj().T
        Y=K@m;Yf=F@m
        target=P@Y
        # Calibrated flat model only; row-by-row uncertainty tests.
        rng=np.random.default_rng(162930)
        calibration=[]
        for eps in [0.001, 0.003, 0.01, 0.02, 0.05, 0.10]:
            amps=[]
            for _ in range(3000):
                # complex circular calibration error in each output bin
                rr=rng.uniform(0,eps,5)*np.exp(2j*np.pi*rng.random(5))
                amps.append(np.linalg.norm(P@(rr*Yf))/max(1e-30,np.linalg.norm(target)))
            calibration.append({'max_rel_g':eps,'random_median_leakage_to_material':float(np.median(amps)),
                                'random_p95_leakage_to_material':float(np.quantile(amps,.95)),
                                'random_max_leakage_to_material':float(np.max(amps))})
        # Worst-case bound: ||P diag(Yf) delta|| <= delta ||P diag(Yf)||_op sqrt(5) if max|delta_i|<=eps
        A=P@np.diag(Yf)
        singular=np.linalg.svd(A,compute_uv=False)
        smax=float(singular[0]); targetnorm=float(np.linalg.norm(target))
        l2threshold=targetnorm/smax
        maxnorms=[]
        for eps in [.01,.03,.05]:
            maxnorms.append({'eps':eps,'upper_l2_ratio':float(eps*np.sqrt(5)*smax/targetnorm)})
        # Flat model with unknown arbitrary spectral dispersion yields different subspace. 5 linearly independent basis vectors reachable.
        # Projected centered dark channel remains exact because B_center:m=0 under flat response.
        calibration_K=F
        rows.append({'eta':case['eta'],'L':case['L'], 'chi_proxy_ratio':case['dual_certificate']['ratio_BSV_to_classical_positive_P_energy_bound'],
                 'target_fraction':targetnorm/np.linalg.norm(Y), 'center_flat_abs':float(abs(Yf[2])),
                 'center_material_abs':float(abs(Y[2])), 'relative_calibration_l2_threshold_for_signal':l2threshold,
                 'calibration_mc':calibration,'bound':maxnorms,
                 'flat_rank_specified':int(np.linalg.matrix_rank(F)),
                 'flat_rank_arbitrary_row_g':int(np.linalg.matrix_rank(np.eye(5)@np.diag(Yf+np.where(np.abs(Yf)<1e-12,1e-5,0)))),
                 'flat_row_values':[pc(x) for x in Yf],
                 'zero_center_flat_significant':bool(abs(Yf[2])<1e-10),
                 'underlying_report_uses_same_P_for_all_g':False})
    return rows

def prep_k(kf,N):
    k=q.geo.k_cart(kf)
    v=q.vertices(k,N)
    E,U=np.linalg.eigh(v['0']);Eg=float(E[0])
    A={name:U.conj().T@val@U for name,val in v.items() if name!='0'}
    return E,Eg,A

def tkernel_e(E,Eg,A,w1,w2,gamma):
    frequencies=[w1,w2,-(w1+w2)]
    verts={frozenset([0]):A['p'],frozenset([1]):A['p'],frozenset([2]):A['h'],
           frozenset([0,1]):A['pp'],frozenset([0,2]):A['ph'],frozenset([1,2]):A['ph'],
           frozenset([0,1,2]):A['pph']}
    t=A['pph'][0,0].copy()
    for ss in [frozenset([0]),frozenset([1]),frozenset([2]),frozenset([0,1]),frozenset([0,2]),frozenset([1,2])]:
        cs=frozenset([0,1,2])-ss
        en=sum(frequencies[i] for i in ss)
        t+=(verts[cs][0,:]@(verts[ss][:,0]/(Eg+en-E+1j*gamma)))
    for p in itertools.permutations((0,1,2)):
        a,b,c=p
        xa=verts[frozenset([a])][:,0]/(Eg+frequencies[a]-E+1j*gamma)
        xb=(verts[frozenset([b])]@xa)/(Eg+frequencies[a]+frequencies[b]-E+1j*gamma)
        t+=verts[frozenset([c])][0,:]@xb
    return t

def compute_t(L,gamma,step=.02):
    mw,_,mf,W,Wh=modes()
    data=[overlap_grid(om,step) for om in bins]
    tvals=[np.zeros(len(x[0]),complex) for x in data]
    start=time.monotonic()
    for i in range(L):
        for j in range(L):
            kfrac=((i+.5)/L,(j+.5)/L)
            E,Eg,A=prep_k(kfrac,1)
            for oi,(w1,w2,wght) in enumerate(data):
                for ai,(u,v) in enumerate(zip(w1,w2)):
                    tvals[oi][ai]+=tkernel_e(E,Eg,A,float(u),float(v),gamma)
    T=[];F=[];chis=[]
    for oi,(w1,w2,wght) in enumerate(data):
        vals=tvals[oi]/L**2
        v=row_project(bins[oi],vals,w1,w2,wght,mw,mf)
        flat=row_project(bins[oi],np.ones_like(vals)*vals[np.argmin(abs(w1-bins[oi]/2))],w1,w2,wght,mw,mf)
        T.append(v);F.append(flat)
        chis.append([pc(x) for x in vals])
    T=np.array(T);F=np.array(F)
    d,r,m,U,W=raw_data()
    Uf,s,Vh=np.linalg.svd(F,full_matrices=True)
    Q=Uf[:,3:].conj().T@T
    yy=Q@m
    control=float(np.linalg.norm(Uf[:,3:].conj().T@F))
    ratio=classical_ratio(Q,yy,W,U)
    result={'L':L,'gamma':gamma,'frequency_step':step,'time_s':time.monotonic()-start,
             'scope':'Unamputated regulated elastic bare T^(3), not asymptotic LSZ-normalized quantum output.',
             'T_matrix':[[pc(x) for x in row] for row in T],
             'Fflat_matrix':[[pc(x) for x in row] for row in F],
             'model_only_signal_over_T':float(np.linalg.norm(Q@m)/np.linalg.norm(T@m)),
             'center_flat_abs':float(abs((F@m)[2])),
             'center_T_abs':float(abs((T@m)[2])),
             'control_projected_F_norm':control,
             'dual':ratio,'flat_singular_values':s.tolist(),
             'material_singular_values':np.linalg.svd(T,compute_uv=False).tolist()}
    return result

def output_modes_audit():
    # exact Gaussian L2 overlaps on the full real axis for identical bandwidth Gaussian rms sigma
    # v_i(Om) ∝ exp[-(Om-Om_i)^2/(4 sigma^2)] wavefunction has intensity std sigma
    return [{'rms_sigma':s,'adjacent_overlap':float(np.exp(-(.02)**2/(8*s*s))),
             'fifth_over_one_overlap':float(np.exp(-(.08)**2/(8*s*s)))} for s in (.002,.003,.004,.005,.007,.01)]

if __name__=='__main__':
    previous=audit_previous()
    Ts=[]
    for l in [9,17,25]:
        print('Running original M0 T; L=',l,flush=True)
        Ts.append(compute_t(l,.08,.02))
        print(' T outcome',l, Ts[-1]['dual'],Ts[-1]['model_only_signal_over_T'], 'seconds',round(Ts[-1]['time_s'],1),flush=True)
    res={'identity':'Q-MATH-16.30 independent production M0 T^(3) vs classical retarded chi2 proxy and calibration robustness',
         'note':'The T calculation is a regulated bare elastic energy-shell coefficient, NOT a fully dressed, bath-normalized, long-time quantum H2 output. Numeric ratios are diagnostics only.',
         'prior_201_calibration':previous,'new_T_scans':Ts,'output_orthogonality':output_modes_audit()}
    (root/'qmath_16_30_bridge_results.json').write_text(json.dumps(res,ensure_ascii=False,indent=2),encoding='utf-8')
    print('DONE wrote result')
