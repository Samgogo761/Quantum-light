#!/usr/bin/env python3
"""Q16.15: spin-rotation/inversion symmetry, actual M0 frozen bond implementation.
Original 2026 Q16.7 source extracted verbatim to sources/QMATH_16_14/Q16_13_original/src_0.
All scattering is gamma-regularized bare T; not LSZ or a predicted observable.
"""
import os
os.environ.update(OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')
import sys,dataclasses,json, numpy as np,time
from pathlib import Path
B=Path(__file__).resolve().parent
SRC=B/'sources/QMATH_16_14/Q16_13_original/src_0'
sys.path.insert(0,str(SRC))
import q167_m0_three_photon_T as q

def set_r0(lam,delta):
    q.m.R0=dataclasses.replace(q.m.R0,lam_so=float(lam),delta=float(delta))
    q.BONDS=q.bonds() # IMPORTANT: SOC Fourier coefficients cached at q.BONDS import

def t3_bz(L, lam, delta, gamma=.25):
    old=q.m.R0;oldb=q.BONDS
    try:
        set_r0(lam,delta)
        tot={1:0j,-1:0j}
        for i in range(L):
            for j in range(L):
                k=((i+0.5)/L,(j+0.5)/L)
                for N in (1,-1):
                    tot[N]+=q.calc(k,N,w1=.16,w2=.20,gamma=gamma)['total']
        return {str(N):[float((tot[N]/(L*L)).real),float((tot[N]/(L*L)).imag)] for N in (1,-1)}
    finally:
        q.m.R0=old;q.BONDS=oldb

def test_symmetries():
    # Spin π x: Ux H(N, lambda) Ux^-1 = H(-N,-lambda) same k.
    sx=np.array([[0,1],[1,0]],complex)
    Ux=np.kron(np.eye(4),sx)
    P=q.m.op_P()
    entries=[]
    for kf in [(0.173,0.271),(.4158,.1683),(-.19,.27)]:
        k=q.geo.k_cart(kf)
        for (lam,delta) in [(.0,.8),(.1,.8),(.1,.4)]:
            old=q.m.R0;oldb=q.BONDS
            try:
                set_r0(lam,delta)
                H0=q.exact_vertex(k,1,[])
                Hminus=q.exact_vertex(-k,-1,[])
                pt_err=np.linalg.norm(P@H0@P.conj().T-Hminus)
                set_r0(-lam,delta)
                Hspin=q.exact_vertex(k,-1,[])
                spin_err=np.linalg.norm(Ux@H0@Ux.conj().T-Hspin)
                entries.append(dict(kf=kf,lam=lam,delta=delta, P_err=float(pt_err),spin_pi_err=float(spin_err)))
            finally:q.m.R0=old;q.BONDS=oldb
    return entries

def main():
    out=dict(model='production Q16.7 code with refreshed cached Peierls bonds',
             scope='formal gamma regularized third-order elastic T; no Maxwell/LSZ',
             exact_spin_symmetries=test_symmetries(), t3_bz=[],coeff_tests=[])
    print('symmetry max P, spin',max(x['P_err'] for x in out['exact_spin_symmetries']),max(x['spin_pi_err'] for x in out['exact_spin_symmetries']),flush=True)
    for L in (5,9,15):
        for lam,delta in [(0,.8),(.02,.8),(.05,.8),(.1,.8),(-.1,.8),(.1,0),(.1,-.8)]:
            t=time.time();v=t3_bz(L,lam,delta)
            z=dict(L=L,lam=lam,delta=delta,Tplus=v['1'],Tminus=v['-1'],secs=round(time.time()-t,3))
            out['t3_bz'].append(z)
            print(f'L={L:2d}  SOC={lam:+.3f} Delta={delta:+.2f} |T|={abs(complex(*v["1"])):.7g} {z["secs"]}s',flush=True)
    for L in (5,9,15):
        get=lambda l,d:complex(*next(a for a in out['t3_bz'] if a['L']==L and a['lam']==l and a['delta']==d)['Tplus'])
        xp=get(.1,.8);xm=get(-.1,.8);xd=get(.1,-.8);xn=get(.1,0.);x0=get(0,.8)
        out['coeff_tests'].append(dict(L=L,nonzero_scale=abs(xp),soc_odd_resid=float(abs(xp+xm)/max(abs(xp),1e-30)),N_odd_resid=float(abs(xp+xd)/max(abs(xp),1e-30)),Delta_zero_resid=float(abs(xn)/max(abs(xp),1e-30)),soc_zero_resid=float(abs(x0)/max(abs(xp),1e-30)),small_SOC_ratio_abs=float(abs(get(.02,.8)/xp)),medium_SOC_ratio_abs=float(abs(get(.05,.8)/xp))))
    (B/'qmath_16_15_soc_symmetry_results.json').write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf-8')
    print('coeff_tests',out['coeff_tests'])
if __name__=='__main__':main()
