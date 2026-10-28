#!/usr/bin/env python3
"""Physically motivated 1/sqrt(omega1*omega2) correction for the Q16.28
CLASSICAL susceptibility contraction PROXY. It is not an input-output S matrix.
Classical GS-P energy witness is exact only for the explicit two canonical modes
and an invertible linear anomalous-moment readout assumed in the proxy.
"""
import json, math, numpy as np
from pathlib import Path
from qmath_16_28_multimode_screen import CACHE, ROOT, kkey, CENTER, OM, SIG

def cp(z):return [float(z.real),float(z.imag)]

def make_rows(cache,L,eta,D,h=.01, sigma=SIG, freq_weight=True):
    w=np.linspace(.10,.26,int(round(.16/h))+1)
    xi=w-CENTER
    tw=np.ones(len(w))*h;tw[[0,-1]]*=.5
    ue=np.exp(-.5*(xi/sigma)**2);ue/=np.sqrt(np.sum(tw*ue**2))
    uo=xi*ue;uo/=np.sqrt(np.sum(tw*uo**2))
    # The photon free energy is not diagonal in the parity basis, even though
    # each diagonal expectation equals the central frequency.
    W=np.array([[np.sum(tw*w*ue*ue),np.sum(tw*w*ue*uo)],
                [np.sum(tw*w*uo*ue),np.sum(tw*w*uo*uo)]])
    xx=np.linspace(.10,.18,int(round(.08/h))+1)
    half=np.array([complex(*cache[kkey(L,D,eta,float(round(x,2)))]) for x in xx])
    chi=np.r_[half,half[-2::-1]]
    factor=1/np.sqrt(w*(OM-w)) if freq_weight else np.ones_like(w)
    v_e=np.sum(tw*factor*ue*ue[::-1])
    v_o=-np.sum(tw*factor*uo*uo[::-1])
    K_e=np.sum(tw*factor*chi*ue*ue[::-1]); K_o=np.sum(tw*factor*chi*uo*uo[::-1])
    K_eo=np.sum(tw*factor*chi*(ue*uo[::-1]+uo*ue[::-1])/2)
    ce=half[-1]
    assert v_e>0 and v_o>0
    assert abs(K_eo)<1e-11
    return dict(ke=K_e,ko=K_o,cross=K_eo,qe=v_e,qo=v_o,W=W,
                material_curvature=K_e/v_e + K_o/v_o,
                chi_center=ce,mode_photon_norm=[np.sum(tw*ue**2),np.sum(tw*uo**2)],
                mode_orth=np.sum(tw*ue*uo))

def derive(cache):
    cases=[(.08,161),(.08,201),(.03,101),(.03,161),(.03,201)]
    data=[]; probes=[]
    ref=make_rows(cache,201,.08,.4)
    qe,qo=ref['qe'],ref['qo']
    m_e=math.sqrt(.1*1.1)
    m_o=m_e*qe/qo
    n_o=.5*(math.sqrt(1+4*m_o*m_o)-1)
    n_e=.1
    m_vec=np.array([m_e,m_o]);N=np.diag([n_e,n_o]); W=ref['W']
    U=float(np.trace(W@N))
    lam_min=float(np.linalg.eigvalsh(W)[0]); classical_max=U/lam_min
    # Explicit ±alpha classical mixture saturates the two-mode energy bound
    ev,V=np.linalg.eigh(W)
    alpha=np.sqrt(U/ev[0])*V[:,0]
    assert abs(alpha.T@alpha-classical_max)<1e-12
    for eta,L in cases:
        K=[]
        for D in (.4,.8):
            z=make_rows(cache,L,eta,D)
            null=z['chi_center']*(z['qe']*m_e-z['qo']*m_o)
            signal=np.dot([z['ke'],z['ko']],m_vec)
            assert abs(null)<1e-12, null
            row=dict(L=L,eta=eta,Delta=D,ke=cp(z['ke']),ko=cp(z['ko']),
                     r_material=cp(z['material_curvature']),cross_abs=float(abs(z['cross'])),
                     uniform_kernel_dark_abs=float(abs(null)),
                     calibrated_dark_output=cp(signal),
                     dark_rel_to_mode_e=float(abs(signal)/(abs(z['ke']*m_e))),
                     qe=z['qe'],qo=z['qo'])
            data.append(row);K.append([z['ke'],z['ko']])
        K=np.asarray(K)
        Q=K/np.linalg.norm(K,axis=1)[:,None]
        s=np.linalg.svd(Q,compute_uv=False)
        # Normalize two calibrated measurements separately (not a claim of noise-resolved performance)
        y=K@m_vec
        reco=np.linalg.solve(K,y)
        witness=float(reco.sum().real)
        flat=np.array([[1*qe,-1*qo],[2*qe,-2*qo]],dtype=complex)
        probes.append(dict(L=L,eta=eta,det_abs=float(abs(np.linalg.det(K))),
                  svals=[float(z) for z in s],condition=float(s[0]/s[1]),
                  linear_estimate=[float(v.real) for v in reco],
                  linear_witness=witness, classical_energy_bound=classical_max,
                  witness_gap=witness-classical_max,
                  relative_witness_excess=witness/classical_max,
                  output_dark_abs=[float(abs(t)) for t in y],
                  uniform_kernel_test_rank=int(np.linalg.matrix_rank(flat))))
    return dict(description='Q16.28 CLASSICAL chi2 proxy with photon-like 1/sqrt(w1 w2) normalization; 2 orthogonal actually occupied squeezing modes; NOT verified quantum HHG emission.',
         input={'band':[.10,.26],'center':CENTER,'sigma':SIG,'omega_out':OM,
                'u_even_and_u_odd_orthogonality':float(abs(ref['mode_orth'])),
                'photon_prefactors':{'qe':qe,'qo':qo,'ratio_qe_over_qo':qe/qo},
                'n_even':n_e,'n_odd':n_o,'m_even':m_e,'m_odd':m_o,
                'mean_free_photon_energy_matrix':W.tolist(),
                'mean_photon_number':n_e+n_o,'fixed_input_energy':U,
                'classical_max_readout_energy_only':classical_max,
                'classical_optimum_pm_coherent_alpha':alpha.tolist(),
                'bsw_anomalous_witness':m_e+m_o,
                'mean_frequency_bottom_eigenvalue':lam_min},
         dataset=data,readout=probes)

if __name__=='__main__':
    cache=json.loads(CACHE.read_text())
    out=derive(cache)
    p=ROOT/'qmath_16_28_photon_normalized_results.json'
    p.write_text(json.dumps(out,indent=2,ensure_ascii=False))
    print(json.dumps({'input':out['input'],'readout':out['readout'],'rows':out['dataset']},ensure_ascii=False,indent=2))
