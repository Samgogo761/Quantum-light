#!/usr/bin/env python3
"""Dual witness and robustness in THE ANOMALOUS-LINEAR PROXY ONLY.
No claim here of a physical quantum HHG output or AFM-exclusive quantum advantage.
"""
from pathlib import Path
import numpy as np, json, math
ROOT=Path(__file__).resolve().parent
j=json.loads((ROOT/'qmath_16_28_photon_normalized_results.json').read_text())
i=j['input']
W=np.array(i['mean_free_photon_energy_matrix'],float)
U=i['fixed_input_energy']
m=np.array([i['m_even'],i['m_odd']],float)
classical=U/np.linalg.eigvalsh(W)[0]
RNG=np.random.default_rng(20261026)
out={'purpose':'Checks classical GS-P dual certificate in idealized 2-mode linear anomalous-moment PROXY with physically motivated photon frequency prefactor. This does not certify quantized HHG.',
     'input_energy':U,'classical_upper_bound':classical,'BSV_witness':float(np.sum(m)),
     'classical_witness_gap':float(np.sum(m)-classical),
     'rows':[]}
# Direct stochastic positive-P source: enforce exactly U per coherent amplitude,
# pair ±alpha to zero mean. The real part of alpha^T alpha is the witness in an invertible readout.
mx=-1e9
for _ in range(10000):
    a=(RNG.normal(size=2)+1j*RNG.normal(size=2))/math.sqrt(2)
    a=a*math.sqrt(U/np.vdot(a,W@a).real)
    value=(a.T@a).real
    mx=max(mx,value)
out['random_classical_coherent_samples']={'count':10000,'max_witness':float(mx),
  'all_below_bound':bool(mx<=classical+1e-12)}
# Exact saturating ± coherent input from real low eigenmode.
ww,vec=np.linalg.eigh(W)
alpha=np.sqrt(U/ww[0])*vec[:,0]
out['exact_achieving_classical_input']={'alpha_pm':alpha.tolist(),
 'zero_mean_by_equal_pm_mix':True, 'mean_energy':float(alpha.T@W@alpha),
 'anomalous_witness':float(alpha.T@alpha)}
for r in j['readout']:
    L,eta=r['L'],r['eta']
    two=sorted([x for x in j['dataset'] if x['L']==L and x['eta']==eta], key=lambda x:x['Delta'])
    G=np.array([[complex(*x['ke']),complex(*x['ko'])] for x in two])
    y=G@m
    inv=np.linalg.inv(G)
    # analytic linear witness: Re [1,1] G^{-1} y <= U/lambda_min(W), positive-P.
    d=np.ones(2)@inv
    quantum=float(np.real(d@y))
    assert abs(quantum-np.sum(m))<1e-9
    # relative noise calibrated to the sum of the mode response magnitudes,
    # i.e., a bright-channel amplitude scale, not the much smaller dark output.
    trials=[]
    scales=np.linalg.norm(G,axis=1)*np.linalg.norm(m)
    for rel in (.001,.005,.01,.02):
        vals=[]
        for _ in range(2000):
            noise=(RNG.normal(size=2)+1j*RNG.normal(size=2))/math.sqrt(2)
            obs=y+rel*scales*noise
            vals.append(np.real(d@obs))
        vals=np.asarray(vals)
        trials.append({'rms_fraction_of_bright_response':rel,
          'witness_5th_percentile':float(np.quantile(vals,.05)),
          'fraction_exceeding_classical_bound':float(np.mean(vals>classical)),
          'witness_standard_deviation':float(np.std(vals))})
    out['rows'].append({'L':L,'eta':eta,
         'dual_measurement_coefficients':[ [float(v.real),float(v.imag)] for v in d],
         'quantum_witness':quantum,'classical_max':classical,'condition':r['condition'],
         'noise_robustness':trials})
(ROOT/'qmath_16_28_classical_dual_results.json').write_text(json.dumps(out,ensure_ascii=False,indent=2))
print('bound',classical,'BSV',sum(m),'random max',mx)
for row in out['rows']:
 print('eta',row['eta'],'L',row['L'],'condition',round(row['condition'],2),'noise 1% pass',row['noise_robustness'][2]['fraction_exceeding_classical_bound'])
