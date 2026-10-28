#!/usr/bin/env python3
"""Q-MATH-16.10: construction and exact moment checks for a positive-Glauber-Sudarshan-P null control.
No M0 Peierls calculation is performed by this script. Python 3 / numpy only.
"""
import json
import numpy as np
from pathlib import Path

n = np.array([0.15, 0.55], dtype=float)
E = float(n.sum())
s = np.sqrt(n*(n+1))
M = np.diag(s)
N = np.diag(n)
F2 = E**2 + np.trace(N@N) + np.trace(M.conj().T@M)
F2 = float(F2.real)
t = float(E/s.sum())
q = E**2/F2       # nonvacuum probability
R = F2/E         # intensity when nonvacuum
w = s/s.sum()    # distribution of mode intensities when nonvacuum
# explicit positive P: vacuum (1-q), or +/-sqrt(R*w_j) independently in each mode (prob q/4 each)
classical_alphas = [np.zeros(2,complex)]
classical_probs = [1-q]
for s1 in [-1,1]:
  for s2 in [-1,1]:
    classical_alphas.append(np.sqrt(R*w)*np.array([s1,s2],complex))
    classical_probs.append(q/4)

# Weighted physical-energy version of the nuclear-norm classical feasibility theorem
omega = np.array([0.16, 0.20])
energy_physical = float(np.dot(omega,n))
weighted_M = np.diag(np.sqrt(omega))@M@np.diag(np.sqrt(omega))
weighted_norm = float(np.linalg.svd(weighted_M, compute_uv=False).sum())
t_energy = energy_physical/weighted_norm
assert t_energy < 1.0
# Takagi sign construction for weighted M, mapped back to original mode amplitudes
M_energy_cl = t_energy*M
assert np.linalg.svd(np.diag(np.sqrt(omega))@M_energy_cl@np.diag(np.sqrt(omega)),compute_uv=False).sum()-energy_physical < 1e-12

E_cl = sum(p*float(np.vdot(a,a).real) for p,a in zip(classical_probs,classical_alphas))
F2_cl = sum(p*float(np.vdot(a,a).real)**2 for p,a in zip(classical_probs,classical_alphas))
M_cl = sum((p*np.outer(a,a) for p,a in zip(classical_probs,classical_alphas)),np.zeros((2,2),complex))
mean_cl = sum((p*a for p,a in zip(classical_probs,classical_alphas)),np.zeros(2,complex))
K_even = np.diag([1.,0.]).astype(complex)
K_odd = np.diag([0.,1.]).astype(complex)
K = [K_even+K_odd,K_even-K_odd,K_even, K_odd, np.array([[.8,.12j],[.12j,1.2]],complex)]
beta_quantum = np.array([np.sum(k*M) for k in K])
beta_classical = np.array([np.sum(k*M_cl) for k in K])
coh_I_q = np.abs(beta_quantum[:2])**2
coh_I_c = np.abs(beta_classical[:2])**2
contrast_q = float((coh_I_q[0]-coh_I_q[1])/sum(coh_I_q))
contrast_c = float((coh_I_c[0]-coh_I_c[1])/sum(coh_I_c))
# diagnostic tensor noncollinearity
D = (np.linalg.norm(K_even)**2*np.linalg.norm(K_odd)**2-abs(np.vdot(K_even,K_odd))**2)
# proportional-control normalised domain contrast: for K_even=c*K_odd it does not depend on M
c = 0.40
P_even = c*(K_even+K_odd)
P_odd = K_even+K_odd
p_bq=np.array([np.sum((P_even+z*P_odd)*M) for z in (1,-1)])
p_bc=np.array([np.sum((P_even+z*P_odd)*M_cl) for z in (1,-1)])
chi=lambda arr:float((abs(arr[0])**2-abs(arr[1])**2)/(abs(arr[0])**2+abs(arr[1])**2))
assert abs(sum(classical_probs)-1)<1e-13
assert abs(E_cl-E)<1e-13
assert abs(F2_cl-F2)<1e-13
assert np.linalg.norm(mean_cl)<1e-13
assert np.max(np.abs(M_cl-t*M))<1e-13
assert np.max(np.abs(beta_classical-t*beta_quantum))<1e-13
assert abs(contrast_c-contrast_q)<1e-13
assert abs(chi(p_bq)-chi(p_bc))<1e-13
assert D>0
out={
 'scope':'Finite two-mode ideal weak pair-conversion model; NOT M0 or measured HHG',
 'quantum_normal_occupations':n.tolist(),
 'input_total_E':E,
 'mode_energies':omega.tolist(),
 'free_photon_energy':energy_physical,
 'weighted_anomaly_nuclear_norm':weighted_norm,
 'fixed_physical_energy_classical_attenuation_t':t_energy,
 'BSV_F2':F2,
 'BSV_anomaly_Takagi_singular_values':s.tolist(),
 'BSV_anomaly_nuclear_norm':float(s.sum()),
 'classical_attenuation_t':t,
 'classical_explicit_P_weights':classical_probs,
 'classical_explicit_P_amplitudes':[[(z.real,z.imag) for z in a] for a in classical_alphas],
 'classical_total_E':E_cl,
 'classical_F2':F2_cl,
 'classical_mean_field_norm':float(np.linalg.norm(mean_cl)),
 'max_classical_anomaly_scaling_error':float(np.max(np.abs(M_cl-t*M))),
 'max_coherent_response_scaling_error':float(np.max(np.abs(beta_classical-t*beta_quantum))),
 'quantum_domain_coherent_contrast':contrast_q,
 'classical_domain_coherent_contrast':contrast_c,
 'K_even_K_odd_gram_det':float(D),
 'proportional_kernel_domain_contrast':chi(p_bq),
 'proportional_kernel_classical_domain_contrast':chi(p_bc),
 'sum_energy_shell_example':{'omega1':.16,'omega2':.20,'Omega':.36,'allowed_elastic_Delta':0.0,'example_inelastic_gap':.08,'inelastic_required_Omega':.28}
}
path=Path(__file__).with_name('qmath_16_10_verification.json')
path.write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf-8')
print('PASS all moment, null-control, kernel-rank and proportionality tests')
print('E,F2:',E,F2,'||M||_*:',s.sum(),'classical scale:',t)
print('domain contrast (BSV vs classical):',contrast_q,contrast_c)
