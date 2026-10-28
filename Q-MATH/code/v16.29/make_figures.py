from pathlib import Path
import json,numpy as np,matplotlib.pyplot as plt
R=Path(__file__).parent
r=json.loads((R/'qmath_16_29_results.json').read_text())
p=json.loads((R/'qmath_16_29_material_dual_results.json').read_text())
fig,ax=plt.subplots(figsize=(7.7,4.5))
for eta in (.08,.03):
 cases=sorted([c for c in r['cases'] if c['eta']==eta and c['Delta']==.8],key=lambda c:c['L'])
 x=[c['L'] for c in cases]; y=[c['analysis']['pair_34_38']['cost_over_U'] for c in cases]
 ax.plot(x,y,marker='o',label=f'2 bins, eta={eta}')
ax.axhline(1.,linestyle='--',color='gray',label='Classical feasibility threshold')
ax.set(xlabel='BZ mesh L x L',ylabel='Minimum classical energy / BSV energy',title='Two H2 output frequencies: classical mimic exists below 1')
ax.legend();fig.tight_layout();fig.savefig(R/'qmath_16_29_two_bin_classical_mimic.png',dpi=170);plt.close(fig)
fig,ax=plt.subplots(figsize=(7.7,4.5))
for eta in (.08,.03):
 cases=sorted([c for c in p['cases'] if c['eta']==eta and c['Delta']==.8],key=lambda c:c['L'])
 x=[c['L'] for c in cases]; y=[c['dual_certificate']['ratio_BSV_to_classical_positive_P_energy_bound'] for c in cases]
 ax.plot(x,y,marker='o',label=f'eta={eta}')
ax.axhline(1.,linestyle='--',color='gray',label='Positive-P classical bound')
ax.set(xlabel='BZ mesh L x L',ylabel='BSV / optimal classical bound (proxy)',title='Five bins: frequency-flat-null, material-selective witness')
ax.legend();fig.tight_layout();fig.savefig(R/'qmath_16_29_material_only_dual.png',dpi=170);plt.close(fig)
fig,ax=plt.subplots(figsize=(7.7,4.5))
c=next(c for c in r['cases'] if c['L']==201 and c['eta']==.08 and c['Delta']==.8)
K=np.array([[complex(*z) for z in row] for row in c['analysis']['K']]);F=np.array([[complex(*z) for z in row] for row in c['analysis']['K_flat']]);
for j,name in enumerate(['M_ee','M_oo','M_eo (off-diagonal)']):
 ax.plot(r['Omega_values'],abs(K[:,j]),marker=['o','s','^'][j],label='M0 '+name)
 if j==2:ax.plot(r['Omega_values'],abs(F[:,j]),linestyle='--',label='Flat-kernel M_eo')
ax.set(xlabel='H2 output frequency Omega',ylabel='Absolute response-kernel coefficient',title='Cross-mode response revives away from central H2 output')
ax.legend();fig.tight_layout();fig.savefig(R/'qmath_16_29_cross_mode_kernel.png',dpi=170);plt.close(fig)
for name in ['qmath_16_29_two_bin_classical_mimic.png','qmath_16_29_material_only_dual.png','qmath_16_29_cross_mode_kernel.png']:
 print(R/name,(R/name).stat().st_size)
