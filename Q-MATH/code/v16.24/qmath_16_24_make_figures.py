from pathlib import Path
import json
import matplotlib.pyplot as plt
R=Path(__file__).parent
j=json.loads((R/'qmath_16_24_ward_sumrule_results.json').read_text())
rows=j['grid']; L=[z['L'] for z in rows]
plt.figure(figsize=(7.1,4.7))
for i,label in [(1,'Lowest 2 transitions retained'),(2,'Lowest 4 transitions retained'),(3,'Lowest 6 transitions retained'),(4,'All 8 transitions')]:
  plt.plot(L,[z['staticK_keep_0_2_4_6_8'][i] for z in rows],marker='o',label=label)
plt.axhline(0,linestyle='--',linewidth=1)
plt.xlabel('Brillouin-zone mesh (L × L)')
plt.ylabel('Static curvature K per k (model units)')
plt.title('M0 static Ward sum rule: missing virtual transitions')
plt.legend(fontsize=8)
plt.grid(alpha=.2);plt.tight_layout()
plt.savefig(R/'ward_static_virtual_truncation.png',dpi=180)
plt.close()

p=json.loads((R/'qmath_16_24_shared_pair_toy_results.json').read_text())['cases']
plt.figure(figsize=(6.9,4.5))
for g in (0.005,0.015,0.025):
 a=[x for x in p if x['gp']==g]
 plt.plot([x['nmax'] for x in a],[x['mean_bare_photon_number'] for x in a],marker='o',label=f'gp={g}')
plt.xlabel('Fock cutoff Nmax')
plt.ylabel('Bare photon population in dressed ground state')
plt.title('Two M0-derived electronic sectors, one shared photon mode')
plt.legend();plt.grid(alpha=.2);plt.tight_layout()
plt.savefig(R/'shared_pair_toy_fock_convergence.png',dpi=180)
plt.close()
