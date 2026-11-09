from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.special import gammaln
R=Path(__file__).resolve().parent
D=R/'data'
plus=np.load(D/'q13_kernel_23_96_Nplus.npz')['kernel'][1,0]
minus=np.load(D/'q13_kernel_23_96_Nminus.npz')['kernel'][1,0]
K=(plus-minus)/2
j=np.arange(len(K)-2);a=np.diag(K,2)*np.exp(-.5*(gammaln(j+1)+gammaln(j+3)))
x=np.linspace(0,3,1500);h=x*np.abs(np.exp(-x)*np.polynomial.polynomial.polyval(x,a))
fig,ax=plt.subplots(figsize=(8.2,5.2))
ax.plot(x,h,label='Coherent probe: $x|F(x)|$')
ax.plot(x,.098+.25*x,label='Certified affine majorant')
ax.scatter([.7],[.2851050319],marker='o',s=55,label='Pure BSV, $\\bar n=0.7$')
ax.set(xlabel='Coherent-state mean photons $x$',ylabel='Two-photon magnetic winding amplitude (source units)',title='M0: universal positive-P certificate at mean photons 0.7')
ax.set_ylim(0,.82);ax.legend();fig.tight_layout();fig.savefig(R/'q15_global_affine_witness.png',dpi=180);plt.close(fig)
with open(R/'q15_certified_results.json') as f:r=json.load(f)
fig,ax=plt.subplots(figsize=(8.2,4.6))
ns=np.array([v['target_mean'] for v in r['scans']]);bsv=np.array([v['BSV_winding_abs'] for v in r['scans']]);old=np.array([v['linear_classic_tight_23Fock'] for v in r['scans']]);aaff=np.array([v['affine_classic_23Fock'] for v in r['scans']]);
ax.loglog(ns,bsv,'o-',label='BSV response');ax.loglog(ns,old,'s--',label='Certified linear bound, exact $C_*$');ax.loglog(ns,aaff,'d:',label='Certified affine bound');
ax.set(xlabel='Mean pump photon count',ylabel='Source winding amplitude (absolute)',title='Two complementary positive-P bounds');ax.legend();fig.tight_layout();fig.savefig(R/'q15_classical_bounds.png',dpi=180);plt.close(fig)
