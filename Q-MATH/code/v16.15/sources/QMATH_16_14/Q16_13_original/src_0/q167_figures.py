#!/usr/bin/env python3
"""Generate neutral matplotlib diagnostics from saved datasets."""
import json,matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
ROOT=Path(__file__).resolve().parent

def load(g):return json.loads((ROOT/f'q167_gamma_{g}.json').read_text())
# independent grids for each gamma
fig,ax=plt.subplots(figsize=(8,5))
measurements=json.loads((ROOT/'q167_bz_verified_points.json').read_text())['data']
for g in (.05,.10,.25):
    subset=sorted((r for r in measurements if abs(r['gamma']-g)<1e-12),key=lambda row:row['grid'])
    xx=[r['grid'] for r in subset]
    yy=[abs(complex(*r['odd'])) for r in subset]
    ax.plot(xx,yy,marker='o',label=f'gamma = {g:g}')
ax.set(xlabel='Paired Brillouin-zone grid (N x N)',ylabel='|magnetically odd three-photon elastic T coefficient|',title='M0: momentum sampling vs resonant regulator')
ax.legend();ax.grid(alpha=.25);fig.tight_layout();fig.savefig(ROOT/'q167_BZ_convergence.png',dpi=190);plt.close(fig)

data=load('0.25_65')['grids'][-1]['components'];fig,ax=plt.subplots(figsize=(8,5))
terms=[('Direct Peierls V3',complex(*data['direct']['1'])),('Mixed V1,V2',complex(*data['mixed']['1'])),('Three V1 vertices',complex(*data['sequential']['1']))]
total=sum(z for _,z in terms)
for name,z in terms:
    ax.scatter([z.real],[z.imag],s=65,label=name);ax.annotate(name,(z.real,z.imag),xytext=(7,6),textcoords='offset points',fontsize=9)
ax.scatter([total.real],[total.imag],marker='*',s=130,label='Sum');ax.annotate('Total',(total.real,total.imag),xytext=(7,6),textcoords='offset points')
ax.axhline(0,linewidth=.7);ax.axvline(0,linewidth=.7);ax.set(xlabel='Real part',ylabel='Imaginary part',title='Three-photon pathway interference, M0 BZ 65x65, gamma=.25');ax.grid(alpha=.3);fig.tight_layout();fig.savefig(ROOT/'q167_vertex_contributions.png',dpi=190);plt.close(fig)

scan=json.loads((ROOT/'q167_frequency_scan_g0.25.json').read_text());a=scan['pairs'];fig,ax=plt.subplots(figsize=(8,5))
ax.plot([z['w1'] for z in a],[z['magnitude'] for z in a],marker='o')
ax.set(xlabel='First input photon frequency omega1 (omega2=0.36-omega1)',ylabel='|BZ-averaged elastic T coefficient|',title='Two-pump frequency redistribution, M0 gamma=.25, BZ 45x45')
ax.grid(alpha=.3);fig.tight_layout();fig.savefig(ROOT/'q167_frequency_redistribution.png',dpi=190);plt.close(fig)
print('created 3 plots')
