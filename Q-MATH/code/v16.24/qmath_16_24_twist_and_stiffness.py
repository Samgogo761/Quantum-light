#!/usr/bin/env python3
"""Exact discrete twist-quadrature identity and pair static photon-stiffness warning."""
import sys,json,math
from pathlib import Path
R=Path(__file__).parent;sys.path.insert(0,str(R))
from qmath_16_24_ward_sumrule import one,pair

def twist(L=5,M=5):
    vals=[]
    for t1 in range(M):
        for t2 in range(M):
            ksum=0.
            for i in range(L):
                for j in range(L):
                    ksum+=one(((i+0.5+t1/M)/L,(j+0.5+t2/M)/L))['K']
            vals.append(ksum/L**2)
    allfine=sum(one(((i+0.5)/(L*M),(j+0.5)/(L*M)))['K'] for i in range(L*M) for j in range(L*M))/(L*M)**2
    return {'L':L,'M':M,'n_total_twist_points':L*L*M*M,'twist_mean_K':sum(vals)/M**2,
            'single_twist_K':vals[0], 'twist_K_std':float(__import__('numpy').std(vals)),
            'equivalent_uniform_fine_grid_K':allfine,'quadrature_equivalence_residual':abs(sum(vals)/M**2-allfine)}

def main():
    pairr=[]
    for f in [(0.4158,.1683),(.37,.1913),(.21,.31)]:
        r=pair(f);w=0.18
        crit=math.sqrt(w/(-2*r['Kpair'])) if r['Kpair']<0 else None
        pairr.append({'frac':list(f),'K_pair':r['Kpair'],'Jpair':r['Jpair'],
                      'omega_cavity':w,'quadratic_softening_critical_g_if_unnormalized':crit,
                      'illustration_at_g_0.08':w+2*(0.08**2)*r['Kpair'],
                      'warning':'This gcrit is adiabatic quadratic estimate at an intentionally unphysical two-k truncation; NOT a real material phase boundary.'})
    res={'pairs':pairr,'twists':[twist(3,3),twist(5,5),twist(9,3)],
         'interpretation':'Twist averages restore quadrature of periodic occupied band; DO NOT commute averaging k with shared quantum-field dynamics. The pair harmonic softening is an adiabatic diagnostic only.'}
    path=R/'qmath_16_24_twist_and_stiffness.json';path.write_text(json.dumps(res,indent=2));print(json.dumps(res,indent=2))
if __name__=='__main__':main()
