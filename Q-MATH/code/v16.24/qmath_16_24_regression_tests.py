#!/usr/bin/env python3
"""Deterministic regression on Q-MATH-16.24 already-computed results; no physical HHG certification."""
from pathlib import Path
import json
R=Path(__file__).parent
J=json.loads((R/'qmath_16_24_ward_sumrule_results.json').read_text())
C=json.loads((R/'qmath_16_24_manybody_crosscheck.json').read_text())
T=json.loads((R/'qmath_16_24_twist_and_stiffness.json').read_text())
P=json.loads((R/'qmath_16_24_shared_pair_toy_results.json').read_text())
assert max(abs(p['Jpair']) for p in J['pair']) <1e-9
assert all(abs(p['Kpair'])>100 for p in J['pair'])
assert max(c['K_residual'] for c in C)<1e-8
r=next(x for x in J['grid'] if x['L']==61)
assert abs(r['staticK_full']) < 1.
assert r['staticK_keep_0_2_4_6_8'][1] > 30.
assert max(t['quadrature_equivalence_residual'] for t in T['twists'])<1e-9
for gp in (0.005,0.015,0.025):
    xs=[x for x in P['cases'] if x['gp']==gp]
    assert len(xs)==4 and all(x['H_dimension']==4*(x['nmax']+1) for x in xs)
    assert max(x['top_fock_probability'] for x in xs if x['nmax']>=16)<1e-10
print('Q-MATH-16.24: 7 groups of tests PASSED. No new HHG quantum-advantage certification.')
