#!/usr/bin/env python3
"""Q18 lightweight independent checks of production M0 determinant and analytic inverse."""
from pathlib import Path
import sys, json, numpy as np
R=Path(__file__).resolve().parent
sys.path.insert(0,str(R))
import qmath_16_18_keldysh_determinant as d
import qmath_16_18_causal_rank_and_inverse as v

a=d.test_exact()
assert max(z['residual'] for z in a)<1e-12
assert max(z['same_branch_residual'] for z in a)<1e-12
s=v.inverse_test(.1)
assert s['rank_2']
assert abs(s['quadratic_fit']['F2']-.13)<1e-10
assert abs(s['quadratic_fit']['F3']-.105)<1e-10
assert s['true']['witness']<0
r=json.load(open(R/'qmath_16_18_final_results.json'))
for L in (3,5,7):
    p=next(z for z in r['production_M0_finite_time_second_fourth_current_response'] if z['L']==L and z['lambda']==.1 and z['Delta']==.8 and z['N']==1)
    n=next(z for z in r['production_M0_finite_time_second_fourth_current_response'] if z['L']==L and z['lambda']==.1 and z['Delta']==.8 and z['N']==-1)
    assert max(abs(x+y) for x,y in zip(p['C2']+p['C4'],n['C2']+n['C4']))<5e-8
    for lam,delta in ((0.,.8),(.1,0.)):
        null=next(z for z in r['production_M0_finite_time_second_fourth_current_response'] if z['L']==L and z['lambda']==lam and z['Delta']==delta and z['N']==1)
        assert max(abs(x) for x in null['C2']+null['C4'])<2e-7
print('PASS: direct original-M0 36d vs 8band Keldysh overlap and same-branch normalization')
print('PASS: recorded M0 magnetic sign and null controls; finite-time C2,C4 diagnostic only')
print('PASS: synthetic coupling-scanned F2/F3 inversion and positive-P witness')
print('Max determinant error:',max(z['residual'] for z in a))
print('Recovered F2,F3:',s['quadratic_fit']['F2'],s['quadratic_fit']['F3'])
