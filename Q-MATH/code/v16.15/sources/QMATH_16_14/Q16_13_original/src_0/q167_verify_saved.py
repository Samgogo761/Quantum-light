#!/usr/bin/env python3
"""Fast CI-style consistency check of delivered numerical results."""
import json, pathlib, math
P=pathlib.Path(__file__).resolve().parent
J=lambda n:json.loads((P/n).read_text())
a=J('q167_photon_hilbert_validation.json')
b=J('q167_slater_condon_validation.json')
c=J('q167_full_peierls_validation.json')
d=J('q167_direct_contact_parity.json')
assert a['different_basis_error']<1e-9
assert b['max_residual']<1e-8
assert d['max_abs_direct_k_plus_direct_minusk']<1e-10
v=c['full_peierls_LS_finite_t']; assert 3.5< v['0.0005']['diff']/v['0.00025']['diff'] <4.5
f=J('q167_gamma_0.25_65.json')['grids'][-1];assert f['inversion_relative_leakage']<1e-9
print('ALL CHECKS PASS',{'photon_hilbert_error':a['different_basis_error'],'max_36to9_error':b['max_residual'],'gamma_025_grid65_odd':f['odd']})
