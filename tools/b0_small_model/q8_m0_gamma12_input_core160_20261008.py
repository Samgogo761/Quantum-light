"""One additional core-quadrature control, preserving the validated 96/128 code.

Uses the existing 4096 backward rows. No new background or forward propagation.
The previous script/hash and all old artifacts stay unchanged. The only new
numerical parameter is n_panel=160 in the original quadrature function.
"""
from pathlib import Path
import os
import json
import time
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('OMP_NUM_THREADS', '1')
import numpy as np
import q8_m0_gamma12_transpose_input_refine_20261008 as base


def run():
    out = base.OUT/'n_panel_160.json'
    if out.exists() or out.with_suffix('.npz').exists():
        raise FileExistsError(out)
    started = time.perf_counter()
    manifest, cache, rows, info, identity = base.load_frozen()
    gate96 = base._check_96_gate(identity)
    oldmeta = json.loads((base.OUT/'n_panel_128.json').read_text(encoding='utf-8'))
    if oldmeta['identity'] != identity or not oldmeta['complete_input_rows']:
        raise ValueError('Previous 128 source identity differs')
    oldpath = base.OUT/'n_panel_128.npz'
    if base.prep.sha256_file(oldpath) != oldmeta['row_sha256']:
        raise ValueError('Previous 128 output changed')
    gamma = float(cache['gamma'])
    pol = np.array(cache['pol'], copy=True)
    prot = base.p1_a2_protocol(base.q.A_STAR)
    op = base.q.omega_grid(prot.omega0, prot.t_pulse)
    op = op[op > 0]
    _, norm = base.q.normalize_phi(op, base.q.phi_unnormalized(op, 2*prot.omega0, prot.t_pulse))
    if not np.isclose(norm, manifest['preparation_checks']['packet_normalizer'], atol=1e-12, rtol=1e-12):
        raise ValueError('Packet definition changed')
    phi_of = lambda nu: base.q.phi_at(nu, prot, norm)
    eta = base.q.eta_of(gamma)
    t0 = float(manifest['time'].get('t0', 0.))
    # Independent extension wiring check on saved 128 frequency rows, including
    # low/H2/mid/high inputs; no new assumptions or output normalization.
    with np.load(oldpath, allow_pickle=False) as ref:
        f = ref['oms']; w = ref['wts']
        fi = np.unique([0, len(f)-1, len(f)//2,
                        *[int(np.argmin(abs(f-x))) for x in (.003, .03, .1, .3, 1., 5.)]])
        ids = (4*fi[:, None]+np.arange(4)).ravel()
        ph, u0 = base.generated_photon_block(cache, gamma, f[fi], w[fi], pol, phi_of, eta)
        got, _ = base._contract(rows, (ph.initial.b, ph.initial.bdag, ph.initial.A),
                                ph.omega, ph.free_A0, t0, 128, 2048)
        checks = {k: base.comparison(got[k], ref[k][:, ids]) for k in base.RAW_KEYS}
        checks['u0'] = base.comparison(u0, ref['u0'][:, ids])
    if not all(r['allclose'] for r in checks.values()):
        raise AssertionError('Extension 128 wiring regression failed')
    del ph, got
    oms, wts = base.q.input_quadrature(cache['gaps'], cache['u0'], gamma, n_panel=160)
    if (np.any(wts <= 0) or np.any(np.diff(oms) <= 0)
            or len(np.unique(np.round(oms, 12))) != len(oms)):
        raise ValueError('Quadrature ordering/weight/key failure')
    nph, ndark = 4*len(oms), manifest['physical_scope']['n_dark_columns']
    old_nph = manifest['physical_scope']['n_photon_columns']
    if ndark != 1976:
        raise ValueError('Dark space changed')
    total = nph+ndark
    vals = {k: np.empty((2, total), complex) for k in base.RAW_KEYS+('u0',)}
    timings = dict(stationary_assembly_s=0., driven_contraction_s=0., undriven_contraction_s=0.)
    print(f'core160: {len(oms)} frequencies, {total} columns; 128 wiring PASS', flush=True)
    for first in range(0, len(oms), 128):
        last = min(first+128, len(oms))
        t = time.perf_counter()
        ph, u0 = base.generated_photon_block(cache, gamma, oms[first:last], wts[first:last], pol, phi_of, eta)
        timings['stationary_assembly_s'] += time.perf_counter()-t
        got, tt = base._contract(rows, (ph.initial.b, ph.initial.bdag, ph.initial.A),
                                 ph.omega, ph.free_A0, t0, 128, 2048)
        dest = slice(4*first, 4*last)
        for key in base.RAW_KEYS:
            vals[key][:, dest] = got[key]
        vals['u0'][:, dest] = u0
        for p in tt:
            timings[p+'_contraction_s'] += tt[p]
        if first == 0 or last % 2048 == 0 or last == len(oms):
            print(f'core160: {last}/{len(oms)} frequencies', flush=True)
    for first in range(0, ndark, 512):
        count = min(512, ndark-first)
        ids = np.arange(old_nph+first, old_nph+first+count)
        initial = tuple(np.array(cache[k][ids].T, copy=True) for k in ('b_cols', 'bdag_cols', 'A_cols'))
        om = np.array(cache['omega'][ids]); free = np.array(cache['free_A0_cols'][ids].T)
        if np.any(om != 0) or np.any(free != 0):
            raise ValueError('Dark columns gained forcing')
        got, tt = base._contract(rows, initial, om, free, t0, 128, 2048)
        dest = slice(nph+first, nph+first+count)
        for key in base.RAW_KEYS:
            vals[key][:, dest] = got[key]
        vals['u0'][:, dest] = cache['U0_canonical'][:, ids]
        for p in tt:
            timings[p+'_contraction_s'] += tt[p]
    du = vals['driven_raw_u']-vals['undriven_raw_u']
    v = vals['driven_raw_v']-vals['undriven_raw_v']
    u = vals['u0']+du
    vals.update(column_id=np.arange(total), oms=oms, wts=wts, U=u, V=v, delta_u=du,
                n=v.conj()@v.T, m=u@v.T, m_U0VT=vals['u0']@v.T, m_source_source=du@v.T,
                normal_contribution=u@u.conj().T-v@v.conj().T,
                anomalous_contribution=u@v.T-v@u.T,
                U0_gram=vals['u0']@vals['u0'].conj().T)
    base.r1._finite_arrays(vals)
    base.r1._write_npz_atomic(out.with_suffix('.npz'), vals)
    changes = {}
    with np.load(oldpath, allow_pickle=False) as ref:
        for key in ('n', 'm', 'normal_contribution', 'anomalous_contribution'):
            a, b = ref[key], vals[key]
            if key == 'normal_contribution':
                a, b = a-np.eye(2), b-np.eye(2)
            changes[key] = dict(maxabs=float(np.max(abs(b-a))),
                                relative_fro_using_160=float(np.linalg.norm(b-a)/max(np.linalg.norm(b), 1e-30)))
    meta = dict(status='resonance_core_refinement_only', n_panel=160,
                identity=identity, extension_sha256=base.prep.sha256_file(Path(__file__)),
                required_96_gate_sha256=gate96, required_128_row_sha256=oldmeta['row_sha256'],
                extension_128_checks=checks, ntpc=4096, nstep=info['nstep'], h=info['h'],
                n_nodes=len(oms), n_photon_columns=nph, n_dark_columns=ndark, n_contracted_columns=total,
                normal_ccr_maxabs=float(np.max(abs(vals['normal_contribution']-np.eye(2)))),
                anomalous_maxabs=float(np.max(abs(vals['anomalous_contribution']))),
                n_trace=float(np.trace(vals['n']).real),
                U0_gram_defect=float(np.max(abs(vals['U0_gram']-np.eye(2)))),
                packet_mass=base.q.packet_mass_dnu(oms,wts,prot,norm),
                changes_vs_128=changes, timings=timings, total_wall_s=time.perf_counter()-started,
                row_sha256=base.prep.sha256_file(out.with_suffix('.npz')),
                n_h2_frozen=base.q.N_H2, tail_panels_frozen=True,
                omega_lo_frozen=.001, omega_hi_frozen=12.,
                production_qualification=False, background_repropagated=False,
                all_checks_pass=None, complete_input_rows=True, smoke_only=False)
    base.prep.write_json_atomic(out, meta)
    print(json.dumps({k:meta[k] for k in ('n_panel','n_nodes','n_trace','normal_ccr_maxabs',
                                        'anomalous_maxabs','total_wall_s')}), flush=True)


if __name__=='__main__':
    run()
