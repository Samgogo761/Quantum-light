#!/usr/bin/env python3
"""Rebuild Q-MATH-16.27 numbers from original Q16.26/Q16.7 M0 source files.
Default --fast: 9x9 waveform and 49x49 retarded test (<5 s).
--full: rebuild all tables; may take roughly 1-3 min depending on CPU.
This program DOES NOT evolve quantum BSV or implement photon LSZ.
"""
import os
for s in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[s]='1'
import argparse, json, time
from pathlib import Path
from qmath_16_27_retarded_kernel import kernel
from qmath_16_27_two_tone import propagate_two
ROOT=Path(__file__).resolve().parent

def fast():
    v,*_=propagate_two(L=9,steps=420,a=.0003,cross_direct=True)
    r=kernel(L=49,eta=.08)
    print('Finite two-tone mixed derivative:',v['sum_frequency'])
    print('Independent direct-Fock-free full Peierls finite difference error:',v['relative_mixed_fd_error'])
    print('Retarded chi2, L=49, eta=.08:',r['chi12'])
    assert v['relative_mixed_fd_error']<2e-4
    assert abs(complex(*r['chi12'])-complex(.3261088348937369,-2.679310228864009))<1e-7
    return {'finite_two_tone':v, 'retarded_49':r,'passed':True}

def full():
    from qmath_16_27_two_tone import run as two_run
    from qmath_16_27_retarded_kernel import main as ret_run
    two_run();ret_run()
    p=ROOT/'qmath_16_27_retarded_results.json';r=json.loads(p.read_text());
    for eta in (.08,.03,.01):
        for L in (129,161,201):r['data'].append(kernel(L=L,eta=eta))
    p.write_text(json.dumps(r,indent=2,ensure_ascii=False))
    shape=[]
    for L,Dvalues in [(161,(.2,.4,.6,.8,1.0)),(201,(.4,.8))]:
        for Delta in Dvalues:
            for w1 in (.1,.12,.14,.16,.18):shape.append(kernel(L=L,Delta=Delta,w1=w1,w2=.36-w1,eta=.08))
    (ROOT/'qmath_16_27_shape_scan.json').write_text(json.dumps({'dataset':shape},indent=2,ensure_ascii=False))
    contact=[]
    for Delta in (.4,.8):
        for eta in (.08,.03):contact.append(kernel(L=201,Delta=Delta,eta=eta))
    soft=[]
    for v in (.08,.04,.02,.01,.005):
        x=kernel(L=201,w1=v,w2=v,eta=v)
        soft.append({'frequency_and_eta':v,'chi12':x['chi12'],'abs_chi':abs(complex(*x['chi12']))})
    (ROOT/'qmath_16_27_contact_ward_checks.json').write_text(json.dumps({'contact_audit':contact,'soft_frequency_ward':soft},ensure_ascii=False,indent=2))
    return {'passed':True,'files':'qmath_16_27_results.json, retarded_results.json, shape_scan.json, contact_ward_checks.json'}

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--full',action='store_true');args=p.parse_args()
    t=time.monotonic();print(json.dumps(full() if args.full else fast(),ensure_ascii=False,indent=2)[:1200]);print('seconds',round(time.monotonic()-t,1))
