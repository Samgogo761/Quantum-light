#!/usr/bin/env python3
"""Exact 36->9 electronic ground + one particle-hole subspace reduction
for the cubic ELASTIC (ground-to-ground) M0 multi-photon T coefficient.
This is exact for a filled Slater initial state, independent-electron H0,
and at most three one-body vertices. All intermediate 2p2h configurations
cannot return to ground in one remaining vertex.
"""
import os
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1'
import numpy as np, json, itertools, time
from pathlib import Path
import q167_m0_three_photon_T as q
ROOT=Path(__file__).resolve().parent

def wedge(u,v):
    return np.array([u[r]*v[s]-u[s]*v[r] for r,s in q.PAIRS],complex)

def slater_subspace(k,N):
    h=q.exact_vertex(k,N,[])
    spins=[];states=[];energies=[]
    for ii in (q.UP,q.DN):
        vals,U=np.linalg.eigh(h[np.ix_(ii,ii)])
        spins.append((ii,vals,U))
    ga=wedge(spins[0][2][:,0],spins[0][2][:,1])
    gb=wedge(spins[1][2][:,0],spins[1][2][:,1]);ground=np.kron(ga,gb)
    Eg=float(sum(np.sum(item[1][:2]) for item in spins))
    states.append(ground);energies.append(Eg)
    for si,(ii,vals,U) in enumerate(spins):
        for i in range(2):
            for a in range(2,4):
                ax=np.zeros((8,8),complex)
                ax[np.ix_(ii,ii)]=np.outer(U[:,a],U[:,i].conj())
                s=q.dgamma(ax)@ground
                if abs(np.linalg.norm(s)-1)>1e-12:raise RuntimeError('non-orthonormal excited Slater states')
                states.append(s);energies.append(Eg+float(vals[a]-vals[i]))
    S=np.stack(states,axis=1)
    assert np.max(abs(S.conj().T@S-np.eye(9)))<3e-13
    return S,np.array(energies)

def run_single(kfrac=(.4158,.1683),N=1,w1=.16,w2=.20,gamma=.1):
    k=q.geo.k_cart(kfrac);S,E=slater_subspace(k,N);Eg=E[0]
    mat={name:S.conj().T@q.dgamma(q.exact_vertex(k,N,dirs))@S for name,dirs in
         [('p',[q.p]),('h',[q.h]),('pp',[q.p,q.p]),('ph',[q.p,q.h]),('pph',[q.p,q.p,q.h]) ]}
    freq={0:w1,1:w2,2:-(w1+w2)}
    def vert(ids):
        x=set(ids)
        if x=={0} or x=={1}:return mat['p']
        if x=={2}:return mat['h']
        if x=={0,1}:return mat['pp']
        if x=={0,2} or x=={1,2}:return mat['ph']
        if x=={0,1,2}:return mat['pph']
        raise RuntimeError()
    def R(v,events):return v/(Eg+sum(freq[i] for i in events)-E+1j*gamma)
    direct=mat['pph'][0,0]
    mixed=0j;sequential=0j
    for group in ({0,1},{0,2},{1,2}):
        other=set((0,1,2))-group
        for first,second in ((group,other),(other,group)):
            mixed+=np.dot(vert(second)[0,:],R(vert(first)[:,0],first))
    for e0,e1,e2 in itertools.permutations((0,1,2)):
        x=R(vert([e0])[:,0],[e0]);x=vert([e1])@x;x=R(x,[e0,e1]);sequential+=np.dot(vert([e2])[0,:],x)
    return {'direct':direct,'mixed':mixed,'sequential':sequential,'total':direct+mixed+sequential,'ground_energy':Eg, 'electronic_reduced_dim':9}

def check():
    vals=[]
    for k,N,w1,w2,gamma in [((.4158,.1683),1,.16,.20,.1),((.2,.4),1,.14,.22,.25),((.5,.7),-1,.18,.18,.07),((.1,.2),-1,.16,.20,.14)]:
        r=run_single(k,N,w1,w2,gamma);ref=q.calc(k,N,w1,w2,gamma)
        err={label:float(abs(r[label]-ref[label])) for label in ('direct','mixed','sequential','total')}
        assert max(err.values())<5e-8,err
        vals.append({'kfrac':list(k),'neel':N,'w1':w1,'w2':w2,'gamma':gamma, 'ref_36':q.cp(ref['total']),'reduced_9':q.cp(r['total']),'errors':err,'electronic_hilbert_dims':[36,9]})
    output={'formula':'ground + spin-allowed one-particle-hole subspace is sufficient for 3 one-body vertices with elastic electronic ground final state', 'tests':vals,'max_residual':max(z for v in vals for z in v['errors'].values())}
    (ROOT/'q167_exact_one_ph_validation.json').write_text(json.dumps(output,indent=2));print(json.dumps(output,indent=2))
if __name__=='__main__':check()
