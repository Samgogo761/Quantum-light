#!/usr/bin/env python3
"""Direct 9x9 Slater-Condon projected elastic 3-photon M0 T matrix.
Avoids constructing 36x36 many-electron operators entirely.
"""
import os
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1'
import numpy as np,json,itertools,time
from pathlib import Path
import q167_m0_three_photon_T as q
ROOT=Path(__file__).resolve().parent

def subspace(k,N):
    h=q.exact_vertex(k,N,[])
    w=np.zeros((8,8),complex);ev=[]
    bands=[]
    for spin_idx,indices in enumerate((q.UP,q.DN)):
        ee,uu=np.linalg.eigh(h[np.ix_(indices,indices)])
        bands.append((ee,uu,indices))
    # occupied orbital order: up0,up1,dn0,dn1; virtual up2,up3,dn2,dn3
    labels=[]
    for occ in (0,1):
        for si in range(2):
            for j in range(2):
                a=j+(0 if occ==0 else 2)
                ee,uu,ii=bands[si]
                orbital=np.zeros(8,complex);orbital[ii]=uu[:,a]
                labels.append((si,a,ee[a],orbital))
    # Above order: occ up0,up1,dn0,dn1 then virt up2,up3,dn2,dn3
    U=np.column_stack([r[3] for r in labels]);energies=np.array([r[2] for r in labels])
    assert np.linalg.norm(U.conj().T@U-np.eye(8))<1e-12
    transitions=[(a,i) for a in range(4,8) for i in range(4) if labels[a][0]==labels[i][0]]
    assert len(transitions)==8
    return U,energies,transitions

def onebody_proj(A,U,transitions):
    ao=U.conj().T@A@U
    groundtr=np.trace(ao[:4,:4])
    z=np.zeros((9,9),complex);z[0,0]=groundtr
    for x,(a,i) in enumerate(transitions,1):
        z[x,0]=ao[a,i];z[0,x]=ao[i,a]
        for y,(b,j) in enumerate(transitions,1):
            z[x,y]=(groundtr if (a==b and i==j) else 0) + (ao[a,b] if i==j else 0) - (ao[j,i] if a==b else 0)
    return z

def calc(kfrac=(.4158,.1683),N=1,w1=.16,w2=.20,gamma=.1):
    k=q.geo.k_cart(kfrac);U,energies,trs=subspace(k,N)
    Eg=float(sum(energies[:4])); Es=np.array([Eg]+[Eg+energies[a]-energies[i] for a,i in trs])
    v={key:onebody_proj(q.exact_vertex(k,N,dirs),U,trs) for key,dirs in [('p',[q.p]),('h',[q.h]),('pp',[q.p,q.p]),('ph',[q.p,q.h]),('pph',[q.p,q.p,q.h])]}
    freq=(w1,w2,-(w1+w2))
    def vert(ids):
        s=set(ids)
        if s=={0} or s=={1}:return v['p']
        if s=={2}:return v['h']
        if s=={0,1}:return v['pp']
        if s=={0,2} or s=={1,2}:return v['ph']
        if s=={0,1,2}:return v['pph']
        raise RuntimeError()
    def resol(s,inds):return s/(Eg+sum(freq[i] for i in inds)-Es+1j*gamma)
    direct=v['pph'][0,0];mixed=0j;sequential=0j
    for subset in ({0,1},{0,2},{1,2}):
        rest={0,1,2}-subset
        for first,second in ((subset,rest),(rest,subset)):
            mixed+=np.dot(vert(second)[0,:],resol(vert(first)[:,0],first))
    for e0,e1,e2 in itertools.permutations((0,1,2)):
        s=resol(vert([e0])[:,0],[e0]);s=vert([e1])@s;s=resol(s,[e0,e1]);sequential+=np.dot(vert([e2])[0,:],s)
    return {'direct':direct,'mixed':mixed,'sequential':sequential,'total':direct+mixed+sequential}

def check():
    comparison=[]
    for k,N,w1,w2,g in [((.4158,.1683),1,.16,.2,.1),((.2,.4),1,.14,.22,.25),((.5,.7),-1,.18,.18,.07),((.1,.2),-1,.16,.20,.14)]:
        x=calc(k,N,w1,w2,g);y=q.calc(k,N,w1,w2,g)
        errors={field:float(abs(x[field]-y[field])) for field in ('direct','mixed','sequential','total')}
        assert max(errors.values())<1e-8,(k,errors)
        comparison.append({'kf':list(k),'N':N,'gamma':g,'frequencies':[w1,w2],'T':q.cp(x['total']),'residuals':errors})
    t0=time.time()
    for i in range(30):calc(((i+.5)/31,((5*i)%31+.5)/31),1)
    fast=time.time()-t0
    t0=time.time()
    for i in range(30):q.calc(((i+.5)/31,((5*i)%31+.5)/31),1)
    slow=time.time()-t0
    out={'exact_36_to_9_fermion_subspace':True,'projected_direct_Slater_Condon':True,'cases':comparison, 'max_residual':max(v for c in comparison for v in c['residuals'].values()), 'benchmark_30k_fast_seconds':fast,'benchmark_30k_36_seconds':slow,'speedup':slow/fast}
    (ROOT/'q167_slater_condon_validation.json').write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))
if __name__=='__main__':check()
