#!/usr/bin/env python3
"""Q-MATH-16.7: Third-order (2 pump absorbed, 1 H2 emitted) elastic T matrix.
No QFT continuum normalized S matrix is claimed: this is the electric-dipole,
periodic-k, small-photonic-coupling M0 matrix element per three external photons.
Crucially includes all photon orderings and mixed Peierls contact vertices.
"""
import os
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1';os.environ['MKL_NUM_THREADS']='1'
import numpy as np, itertools, json, argparse, sys, time
from pathlib import Path
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'deps'))
import q8_m0_model as m
geo=m.build_geometry(ROOT/'deps/CrI3_scf_soc_a1h_clean.in')
p=np.asarray(geo.a2_cart[:2]);p=p/np.linalg.norm(p)
h=np.array([p[1],-p[0]])
UP=np.array([0,2,4,6]); DN=np.array([1,3,5,7]); I6=np.eye(6)
PAIRS=list(itertools.combinations(range(4),2))

def exterior2(a):
    # Second quantization dGamma(a) in wedge^2 C^4
    z=np.zeros((6,6),complex)
    for i,(r,s) in enumerate(PAIRS):
        for j,(t,u) in enumerate(PAIRS):
            z[i,j]=a[r,t]*(s==u)-a[r,u]*(s==t)+a[s,u]*(r==t)-a[s,t]*(r==u)
    return z

def dgamma(a):
    if np.max(np.abs(a[np.ix_(UP,DN)]))>1e-12 or np.max(np.abs(a[np.ix_(DN,UP)]))>1e-12:
        raise ValueError('spin mixing violates 36-dimensional fixed spin model')
    return np.kron(exterior2(a[np.ix_(UP,UP)]),I6)+np.kron(I6,exterior2(a[np.ix_(DN,DN)]))

def bonds():
    """Reconstruct M0 exactly from Hermitian Fourier bond pieces.
    H(k) = onsite + sum [ B_b exp(i k.d_b) + B_b^dag exp(-i k.d_b)].
    """
    out=[]
    for d in geo.nn_deltas_cart:
        a=np.zeros((8,8),complex)
        for x,y in [(0,2),(1,3),(4,6),(5,7)]:a[x,y]+=m.R0.t_h
        out.append((np.array(d[:2],float),a,'NN'))
    # SOC = 2 lambda sum nu sin(k.b) sigma_z s_z, top and bottom.
    sigma_sz=np.array([1,-1,-1,1,1,-1,-1,1])
    for nu,d in geo.nnn_terms:
        a=np.diag(-1j*m.R0.lam_so*nu*sigma_sz)
        out.append((np.array(d[:2],float),a,'SOC'))
    o={'tA':0,'tB':2,'bA':4,'bB':6}
    for b in geo.interlayer_bonds:
        a=np.zeros((8,8),complex)
        for spin in (0,1):a[o[b['from']]+spin,o[b['to']]+spin]=getattr(m.R0,b['amplitude'])
        out.append((np.array(b['delta_cart'][:2],float),a,'interlayer'))
    return out
BONDS=bonds()

def exact_vertex(k,N,dirs):
    """Analytic mixed derivative D_{directions} H(k), all bonds, including contact."""
    k=np.asarray(k)
    if len(dirs)==0:
        out=np.diag([m.R0.delta*N,-m.R0.delta*N,m.R0.delta*N,-m.R0.delta*N,
                    -m.R0.delta*N,m.R0.delta*N,-m.R0.delta*N,m.R0.delta*N]).astype(complex)
    else:out=np.zeros((8,8),complex)
    for delta,a,_kind in BONDS:
        coeff=1j**len(dirs)*np.prod([np.dot(delta,v) for v in dirs])
        term=coeff*np.exp(1j*np.dot(k,delta))*a
        out+=term+term.conj().T
    return out

def vertices(k,N):
    return {
        '0':dgamma(exact_vertex(k,N,[])),
        'p':dgamma(exact_vertex(k,N,[p])),
        'h':dgamma(exact_vertex(k,N,[h])),
        'pp':dgamma(exact_vertex(k,N,[p,p])),
        'ph':dgamma(exact_vertex(k,N,[p,h])),
        'pph':dgamma(exact_vertex(k,N,[p,p,h]))
    }

def validate_bonds(kf=(.4158,.1683),N=1):
    k=geo.k_cart(kf)
    residual0=np.max(np.abs(exact_vertex(k,N,[])-m.h8(geo,m.R0,k,N)))
    jp= np.einsum('a,aij->ij',p,m.dh8(geo,m.R0,k,N))
    jh= np.einsum('a,aij->ij',h,m.dh8(geo,m.R0,k,N))
    residualp=np.max(np.abs(exact_vertex(k,N,[p])-jp))
    residualh=np.max(np.abs(exact_vertex(k,N,[h])-jh))
    delta=1e-4
    pp_fd=(exact_vertex(k+delta*p,N,[p])-exact_vertex(k-delta*p,N,[p]))/(2*delta)
    pph_fd=(exact_vertex(k+delta*p,N,[p,h])-exact_vertex(k-delta*p,N,[p,h]))/(2*delta)
    return dict(h0=residual0,first_parallel=residualp,first_perp=residualh,second_fd=float(np.max(abs(pp_fd-exact_vertex(k,N,[p,p])))),third_fd=float(np.max(abs(pph_fd-exact_vertex(k,N,[p,p,h])))))

EVENTS=(0,1,2)
FREQS=(.16,.20,-.36) # photon energy transferred *to* electrons by each vertex
EVENT_NAME=('p1','p2','Hout')

def calc(kfrac=(.4158,.1683),N=1,w1=.16,w2=.20,gamma=.10,decompose=False):
    k=geo.k_cart(kfrac);v=vertices(k,N)
    E,U=np.linalg.eigh(v['0']);Eg=E[0];g=U[:,0]
    # Degenerate occupied Slater states? inspect with Eg, excited lower bound
    A={key:U.conj().T@val@U for key,val in v.items() if key!='0'}
    efreq=[w1,w2,-(w1+w2)]
    def resolvent(source,consumed):
        den=(Eg+sum(efreq[i] for i in consumed)-E+1j*gamma)
        return source/den
    def vert(ids):
        s=set(ids)
        if s=={0} or s=={1}:return A['p']
        if s=={2}:return A['h']
        if s=={0,1}:return A['pp']
        if s=={0,2} or s=={1,2}:return A['ph']
        if s=={0,1,2}:return A['pph']
        raise ValueError(ids)
    direct=A['pph'][0,0]
    two=0j
    for subset in ({0,1},{0,2},{1,2}):
        other=tuple(set(EVENTS)-subset)
        for first,second in ((subset,other),(other,subset)):
            s=vert(first)[:,0]
            s=resolvent(s,first)
            two+=np.dot(vert(second)[0,:],s)
    three=0j
    sequence_terms=[]
    for ordering in itertools.permutations(EVENTS):
        e0,e1,e2=ordering
        s=vert([e0])[:,0]
        s=resolvent(s,[e0]);s=vert([e1])@s
        s=resolvent(s,[e0,e1])
        term=np.dot(vert([e2])[0,:],s)
        sequence_terms.append((tuple(EVENT_NAME[i] for i in ordering),term))
        three+=term
    tot=direct+two+three
    ans={'kfrac':[float(x) for x in kfrac],'N':int(N),'w1':w1,'w2':w2,'gamma':gamma,
         'Eground':float(Eg),'excitation_gap':float(E[1]-E[0]),
         'direct':complex(direct),'mixed':complex(two),'sequential':complex(three),'total':complex(tot)}
    if decompose:ans['permutation_terms']=sequence_terms
    return ans

def cp(z):return [float(np.real(z)),float(np.imag(z))]
def jsonify(obj):
    if isinstance(obj,dict):return {k:jsonify(v) for k,v in obj.items()}
    if isinstance(obj,(list,tuple)):return [jsonify(v) for v in obj]
    if isinstance(obj,(complex,np.complexfloating)):return cp(obj)
    if isinstance(obj,(np.floating,np.integer)):return obj.item()
    return obj

if __name__=='__main__':
    pa=argparse.ArgumentParser();pa.add_argument('--grids',default='5,9,15');pa.add_argument('--gamma',type=float,default=.10)
    a=pa.parse_args();
    audit=validate_bonds(); print('vertex audit',audit,flush=True)
    one=calc(decompose=True,gamma=a.gamma); print('single',jsonify({x:one[x] for x in ('total','direct','mixed','sequential','excitation_gap')}),flush=True)
    x=calc(w1=.20,w2=.16,gamma=a.gamma)
    print('exchange check',abs(one['total']-x['total']),flush=True)
    ka=geo.k_cart((.4158,.1683));inv=calc(geo.k_frac(-ka),N=-1,gamma=a.gamma)
    print('inversion check',abs(one['total']+inv['total']),flush=True)
    grid_out=[]
    for L in [int(v) for v in a.grids.split(',')]:
        ts=time.time();s={1:0j,-1:0j};d={1:0j,-1:0j};mix={1:0j,-1:0j};seq={1:0j,-1:0j}
        for i in range(L):
            for j in range(L):
                kf=((i+.5)/L,(j+.5)/L)
                for neel in (1,-1):
                    rr=calc(kf,neel,gamma=a.gamma)
                    s[neel]+=rr['total'];d[neel]+=rr['direct'];mix[neel]+=rr['mixed'];seq[neel]+=rr['sequential']
        result={'grid':L,'seconds':time.time()-ts,'plus':cp(s[1]/L**2),'minus':cp(s[-1]/L**2),'odd':cp((s[1]-s[-1])/(2*L**2)), 'even':cp((s[1]+s[-1])/(2*L**2)),
                'inversion_relative_leakage':float(abs(s[1]+s[-1])/max(abs(s[1]-s[-1]),1e-30)),
                'components':{key:{str(n):cp(z[n]/L**2) for n in (1,-1)} for key,z in [('direct',d),('mixed',mix),('sequential',seq)]}}
        grid_out.append(result)
        print('grid',L,'odd',result['odd'],'even leakage',result['inversion_relative_leakage'],'sec',result['seconds'],flush=True)
    output={'scope':'genuine eight-band M0 fixed-spin 36-electron matrix, lowest 3-photon elastic LS T coefficient; photon plane-wave dipole approx only; no full continuum scattering or BSV quantum advantage','vertex_audit':audit,'single_k':jsonify(one), 'pump_frequency_exchange_diff':float(abs(one['total']-x['total'])),'inversion_single_k_diff':float(abs(one['total']+inv['total'])),'grids':grid_out}
    (ROOT/'q167_results.json').write_text(json.dumps(output,indent=2,ensure_ascii=False))
