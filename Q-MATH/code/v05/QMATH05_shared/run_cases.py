import os
os.environ['OPENBLAS_NUM_THREADS']='1'
import shared,json,sys
from pathlib import Path
out=Path(sys.argv[1]);cfg=json.loads(sys.argv[2]);cases=json.loads(sys.argv[3]);results=[]
for t,ph,N in cases:
 r=shared.calc(N,t,ph,**cfg)
 results.append(r)
 print(t,ph,N,'n',r['n_parallel'],r['n_perp'],'m',r['m_cross'],'edge',r['photon_corner'],r['pump_edge'],'norm',r['norm_error'],'s',r['wall'],flush=True)
 out.write_text(json.dumps(results,indent=2))
