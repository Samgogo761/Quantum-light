import sys, numpy as np
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent/'QMATH13_M0_H2H4'))
from q13_M0_BSV_weak_radiation import M0Source
s=M0Source(5,1)
Z=0
for delta,O in s.terms:
 d=np.asarray(delta[:2],float); strength=2*abs(float(s.et@d))*np.linalg.norm(O.toarray(),2)
 Z+=strength
 print('d',np.round(d,2),'O_norm',np.linalg.norm(O.toarray(),2),'J_strength',strength)
T=4*np.pi/s.omega
print('J norm <=',Z,'integrated bound B',Z*T/2,'C_full <',Z*T/(2*np.sqrt(2)))
