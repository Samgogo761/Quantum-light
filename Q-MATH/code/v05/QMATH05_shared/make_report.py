import json,math,zipfile
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
p=Path(__file__).resolve().parent

def read(f):return json.loads((p/f).read_text())
def odd(a,b):return (complex(*a['m_cross'])-complex(*b['m_cross']))/2
pairs={}
for key,f in [('vac_bsv','results_even_11_3_80.json'),('seed','results_seeded_11_3_80.json'),('cohpi2','results_inversion_11_3_80.json'),('coh0','results_coherent0_11_3_80.json')]:pairs[key]=read(f)
vs=pairs['vac_bsv'];ss=pairs['seed'];ph=pairs['cohpi2'];ch=pairs['coh0']
xs=[('真空',vs[0],vs[1]),('压缩真空',vs[2],vs[3]),('相干 0',ch[0],ch[1]),('相干 π/2',ph[2],ph[3]),('位移压缩 0',ss[2],ss[3]),('位移压缩 π/2',ss[0],ss[1])]
labels=[x[0] for x in xs];oddvals=[abs(odd(x[1],x[2])) for x in xs]
ndata=[.5*(x[1]['n_parallel']+x[2]['n_parallel']) for x in xs]
fig,ax=plt.subplots(figsize=(9,4.3))
ax.bar(range(len(xs)),np.array(oddvals)*1e4)
ax.set_ylabel('|m_odd| × 10⁴');ax.set_xticks(range(len(xs)));ax.set_xticklabels(['vacuum','BSV','coherent 0','coherent π/2','seeded 0','seeded π/2'],rotation=13)
ax.set_title('Shared-field two-k benchmark | magnetic-odd H2 correlation')
fig.tight_layout();fig.savefig(p/'shared_two_k_odd.png',dpi=160);plt.close(fig)

configs=[('11×3,80','results_seeded_11_3_80.json'),('11×3,100','results_seeded_11_3_100.json'),('11×3,120','results_seeded_11_3_120.json'),('11×4,100','results_seeded_11_4_100.json'),('13×3,100','results_seeded_13_3_100.json'),('13×4,100','results_seeded_13_4_100.json')]
cut=[]
for label,f in configs:
 a,b=read(f)[:2];cut.append({'config':label,'abs_odd':abs(odd(a,b)),'pump_edge':a['pump_edge'],'h2_edge':a['photon_corner'],'mean_input':a['actual_mean']})
a=read('results_seeded_plus_15_4_100.json')[0];b=read('results_seeded_minus_15_4_100.json')[0]
cut.append({'config':'15×4,100','abs_odd':abs(odd(a,b)),'pump_edge':a['pump_edge'],'h2_edge':a['photon_corner'],'mean_input':a['actual_mean']})
fig,ax=plt.subplots(figsize=(8.2,4.2));ax.plot(range(len(cut)),[z['abs_odd']*1e4 for z in cut],marker='o')
ax.set_xticks(range(len(cut)));ax.set_xticklabels([c['config'] for c in cut],rotation=20)
ax.set_ylabel('|m_odd| × 10⁴');ax.set_title('Pump and H2 cutoffs / Strang steps: not yet fully converged')
fig.tight_layout();fig.savefig(p/'cutoff_convergence.png',dpi=160);plt.close(fig)

src={}
for label,a,b in xs:
 z=odd(a,b)
 src[label]={'N_plus_nparallel':a['n_parallel'],'N_minus_nparallel':b['n_parallel'],
             'm_odd_re':z.real,'m_odd_im':z.imag,'abs_m_odd':abs(z),'input_actual_mean':a['actual_mean'],
             'pump_edge':a['pump_edge'],'h2_edge_sum':a['photon_corner']}
inv=read('results_inversion_11_3_80.json')
err_pi2=max(abs(ss[1]['n_parallel']-inv[0]['n_parallel']),abs(complex(*ss[1]['m_cross'])-complex(*inv[0]['m_cross'])))
err_0=max(abs(ss[3]['n_parallel']-inv[1]['n_parallel']),abs(complex(*ss[3]['m_cross'])-complex(*inv[1]['m_cross'])))
records={'settings':{'kfrac':[.4158,.1683],'paired_kfrac':[-.4158,-.1683],'omega':.18,'cycles':2,'azp':.05,'azh':.005,'nominal_pump_n':.7,'main_dim':[36,36,11,3,3],'main_steps':80},'main':src,'convergence':cut,'inversion_error_max':max(err_pi2,err_0),'unit_test':read('unit_test_output.json')}
(p/'SUMMARY.json').write_text(json.dumps(records,indent=2,ensure_ascii=False))
print('summary',list(src.keys()))
