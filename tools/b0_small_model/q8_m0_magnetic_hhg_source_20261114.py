#!/usr/bin/env python3
"""Task A only: read archived a2 currents, produce two source-spectrum figures.

No propagation, no input-state change, no photon-output calculation.
"""
from __future__ import annotations
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np
import q8_m0_classical as cl
from q8_m0_model import R0, build_geometry
from q8_m0_s1_kernel import parallel_perp_basis

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'results/q_line/magnetic_hhg_source_20261114'
CAP=120.
LABELS=('Gamma12','Gamma48','shift48')
COLORS=('#7c8797','#1464a0','#ce7028')


def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(2**20),b''):h.update(b)
    return h.hexdigest()


def serial(x):
    if isinstance(x,np.ndarray):return serial(x.tolist())
    if isinstance(x,dict):return {str(k):serial(v) for k,v in x.items()}
    if isinstance(x,(list,tuple)):return [serial(v) for v in x]
    if isinstance(x,(complex,np.complexfloating)):return {'re':float(x.real),'im':float(x.imag)}
    if isinstance(x,np.generic):return x.item()
    return x


def dump(p,x):
    with Path(p).open('x',encoding='utf-8') as f:
        json.dump(serial(x),f,ensure_ascii=False,indent=1,allow_nan=False)


def angle_deg(z):return np.angle(z,deg=True)


def wrap_deg(d):return (d+180.)%360.-180.


def obs(p,m):
    """p,m (...,2), in the frozen parallel/perpendicular basis."""
    e,o=(p+m)/2,(p-m)/2
    result={'J_plus':p,'J_minus':m,'J_even':e,'J_odd':o}
    for name,j in [('plus',p),('minus',m)]:
        s0=np.sum(abs(j)**2,axis=-1)
        s1=abs(j[...,0])**2-abs(j[...,1])**2
        s2=2*np.real(j[...,0].conj()*j[...,1])
        s3=2*np.imag(j[...,0].conj()*j[...,1])
        result[name]={'S0':s0,'S1':s1,'S2':s2,'S3':s3,
            'P_parallel':abs(j[...,0])**2,'P_perp':abs(j[...,1])**2,
            'P45':abs((j[...,0]+j[...,1])/np.sqrt(2))**2,
            'P135':abs((j[...,0]-j[...,1])/np.sqrt(2))**2,
            'phase_perp_minus_parallel_deg':angle_deg(j[...,1]*j[...,0].conj())}
    for key in ('S0','S1','S2','S3','P_parallel','P_perp','P45','P135'):
        result['D_'+key]=result['plus'][key]-result['minus'][key]
    for key in ('P_parallel','P_perp','P45','P135'):
        den=result['plus'][key]+result['minus'][key]
        # Null means no defined relative contrast; no physical floor is imposed.
        result['contrast_'+key]=np.divide(result['D_'+key],den,out=np.zeros_like(den),where=den>0)
        result['sum_'+key]=den
    den=result['plus']['S0']+result['minus']['S0']
    for key in ('S2','S3'):
        result['D_'+key+'_over_sumS0']=np.divide(result['D_'+key],den,out=np.zeros_like(den),where=den>0)
    result['phase_odd_minus_even_deg']=angle_deg(o*e.conj())
    result['phase_oddperp_minus_evenpar_deg']=angle_deg(o[...,1]*e[...,0].conj())
    result['phase_even_perp_minus_parallel_deg']=angle_deg(e[...,1]*e[...,0].conj())
    result['phase_odd_perp_minus_parallel_deg']=angle_deg(o[...,1]*o[...,0].conj())
    for key,v in [('parallel',np.array([1.,0.])),('perp',np.array([0.,1.])),
                  ('45',np.array([1.,1.])/np.sqrt(2)),('135',np.array([1.,-1.])/np.sqrt(2))]:
        result['D_identity_'+key]=4*np.real((e@v).conj()*(o@v))
    return result


def load(label,geo,prot):
    paths=[]
    if label=='Gamma12':
        folder=ROOT/'results/q_line/h2_class_pmN_run_20261025'
        jp=folder/'h2_class_pmN.json';p=folder/'h2_class_pmN_traj.npz'
        meta=json.loads(jp.read_text(encoding='utf-8'))['protocol']
        assert meta['drive']=='p1_a2_protocol(0.8)' and meta['n_cycles_envelope']==8
        with np.load(p,allow_pickle=False) as z:
            t=z['times_plus'];tm=z['times_minus']
            jplus=z['I_cell_plus']/geo.a_cell_ang2
            jminus=z['I_cell_minus']/geo.a_cell_ang2
            par,perp=parallel_perp_basis(geo)
            np.testing.assert_allclose(z['e_par'],par,atol=1e-14)
            np.testing.assert_allclose(z['e_perp'],perp,atol=1e-14)
        np.testing.assert_array_equal(t,tm)
        original_end=float(t[-1]);keep=t<=prot.t_pulse+1e-10
        t,jplus,jminus=t[keep],jplus[keep],jminus[keep]
        paths=[jp,p]
        info={'nk':12,'ntpc':1024,'shift':0.,'original_end':original_end,
              'tail_cropped':True,'unit_conversion':'I_cell/A_cell -> J_sheet'}
    else:
        name='V48' if label=='Gamma48' else 'V48S'
        folder=ROOT/'results/q_line/cjj_m0_p1_a2_s1_20260917'/name
        currents=[];metas=[];ts=[]
        for N in (1,-1):
            p=folder/f'{name}_N{N:+d}_A+1.npz'
            jp=p.with_suffix('.meta.json');meta=json.loads(jp.read_text(encoding='utf-8'))
            assert meta['protocol']==prot.__dict__
            with np.load(p,allow_pickle=False) as z:
                ts.append(z['times']);currents.append(z['J_sheet'])
            paths.extend([jp,p]);metas.append(meta)
        np.testing.assert_array_equal(ts[0],ts[1]);t=ts[0];jplus,jminus=currents
        info={'nk':48,'ntpc':400,'shift':metas[0]['grid_shift'],'tail_cropped':False,
              'unit_conversion':'stored J_sheet retained'}
    if abs(t[0])>1e-12 or abs(t[-1]-prot.t_pulse)>1e-9:
        raise ValueError('time origin or endpoint mismatch')
    np.testing.assert_allclose(np.diff(t),prot.period/info['ntpc'],atol=2e-12,rtol=0)
    info.update(n_times=len(t),t_start=float(t[0]),t_end=float(t[-1]),
                initial_current_plus=jplus[0],initial_current_minus=jminus[0],
                baseline_subtracted=False,paths=[str(p) for p in paths])
    return t,jplus,jminus,info,paths


def band_power(nu,p,q,halfwidth=.25):
    lo,hi=q-halfwidth,q+halfwidth
    keep=(nu>lo)&(nu<hi)
    xx=np.r_[lo,nu[keep],hi]
    yy=np.r_[np.interp(lo,nu,p),p[keep],np.interp(hi,nu,p)]
    return float(np.trapezoid(yy,xx))


def harmonic_rows(label,h,bands):
    rows=[]
    for i,q in enumerate(range(1,10)):
        row={'grid':label,'harmonic':q}
        for who in ('plus','minus','even','odd'):
            J=h['J_'+who][i]
            for a,axis in enumerate(('parallel','perp')):
                z=J[a]
                row[f'J_{who}_{axis}_re']=z.real;row[f'J_{who}_{axis}_im']=z.imag
                row[f'J_{who}_{axis}_abs']=abs(z);row[f'J_{who}_{axis}_arg_deg']=angle_deg(z)
        for name in ('phase_odd_minus_even_deg',):
            for a,axis in enumerate(('parallel','perp')):row[name+'_'+axis]=h[name][i,a]
        for name in ('phase_oddperp_minus_evenpar_deg','phase_even_perp_minus_parallel_deg','phase_odd_perp_minus_parallel_deg'):
            row[name]=h[name][i]
        for who in ('plus','minus'):
            for k in ('S0','S1','S2','S3','P_parallel','P_perp','P45','P135','phase_perp_minus_parallel_deg'):
                row[who+'_'+k]=h[who][k][i]
        for k,v in h.items():
            if k.startswith(('D_','contrast_','sum_')) and not k.startswith('D_identity_'):
                row[k]=v[i]
        row.update(bands[i]);rows.append(row)
    return rows


def make_figures(blocks):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FixedLocator
    plt.rcParams.update({'font.family':['Microsoft YaHei','DejaVu Sans'],
        'axes.unicode_minus':False,'font.size':9,'axes.spines.top':False,
        'axes.spines.right':False,'axes.grid':True,'grid.alpha':.15})
    fig,axs=plt.subplots(2,3,figsize=(14.4,8.3),layout='constrained')
    for label,color in zip(LABELS,COLORS):
        b=blocks[label];nu=b['nu'];s=b['spectrum'];h=b['harmonics'];q=np.arange(1,10)
        for a in range(2):
            axs[0,a].semilogy(nu,np.maximum(abs(s['J_even'][:,a]),1e-16),color=color,lw=1.2,label=label+' even')
            axs[0,a].semilogy(nu,np.maximum(abs(s['J_odd'][:,a]),1e-16),color=color,lw=1.,ls='--',label=label+' odd')
        ratio=abs(h['J_odd'][:,1])/np.maximum(abs(h['J_even'][:,0]),1e-30)
        axs[0,2].semilogy(q,ratio,'o-',color=color,ms=3,label=label)
        for ax,j in [(axs[1,0],h['J_even'][:,0]),(axs[1,1],h['J_odd'][:,1])]:
            ax.plot(q,j.real,'o-',color=color,ms=3,label=label+' Re')
            ax.plot(q,j.imag,'s--',color=color,ms=3,alpha=.8,label=label+' Im')
        axs[1,2].plot(q,h['phase_oddperp_minus_evenpar_deg'],'o-',color=color,ms=3,label=label)
    for a,title in enumerate(('平行分量：磁偶 / 磁奇源谱','垂直分量：磁偶 / 磁奇源谱')):
        axs[0,a].set(title=title,xlabel=r'$\Omega/\omega_0$',ylabel=r'$|\mathcal{J}|$（片电流模型单位）',xlim=(.5,9.5),ylim=(1e-12,2e-2))
        axs[0,a].set_xticks(range(1,10));axs[0,a].legend(fontsize=7,ncol=2)
        for q in (4,7):axs[0,a].axvspan(q-.25,q+.25,color='#777',alpha=.07)
    axs[0,2].set(title='整数谐波的偏振结构',xlabel='谐波阶 q',ylabel=r'$|\mathcal{J}^o_\perp|/|\mathcal{J}^e_\parallel|$')
    for a,title in enumerate(('磁偶平行分量：复振幅','磁奇垂直分量：复振幅')):
        axs[1,a].set(title=title,xlabel='谐波阶 q',ylabel='Re / Im（片电流模型单位）')
        axs[1,a].set_yscale('symlog',linthresh=1e-8)
        axs[1,a].legend(fontsize=6.8,ncol=2)
    axs[1,2].set(title='干涉相位：磁奇⊥ 相对磁偶∥',xlabel='谐波阶 q',ylabel='相位差（度）',ylim=(-190,190),yticks=(-180,-90,0,90,180))
    for ax in [axs[0,2],axs[1,2]]:ax.legend(fontsize=8)
    for ax in axs.flat:
        if ax not in (axs[0,0],axs[0,1]):ax.set_xticks(range(1,10))
    fig.suptitle('图一｜八周期 PT-AFM 相干源谱与偏振结构\n同一 a₂ 驱动；移位差异反映未收敛的 BZ 求积，未作材料认证',fontsize=13)
    for suffix in ('png','pdf'):fig.savefig(OUT/f'figure1_source_spectra.{suffix}',dpi=170)
    plt.close(fig)
    fig,axs=plt.subplots(3,3,figsize=(14.4,10.4),layout='constrained',sharex=True)
    scale=max(float(np.max(abs(blocks[g]['harmonics']['D_P45']))) for g in LABELS)
    for c,(label,color) in enumerate(zip(LABELS,COLORS)):
        h=blocks[label]['harmonics'];q=np.arange(1,10)
        axs[0,c].plot(q,h['D_P45'],'o-',color='#185c9c',ms=4,label=r'$D_{45}$')
        axs[0,c].plot(q,h['D_P135'],'s--',color='#b85b2a',ms=4,label=r'$D_{135}$')
        axs[0,c].set_yscale('symlog',linthresh=max(scale*1e-5,1e-18))
        axs[0,c].yaxis.set_major_locator(FixedLocator([-1e-6,-1e-8,-1e-10,0,1e-10,1e-8,1e-6]))
        axs[0,c].set_title(label+'｜绝对磁畴差')
        axs[1,c].plot(q,h['contrast_P45'],'o-',color='#185c9c',ms=4,label=r'$C_{45}$')
        axs[1,c].plot(q,h['contrast_P135'],'s--',color='#b85b2a',ms=4,label=r'$C_{135}$')
        bd=blocks[label]['bands']
        band45=np.array([(v['band_plus_P45']-v['band_minus_P45'])/(v['band_plus_P45']+v['band_minus_P45']) for v in bd])
        axs[1,c].plot(q,band45,'^:',color='#82599b',ms=4,label=r'$C_{45}$ 带限')
        axs[1,c].set_ylim(-1.08,1.08);axs[1,c].set_title('归一化反差与源信号强度')
        right=axs[1,c].twinx()
        right.semilogy(q,h['plus']['S0']+h['minus']['S0'],':',color='#8a8a8a',lw=1.,label=r'$\Sigma S_0$')
        right.set_ylabel(r'$\Sigma S_0$（灰虚线）',color='#777',fontsize=8)
        axs[2,c].plot(q,h['D_S2_over_sumS0'],'o-',color='#25745a',ms=4,label=r'$\Delta S_2/\Sigma S_0$')
        axs[2,c].plot(q,h['D_S3_over_sumS0'],'s--',color='#83539a',ms=4,label=r'$\Delta S_3/\Sigma S_0$')
        axs[2,c].set_title('磁序翻转下的偏振干涉变化')
        for a in range(3):
            axs[a,c].axhline(0,color='#777',lw=.5);axs[a,c].legend(fontsize=8)
            axs[a,c].set_xticks(range(1,10))
        axs[2,c].set_xlabel('谐波阶 q')
    axs[0,0].set_ylabel('源强度代理差（模型单位²）')
    axs[1,0].set_ylabel(r'$C=D/(P_++P_-)$')
    axs[2,0].set_ylabel('Stokes 差 / 两畴总 S₀')
    fig.suptitle('图二｜磁畴翻转的分析器反差与 Stokes 变化\n相对反差必须结合信号强度与 Γ48 / shift48 一致性解读',fontsize=13)
    for suffix in ('png','pdf'):fig.savefig(OUT/f'figure2_domain_contrast.{suffix}',dpi=170)
    plt.close(fig)


def run():
    start=time.perf_counter()
    def check():
        if time.perf_counter()-start>CAP-2:raise TimeoutError('120 s analysis budget')
    if OUT.exists() and any(OUT.iterdir()):raise FileExistsError('result directory is not empty')
    OUT.mkdir(parents=True,exist_ok=True)
    geo=build_geometry();prot=cl.p1_a2_protocol(.8);par,perp=parallel_perp_basis(geo)
    basis=np.stack((par,perp));np.testing.assert_allclose(basis@basis.T,np.eye(2),atol=1e-14)
    blocks={};rows=[];before={};source_checks={}
    for label in LABELS:
        check();t,jp,jm,info,paths=load(label,geo,prot)
        before.update({str(p):sha(p) for p in paths})
        np.testing.assert_allclose(t[-1],prot.t_pulse,atol=1e-9,rtol=0)
        nu,sp=cl.spectrum(t,jp@basis.T,prot.omega0,wmax=9.5,dw=.02)
        _,sm=cl.spectrum(t,jm@basis.T,prot.omega0,wmax=9.5,dw=.02)
        hp=np.stack([cl.project_harmonic(t,jp@basis.T,q*prot.omega0,'sin2') for q in range(1,10)])
        hm=np.stack([cl.project_harmonic(t,jm@basis.T,q*prot.omega0,'sin2') for q in range(1,10)])
        spectra=obs(sp,sm);harm=obs(hp,hm);bands=[]
        for q in range(1,10):
            bands.append({f'band_H{q}_{who}_{key}':band_power(nu,spectra[who][key],q)
                for who in ('plus','minus') for key in ('P45','P135','S0')})
        # Keep identical column names across harmonic rows.
        bands=[{k.replace(f'H{q}_',''):v for k,v in b.items()} for q,b in enumerate(bands,1)]
        blocks[label]={'nu':nu,'spectrum':spectra,'harmonics':harm,'bands':bands,'info':info}
        rows.extend(harmonic_rows(label,harm,bands))
        max_identity=max(float(np.max(abs(harm['D_'+power]-harm['D_identity_'+key])))
            for power,key in (('P_parallel','parallel'),('P_perp','perp'),('P45','45'),('P135','135')))
        dense_match=max(float(np.max(abs(sp[50*q]-hp[q-1]))) for q in range(1,10))
        source_checks[label]={'contrast_identity_max_abs':max_identity,'integer_vs_dense_max_abs':dense_match,
            'initial_baseline_max_abs':float(max(np.max(abs(jp[0])),np.max(abs(jm[0])))),
            'baseline_subtracted':False}
        print(label,'analysed',len(t),'points',flush=True)
    comparisons=[]
    for q in range(1,10):
        a=blocks['Gamma48']['harmonics'];b=blocks['shift48']['harmonics'];i=q-1
        row={'q':q}
        for who in ('even','odd'):
            x,y=a['J_'+who][i],b['J_'+who][i]
            row[who+'_vector_relative_difference']=float(np.linalg.norm(y-x)/max(np.linalg.norm(x),np.linalg.norm(y),1e-30))
            for c,axis in enumerate(('parallel','perp')):
                row[f'{who}_{axis}_shift_over_Gamma_abs']=float(abs(y[c])/max(abs(x[c]),1e-30))
                row[f'{who}_{axis}_phase_shift_deg']=float(angle_deg(y[c]*x[c].conj()))
        for key in ('P45','P135'):
            row['Gamma48_D_'+key]=float(a['D_'+key][i]);row['shift48_D_'+key]=float(b['D_'+key][i])
            row['Gamma48_C_'+key]=float(a['contrast_'+key][i]);row['shift48_C_'+key]=float(b['contrast_'+key][i])
        row['cross_phase_change_deg']=float(wrap_deg(b['phase_oddperp_minus_evenpar_deg'][i]-a['phase_oddperp_minus_evenpar_deg'][i]))
        comparisons.append(row)
    check();make_figures(blocks);check()
    with (OUT/'harmonics_H1_H9.tsv').open('x',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]),delimiter='\t');w.writeheader();w.writerows(rows)
    np.savez_compressed(OUT/'complex_source_spectra.npz',nu=blocks['Gamma48']['nu'],
        **{label+'_'+k:v for label,b in blocks.items() for k,v in b['spectrum'].items() if isinstance(v,np.ndarray)})
    after={p:sha(p) for p in before}
    report={'task_label':'20261114','executed_at_utc':datetime.now(timezone.utc).isoformat(),
        'status':'ok','new_propagations':0,'wall_cap_s':CAP,'protocol':prot.__dict__,
        'object':'coherent_current_source_spectrum_not_total_photon_spectrum',
        'window':'sin2 on [0,Tp], 1/Tp normalization, exp(+i Omega t)',
        'frequency_sampling_step_nu':.02,'basic_resolution_scale_nu':1/8,
        'e_parallel':par,'e_perp':perp,'A_cell_Ang2':geo.a_cell_ang2,
        'blocks':blocks,'Gamma48_vs_shift48':comparisons,'checks':source_checks,
        'inputs_sha256_before':before,'inputs_unchanged':before==after,
        'source_sha256':sha(Path(__file__)),'wall_s':time.perf_counter()-start,
        'B_C_D_started':False,'stopped_after_A':True}
    report['checks_pass']=bool(report['inputs_unchanged'] and all(
        c['contrast_identity_max_abs']<1e-18 and c['integer_vs_dense_max_abs']<1e-13 for c in source_checks.values()))
    dump(OUT/'source_analysis.json',report)
    print('status',report['status'],'checks_pass',report['checks_pass'],'wall',report['wall_s'],flush=True)
    return report


if __name__=='__main__':run()
