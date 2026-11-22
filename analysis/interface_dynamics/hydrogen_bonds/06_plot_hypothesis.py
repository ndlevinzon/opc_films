#!/usr/bin/env python3
"""Aggregate independent replicas and plot V6 defect-identity hypotheses.
All temperature axes hot left -> cold right; bulk solid, AWI dashed; legends outside.
"""
import argparse, json, sys
from pathlib import Path
import numpy as np,pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from common.paths import ensure_analysis_on_path

ensure_analysis_on_path(__file__)
from common.manifest import load_manifest, plotted_rate
from common.contrast import bulk_minus_awi
from common.constants import REGION_STYLE
from common.uncertainty import contrast_caption, paired_student_interval
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

KEYS=['replica','rate_K_ns','temp_mid_K','region','awi_inner_A','awi_outer_A']
METRICS={
'p_defect':'Any coordination defect, $n_{HB}\\ne4$',
'p_D2A1':'Donor 2 / acceptor 1',
'p_D1A2':'Donor 1 / acceptor 2',
'p_D1A1':'Donor 1 / acceptor 1',
'p_under_other':'Other undercoordinated states',
'p_over':'Overcoordinated states',
'p_acceptor_deficient_donor_saturated':'Donor-saturated / acceptor-deficient water',
'p_donor_deficient_acceptor_saturated':'Acceptor-saturated / donor-deficient water',
'p_defect_outOH_joint':'Defective water with outward free OH (all-water denominator)',
'p_outOH_given_defect':'Outward free OH given defect',
'p_anyfree_given_defect':'Any free OH given defect',
'p_acceptor_proxy_outward_given_acceptor_deficit':'Outward oxygen-side proxy given acceptor deficit',
'p_acceptor_proxy_outward_given_acceptor_normal':'Outward oxygen-side proxy with normal acceptor coordination',
'p_under_with_four_nearby':'Undercoordinated water with $\\geq4$ O neighbors',
'p_under_with_lt_four_nearby':'Undercoordinated water with $<4$ O neighbors',
'p_acceptor_deficit_with_four_nearby':'Acceptor-deficient water with $\\geq4$ O neighbors',
'p_freeOH_close_O_nearby_joint_per_OH':'Free OH with a nearby oxygen (per OH)',
'p_freeOH_no_close_O_per_OH':'Free OH without nearby oxygen (per OH)',
}

def save(fig,path):
    """Write a PNG and a PDF and close the figure.

    Parameters
    ----------
    fig : 
        Matplotlib figure.
    path : 
        Input file or directory.

    Returns
    -------
    result
        Write a PNG and a PDF and close the figure.
    """
    fig.tight_layout();path.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(path.with_suffix('.png'),dpi=300,bbox_inches='tight')
    fig.savefig(path.with_suffix('.pdf'),bbox_inches='tight')
    plt.close(fig)

def setup():
    """A new figure with the shared style.

    Returns
    -------
    result
        A new figure with the shared style.
    """
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,
      'axes.spines.right':False,'pdf.fonttype':42,'axes.labelsize':11})
    fig,ax=plt.subplots(figsize=(8.2,4.85));ax.grid(alpha=.13)
    return fig,ax

def curves(rep,metric,out,band=(0,6),rate=4.44):
    """One metric against temperature for the bulk and the interface.

    Parameters
    ----------
    rep : 
        Per-replica table.
    metric : 
        Observable name.
    out : 
        Output path or directory.
    band : 
        Interface depth band ``(inner, outer)`` in angstroms.
    rate : 
        Cooling-rate code or rate in K/ns, as stored in the table.

    Returns
    -------
    result
        One metric against temperature for the bulk and the interface.
    """
    x=rep[(np.isclose(rep.awi_inner_A,band[0]))&(np.isclose(rep.awi_outer_A,band[1]))&
      (np.isclose(rep.rate_K_ns,rate))]
    if x.empty or metric not in x:return False
    fig,ax=setup();drawn=0
    for reg, marker in (('interior', 'o'), ('awi', 's')):
        lab, style = REGION_STYLE[reg]
        d=x[x.region==reg].groupby('temp_mid_K')[metric].agg(['mean','std','count']).sort_index()
        if d.empty:continue
        T=d.index.to_numpy(float);y=d['mean'].to_numpy(float);sd=d['std'].fillna(0).to_numpy(float)
        ax.plot(T,y,ls=style,marker=marker,color='black',ms=3.50,label=lab)
        ax.fill_between(T,y-sd,y+sd,color='black',alpha=.10 if reg=='interior' else .055,lw=0,label='Replica mean ± 1 SD' if drawn==0 else None)
        drawn+=1
    if not drawn:plt.close(fig);return False
    ax.invert_xaxis();ax.set(xlabel='Cooling-window midpoint temperature (K)',ylabel=METRICS[metric],title=f'AWI {band[0]:g}–{band[1]:g} Å')
    ax.legend(loc='upper left',bbox_to_anchor=(1.02,1),frameon=False)
    save(fig,out/f'{metric}_0to6A');return True

def paired(rep,metric,out,rate,band=(0,6)):
    """Paired interface-minus-bulk difference of one metric.

    Parameters
    ----------
    rep : 
        Per-replica table.
    metric : 
        Observable name.
    out : 
        Output path or directory.
    rate : 
        Cooling-rate code or rate in K/ns, as stored in the table.
    band : 
        Interface depth band ``(inner, outer)`` in angstroms.

    Returns
    -------
    result
        Paired interface-minus-bulk difference of one metric.
    """
    x=rep[np.isclose(rep.rate_K_ns,rate)&np.isclose(rep.awi_inner_A,band[0])&np.isclose(rep.awi_outer_A,band[1])]
    if x.empty:return False
    k=['replica','temp_mid_K']
    d=x.pivot_table(index=k,columns='region',values=metric,aggfunc='first').dropna(subset=['interior','awi']).reset_index()
    if d.empty:return False
    d['delta']=bulk_minus_awi(d.interior, d.awi)
    rows=[]
    for temperature, group in d.groupby('temp_mid_K'):
        mean, _sd, half, count = paired_student_interval(group.delta)
        rows.append({'temp_mid_K': temperature, 'mean': mean, 'halfwidth': half, 'count': count})
    summary=pd.DataFrame(rows).sort_values('temp_mid_K')
    fig,ax=setup();T=summary.temp_mid_K.to_numpy(float);y=summary['mean'].to_numpy(float);half=summary.halfwidth.to_numpy(float)
    ax.plot(T,y,'o-',color='black',ms=4,label='Bulk − AWI')
    ax.fill_between(T,y-half,y+half,alpha=.14,color='black',label=contrast_caption());ax.axhline(0,color='.5',ls=':')
    ax.invert_xaxis();ax.set(xlabel='Cooling-window midpoint temperature (K)',ylabel='Bulk − AWI: '+METRICS[metric],title=f'AWI {band[0]:g}–{band[1]:g} Å')
    ax.legend(loc='upper left',bbox_to_anchor=(1.02,1),frameon=False)
    save(fig,out/f'paired_{metric}_bulk_minus_AWI');return True

def decomposition(rep,out,rate,band=(0,6)):
    """Paired difference of each donor-acceptor class.

    Parameters
    ----------
    rep : 
        Per-replica table.
    out : 
        Output path or directory.
    rate : 
        Cooling-rate code or rate in K/ns, as stored in the table.
    band : 
        Interface depth band ``(inner, outer)`` in angstroms.

    Returns
    -------
    result
        Paired difference of each donor-acceptor class.
    """
    x=rep[np.isclose(rep.rate_K_ns,rate)&np.isclose(rep.awi_inner_A,band[0])&np.isclose(rep.awi_outer_A,band[1])]
    classes=['p_D2A1','p_D1A2','p_D1A1','p_under_other','p_over','p_other_four']
    if x.empty:return False
    deltas=[]
    for met in classes:
        w=x.pivot_table(index=['replica','temp_mid_K'],columns='region',values=met,aggfunc='first').dropna(subset=['interior','awi'])
        if w.empty:continue
        for (r,T),v in w.iterrows():deltas.append({'replica':r,'T':T,'class':met,'delta':float(bulk_minus_awi(v.interior, v.awi))})
    if not deltas:return False
    d=pd.DataFrame(deltas);d.to_csv(out/'paired_class_decomposition_bulk_minus_AWI.csv',index=False)
    fig,ax=setup()
    for met in classes:
        sub=d[d['class']==met]
        temperatures=[]; means=[]; halves=[]
        for temperature, group in sub.groupby('T'):
            mean, _sd, half, _count = paired_student_interval(group.delta)
            temperatures.append(temperature); means.append(mean); halves.append(half)
        ax.errorbar(temperatures, means, yerr=halves, marker='o', ms=3, capsize=2, label=met.replace('p_',''))
    ax.invert_xaxis();ax.axhline(0,color='.5',ls=':');ax.set(xlabel='Cooling-window midpoint temperature (K)',
       ylabel='Bulk − AWI population fraction', title=f'AWI {band[0]:g}–{band[1]:g} Å')
    ax.legend(loc='upper left',bbox_to_anchor=(1.02,1),frameon=False,title='Exclusive state')
    save(fig,out/'paired_exclusive_defect_decomposition_bulk_minus_AWI');return True

def depth_sensitivity(rep,out,rate,metric='p_outOH_given_defect'):
    """One interface metric in every requested depth band.

    Parameters
    ----------
    rep : 
        Per-replica table.
    out : 
        Output path or directory.
    rate : 
        Cooling-rate code or rate in K/ns, as stored in the table.
    metric : 
        Observable name.

    Returns
    -------
    result
        One interface metric in every requested depth band.
    """
    x=rep[np.isclose(rep.rate_K_ns,rate)&(rep.region=='awi')]
    if x.empty:return False
    fig,ax=setup()
    for (a,b),g in x.groupby(['awi_inner_A','awi_outer_A']):
        v=g.groupby('temp_mid_K')[metric].agg(['mean','std']).sort_index()
        xx=v.index.to_numpy(float);yy=v['mean'].to_numpy(float);sd=v['std'].fillna(0).to_numpy(float)
        ax.plot(xx,yy,'o-',ms=3,label=f'{a:g}–{b:g} Å AWI')
        ax.fill_between(xx,yy-sd,yy+sd,alpha=.10)
    ax.invert_xaxis();ax.set(xlabel='Cooling-window midpoint temperature (K)',ylabel=METRICS[metric])
    ax.legend(loc='upper left',bbox_to_anchor=(1.02,1),frameon=False)
    save(fig,out/f'depth_sensitivity_{metric}');return True

def main():
    """Average replicas and write the coordination figures.

    Returns
    -------
    result
        Average replicas and write the coordination figures.
    """
    p=argparse.ArgumentParser();p.add_argument('--manifest',type=Path,required=True)
    p.add_argument('--outroot',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--rate',type=float,default=4.44,help='Physical K/ns, not directory rate code')
    p.add_argument('--allow-incomplete',action='store_true')
    a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
    rows=load_manifest(a.manifest);frames=[];qcs=[];missing=[];kinds=set()
    for r in rows:
        # The defects rule runs on ramps and holds only. A 78 K endpoint carries
        # its source ramp rate but has no defects output, so it is not expected.
        if r['kind']=='endpoint':continue
        rate=plotted_rate(r)
        if not rate or not np.isclose(float(rate),a.rate):continue
        root=a.outroot/str(r['row_id'])
        band_dirs=sorted(path for path in root.glob('awi_*') if path.is_dir())
        if not band_dirs and (root/'hypotheses_frame.csv').exists():
            band_dirs=[root]
        if not band_dirs:
            missing.append(str(root));continue
        for band in band_dirs:
            f=band/'hypotheses_frame.csv';q=band/'free_oh_qc.csv'
            if not q.exists():
                q=band/'hypotheses_qc.csv'
            if not f.exists() or not q.exists():
                missing.append(str(band));continue
            kinds.add(r['kind'])
            d=pd.read_csv(f);d['replica']=r['replica'];d['rate_K_ns']=float(rate);d['temp_mid_K']=float(r['temperature_K'])
            frames.append(d)
            z=pd.read_csv(q);z['replica']=r['replica'];z['rate_K_ns']=float(rate);z['temp_mid_K']=float(r['temperature_K']);qcs.append(z)
    if any(row['kind']=='hold' and plotted_rate(row) and np.isclose(float(plotted_rate(row)), a.rate) for row in rows) and 'hold' not in kinds:
        raise SystemExit(
            'The manifest has hold rows at this rate, and this figure has none. '
            + (f'Missing: {missing}' if missing else '')
        )
    if missing:
        (a.out/'missing_chunks.txt').write_text('\n'.join(missing)+'\n')
        if not a.allow_incomplete:raise RuntimeError(f'{len(missing)} missing chunks. See {a.out}/missing_chunks.txt; pass --allow-incomplete only for exploration')
    if not frames:raise RuntimeError('No frame analyses found')
    df=pd.concat(frames,ignore_index=True);qc=pd.concat(qcs,ignore_index=True)
    # IMPORTANT average frames/chunks within each replica; then across replicas.
    numeric=[c for c in df.select_dtypes(include='number').columns if c not in ['frame','rate_K_ns','temp_mid_K','awi_inner_A','awi_outer_A']]
    rep=df.groupby(KEYS,as_index=False)[numeric].mean()
    rep.to_csv(a.out/'replica_means.csv',index=False)
    qc.to_csv(a.out/'all_overlap_qc.csv',index=False)
    s=rep.groupby(['rate_K_ns','temp_mid_K','region','awi_inner_A','awi_outer_A'])[numeric].agg(['mean','std','count'])
    s.to_csv(a.out/'between_replica_summary.csv')
    # Check probability-mass conservation among exclusive class labels.
    labels=['p_D2A2','p_D2A1','p_D1A2','p_D1A1','p_under_other','p_over','p_other_four']
    mass=rep[labels].sum(axis=1)
    if not np.allclose(mass,1,atol=1e-8):raise AssertionError('Exclusive classes do not sum to one')
    made=[]
    for metric in METRICS:
        if curves(rep,metric,a.out):made.append(metric)
    for metric in ['p_D2A1','p_D1A2','p_acceptor_deficient_donor_saturated','p_defect_outOH_joint',
      'p_outOH_given_defect','p_under_with_four_nearby','p_under_with_lt_four_nearby']:
        paired(rep,metric,a.out,a.rate)
    decomposition(rep,a.out,a.rate)
    depth_sensitivity(rep,a.out,a.rate)
    (a.out/'V6_plot_manifest.json').write_text(json.dumps({'rate_K_ns':a.rate,'metrics':made,'missing':missing,
      'note':'Results are structural associations, not causal decomposition; 3 independent replicas define uncertainty.'},indent=2))
    print('Plotted',len(made),'single-metric figures and comparisons into',a.out)
if __name__=='__main__':main()
