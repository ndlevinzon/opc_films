#!/usr/bin/env python3
"""Finalize HB regional persistence separation by T and lag, with optional relaxation comparison.
No MD reads. Independent unit = matched replica (AWI, interior).
"""
import argparse
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from opcfilm.plotting.conventions import install_figure_conventions
install_figure_conventions()  # temperature axes hot to cold; legends outside panels
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.paths import ensure_analysis_on_path

ensure_analysis_on_path(__file__)
from common.contrast import bulk_minus_awi
from common.rates import canonicalize_rate, stored_rate_label
from common.trajectory import cooling_window_lag_cap_ps
from common.uncertainty import contrast_caption, paired_student_interval

p=argparse.ArgumentParser()
p.add_argument('--hb-summary',type=Path,required=True)
p.add_argument('--tau',type=Path,default=None,help='tau_censoring_corrected.csv (optional)')
p.add_argument('--out',type=Path,required=True)
p.add_argument('--rate', type=float, default=4.44, help='Physical cooling rate in K/ns.')
p.add_argument('--lags',default='40,80,160,320,640')
p.add_argument('--hold',action='store_true',help='Isothermal hold. Do not drop lags that exceed the cooling-window cap.')
p.add_argument('--temps',default='',help='Optional comma-separated temperatures; default all available')
p.add_argument('--effect-threshold',type=float,default=.03,help='Practical separation in retention fraction')
p.add_argument('--lag-tol',type=float,default=.15,help='Maximum relative difference from requested lag')
p.add_argument('--min-pairs',type=int,default=3)
a=p.parse_args();a.rate=canonicalize_rate(a.rate);a.out.mkdir(parents=True,exist_ok=True)

def summary(x):
    """Paired Student-t summary of one contrast.

    Parameters
    ----------
    x : array_like
        One paired difference per replica.

    Returns
    -------
    dict
        Mean, sample standard deviation, 95 percent limits, and the count.
    """
    mean, sd, half, count = paired_student_interval(x)
    return dict(mean=mean, sd=sd, ci_low=mean-half, ci_high=mean+half, n=count)

d=pd.read_csv(a.hb_summary)
cols={'replica','rate','temp_mid_K','region','lag_ps','C_HB_intermittent'}
if cols-set(d):raise SystemExit(f'H-bond input missing columns: {sorted(cols-set(d))}')
d['rate']=d.rate.map(stored_rate_label);d=d[(d.rate==a.rate)&d.region.isin(['awi','interior'])].copy()
if a.temps:
    wanted=[float(t) for t in a.temps.split(',')]
    d=d[d.temp_mid_K.apply(lambda t:any(np.isclose(t,w,atol=.01) for w in wanted))]
if d.empty:raise SystemExit('No matching H-bond data; check --rate and --temps')
key=['replica','temp_mid_K','lag_ps','region']
if d.duplicated(key).any():raise SystemExit('Duplicate entries per replica/T/lag/region. Check upstream HB summary.')
wide=d.pivot(index=['replica','temp_mid_K','lag_ps'],columns='region',values='C_HB_intermittent').reset_index()
wide=wide.dropna(subset=['awi','interior']).copy()
wide['delta_HB']=bulk_minus_awi(wide.interior, wide.awi)
if wide.empty:raise SystemExit('No paired AWI and interior measurements.')
requested=np.array([float(v) for v in a.lags.split(',')])
if not a.hold:
    cap_ps=cooling_window_lag_cap_ps(10.0/float(a.rate)*1000.0)
    kept=requested[requested<=cap_ps]
    if not len(kept):raise SystemExit(f'Every requested lag exceeds the {cap_ps:g} ps cooling-window cap. Pass --hold for an isothermal table.')
    if len(kept)<len(requested):print(f'Dropped cooling-window lags above {cap_ps:g} ps.')
    requested=kept
rows=[]
# Choose a common *actual* lag for all replicas at a temperature.
for T,z in wide.groupby('temp_mid_K'):
    reps=set(z.replica.unique());
    for desired in requested:
        lag_counts=z.groupby('lag_ps').replica.nunique()
        candidates=lag_counts[lag_counts>=a.min_pairs].index.to_numpy(float)
        if not len(candidates):continue
        actual=candidates[np.argmin(abs(candidates-desired))]
        if abs(actual-desired)>max(1e-8,a.lag_tol*desired):continue
        zz=z[np.isclose(z.lag_ps,actual)]
        for _,rr in zz.iterrows():
            rows.append(dict(rate=str(a.rate),replica=rr.replica,temperature_K=float(T),requested_lag_ps=desired,
                actual_lag_ps=actual,awi=float(rr.awi),interior=float(rr.interior),delta_HB=float(rr.delta_HB)))
paired=pd.DataFrame(rows)
if paired.empty:raise SystemExit('No acceptable common lags. Inspect hb_summary lag_ps; adjust --lags or --lag-tol.')
paired.to_csv(a.out/'HB_onset_paired_replicas.csv',index=False)
ss=[]
for (temp,lag),z in paired.groupby(['temperature_K','requested_lag_ps']):
    s=summary(z.delta_HB)
    ss.append(dict(rate=str(a.rate),temperature_K=temp,lag_requested_ps=lag,
        lag_actual_ps=float(z.actual_lag_ps.iloc[0]),**s,
        all_positive=bool((z.delta_HB>0).all()),
        practical_separation=bool(s['n']>=a.min_pairs and s['mean']>=a.effect_threshold and (z.delta_HB>0).all()),
        ci_excludes_zero=bool(s['n']>1 and s['ci_low']>0)))
res=pd.DataFrame(ss).sort_values(['temperature_K','lag_requested_ps']);res.to_csv(a.out/'HB_onset_temperature_lag_table.csv',index=False)
# Compact final table: all lags as mean ± SD; one row per temperature.
compact=res.pivot(index='temperature_K',columns='lag_requested_ps',values='mean')
compact.index.name='temperature_K';compact.columns=[f'deltaHB_{v:g}ps' for v in compact.columns]
compact.reset_index().sort_values('temperature_K',ascending=False).to_csv(a.out/'TABLE_HB_onset_compact.csv',index=False)
fig,ax=plt.subplots(figsize=(8.4,5.1))
col=plt.get_cmap('viridis')(np.linspace(.12,.85,len(requested)))
for ix,lag in enumerate(requested):
    z=res[res.lag_requested_ps==lag].sort_values('temperature_K')
    if z.empty:continue
    half=z['mean'].to_numpy(float)-z.ci_low.to_numpy(float)
    ax.errorbar(z.temperature_K,z['mean'],yerr=half,fmt='o-',lw=1.7,ms=4.5,capsize=2.8,
                color=col[ix],label=f'{lag:g} ps')
ax.axhline(0,color='gray',ls=':',lw=1)
ax.axhline(a.effect_threshold,color='gray',ls='--',lw=.8,alpha=.7)
ax.invert_xaxis()
ax.set(xlabel='Cooling-window midpoint temperature (K)',ylabel=r'Bulk − AWI retention, 3–12 Å',
       title=f'Hydrogen-bond persistence, {a.rate:g} K/ns\n{contrast_caption()}')
ax.grid(alpha=.15);ax.legend(title='Observed lag',bbox_to_anchor=(1.02,.5),loc='center left',frameon=False)
fig.tight_layout();fig.savefig(a.out/'FIG_HB_temperature_lag.png',dpi=300,bbox_inches='tight');fig.savefig(a.out/'FIG_HB_temperature_lag.pdf',bbox_inches='tight');plt.close(fig)
# Optional clean pairing of region-specific resolved relaxation times at same replica and T.
if a.tau:
    t=pd.read_csv(a.tau);need={'replica','rate','temperature_K','region','metric','status','tau_ps'}
    if need-set(t):raise SystemExit(f'Tau input missing: {sorted(need-set(t))}')
    t=t[(t.rate.astype(str)==str(a.rate))&(t.region.isin(['awi','interior'])) & (t.metric.isin(['Fs','C1']))].copy()
    good=t[t.status=='resolved'];w=good.pivot_table(index=['replica','temperature_K','metric'],columns='region',values='tau_ps',aggfunc='first').dropna(subset=['awi','interior']).reset_index()
    w['log10_tau_interior_over_awi']=np.log10(w.interior/w.awi)
    w.to_csv(a.out/'RELAX_region_paired_replicas.csv',index=False)
    rr=[]
    for (temp,metric),z in w.groupby(['temperature_K','metric']):
        rr.append(dict(temperature_K=temp,metric=metric,**summary(z.log10_tau_interior_over_awi)))
    pd.DataFrame(rr).to_csv(a.out/'RELAX_region_contrast_summary.csv',index=False)
# Bracket, not interpolated Tg/onset: only complete paired replicates and prespecified threshold.
notes=[]
for lag,z in res.groupby('lag_requested_ps'):
    z=z.sort_values('temperature_K',ascending=False)
    hit=z[z.practical_separation]
    if hit.empty:notes.append(f'{lag:g} ps: no operational separation meeting threshold {a.effect_threshold:.3g}.');continue
    onset=float(hit.temperature_K.max())
    hotter=z[(z.temperature_K>onset)&(z.n>=a.min_pairs)]
    nearest=float(hotter.temperature_K.min()) if len(hotter) else None
    notes.append(f'{lag:g} ps: highest qualifying midpoint {onset:g} K; '+
      (f'bracketed by {onset:g} and {nearest:g} K (not a fitted transition).' if nearest is not None else 'no hotter tested qualifying bracket.'))
(a.out/'ONSET_README.txt').write_text('Operational HB separation: mean delta >= threshold; all three paired replicas delta>0.\n'+
  'Only measured common lag points within relative tolerance are used. Means and SD are across independent replicas.\n'+
  'Temperature windows are continuously cooling; this is not an equilibrium phase-transition temperature.\n'+
  '\n'.join(notes)+'\n')
print('\n'.join(notes));print('Outputs:',a.out)
