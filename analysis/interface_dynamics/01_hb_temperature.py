#!/usr/bin/env python3
"""Temperature comparison of V3 intermittent HB partner retention; replica is uncertainty unit."""
import argparse
import sys
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from opcfilm.plotting.conventions import install_figure_conventions
install_figure_conventions()  # temperature axes hot to cold; legends outside panels
from matplotlib.lines import Line2D
from matplotlib.colors import Normalize

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.paths import ensure_analysis_on_path

ensure_analysis_on_path(__file__)
from common.constants import REGION_STYLE
from common.rates import canonicalize_rate, stored_rate_label
from common.uncertainty import contrast_caption, paired_student_interval

ap=argparse.ArgumentParser()
ap.add_argument('--hb-summary',required=True,type=Path,help='hb_intermittent_replica.csv')
ap.add_argument('--out',required=True,type=Path)
ap.add_argument('--rate', type=float, default=4.44, help='Physical cooling rate in K/ns.')
ap.add_argument('--temps',default='235,215,195')
ap.add_argument('--lag',type=float,default=320,help='Selected reference lag in ps for temperature trends')
a=ap.parse_args();a.rate=canonicalize_rate(a.rate);a.out.mkdir(parents=True,exist_ok=True)
d=pd.read_csv(a.hb_summary)
req={'replica','rate','temp_mid_K','region','lag_ps','C_HB_intermittent'}
if not req.issubset(d):raise SystemExit(f'Missing columns: {sorted(req-set(d))}')
d['rate']=d.rate.map(stored_rate_label)
d=d[(d.rate==a.rate)&d.region.isin(['awi','interior'])].copy()
temps=np.array([float(s) for s in a.temps.split(',')]);avail=np.sort(d.temp_mid_K.unique())
if not all(np.any(np.isclose(avail,T,atol=.01)) for T in temps):
    raise SystemExit(f'Requested midpoint(s) absent. Available: {avail.tolist()}')
d=d[d.temp_mid_K.isin(temps)]
# Prevent accidental overrepresentation of duplicate rows, use one within-replica value per lag.
if d.duplicated(['replica','rate','temp_mid_K','region','lag_ps']).any():
    raise SystemExit('Duplicate replica/temperature/region/lag entries. Inspect aggregation.')
mean=d.groupby(['temp_mid_K','region','lag_ps'],as_index=False).C_HB_intermittent.agg(['mean','std','count']).reset_index()
mean.to_csv(a.out/'hb_temp_replica_summary.csv',index=False)
# Numeric temperature codes map smoothly and consistently: hottest = red, coldest = blue.
norm=Normalize(vmin=min(temps),vmax=max(temps));cmap=plt.get_cmap('coolwarm')
fig,ax=plt.subplots(figsize=(8.8,5.3))
for T in sorted(temps,reverse=True):
    for reg, (_label, ls) in REGION_STYLE.items():
        z=mean[(mean.temp_mid_K==T)&(mean.region==reg)].sort_values('lag_ps')
        if len(z):
            ax.plot(z.lag_ps,z['mean'],ls,color=cmap(norm(T)),lw=2.2)
            # ± one SD across independent replicas, not a CI.
            if (z['count']>=2).all():
                low=np.clip(z['mean'].to_numpy(float)-z['std'].to_numpy(float),0,1)
                high=np.clip(z['mean'].to_numpy(float)+z['std'].to_numpy(float),0,1)
                ax.fill_between(z.lag_ps.to_numpy(),low,high,color=cmap(norm(T)),alpha=.10)
ax.set(xscale='log',xlabel='Lag (ps)',ylabel='Intermittent original-partner retention, 3–12 Å',title=f'H-bond persistence, {a.rate:g} K/ns')
ax.grid(alpha=.18)
handles=[Line2D([0],[0],color=cmap(norm(T)),lw=2,label=f'{T:g} K') for T in sorted(temps,reverse=True)]
handles += [Line2D([0],[0],color='black',ls=ls,lw=2,label=label) for label, ls in REGION_STYLE.values()]
ax.legend(handles=handles,loc='center left',bbox_to_anchor=(1.02,.5),frameon=False)
fig.tight_layout();fig.savefig(a.out/'HB_persistence_by_temperature.png',dpi=250,bbox_inches='tight');fig.savefig(a.out/'HB_persistence_by_temperature.pdf',bbox_inches='tight');plt.close(fig)
# Paired difference at nearest *actual common* lag for each temperature; across replicas.
records=[]
for (rep,T),z in d.groupby(['replica','temp_mid_K']):
    wide=z.pivot_table(index='lag_ps',columns='region',values='C_HB_intermittent')
    wide=wide.dropna(subset=['awi','interior'])
    if wide.empty:continue
    lag=float(wide.index[np.argmin(abs(wide.index.to_numpy()-a.lag))]);rr=wide.loc[lag]
    records.append(dict(replica=rep,temp_mid_K=T,lag_ps=lag,AWI=rr.awi,interior=rr.interior,interior_minus_AWI=rr.interior-rr.awi))
contrast=pd.DataFrame(records)
contrast.to_csv(a.out/'HB_persistence_paired_contrast.csv',index=False)
if not contrast.empty:
    rows=[]
    for temperature, group in contrast.groupby('temp_mid_K'):
        mean, sd, half, count = paired_student_interval(group.interior_minus_AWI)
        rows.append(dict(temp_mid_K=temperature, mean=mean, sd=sd, ci95_halfwidth=half, n=count))
    summary=pd.DataFrame(rows)
    summary.to_csv(a.out/'HB_persistence_paired_contrast_summary.csv',index=False)
    fig,ax=plt.subplots(figsize=(6.4,4.2))
    ax.errorbar(summary.temp_mid_K, summary['mean'], yerr=summary.ci95_halfwidth, fmt='o-', color='black', capsize=3)
    ax.axhline(0, color='gray', ls=':')
    ax.invert_xaxis()
    ax.set(xlabel='Cooling-window midpoint (K)', ylabel='Bulk − AWI retention, 3–12 Å', title=contrast_caption())
    fig.tight_layout(); fig.savefig(a.out/'HB_persistence_paired_contrast.png', dpi=250, bbox_inches='tight'); plt.close(fig)
print('Wrote',a.out,'; caution: temperature windows are continuously cooled, not stationary holds.')
