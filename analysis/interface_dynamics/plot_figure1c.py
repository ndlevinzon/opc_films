#!/usr/bin/env python3
"""Figure 1C: paired log10(interior/AWI relaxation-time contrasts).
Reads V3 tau_censoring_corrected.csv; no trajectories are needed.
Only pairs in which BOTH AWI and interior relaxation times are resolved are used.
"""
import argparse
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import t as t_dist

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.paths import ensure_analysis_on_path

ensure_analysis_on_path(__file__)
from common.rates import canonicalize_rate, stored_rate_label


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--tau', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True, help='Output directory')
    p.add_argument('--rate', type=float, default=4.44, help='Physical cooling rate in K/ns.')
    p.add_argument('--temp-min', type=float, default=195)
    p.add_argument('--temp-max', type=float, default=265)
    p.add_argument('--min-pairs', type=int, default=3, help='Require three paired replicas by default')
    p.add_argument('--show-replicas', action='store_true')
    a=p.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    a.rate = canonicalize_rate(a.rate)
    df=pd.read_csv(a.tau)
    needed={'replica','rate','temperature_K','region','metric','status','tau_ps'}
    if missing:=needed-set(df.columns):
        p.error(f'Missing columns: {sorted(missing)}')
    df['rate']=df.rate.map(stored_rate_label)
    df=df[(df.rate==a.rate)&
          df.region.isin(['awi','interior'])&df.metric.isin(['Fs','C1'])].copy()
    if df.empty: p.error(f'No Fs/C1 AWI/interior records for rate {a.rate}')
    # Strictly pair same replica, rate, temperature, metric. No interpolation across temperatures.
    for c in ['replica','region','metric','status']:
        df[c]=df[c].astype(str)
    keys=['replica','rate','temperature_K','metric']
    if df.duplicated(keys+['region']).any():
        p.error('Duplicate replica/rate/temperature/metric/region rows found: investigate input')
    wide=df.pivot(index=keys,columns='region',values=['status','tau_ps'])
    rows=[]
    for idx, row in wide.iterrows():
        rep,rate,temp,metric=idx
        awis=row.get(('status','awi'))
        ints=row.get(('status','interior'))
        ta=pd.to_numeric(row.get(('tau_ps','awi')),errors='coerce')
        ti=pd.to_numeric(row.get(('tau_ps','interior')),errors='coerce')
        is_good=(awis=='resolved' and ints=='resolved' and np.isfinite(ta) and
                 np.isfinite(ti) and ta>0 and ti>0)
        rows.append({'replica':rep,'rate':rate,'temperature_K':temp,'metric':metric,
                     'awi_status':awis,'interior_status':ints,'tau_awi_ps':ta,
                     'tau_interior_ps':ti,'paired_resolved':bool(is_good),
                     'log10_interior_over_awi':np.log10(ti/ta) if is_good else np.nan})
    audit=pd.DataFrame(rows)
    audit.to_csv(a.out/'Fig1C_replica_pair_audit.csv',index=False)
    summary=[]
    for (temp,metric),g in audit.groupby(['temperature_K','metric'],sort=True):
        vals=g.loc[g.paired_resolved,'log10_interior_over_awi'].to_numpy(dtype=float)
        n=len(vals); n_total=len(g)
        mu=float(vals.mean()) if n else np.nan
        sd=float(vals.std(ddof=1)) if n>=2 else np.nan
        hw=float(t_dist.ppf(.975,n-1)*sd/np.sqrt(n)) if n>=2 else np.nan
        summary.append({'temperature_K':temp,'metric':metric,'n_pairs':n,
                        'n_total_replicas':n_total,'all_replicas_resolved':n==n_total,
                        'mean_log10_interior_over_awi':mu,
                        'sd_log10':sd,'ci95_low':mu-hw if n>=2 else np.nan,
                        'ci95_high':mu+hw if n>=2 else np.nan,
                        'ratio_interior_over_awi':10**mu if n else np.nan})
    s=pd.DataFrame(summary).sort_values(['metric','temperature_K'])
    s.to_csv(a.out/'Fig1C_region_contrast_summary.csv',index=False)
    plt.rcParams.update({'font.size':11,'axes.spines.top':False,'axes.spines.right':False,
                         'pdf.fonttype':42,'ps.fonttype':42})
    fig,ax=plt.subplots(figsize=(7.5,4.6),layout='constrained')
    styles={'Fs':("#FF9100",'o',r'Translation, $F_s^\parallel$'),
            'C1':("#00D9F6",'s',r'Rotation, $C_1$')}
    plotted=0
    for metric,(color,mark,name) in styles.items():
        sub=s[(s.metric==metric)&(s.temperature_K.between(a.temp_min,a.temp_max))&
              (s.n_pairs>=a.min_pairs)&s.ci95_low.notna()].copy()
        if sub.empty:continue
        sub=sub.sort_values('temperature_K',ascending=False)
        y=sub.mean_log10_interior_over_awi.to_numpy(float)
        err=np.vstack([y-sub.ci95_low.to_numpy(float),sub.ci95_high.to_numpy(float)-y])
        ax.errorbar(sub.temperature_K,y,yerr=err,color=color,marker=mark,
                    ms=6,lw=1.8,elinewidth=1.1,capsize=3,label=name)
        if a.show_replicas:
            aa=audit[(audit.metric==metric)&audit.paired_resolved&
                     audit.temperature_K.isin(sub.temperature_K)]
            ax.scatter(aa.temperature_K,aa.log10_interior_over_awi,
                       color=color,s=15,alpha=.20,zorder=1)
        plotted+=1
    if plotted==0:
        p.error('No temperatures with enough paired resolved replicas in range. '
                'Try --min-pairs 2 or expand temperature range, and inspect audit CSV.')
    ax.axhline(0,color='0.4',ls=':',lw=1)
    ax.set_xlabel('Temperature (K)')
    ax.set_ylabel(r'$\log_{10}(\tau_{\mathrm{interior}}/\tau_{\mathrm{AWI}})$')
    ax.set_xlim(a.temp_max+2,a.temp_min-2)  # cooling hot -> cold left -> right
    ax.grid(axis='y',alpha=.17)
    ax.legend(loc='center left',bbox_to_anchor=(1.02,.5),frameon=False)
    fig.savefig(a.out/'Fig1C_region_relaxation_contrast.png',dpi=300,bbox_inches='tight')
    fig.savefig(a.out/'Fig1C_region_relaxation_contrast.pdf',bbox_inches='tight')
    plt.close(fig)
    print('Wrote:',a.out.resolve())
    print('Paired resolved counts in plotted interval:')
    print(s[s.temperature_K.between(a.temp_min,a.temp_max)][['temperature_K','metric','n_pairs','ratio_interior_over_awi']].to_string(index=False))

if __name__=='__main__':main()
