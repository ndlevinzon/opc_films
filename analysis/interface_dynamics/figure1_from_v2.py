#!/usr/bin/env python3
"""V2 -> manuscript Figure 1 with replicate-level uncertainty and censoring."""
import argparse, json, re, sys
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from opcfilm.plotting.conventions import install_figure_conventions
install_figure_conventions()  # temperature axes hot to cold; legends outside panels
from scipy.stats import t as tdist

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.paths import ensure_analysis_on_path

ensure_analysis_on_path(__file__)
from common.constants import REGION_STYLE
from common.rates import canonicalize_rate, stored_rate_label
from opcfilm.config import CONFIG
TH=np.exp(-1)
_REPLICA='|'.join(re.escape(str(name)) for name in CONFIG.film.replicas)
PAT=re.compile(rf'[/\\]({_REPLICA})[/\\]ramp_nvt[/\\](\d+)K_ns[/\\]([^/\\]+)[/\\]origin_relaxation\.csv$')
PAT_KIND=re.compile(rf'[/\\](ramp|hold|endpoint)[/\\]({_REPLICA})[/\\]([^/\\]+)Kns[/\\]([^/\\]+)K[/\\]([^/\\]+)[/\\]origin_relaxation\.csv$')
TEMP=re.compile(r'(\d{2,3})_to_(\d{2,3})K')
def parse(p):
    """Read replica, rate, label, and temperature from a result path.

    Parameters
    ----------
    p : path-like
        ``origin_relaxation.csv``. A manifest path encodes ``kind``. An older
        path encodes ``ramp_nvt`` and takes the temperature from ``_to_``.

    Returns
    -------
    tuple or None
        Replica, rate label, chunk label, and temperature in kelvin.
        ``None`` when the path is neither form.
    """
    kind=PAT_KIND.search(str(p))
    if kind:
        rate=kind[3]
        return kind[2], rate if rate not in ('unset', 'nan') else '', kind[5], float(kind[4])
    m=PAT.search(str(p)); tm=TEMP.search(m[3]) if m else None
    if not tm: return None
    return m[1],m[2],m[3],(int(tm[1])+int(tm[2]))/2

def tau_cross(df, metric):
    """Cross 1/e with the shared censoring vocabulary.

    Parameters
    ----------
    df : pandas.DataFrame
        One replica, temperature, and region. ``lag_ps`` is the time.
    metric : str
        Column that holds the correlation.

    Returns
    -------
    tau, status, max_lag : tuple
        ``status`` keeps the names this script's tables already use.
        The shared function's ``fast_censored`` and ``slow_censored`` are
        mapped back to ``left_censored`` and ``right_censored``.
    """
    from opcfilm.dynamics.relaxation import classify_tau

    ordered = df.sort_values("lag_ps")
    times = ordered.lag_ps.to_numpy(float)
    values = ordered[metric].to_numpy(float)
    finite = np.isfinite(times) & np.isfinite(values) & (times > 0)
    if int(np.count_nonzero(finite)) < 2:
        return np.nan, "insufficient", np.nan
    status, tau, bound = classify_tau(times, values, level=TH)
    mapped = {
        "resolved": "resolved",
        "fast_censored": "left_censored",
        "slow_censored": "right_censored",
        "unresolved": "insufficient",
    }[status]
    if mapped == "resolved":
        return tau, mapped, float(times[finite][-1])
    return tau, mapped, bound

def write_corrected_tables(times, out):
    """Write the relaxation tables the regional figure scripts read.

    ``plot_all.py``, ``plot_figure1c.py``, and ``close_hb_onset.py --tau``
    read ``tau_censoring_corrected.csv``, ``tau_summary_corrected.csv``,
    and ``decoupling_paired_resolved.csv``. A time that crosses ``1/e``
    before the first lag is ``fast_censored``. A time that has not crossed
    by the last lag is ``slow_censored``. ``D_TR,1`` is ``log10(tau_C1/tau_Fs)``
    and ``D_TR,2`` is ``log10(tau_C2/tau_Fs)``. A decoupling row is written
    only when all three times are resolved.

    Parameters
    ----------
    times : pandas.DataFrame
        One row per replica, temperature, region, and metric. ``status``
        uses ``resolved``, ``left_censored``, and ``right_censored``.
        ``max_lag_ps`` is the censoring bound for a censored trace.
    out : pathlib.Path
        Directory passed to ``plot_all.py --corrected-dir``.

    Returns
    -------
    None
        Writes the three CSV files.
    """
    labels = {'resolved': 'resolved', 'left_censored': 'fast_censored', 'right_censored': 'slow_censored'}
    cens = times[times.status.isin(labels)].copy()
    cens['status'] = cens.status.map(labels)
    cens['temperature_K'] = cens.temp_mid_K
    cens['bound_ps'] = np.where(cens.status.eq('resolved'), np.nan, cens.max_lag_ps)
    cens.loc[~cens.status.eq('resolved'), 'tau_ps'] = np.nan
    cens[['replica', 'rate', 'temperature_K', 'region', 'metric', 'status', 'tau_ps', 'bound_ps']].to_csv(
        out / 'tau_censoring_corrected.csv', index=False)
    summary = []
    for key, group in times.groupby(['rate', 'temp_mid_K', 'region', 'metric']):
        resolved = group.loc[group.status.eq('resolved'), 'tau_ps']
        mean, _sd, low, high, n = interval(np.log10(resolved))
        summary.append(dict(zip(
            ['rate', 'temperature_K', 'region', 'metric'], key),
            n_resolved=n, n_replicas=len(group),
            log10_tau_mean=mean, log10_ci_low=low, log10_ci_high=high))
    pd.DataFrame(summary).to_csv(out / 'tau_summary_corrected.csv', index=False)
    resolved = times[times.status.eq('resolved')]
    wide = resolved.pivot_table(
        index=['replica', 'rate', 'temp_mid_K', 'region'],
        columns='metric', values='tau_ps', aggfunc='first').reset_index()
    for metric in ('Fs', 'C1', 'C2'):
        if metric not in wide:
            wide[metric] = np.nan
    wide = wide.dropna(subset=['Fs', 'C1', 'C2'])
    wide = wide[(wide[['Fs', 'C1', 'C2']] > 0).all(axis=1)].copy()
    wide['temperature_K'] = wide.temp_mid_K
    wide['D_TR1_log10'] = np.log10(wide.C1 / wide.Fs)
    wide['D_TR2_log10'] = np.log10(wide.C2 / wide.Fs)
    wide[['replica', 'rate', 'temperature_K', 'region', 'Fs', 'C1', 'C2', 'D_TR1_log10', 'D_TR2_log10']].to_csv(
        out / 'decoupling_paired_resolved.csv', index=False)


def interval(v):
    v=np.asarray(v,float);v=v[np.isfinite(v)];n=len(v)
    if n==0:return (np.nan,np.nan,np.nan,np.nan,0)
    mean=float(v.mean());sd=float(v.std(ddof=1)) if n>1 else np.nan
    delta=tdist.ppf(.975,n-1)*sd/np.sqrt(n) if n>1 else np.nan
    return mean,sd,mean-delta,mean+delta,n

def manifest_hits(path):
    """Result files named by the project manifest.

    Parameters
    ----------
    path : path-like
        ``outputs/manifest.tsv``.

    Returns
    -------
    list of tuple
        ``(csv, replica, rate, label, temperature_K)`` for every row whose
        output directory contains ``origin_relaxation.csv``.
    """
    from common.manifest import load_manifest, plotted_rate, result_directory
    hits=[]
    for row in load_manifest(path):
        candidate=result_directory(Path(__file__).resolve().parents[2], 'regional_dyn', row)/'origin_relaxation.csv'
        if not candidate.is_file():
            candidate=Path(row['outdir'])/'origin_relaxation.csv'
        if candidate.is_file() and row['temperature_K']!='':
            hits.append((candidate,row['replica'],plotted_rate(row),row['label'],float(row['temperature_K']),row['kind']))
    return hits

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--manifest',default=None);ap.add_argument('--v2-root',default=None);ap.add_argument('--out',required=True);ap.add_argument('--rate',type=float,default=4.44,help='Physical cooling rate in K/ns.');ap.add_argument('--temps',default='275,235,195,155');ap.add_argument('--stationarity-max-drift',type=float,default=.10);a=ap.parse_args();a.rate=canonicalize_rate(a.rate)
    if a.manifest is None and a.v2_root is None:ap.error('Pass --manifest or --v2-root.')
    out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    origins=[];qc=[];seen=set()
    discovered=[]
    if a.manifest:discovered.extend(manifest_hits(a.manifest))
    if a.v2_root:
        for f in Path(a.v2_root).rglob('origin_relaxation.csv'):
            v=parse(f)
            if v is not None:discovered.append((f,)+v+('',))
    if a.manifest:
        from common.manifest import load_manifest, plotted_rate
        rows=load_manifest(a.manifest)
        wanted={plotted_rate(row) for row in rows if row['kind']=='hold'}
        found={kind for *_, kind in discovered}
        if str(a.rate) in wanted and 'hold' not in found:
            raise SystemExit('The manifest has hold rows at this rate, and this figure has none.')
    for f,rep,rate,chunk,T,kind in discovered:
        if f in seen:continue
        seen.add(f)
        d=pd.read_csv(f)
        needed={'origin','lag_ps','region','n','fs_sum','c1_sum','c2_sum'}
        if not needed.issubset(d):raise RuntimeError(f'{f}: missing {needed-set(d.columns)}')
        rate_label=stored_rate_label(rate) if str(rate) not in ('', 'unset', 'nan') else ''
        d=d.assign(replica=rep,rate=rate_label,chunk=chunk,temp_mid_K=T,kind=kind)
        origins.append(d)
        md=f.parent/'metadata.json'
        qc.append(dict(replica=rep,rate=rate_label,chunk=chunk,temp_mid_K=T,metadata_present=md.exists(),n_origins=d.origin.nunique(),n_rows=len(d),source=str(f)))
    if not origins:raise SystemExit('No V2 origin_relaxation.csv found')
    d=pd.concat(origins,ignore_index=True);pd.DataFrame(qc).to_csv(out/'coverage.csv',index=False)
    # origins are the natural first-level time blocks; missing lag-specific data excluded.
    keys=['replica','rate','chunk','temp_mid_K','region','origin','lag_ps']
    g=d.groupby(keys,as_index=False)[['n','fs_sum','c1_sum','c2_sum']].sum()
    for name,ss in [('Fs','fs_sum'),('C1','c1_sum'),('C2','c2_sum')]:g[name]=g[ss]/g['n']
    g.to_csv(out/'origin_curves.csv',index=False)
    # Per-chunk drift diagnostic of observable at shortest common lag: early vs late origins
    drift=[]
    for key,dd in g.groupby(['replica','rate','chunk','temp_mid_K','region']):
        l0=dd.lag_ps.min();v=dd[dd.lag_ps==l0].sort_values('origin')
        n=len(v)
        for metric in ['Fs','C1']:
            k=max(1,n//3);early=v[metric].iloc[:k].mean();late=v[metric].iloc[-k:].mean()
            drift.append(dict(zip(['replica','rate','chunk','temp_mid_K','region'],key),metric=metric,lag_ps=l0,origins=n,early=early,late=late,abs_change=abs(late-early),flag=bool(abs(late-early)>a.stationarity_max_drift)))
    pd.DataFrame(drift).to_csv(out/'stationarity_screen.csv',index=False)
    # aggregate origins within chunks, chunks within replicas; do not merge physical replicas
    ck=['replica','rate','chunk','temp_mid_K','region','lag_ps']
    c=g.groupby(ck,as_index=False)[['n','fs_sum','c1_sum','c2_sum']].sum()
    for name,ss in [('Fs','fs_sum'),('C1','c1_sum'),('C2','c2_sum')]:c[name]=c[ss]/c['n']
    c.to_csv(out/'chunk_curves.csv',index=False)
    rk=['replica','rate','temp_mid_K','region','lag_ps']
    r=c.groupby(rk,as_index=False)[['n','fs_sum','c1_sum','c2_sum']].sum()
    for name,ss in [('Fs','fs_sum'),('C1','c1_sum'),('C2','c2_sum')]:r[name]=r[ss]/r['n']
    r=r.assign(temperature_K=r.temp_mid_K)
    r.to_csv(out/'replica_curves.csv',index=False)
    sums=[]
    for key,dd in r.groupby(['rate','temp_mid_K','region','lag_ps']):
        row=dict(zip(['rate','temp_mid_K','region','lag_ps'],key))
        for metric in ['Fs','C1','C2']:
            mean,sd,lo,hi,n=interval(dd[metric]);row.update({f'{metric}_{k}':v for k,v in dict(mean=mean,sd=sd,ci_low=lo,ci_high=hi,nrep=n).items()})
        sums.append(row)
    pd.DataFrame(sums).to_csv(out/'replica_mean_curves.csv',index=False)
    # IMPORTANT physics sanity gate: colder water should not systematically relax faster than hot liquid
    thermal=[]
    for (rate,region,metric),dd in pd.DataFrame(sums).melt(id_vars=['rate','temp_mid_K','region','lag_ps'],value_vars=['Fs_mean','C1_mean'],var_name='metric',value_name='value').groupby(['rate','region','metric']):
        temps=sorted(dd.temp_mid_K.unique())
        if len(temps)<2:continue
        midlag=float(np.median(dd.lag_ps.unique()))
        z=dd.iloc[(dd.lag_ps-midlag).abs().argsort()[:len(temps)*2]]
        low=z[z.temp_mid_K==temps[0]].sort_values('lag_ps');high=z[z.temp_mid_K==temps[-1]].sort_values('lag_ps')
        common=np.intersect1d(low.lag_ps.values,high.lag_ps.values)
        if not len(common):continue
        lag=common[len(common)//2];vl=float(low.loc[low.lag_ps==lag,'value'].iloc[0]);vh=float(high.loc[high.lag_ps==lag,'value'].iloc[0])
        thermal.append(dict(rate=rate,region=region,metric=metric,lag_ps=lag,cold_K=temps[0],hot_K=temps[-1],cold_value=vl,hot_value=vh,cold_relaxes_faster=bool(vl<vh),warning='CHECK TEMPERATURE/COORDINATES' if vl<vh else ''))
    pd.DataFrame(thermal).to_csv(out/'physics_sanity_temperature_order.csv',index=False)

    tr=[]
    for key,dd in r.groupby(['replica','rate','temp_mid_K','region']):
        for metric in ['Fs','C1','C2']:
            tau,status,maxlag=tau_cross(dd,metric)
            tr.append(dict(zip(['replica','rate','temp_mid_K','region'],key),metric=metric,tau_ps=tau,status=status,max_lag_ps=maxlag))
    ts=pd.DataFrame(tr);ts.to_csv(out/'tau_by_replica.csv',index=False)
    stat=[]
    for key,dd in ts.groupby(['rate','temp_mid_K','region','metric']):
        resolved=dd[dd.status=='resolved'].tau_ps
        mean,sd,lo,hi,n=interval(np.log10(resolved))
        stat.append(dict(zip(['rate','temp_mid_K','region','metric'],key),nrep_total=len(dd),nrep_resolved=n,nrep_right_censored=int((dd.status=='right_censored').sum()),log10_tau_mean=mean,log10_tau_sd=sd,log10_tau_ci_low=lo,log10_tau_ci_high=hi,tau_geom_ps=10**mean if np.isfinite(mean) else np.nan,all_resolved=bool(n==len(dd))))
    pd.DataFrame(stat).to_csv(out/'tau_summary.csv',index=False)
    write_corrected_tables(ts, out)
    w=ts.pivot_table(index=['replica','rate','temp_mid_K','region'],columns='metric',values='tau_ps',aggfunc='first').reset_index()
    w['D_TR1_log10']=np.log10(w['C1']/w['Fs'])
    w['D_TR2_log10']=np.log10(w['C2']/w['Fs'])
    w['D_TR_log10']=w['D_TR1_log10']
    w.to_csv(out/'decoupling_by_replica.csv',index=False)
    dec=[]
    for key,dd in w.groupby(['rate','temp_mid_K','region']):
        mean,sd,lo,hi,n=interval(dd.D_TR_log10)
        dec.append(dict(zip(['rate','temp_mid_K','region'],key),mean=mean,sd=sd,ci_low=lo,ci_high=hi,nrep_resolved=n,nrep_total=len(dd)))
    pd.DataFrame(dec).to_csv(out/'decoupling_summary.csv',index=False)
    rs=r[r.rate==str(a.rate)]; temperatures=[float(x) for x in a.temps.split(',')]
    avail=np.sort(rs.temp_mid_K.unique()); selected=[float(avail[np.argmin(abs(avail-t))]) for t in temperatures];selected=list(dict.fromkeys(selected))
    def nice(ax):ax.spines[['top','right']].set_visible(False);ax.grid(alpha=.14)
    for metric,letter in [('Fs','A'),('C1','B')]:
        fig,ax=plt.subplots(figsize=(6.5,4.8)); colors=plt.cm.coolwarm(np.linspace(.05,.95,len(selected)))
        for T,color in zip(selected,colors):
            for reg,(label,ls) in REGION_STYLE.items():
                dd=rs[(rs.temp_mid_K==T)&(rs.region==reg)].groupby('lag_ps',as_index=False)[metric].mean()
                if len(dd):ax.plot(dd.lag_ps,dd[metric],ls,color=color,label=f'{T:g} K {label}')
        ax.axhline(TH,color='gray',ls=':',lw=1);ax.set(xscale='log',xlabel='Lag (ps)',ylabel=metric,title=f'{metric}: {a.rate} K/ns');nice(ax);ax.legend(fontsize=8,ncol=2,frameon=False)
        fig.tight_layout();fig.savefig(out/f'Fig1{letter}_{metric}.png',dpi=220);plt.close(fig)
    fig,ax=plt.subplots(figsize=(6.5,4.8))
    for (reg,met),dd in ts[(ts.rate==str(a.rate))&(ts.status=='resolved')].groupby(['region','metric']):
        if reg not in ('awi','interior') or met not in ('Fs','C1'):continue
        c={'awi':'tab:blue','interior':'tab:orange'}[reg];mk={'Fs':'o','C1':'s'}[met]
        ax.scatter(dd.temp_mid_K,dd.tau_ps,color=c,marker=mk,alpha=.6,label=f'{reg} {met}')
    ax.set(yscale='log',xlabel='Window midpoint (K)',ylabel='Resolved relaxation time (ps)',title=f'Rate {a.rate}: resolved times only');nice(ax);ax.legend(frameon=False)
    fig.tight_layout();fig.savefig(out/'Fig1C_tau.png',dpi=220);plt.close(fig)
    fig,ax=plt.subplots(figsize=(6.5,4.8))
    for reg,dd in w[(w.rate==str(a.rate))&w.region.isin(['awi','interior'])].groupby('region'):
        ax.scatter(dd.temp_mid_K,dd.D_TR_log10,label=reg,alpha=.7)
    ax.axhline(0,color='gray',ls=':');ax.set(xlabel='Window midpoint (K)',ylabel=r'log$_{10}$($\tau_{rot}/\tau_{trans}$)',title=f'Rate {a.rate}: both times resolved');nice(ax);ax.legend(frameon=False)
    fig.tight_layout();fig.savefig(out/'Fig1D_decoupling.png',dpi=220);plt.close(fig)
    (out/'README_RESULTS.txt').write_text('Exploratory V2 Figure 1. Curve means are averaged across replicas for the plots; errors and origin-level curves are in CSV. Cooling chunks are nonequilibrium. Censored tau excluded from ordinary scatter and retained explicitly in tau_by_replica.csv. Stationarity screen is diagnostic only. F_s at the first lag can be near 0.5 in a hot window: q* is 1.925 A^-1, so a 2 ps displacement already lowers the Bessel average. That is the scattering function, not a normalization error.\n')
    print('Wrote Figure 1 and validation tables:',out)
if __name__=='__main__':main()
