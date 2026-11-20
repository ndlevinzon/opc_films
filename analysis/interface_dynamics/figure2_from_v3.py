#!/usr/bin/env python3
"""Aggregate HB counts, intermittent pair retention, and event enrichment for V3.
Association is not causal; chronological same-origin join when possible.
"""
import argparse,re,sys
from pathlib import Path
import numpy as np,pandas as pd, matplotlib

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.paths import ensure_analysis_on_path

ensure_analysis_on_path(__file__)
from common.constants import REGION_STYLE
from common.rates import canonicalize_rate, directory_code, stored_rate_label
matplotlib.use('Agg')
import matplotlib.pyplot as plt
PAT=re.compile(r'[/\\](r[123])[/\\]ramp_nvt[/\\](\d+)K_ns[/\\](.+)[/\\]hb_coord_samples\.csv$')
PAT_KIND=re.compile(r'[/\\](ramp|hold|endpoint)[/\\](r[123])[/\\]([^/\\]+)Kns[/\\]([^/\\]+)K[/\\]([^/\\]+)[/\\]hb_coord_samples\.csv$')
T=re.compile(r'(\d{2,3})_to_(\d{2,3})K')
def meta(p):
    """Read replica, rate, label, and temperature from a hydrogen-bond path.

    Parameters
    ----------
    p : path-like
        ``hb_coord_samples.csv``.

    Returns
    -------
    tuple or None
        Replica, physical rate, chunk label, and temperature in kelvin.
    """
    kind=PAT_KIND.search(str(p))
    if kind:
        rate=kind[3]
        label=stored_rate_label(rate) if rate not in ('unset', 'nan', '') else ''
        return kind[2], label, kind[5], float(kind[4])
    m=PAT.search(str(p));t=T.search(m[3]) if m else None
    if not t:return None
    return m[1],stored_rate_label(m[2]),m[3],(int(t[1])+int(t[2]))/2

def manifest_hits(path):
    """Hydrogen-bond tables named by the project manifest.

    Parameters
    ----------
    path : path-like
        ``outputs/manifest.tsv``.

    Returns
    -------
    list of tuple
        ``(csv, replica, rate, label, temperature_K)`` for rows that have
        ``hb_coord_samples.csv`` in their output directory.
    """
    from common.manifest import load_manifest, plotted_rate, result_directory
    hits=[]
    for row in load_manifest(path):
        candidate=result_directory(Path(__file__).resolve().parents[2], 'hbond', row)/'hb_coord_samples.csv'
        if not candidate.is_file():
            candidate=Path(row['outdir'])/'hb_coord_samples.csv'
        if candidate.is_file() and row['temperature_K']!='':
            rate=plotted_rate(row)
            rate=stored_rate_label(rate) if rate not in ('', 'unset') else ''
            hits.append((candidate,row['replica'],rate,row['label'],float(row['temperature_K']),row['kind']))
    return hits

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--manifest',default=None);ap.add_argument('--hb-root',default=None);ap.add_argument('--v2-root',default=None);ap.add_argument('--out',required=True);ap.add_argument('--rate',type=float,default=4.44,help='Physical cooling rate in K/ns.');ap.add_argument('--temps',default='215,78',help='Temperatures in K. Each is matched to the nearest stored midpoint.');a=ap.parse_args();a.rate=canonicalize_rate(a.rate)
    if a.manifest is None and a.hb_root is None:ap.error('Pass --manifest or --hb-root.')
    out=Path(a.out);out.mkdir(parents=True,exist_ok=True);counts=[];curves=[];joins=[];seen=set();discovered=[]
    if a.manifest:discovered.extend(manifest_hits(a.manifest))
    if a.hb_root:
        for f in Path(a.hb_root).rglob('hb_coord_samples.csv'):
            x=meta(f)
            if x is not None:discovered.append((f,)+x+('',))
    if a.manifest:
        from common.manifest import load_manifest, plotted_rate
        rows=load_manifest(a.manifest)
        wanted={plotted_rate(row) for row in rows if row['kind']=='hold'}
        found={kind for *_, kind in discovered}
        if str(a.rate) in wanted and 'hold' not in found:
            raise SystemExit('The manifest has hold rows at this rate, and this figure has none.')
    for f,rep,rate,chunk,temp,kind in discovered:
        if f in seen:continue
        seen.add(f)
        ident=dict(replica=rep,rate=rate,chunk=chunk,temp_mid_K=temp,kind=kind)
        d=pd.read_csv(f).assign(**ident);counts.append(d)
        cf=f.parent/'hb_intermit_pairs.csv'
        if cf.exists() and cf.stat().st_size:curves.append(pd.read_csv(cf).assign(**ident))
        ef=f.parent/'hb_event_samples.csv'
        vf=f.parent/'event_samples.csv'
        if not vf.exists() and a.v2_root and rate not in ('',):
            vf=Path(a.v2_root)/rep/'ramp_nvt'/f"{directory_code(rate)}K_ns"/chunk/'event_samples.csv'
        if ef.exists() and vf.exists():
            hb=pd.read_csv(ef);ev=pd.read_csv(vf)
            common=['origin','ref','region'];needed=set(common+['ref_angle_late_rad','shell_move_early_A'])
            if needed.issubset(ev) and set(common).issubset(hb):
                merged=ev.merge(hb,on=common,how='inner',validate='one_to_one')
                if len(merged):joins.append(merged.assign(**ident))
    if not counts:raise SystemExit('No HB CSVs found')
    c=pd.concat(counts,ignore_index=True);c.to_csv(out/'hb_coord_all.csv',index=False)
    distributions=c.groupby(['replica','rate','temp_mid_K','region','n_hb']).size().rename('count').reset_index()
    distributions['prob']=distributions['count']/distributions.groupby(['replica','rate','temp_mid_K','region'])['count'].transform('sum')
    distributions.to_csv(out/'hb_coord_distributions.csv',index=False)
    if curves:
        d=pd.concat(curves,ignore_index=True)
        k=['replica','rate','temp_mid_K','region','lag_ps'];p=d.groupby(k,as_index=False)[['n_initial','n_retained']].sum()
        p['C_HB_intermittent']=p.n_retained/p.n_initial
        p.to_csv(out/'hb_intermittent_replica.csv',index=False)
        m=p.groupby(['rate','temp_mid_K','region','lag_ps']).C_HB_intermittent.agg(['mean','std','count']).reset_index()
        m.to_csv(out/'hb_intermittent_mean.csv',index=False)
    else:m=pd.DataFrame()
    rr=distributions[distributions.rate==str(a.rate)]
    available=np.sort(rr.temp_mid_K.unique())
    if len(available)==0:raise SystemExit('No coordination rows for this rate.')
    requested=[float(v) for v in a.temps.split(',') if v.strip()]
    chosen=[]
    for t in requested:
        match=float(available[np.argmin(np.abs(available-t))])
        if match not in chosen:chosen.append(match)
    for t in chosen:
        fig,ax=plt.subplots(figsize=(6.5,4.3))
        for reg,(label,ls) in REGION_STYLE.items():
            z=rr[(np.isclose(rr.temp_mid_K,t))&(rr.region==reg)].groupby('n_hb').prob.mean()
            ax.plot(z.index,z.values,ls+'o',label=label)
        ax.set(xlabel='H bonds per water',ylabel='P(nHB), mean of replica histograms',title=f'{a.rate} K/ns, {t:g} K, AWI 3–12 Å');ax.legend(frameon=False);fig.tight_layout();fig.savefig(out/f'Fig2A_hbond_distribution_T{t:g}K.png',dpi=200);plt.close(fig)
        if not m.empty:
            fig,ax=plt.subplots(figsize=(6.5,4.3))
            for reg,(label,ls) in REGION_STYLE.items():
                z=m[(m.rate==str(a.rate))&(np.isclose(m.temp_mid_K,t))&(m.region==reg)];ax.plot(z.lag_ps,z['mean'],ls+'o',label=label)
            ax.set(xscale='log',xlabel='Lag (ps)',ylabel='Intermittent original-partner retention',title=f'{a.rate} K/ns, {t:g} K, AWI 3–12 Å. Compared at 320 ps.');ax.legend(frameon=False);fig.tight_layout();fig.savefig(out/f'Fig2B_hbond_persistence_T{t:g}K.png',dpi=200);plt.close(fig)
    if joins:
        j=pd.concat(joins,ignore_index=True);j.to_csv(out/'aligned_event_samples.csv',index=False)
        # Empirical threshold is for exploration; matched control / conditional modeling in stage D.
        j['reorient30']=(j.ref_angle_late_rad>np.pi/6).astype(int)
        event=[]
        for k,z in j.groupby(['replica','rate','temp_mid_K','region']):
            active=z.hb_break_early>0
            for typ,mask in [('HB_broken',active),('no_HB_break',~active)]:
                zz=z[mask];event.append(dict(zip(['replica','rate','temp_mid_K','region'],k),condition=typ,n=len(zz),prob=float(zz.reorient30.mean()) if len(zz)>0 else np.nan))
        pd.DataFrame(event).to_csv(out/'reorientation_given_HB.csv',index=False)
    else:
        (out/'join_warning.txt').write_text('No event joins. V2 and V3 origins/reference IDs must match exactly; use identical origin counts, seed, and reference selection or rerun coordinated extraction. Do not present matched controls or predictions from unmatched data.\n')
    print('Wrote HB results',out,'aligned event rows',sum(map(len,joins)))
if __name__=='__main__':main()
