#!/usr/bin/env python3
"""Stratified matched observed HB loss vs no-loss; descriptive, not causal.
Exact matching in replica, temperature window, region, and prior-mobility quantile.
Each stratum weighted equally by available pairs; clusters = replicas.
"""
import argparse
import sys
from pathlib import Path
import numpy as np,pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.paths import ensure_analysis_on_path

ensure_analysis_on_path(__file__)
from common.rates import canonicalize_rate, stored_rate_label
ap=argparse.ArgumentParser()
ap.add_argument('--events',required=True,type=Path);ap.add_argument('--out',required=True,type=Path)
ap.add_argument('--rate', type=float, default=4.44, help='Physical cooling rate in K/ns.');ap.add_argument('--angle-deg',type=float,default=30.)
ap.add_argument('--mobility-bins',type=int,default=5);ap.add_argument('--min-pairs',type=int,default=5)
a=ap.parse_args();a.rate=canonicalize_rate(a.rate);a.out.mkdir(parents=True,exist_ok=True)
d=pd.read_csv(a.events)
req={'replica','rate','region','temp_mid_K','ref_move_early_A','ref_angle_late_rad','hb_break_early','origin','ref'}
if not req.issubset(d):raise SystemExit(f'Missing {sorted(req-set(d))}. Re-run V3 HB with --v2-events to align identifiers.')
keys=['replica','rate','temp_mid_K','region','origin','ref']
if d.duplicated(keys).any():raise SystemExit('Duplicate joined records: inspect V2/V3 origin/ref alignment.')
d['rate']=d.rate.map(stored_rate_label);d=d[d.rate==a.rate].copy()
for c in ['ref_move_early_A','ref_angle_late_rad','hb_break_early']:
    d[c]=pd.to_numeric(d[c],errors='coerce')
d=d.dropna(subset=['ref_move_early_A','ref_angle_late_rad','hb_break_early'])
if (d.region=='subsurface').any():
    raise SystemExit('Region name subsurface is retired. Use awi for the 3–12 Å band.')
d=d[d.region.isin(['awi','interior'])]
d['exchange']=(d.hb_break_early>0).astype(int) # loss of at least one initial HB; not necessarily replacement
# hb_partner_exchange_early (loss AND gain) is a different, stricter outcome; optionally replace exposure below.
d['reorient']=(d.ref_angle_late_rad>np.deg2rad(a.angle_deg)).astype(int)
# Quantile bins calculated INSIDE each replica/temperature/region, which are exact match variables.
blocks=['replica','rate','temp_mid_K','region']
def assign_bins(x):
    # Ranking avoids duplicate qcut edges in highly discrete mobility.
    x=x.copy();rank=x.ref_move_early_A.rank(pct=True,method='average')
    x['mobility_bin']=np.minimum(a.mobility_bins-1,np.floor(rank*a.mobility_bins).astype(int))
    return x
# Explicit group loop avoids pandas version-specific groupby apply behavior.
parts=[]
for _,x in d.groupby(blocks,sort=False):
    x=x.copy();rank=x.ref_move_early_A.rank(pct=True,method='average')
    x['mobility_bin']=np.minimum(a.mobility_bins-1,np.floor(rank*a.mobility_bins).astype(int))
    parts.append(x)
d=pd.concat(parts,ignore_index=True)
strata=[]
for k,z in d.groupby(blocks+['mobility_bin']):
    one=z[z.exchange==1];zero=z[z.exchange==0]
    npairs=min(len(one),len(zero))
    if npairs<a.min_pairs:continue
    strata.append(dict(zip(blocks+['mobility_bin'],k),n_exposed=len(one),n_control=len(zero),matched_pairs=npairs,
        p_after_break=one.reorient.mean(),p_without_break=zero.reorient.mean(),
        risk_difference=one.reorient.mean()-zero.reorient.mean(),
        prior_mobility_exposed=one.ref_move_early_A.mean(),prior_mobility_control=zero.ref_move_early_A.mean()))
s=pd.DataFrame(strata)
if s.empty:raise SystemExit('No strata with sufficient exposed/control records; lower --min-pairs or check HB alignment.')
s.to_csv(a.out/'matched_exchange_strata.csv',index=False)
# Weight each stratum by number of matchable pairs; samples are not treated as independent replicate estimates.
def weighted(z,col):return np.average(z[col],weights=z.matched_pairs)
rep=[]
for k,z in s.groupby(blocks):
    rep.append(dict(zip(blocks,k),n_strata=len(z),matched_pairs=int(z.matched_pairs.sum()),
        p_after_break=weighted(z,'p_after_break'),p_without_break=weighted(z,'p_without_break'),
        matched_risk_difference=weighted(z,'risk_difference'),
        mobility_balance_A=weighted(z,'prior_mobility_exposed')-weighted(z,'prior_mobility_control')))
r=pd.DataFrame(rep);r.to_csv(a.out/'matched_exchange_replica.csv',index=False)
summary=r.groupby(['rate','temp_mid_K','region']).matched_risk_difference.agg(['mean','std','count']).reset_index()
summary.to_csv(a.out/'matched_exchange_summary.csv',index=False)
print('Wrote',a.out,'; exposure=early loss of >=1 initial HB; no causal interpretation.')
