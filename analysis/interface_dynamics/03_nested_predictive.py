#!/usr/bin/env python3
"""Four nested logistic models; LO-replica-out prediction. No molecular iid significance claims."""
import argparse
import sys
from pathlib import Path
import numpy as np,pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss,brier_score_loss

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.paths import ensure_analysis_on_path

ensure_analysis_on_path(__file__)
from common.rates import canonicalize_rate, stored_rate_label
ap=argparse.ArgumentParser();ap.add_argument('--events',type=Path,required=True);ap.add_argument('--out',type=Path,required=True)
ap.add_argument('--rate', type=float, default=4.44, help='Physical cooling rate in K/ns.');ap.add_argument('--angle-deg',type=float,default=30)
ap.add_argument('--min-test-events',type=int,default=10)
a=ap.parse_args();a.rate=canonicalize_rate(a.rate);a.out.parent.mkdir(parents=True,exist_ok=True)
d=pd.read_csv(a.events)
base=['temp_mid_K','ref_move_early_A','ref_angle_early_rad','coord','depth_A']
neighbor=['shell_move_early_A'];hb=['hb_break_early','hb_initial']
need=set(['replica','rate','region','ref_angle_late_rad','origin','ref']+base+neighbor+hb)
if not need.issubset(d):raise SystemExit(f'Missing {sorted(need-set(d))}. Existing joins may omit HB fields; verify --v2-events and rerun HB on SAME refs/time origins.')
key=['replica','rate','temp_mid_K','region','origin','ref']
if d.duplicated(key).any():raise SystemExit('Nonunique origin/ref rows: inspect V2↔V3 join before modelling.')
d.rate=d.rate.map(stored_rate_label);d=d[d.rate==a.rate].copy()
features=base+neighbor+hb+['ref_angle_late_rad']
for c in features:d[c]=pd.to_numeric(d[c],errors='coerce')
# Use exactly the same COMPLETE rows for all four models to avoid comparisons induced by missingness.
before=len(d);d=d.replace([np.inf,-np.inf],np.nan).dropna(subset=features)
print(f'Complete cases {len(d):,}/{before:,} ({len(d)/max(before,1):.1%})')
d['Y']=(d.ref_angle_late_rad>np.deg2rad(a.angle_deg)).astype(int)
models={'baseline':base,'neighbor_translation':base+neighbor,'HB_exchange':base+hb,'both':base+neighbor+hb}
rows=[]
for (rate,region),grp in d.groupby(['rate','region']):
    reps=sorted(grp.replica.unique())
    if len(reps)<3:print('SKIP',rate,region,'only',len(reps),'replicas');continue
    for held in reps:
        train=grp[grp.replica!=held];test=grp[grp.replica==held]
        if len(train)<100 or len(test)<30 or train.Y.nunique()!=2 or test.Y.sum()<a.min_test_events:
            print('SKIP held-out',held,region,'insufficient samples/classes');continue
        for name,fields in models.items():
            mod=make_pipeline(SimpleImputer(strategy='median'),StandardScaler(),LogisticRegression(C=1,max_iter=1000))
            mod.fit(train[fields],train.Y);p=np.clip(mod.predict_proba(test[fields])[:,1],1e-7,1-1e-7)
            rows.append(dict(rate=rate,region=region,heldout_replica=held,model=name,n_train=len(train),n_test=len(test),n_events=int(test.Y.sum()),prevalence=float(test.Y.mean()),
                logloss=log_loss(test.Y,p,labels=[0,1]),brier=brier_score_loss(test.Y,p)))
res=pd.DataFrame(rows)
if res.empty:raise SystemExit('No held-out results. Verify joined event data and minimum events.')
res.to_csv(a.out,index=False)
idx=['rate','region','heldout_replica']
w=res.pivot(index=idx,columns='model',values=['logloss','brier'])
delta=[]
for k,row in w.iterrows():
    z=dict(zip(idx,k))
    for metric in ['logloss','brier']:
        a0=row[(metric,'baseline')];an=row[(metric,'neighbor_translation')];ah=row[(metric,'HB_exchange')];ab=row[(metric,'both')]
        z[metric+'_gain_neighbor']=a0-an;z[metric+'_gain_HB']=a0-ah;z[metric+'_gain_both']=a0-ab
        z[metric+'_increment_HB_given_neighbor']=an-ab
        z[metric+'_increment_neighbor_given_HB']=ah-ab
        z[metric+'_interaction']=an+ah-a0-ab
    delta.append(z)
pd.DataFrame(delta).to_csv(a.out.with_name(a.out.stem+'_contrasts.csv'),index=False)
print('Wrote',a.out,'and paired contrasts; gains are descriptive, not statistical causal effects.')
