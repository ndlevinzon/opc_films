#!/usr/bin/env python3
"""Leave-one-replica-out conditional prediction; needs aligned_event_samples.csv.
Not causal proof. Grouped held-out evaluation only, no pooled p-values.
"""
import argparse
from pathlib import Path
import numpy as np,pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss,brier_score_loss
p=argparse.ArgumentParser();p.add_argument('--events',required=True);p.add_argument('--out',required=True);p.add_argument('--angle-deg',type=float,default=30);p.add_argument('--min-test-events',type=int,default=10);a=p.parse_args()
d=pd.read_csv(a.events)
req=['replica','rate','region','temp_mid_K','ref_move_early_A','ref_angle_early_rad','coord','depth_A','shell_move_early_A','ref_angle_late_rad']
missing=set(req)-set(d)
if missing:raise SystemExit(f'Missing {sorted(missing)}')
d['Y']=(d.ref_angle_late_rad>np.deg2rad(a.angle_deg)).astype(int)
base=['temp_mid_K','ref_move_early_A','ref_angle_early_rad','coord','depth_A'];expanded=base+['shell_move_early_A']
if 'hb_break_early' in d.columns:
    base+=['hb_initial'];expanded=base+['shell_move_early_A','hb_break_early']
results=[]
for (rate,region),grp in d.groupby(['rate','region']):
    reps=grp.replica.unique()
    if len(reps)<3:continue
    for held in reps:
        train=grp[grp.replica!=held];test=grp[grp.replica==held]
        if len(train)<100 or len(test)<30 or train.Y.nunique()<2 or test.Y.sum()<a.min_test_events:continue
        scores=[]
        for fields in (base,expanded):
            mod=make_pipeline(SimpleImputer(strategy='median',add_indicator=False),StandardScaler(),LogisticRegression(C=1,max_iter=500,class_weight=None))
            mod.fit(train[fields],train.Y);prob=np.clip(mod.predict_proba(test[fields])[:,1],1e-6,1-1e-6)
            scores.append((log_loss(test.Y,prob,labels=[0,1]),brier_score_loss(test.Y,prob)))
        results.append(dict(rate=rate,region=region,heldout_replica=held,n_train=len(train),n_test=len(test),n_events=int(test.Y.sum()),baseline_logloss=scores[0][0],expanded_logloss=scores[1][0],gain_logloss=scores[0][0]-scores[1][0],baseline_brier=scores[0][1],expanded_brier=scores[1][1],gain_brier=scores[0][1]-scores[1][1]))
out=Path(a.out);out.parent.mkdir(parents=True,exist_ok=True);pd.DataFrame(results).to_csv(out,index=False)
print('Held-out results',len(results),'rows ->',out)
