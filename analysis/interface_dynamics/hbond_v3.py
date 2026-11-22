#!/usr/bin/env python3
"""Sparse-frame HB analysis for the original trajectory; interfacial origin populations.
Intermittent partner persistence, not continuous survival. Full coordinate records required.
"""
import argparse, json, sys
from pathlib import Path
import numpy as np, pandas as pd, MDAnalysis as mda
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.paths import ensure_analysis_on_path

ensure_analysis_on_path(__file__)
from common.constants import (
    HBOND_ANGLE_DEG, HBOND_COMPARISON_LAG_PS, HBOND_LAGS_PS, HBOND_OO_CUTOFF_A,
    HOLD_DISCARD_FRACTION, REGION_BIN_A, SEEDS,
)
from opcfilm.config import CONFIG
from common.slab import directed_hydrogen_bonds
from common.trajectory import cooling_window_lag_cap_ps, continuation_frame_spacing_ps, discard_frame_count, lag_pair_ok
from analyze_v2 import waters, frame_arrays, gds_depth, sample_refs

def hb_pairs(O, H, L, refs, dist_cut=HBOND_OO_CUTOFF_A, angle_cut=HBOND_ANGLE_DEG):
    """Directed hydrogen bonds from the shared slab geometry.

    Parameters
    ----------
    O, H, L, refs
        Oxygen coordinates, hydrogen coordinates, box lengths, and donor indices.
    dist_cut : float, optional
        Oxygen-oxygen cutoff in angstroms.
    angle_cut : float, optional
        Minimum O-H···O angle in degrees.

    Returns
    -------
    set of tuple of int
        ``(donor, acceptor)`` pairs.
    """
    return directed_hydrogen_bonds(O, H, L, refs, dist_cut, angle_cut)

def get_frame(u,frame,oi,hi):
    ts=u.trajectory[int(frame)];O,M,L=frame_arrays(ts,oi,hi)
    H=np.asarray(ts.positions[hi],float)
    return O,H,L

def main():
    p=argparse.ArgumentParser();p.add_argument('--top',required=True);p.add_argument('--traj',action='append',required=True);p.add_argument('--out',required=True)
    p.add_argument('--origins',type=int,default=CONFIG.hbond.origins);p.add_argument('--nref-region',type=int,default=CONFIG.dynamics.references_per_region)
    p.add_argument('--lags-ps', default=",".join(f"{lag:g}" for lag in CONFIG.hbond.lags_ps), help=f'Retention is compared at {CONFIG.hbond.comparison_lag_ps:g} ps. The list ends at 640 ps and is not extended in hold mode.')
    p.add_argument('--event-window-ps',type=float,default=CONFIG.dynamics.event_window_ps)
    p.add_argument('--hold',action='store_true',help='Isothermal hold. Do not cap the lags.')
    p.add_argument('--window-temperature-drop-K',type=float,default=10)
    p.add_argument('--max-lag-temperature-drop-K',type=float,default=CONFIG.dynamics.cooling_lag_cap_K,help='Cooling-window lag cap. Ignored with --hold.')
    p.add_argument('--v2-events',default=None,help='Use exact V2 origin and reference water identities for event-model join');p.add_argument('--distance',type=float,default=HBOND_OO_CUTOFF_A);p.add_argument('--angle',type=float,default=HBOND_ANGLE_DEG);p.add_argument('--seed',type=int,default=SEEDS['hydrogen_bond_persistence'])
    args=p.parse_args();u=mda.Universe(args.top,args.traj if len(args.traj)>1 else args.traj[0],format="NCDF");oi,hi=waters(u)
    frame_times=np.array([float(ts.time) for ts in u.trajectory])
    dt,njoin=continuation_frame_spacing_ps(frame_times)
    nframe=len(u.trajectory)
    duration_ps=(nframe-1)*dt
    max_lag_ps=duration_ps if args.hold else cooling_window_lag_cap_ps(
        duration_ps, args.max_lag_temperature_drop_K, args.window_temperature_drop_K)
    lags=sorted(set(round(float(s)/dt) for s in args.lags_ps.split(',')))
    lags=[int(v) for v in lags if 1<=v and v*dt<=max_lag_ps]
    if not lags:raise ValueError('No lag fits the trajectory. Use --hold for an isothermal trajectory.')
    evdelta=max(1,round(args.event_window_ps/dt));maxlag=max(lags+[2*evdelta]);maxstart=len(u.trajectory)-1-maxlag
    if maxstart<0:raise ValueError('Trajectory too short')
    first=discard_frame_count(nframe, "hold" if args.hold else "ramp", HOLD_DISCARD_FRACTION)
    if first>maxstart:raise ValueError('The equilibration discard leaves no origin.')
    origins=np.unique(np.linspace(first,maxstart,min(args.origins,maxstart-first+1),dtype=int))
    rng=np.random.default_rng(args.seed);curves=[];dist=[];events=[];qc=[]
    if args.v2_events:
        prior=pd.read_csv(args.v2_events)
        if not {"origin","ref","region"}.issubset(prior):raise ValueError("Invalid V2 event data")
        origins=np.sort(prior.origin.astype(int).unique())
        if int(origins.max())+maxlag>=len(u.trajectory):raise ValueError("V2 origins too late for HB lags; shorten lags")
    for origin in origins:
        O0,H0,L=get_frame(u,origin,oi,hi)
        depth,_,_,z_relative=gds_depth(O0,L)
        if args.v2_events:
            ps=prior[prior.origin==origin].drop_duplicates("ref")
            refs=ps.ref.to_numpy(dtype=int)
            idmap={int(ref):str(region) for ref,region in zip(refs,ps.region)}
        else:
            refs,names=sample_refs(z_relative,depth,args.nref_region,rng)
            idmap={int(ref):str(name) for ref,name in zip(refs,names)}
        if len(refs)==0:continue
        # For each ref, count both donors and acceptors by searching *all* waters near each ref;
        # donor refs and their neighbors suffice to evaluate all directed contacts involving refs.
        tree=cKDTree(O0,boxsize=L);neighbors=tree.query_ball_point(O0[refs],args.distance)
        surround=np.unique(np.concatenate([refs]+[np.asarray(x,dtype=int) for x in neighbors]))
        initial=hb_pairs(O0,H0,L,surround,args.distance,args.angle)
        R=set(map(int,refs));initial={pair for pair in initial if pair[0] in R or pair[1] in R}
        def assigned_pairs(bonds,ref):return {b for b in bonds if ref in b}
        for ref in refs:
            ref=int(ref);b=assigned_pairs(initial,ref)
            dist.append(dict(origin=int(origin),ref=ref,region=idmap[ref],n_hb=len(b),n_donor=sum(x[0]==ref for x in b),n_accept=sum(x[1]==ref for x in b)))
        for lag in lags:
            # A pair across a segment join is used only when its true separation matches the lag.
            if not lag_pair_ok(frame_times,origin,lag,dt):continue
            O1,H1,L1=get_frame(u,origin+lag,oi,hi)
            if np.max(abs(L-L1))>.02:raise ValueError('Box varies')
            tree1=cKDTree(O1,boxsize=L1)
            nn1=tree1.query_ball_point(O1[refs],args.distance)
            subjects=np.unique(np.concatenate([refs]+[np.asarray(x,dtype=int) for x in nn1]))
            b1=hb_pairs(O1,H1,L1,subjects,args.distance,args.angle)
            for ref in refs:
                ref=int(ref);old=assigned_pairs(initial,ref)
                if not old:continue
                present=len(old & b1)
                curves.append(dict(origin=int(origin),ref=ref,region=idmap[ref],lag_ps=lag*dt,n_initial=len(old),n_retained=present))
        if origin+2*evdelta<len(u.trajectory) and lag_pair_ok(frame_times,origin,evdelta,dt):
            Om,Hm,Lm=get_frame(u,origin+evdelta,oi,hi)
            treem=cKDTree(Om,boxsize=Lm)
            nnm=treem.query_ball_point(Om[refs],args.distance)
            subjects_m=np.unique(np.concatenate([refs]+[np.asarray(x,dtype=int) for x in nnm]))
            bm=hb_pairs(Om,Hm,Lm,subjects_m,args.distance,args.angle)
            for ref in refs:
                ref=int(ref);old=assigned_pairs(initial,ref);new=assigned_pairs(bm,ref)
                events.append(dict(origin=int(origin),ref=ref,region=idmap[ref],hb_initial=len(old),hb_break_early=len(old-new),hb_form_early=len(new-old),hb_partner_exchange_early=int(bool(old-new) and bool(new-old))))
        qc.append(dict(origin=int(origin),nrefs=len(refs),dt_ps=dt))
        print('Origin',origin,'processed',flush=True)
    out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    pd.DataFrame(curves).to_csv(out/'hb_intermit_pairs.csv',index=False)
    pd.DataFrame(dist).to_csv(out/'hb_coord_samples.csv',index=False)
    pd.DataFrame(events).to_csv(out/'hb_event_samples.csv',index=False)
    pd.DataFrame(qc).to_csv(out/'hb_qc.csv',index=False)
    (out/'hb_metadata.json').write_text(json.dumps(dict(traj=args.traj,dt_ps=dt,segment_joins=njoin,hold=bool(args.hold),max_lag_ps=max_lag_ps,origins=len(origins),distance_A=args.distance,angle_deg=args.angle,seed=args.seed,region_bin_A=REGION_BIN_A,notes='Intermittent original-partner persistence at lagged snapshots. Not continuous survival. Cooling-window lags use the same temperature cap as analyze_v2.py unless --hold is set. Retention is compared at 320 ps; the stored list ends at 640 ps and is not extended on a hold. Snapshot resolution limits exchange detection. Water IDs follow topology residue order. Region at initial origin. Hydrogen bonds use slab.directed_hydrogen_bonds.'),indent=2))
if __name__=='__main__':main()
