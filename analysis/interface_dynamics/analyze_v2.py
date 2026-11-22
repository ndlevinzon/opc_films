#!/usr/bin/env python3
"""OPC slab V2: origin-defined region relaxation and mobility/rotation event sequence.

Observational, non-causal. All observations use initial-region labels and initial-shell
membership. End-point minimum-image displacement is used; see README caveats.
"""
import argparse, json, sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from scipy.special import j0
import MDAnalysis as mda

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.paths import ensure_analysis_on_path

ensure_analysis_on_path(__file__)
from common.constants import HOLD_DISCARD_FRACTION, REGION_BIN_A, SEEDS, load_adopted_qstar
from opcfilm.config import CONFIG
from common.slab import region_masks, slab_geometry, water_indices
from common.trajectory import cooling_window_lag_cap_ps, continuation_frame_spacing_ps, discard_frame_count, hold_lag_times_ps, lag_pair_ok

REGIONS=('awi','awi_outer','interior')

def waters(u):
    """Oxygen and hydrogen indices from ``common.slab.water_indices``.

    Parameters
    ----------
    u : MDAnalysis.Universe
        Topology containing WAT, HOH, or OPC residues.

    Returns
    -------
    oxygen, hydrogen : tuple of numpy.ndarray
        One oxygen and two hydrogens per water.
    """
    return water_indices(u)

def wrap_diff(x,L): return x-L*np.rint(x/L)

def frame_arrays(ts, oi, hi):
    box=ts.dimensions.astype(float)
    if not np.allclose(box[3:],90,atol=.01):raise RuntimeError('Requires orthorhombic box')
    L=box[:3]; coords=ts.positions
    O=np.asarray(coords[oi],dtype=np.float64)%L
    H=np.asarray(coords[hi],dtype=np.float64)
    v=wrap_diff(H[:,0]-O,L)+wrap_diff(H[:,1]-O,L)
    norm=np.linalg.norm(v,axis=1)
    return O,v/np.maximum(norm[:,None],1.e-12),L

def gds_depth(O, L):
    """Depth from the shared half-density surface.

    Parameters
    ----------
    O : numpy.ndarray
        Oxygen coordinates.
    L : array_like
        Orthorhombic box lengths.

    Returns
    -------
    depth : numpy.ndarray
        Inward depth in angstroms.
    gds : numpy.ndarray
        Lower and upper half-density surfaces.
    center : float
        Slab center.
    z_relative : numpy.ndarray
        Coordinate from the slab center.
    """
    z_relative, depth, bounds, center = slab_geometry(O, L)
    return depth, np.asarray(bounds), center, z_relative

def shell_indices(O,refs,L,cutoff):
    tree=cKDTree(O,boxsize=L)
    neighborhoods=tree.query_ball_point(O[refs],cutoff)
    out=[]
    for ref, ids in zip(refs,neighborhoods):
        ids=np.array([i for i in ids if i!=ref],dtype=np.int32)
        if len(ids):
            r=np.linalg.norm(wrap_diff(O[ids]-O[ref],L),axis=1)
            ids=ids[r<cutoff]
        out.append(ids)
    return out

def sample_refs(z_relative, depth, nref, rng):
    """Sample the bulk and both interface bands.

    Parameters
    ----------
    z_relative, depth : numpy.ndarray
        Slab coordinates from :func:`gds_depth`.
    nref : int
        Maximum waters drawn from each band.
    rng : numpy.random.Generator
        Sampler. A water that is in both interface bands can be drawn for each.

    Returns
    -------
    refs : numpy.ndarray of int
        Sampled water indices.
    names : numpy.ndarray
        Region name of each sample. ``awi`` is the 3–12 Å band.
    """
    masks = region_masks(z_relative, depth)
    refs = []
    names = []
    for name in REGIONS:
        available = np.flatnonzero(masks[name])
        if not len(available):
            continue
        chosen = rng.choice(available, size=min(len(available), nref), replace=False)
        refs.append(chosen)
        names.append(np.full(len(chosen), name, dtype=object))
    if not refs:
        return np.array([], dtype=int), np.array([], dtype=object)
    return np.concatenate(refs), np.concatenate(names)

def analyze(args):
    u=mda.Universe(args.top,args.traj,format="NCDF");oi,hi=waters(u)
    nframe=len(u.trajectory)
    frame_times=np.array([float(ts.time) for ts in u.trajectory])
    dt,njoin=continuation_frame_spacing_ps(frame_times)
    duration_ps=(nframe-1)*dt
    if args.hold and not args.lags_ps:
        lag_ps=hold_lag_times_ps(duration_ps, dt)
        max_lag_ps=max(lag_ps)
    else:
        lag_text = args.lags_ps or ",".join(f"{lag:g}" for lag in CONFIG.dynamics.cooling_lags_ps)
        lag_ps=[float(s) for s in lag_text.split(",") if s.strip()]
        max_lag_ps=duration_ps if args.hold else cooling_window_lag_cap_ps(
            duration_ps, args.max_lag_temperature_drop_K, args.window_temperature_drop_K)
    lags=sorted(set(int(round(value/dt)) for value in lag_ps if value/dt>=1 and value<=max_lag_ps+1e-6))
    if not lags:raise RuntimeError('No lag fits trajectory; adjust --lags-ps or the temperature-drop cap')
    event_delta=max(1,int(round(args.event_window_ps/dt)))
    max_lag=max(lags+[2*event_delta]);last_origin=nframe-max_lag-1
    if last_origin<0:raise RuntimeError('Trajectory too short')
    first_origin=discard_frame_count(nframe, "hold" if args.hold else "ramp", HOLD_DISCARD_FRACTION)
    if first_origin>last_origin:raise RuntimeError('The equilibration discard leaves no origin.')
    origins=np.unique(np.linspace(first_origin,last_origin,min(args.origins,last_origin-first_origin+1),dtype=int))
    rng=np.random.default_rng(args.seed)
    # Plan is sparse on-demand: only origins, relevant endpoints and event midpoints.
    # Per-origin independent time blocks still require convergence assessment.
    accum=[]; ev=[]; qc=[]
    for ix,t0 in enumerate(origins):
        ts=u.trajectory[int(t0)]; O0,M0,L=frame_arrays(ts,oi,hi)
        depth,gds,center,z_relative=gds_depth(O0,L)
        refs,groups=sample_refs(z_relative,depth,args.nref_per_region,rng)
        shell=shell_indices(O0,refs,L,args.shell_cutoff)
        # Restrict reference to waters with a nonempty first shell
        keep=np.array([len(x)>0 for x in shell]); refs=refs[keep];groups=groups[keep];shell=[s for s,k in zip(shell,keep) if k]
        if not len(refs):continue
        r0=O0[refs].copy();m0=M0[refs].copy()
        # cache initial shell weighted dipoles and coordination, for new hypotheses
        coord=np.array([len(x) for x in shell],float)
        # Regular relaxation curves
        for lag in lags:
            # A pair across a segment join is used only when its true separation matches the lag.
            if not lag_pair_ok(frame_times,t0,lag,dt):continue
            ts=u.trajectory[int(t0+lag)];O1,M1,L1=frame_arrays(ts,oi,hi)
            if np.max(abs(L1-L))>args.box_tolerance:raise RuntimeError('Box length varies; no constant-cell endpoint correction')
            dr=wrap_diff(O1[refs]-r0,L); drxy=np.linalg.norm(dr[:,:2],axis=1)
            dot=np.sum(m0*M1[refs],axis=1).clip(-1,1)
            fs=j0(args.q_anginv*drxy);c1=dot;c2=(3*dot**2-1)/2
            for name in REGIONS:
                sel=groups==name
                if not np.any(sel):continue
                accum.append(dict(origin=int(t0),time_ps=t0*dt,lag_frames=int(lag),lag_ps=lag*dt,region=name,n=int(sel.sum()),
                                  fs_sum=float(fs[sel].sum()),c1_sum=float(c1[sel].sum()),c2_sum=float(c2[sel].sum()),
                                  drxy2_sum=float((drxy[sel]**2).sum()),coord_sum=float(coord[sel].sum())))
        # Two consecutive, nonoverlapping event intervals, origin->mid->end
        if t0+2*event_delta<nframe and lag_pair_ok(frame_times,t0,event_delta,dt) and lag_pair_ok(frame_times,t0+event_delta,event_delta,dt):
            tm=u.trajectory[int(t0+event_delta)]; Om,Mm,Lm=frame_arrays(tm,oi,hi)
            te=u.trajectory[int(t0+2*event_delta)]; Oe,Me,Le=frame_arrays(te,oi,hi)
            if max(np.max(abs(Lm-L)),np.max(abs(Le-L)))>args.box_tolerance:raise RuntimeError('Box changes')
            shell_mob=np.array([np.mean(np.linalg.norm(wrap_diff(Om[s]-O0[s],L)[:,:2],axis=1)) for s in shell]);
            ref_mob1=np.linalg.norm(wrap_diff(Om[refs]-r0,L)[:,:2],axis=1)
            ref_mob2=np.linalg.norm(wrap_diff(Oe[refs]-Om[refs],L)[:,:2],axis=1)
            ref_ang1=np.arccos(np.clip(np.sum(m0*Mm[refs],axis=1),-1,1))
            ref_ang2=np.arccos(np.clip(np.sum(Mm[refs]*Me[refs],axis=1),-1,1))
            shell_mob2=np.array([np.mean(np.linalg.norm(wrap_diff(Oe[s]-Om[s],L)[:,:2],axis=1)) for s in shell]);
            for name in REGIONS:
                sel=groups==name
                if not np.any(sel):continue
                ev.extend(dict(origin=int(t0),region=name,ref=int(ref),depth_A=float(depth[ref]),coord=float(co),
                               shell_move_early_A=float(sm),shell_move_late_A=float(sm2),
                               ref_move_early_A=float(rm),ref_move_late_A=float(rm2),
                               ref_angle_early_rad=float(a1),ref_angle_late_rad=float(a2))
                          for ref,co,sm,sm2,rm,rm2,a1,a2 in zip(refs[sel],coord[sel],shell_mob[sel],shell_mob2[sel],ref_mob1[sel],ref_mob2[sel],ref_ang1[sel],ref_ang2[sel]))
        qc.append(dict(origin=int(t0),time_ps=float(t0*dt),gds_left_A=float(gds[0]),gds_right_A=float(gds[1]),nrefs=int(len(refs)),n_awi=int(np.sum(groups=='awi')),n_awi_outer=int(np.sum(groups=='awi_outer')),n_interior=int(np.sum(groups=='interior'))))
        if (ix+1)%10==0:print(f'Origins {ix+1}/{len(origins)}',flush=True)
    if not accum:raise RuntimeError('No valid results')
    out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    pd.DataFrame(accum).to_csv(out/'origin_relaxation.csv',index=False)
    pd.DataFrame(ev).to_csv(out/'event_samples.csv',index=False)
    pd.DataFrame(qc).to_csv(out/'origin_qc.csv',index=False)
    meta=dict(top=args.top,traj=args.traj,nwater=len(oi),nframes=nframe,dt_ps=dt,segment_joins=njoin,lag_ps=[float(l*dt) for l in lags],
              event_interval_ps=event_delta*dt,norigins=len(qc),nref_per_region=args.nref_per_region,
              seed=args.seed,q_anginv=args.q_anginv,shell_cutoff_A=args.shell_cutoff,
              hold=bool(args.hold),max_lag_ps=max_lag_ps,window_temperature_drop_K=args.window_temperature_drop_K,
              region_bin_A=REGION_BIN_A,
              hydrogen_bonds='not computed here; hbond_v3 uses slab.directed_hydrogen_bonds',
              assumptions=['orthorhombic constant box','planar GDS','initial-region and initial-shell definitions',
              'endpoint minimum image xy displacement; verify no > half-box crossings',
              'OH bisector orientational vector, not full electrostatic dipole',
              'cooling chunks may be nonstationary','overlapping origins are correlated',
              'cooling windows drop about 10 K and lags are capped unless --hold is set'])
    (out/'metadata.json').write_text(json.dumps(meta,indent=2))
    print(f'Wrote {out} with {len(accum)} origin/region/lag records and {len(ev)} event observations',flush=True)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest',type=Path,default=None);p.add_argument('--row',type=int,default=None)
    p.add_argument('--top',default=None);p.add_argument('--traj',default=None);p.add_argument('--out',default=None)
    p.add_argument('--origins',type=int,default=CONFIG.dynamics.origins);p.add_argument('--nref-per-region',type=int,default=CONFIG.dynamics.references_per_region)
    p.add_argument('--lags-ps',default=None,help='Cooling-window lags in ps. With --hold, the lags run to 25%% of the hold unless this is set.')
    p.add_argument('--event-window-ps',type=float,default=CONFIG.dynamics.event_window_ps)
    p.add_argument('--window-temperature-drop-K',type=float,default=10,help='Temperature change across one cooling window.')
    p.add_argument('--hold',action='store_true',help='Isothermal hold. Do not cap the lags.')
    p.add_argument('--max-lag-temperature-drop-K',type=float,default=CONFIG.dynamics.cooling_lag_cap_K,help='Cooling-window lag cap, as a temperature change. Ignored with --hold.')
    p.add_argument('--q-anginv',type=float,default=None,help='Must match qstar_adopted_Ainv in --qstar-json.')
    p.add_argument('--qstar-json',type=Path,default=Path('outputs/qstar.json'))
    p.add_argument('--shell-cutoff',type=float,default=3.3)
    p.add_argument('--seed',type=int,default=SEEDS['regional_dynamics']);p.add_argument('--box-tolerance',type=float,default=0.02)
    args=p.parse_args()
    if args.manifest is not None:
        from common.manifest import load_manifest, result_directory, select_row, trajectory_list
        row=select_row(load_manifest(args.manifest), args.row)
        args.top=row['topology']
        paths=trajectory_list(row)
        args.traj=[str(path) for path in paths]
        args.out=str(result_directory(Path(__file__).resolve().parents[2], 'regional_dyn', row))
        if row['kind'] in ('hold','endpoint'):
            args.hold=True
    elif not (args.top and args.traj and args.out):
        p.error('Pass --manifest and --row, or --top, --traj, and --out.')
    args.q_anginv=load_adopted_qstar(args.qstar_json, args.q_anginv)
    analyze(args)
if __name__=='__main__':main()
