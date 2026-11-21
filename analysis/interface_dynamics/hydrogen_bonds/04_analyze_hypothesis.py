#!/usr/bin/env python3
"""Donor/acceptor states, free OH groups, and near-neighbor geometry.

One water is the structural unit. Independent replica trajectories are the
uncertainty unit. Hydrogen bonds use an oxygen-oxygen cutoff of 3.5 angstroms
and an O-H···O angle above 150 degrees unless those defaults are changed.
The acceptor-side vector is the negative OH bisector. It is a geometric
proxy, not an explicit lone pair.
"""
import argparse, json, sys
from pathlib import Path
import numpy as np, pandas as pd
import MDAnalysis as mda
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from common.paths import ensure_analysis_on_path

ensure_analysis_on_path(__file__)
from common.slab import (
    assign_regions,
    hydrogen_bond_counts as hb_characterize,
    minimum_image as mic,
    slab_geometry,
    water_indices,
)

CLASSES=('D2A2','D2A1','D1A2','D1A1','under_other','over','other_four')

def class_label(D,A):
    """Donor-acceptor class of each water.

    Parameters
    ----------
    D : 
        Number of hydrogen bonds donated by each sampled water.
    A : 
        Activation parameter or amplitude, in the units of the calling fit.

    Returns
    -------
    result
        Donor-acceptor class of each water.
    """
    ans=np.full(len(D),'other_four',dtype=object)
    n=D+A
    ans[n<4]='under_other';ans[n>4]='over'
    for d,a in ((2,2),(2,1),(1,2),(1,1)):
        ans[(D==d)&(A==a)]=f'D{d}A{a}'
    return ans

def diagnostics(O,H,L,refs,D,A,occupied,cutoff,angle):
    """Coordinates and geometric near-neighbor diagnostics for sampled waters.
    No attempt to assign a unique oxygen lone-pair bonding site.

    Parameters
    ----------
    O : 
        Oxygen coordinates.
    H : 
        Hydrogen coordinates with shape (n_water, 2, 3).
    L : 
        Orthorhombic box lengths.
    refs : 
        Indices of the waters being scored.
    D : 
        Number of hydrogen bonds donated by each sampled water.
    A : 
        Activation parameter or amplitude, in the units of the calling fit.
    occupied : 
        Boolean array marking hydrogens that donate a bond.
    cutoff : 
        Oxygen-oxygen cutoff in angstroms.
    angle : 
        Minimum O-H...O angle of a hydrogen bond, in degrees.

    Returns
    -------
    result
        Coordinates and geometric near-neighbor diagnostics for sampled waters.
    """
    n=len(refs);coords=O%L;tree=cKDTree(coords,boxsize=L)
    neigh=tree.query_ball_point(coords[refs],cutoff)
    near=np.zeros(n,dtype=int)
    # Quantify each free OH if it sees any close oxygen but misses the angle.
    # A candidate donor angle is measured even if not above HB threshold.
    missing_near=np.zeros((n,2),bool)
    best_donor_angle=np.full((n,2),np.nan)
    # Reference acceptor: count nearby waters with at least one OH
    # geometrically pointing toward it, versus neighbors with angular failure.
    incoming_candidate=np.zeros(n,int)
    incoming_failed=np.zeros(n,int)
    oh=mic(H-O[:,None,:],L)
    for k,(ref,ns) in enumerate(zip(refs,neigh)):
        ns=np.asarray([j for j in ns if j!=ref],int)
        if not len(ns):continue
        dr=mic(O[ns]-O[ref],L)
        dist=np.linalg.norm(dr,axis=1)
        keep=(dist>0)&(dist<cutoff);ns=ns[keep];dr=dr[keep]
        near[k]=len(ns)
        if not len(ns):continue
        for h in (0,1):
            # O_ref-H_ref ... O_ns angle: between H->O_ref and H->O_ns.
            v=oh[ref,h]; hv=dr-v
            c=np.clip(np.einsum('ij,j->i',hv,-v)/(np.linalg.norm(hv,axis=1)*max(np.linalg.norm(v),1e-12)),-1,1)
            angles=np.degrees(np.arccos(c))
            best_donor_angle[k,h]=float(np.max(angles))
            if not occupied[k,h]:missing_near[k,h]=True
        # Incoming HB candidate: for each neighboring donor, check both H.
        # At a close oxygen distance, failure of angle is quantified, not called
        # a specific "missing lone pair" because that is undefined by OPC topology.
        v=oh[ns]; dest=-dr[:,None,:]-v
        num=np.sum(-v*dest,axis=2)
        den=np.linalg.norm(v,axis=2)*np.linalg.norm(dest,axis=2)
        ang=np.degrees(np.arccos(np.clip(num/np.maximum(den,1e-15),-1,1)))
        good=np.any(ang>angle,axis=1)
        incoming_candidate[k]=int(good.sum());incoming_failed[k]=int((~good).sum())
    return dict(n_near=near,free_oh_with_close_acceptor=missing_near,
                donor_best_angle=best_donor_angle, incoming_geometric=incoming_candidate,
                incoming_angular_fail=incoming_failed)

def frame_metrics(O,H,L,refs,zr,depth,cutoff,angle,outward_cos,batch):
    """Coordination, free OH, and neighbor-geometry fractions for one frame.

    Parameters
    ----------
    O : 
        Oxygen coordinates.
    H : 
        Hydrogen coordinates with shape (n_water, 2, 3).
    L : 
        Orthorhombic box lengths.
    refs : 
        Indices of the waters being scored.
    zr : 
        Slab-centered z coordinate of every water.
    depth : 
        Inward depth from the half-density surface, in angstroms.
    cutoff : 
        Oxygen-oxygen cutoff in angstroms.
    angle : 
        Minimum O-H...O angle of a hydrogen bond, in degrees.
    outward_cos : 
        Minimum cosine with the outward surface normal.
    batch : 
        Number of reference waters per neighbor-list query.

    Returns
    -------
    result
        Coordination, free OH, and neighbor-geometry fractions for one frame.
    """
    D,A,occupied=hb_characterize(O,H,L,refs,cutoff,angle,batch)
    diag=diagnostics(O,H,L,refs,D,A,occupied,cutoff,angle)
    near=diag['n_near']; n=D+A; defective=n!=4
    oh=mic(H[refs]-O[refs,None,:],L)
    uv=oh/np.maximum(np.linalg.norm(oh,axis=2,keepdims=True),1e-12)
    sign=np.where(zr[refs]>=0,1.,-1.)
    cos_oh=uv[:,:,2]*sign[:,None]
    free=~occupied
    outfree=free&(cos_oh>=outward_cos)
    # -bisector points toward acceptor side, not an explicit lone-pair vector.
    bis=np.sum(uv,axis=1)
    acceptor_proxy=-bis/np.maximum(np.linalg.norm(bis,axis=1,keepdims=True),1e-12)
    acceptor_cos=acceptor_proxy[:,2]*sign
    exposed_acceptor=acceptor_cos>=outward_cos
    label=class_label(D,A)
    free_count=np.sum(free,axis=1)
    with_outfree=np.any(outfree,axis=1)
    with_free=np.any(free,axis=1)
    rows={
      'n_water':len(refs),'p_defect':np.mean(defective),'p_under':np.mean(n<4),'p_over':np.mean(n>4),
      'mean_donor':np.mean(D),'mean_acceptor':np.mean(A),'mean_OO_contacts':np.mean(near),
      'p_donor_missing':np.mean(D<2),'p_acceptor_missing':np.mean(A<2),
      'p_acceptor_deficient_donor_saturated':np.mean((D>=2)&(A<2)),
      'p_donor_deficient_acceptor_saturated':np.mean((D<2)&(A>=2)),
      'p_defect_outOH_joint':np.mean(defective&with_outfree),
      'p_defect_any_freeOH_joint':np.mean(defective&with_free),
      'p_outOH_given_defect':np.mean(with_outfree[defective]) if defective.any() else np.nan,
      'p_anyfree_given_defect':np.mean(with_free[defective]) if defective.any() else np.nan,
      'p_outOH_water':np.mean(with_outfree),'p_freeOH_per_OH':np.mean(free),
      'p_acceptor_proxy_outward':np.mean(exposed_acceptor),
      'p_acceptor_proxy_outward_given_acceptor_deficit':np.mean(exposed_acceptor[A<2]) if np.any(A<2) else np.nan,
      'p_acceptor_proxy_outward_given_acceptor_normal':np.mean(exposed_acceptor[A>=2]) if np.any(A>=2) else np.nan,
      'mean_acceptor_proxy_cos_deficient':np.mean(acceptor_cos[A<2]) if np.any(A<2) else np.nan,
      'mean_acceptor_proxy_cos_normal':np.mean(acceptor_cos[A>=2]) if np.any(A>=2) else np.nan,
      # This tests geometric opportunity, NOT a causal decomposition of missing bonds.
      'p_under_with_four_nearby':np.mean((n<4)&(near>=4)),
      'p_under_with_lt_four_nearby':np.mean((n<4)&(near<4)),
      'p_acceptor_deficit_with_four_nearby':np.mean((A<2)&(near>=4)),
      'p_donor_deficit_with_four_nearby':np.mean((D<2)&(near>=4)),
      'p_freeOH_close_O_nearby_joint_per_OH':np.mean(free&diag['free_oh_with_close_acceptor']),
      'p_freeOH_no_close_O_per_OH':np.mean(free&~diag['free_oh_with_close_acceptor']),
      'p_no_geometry_incoming_per_water':np.mean(diag['incoming_geometric']==0),
      'mean_incoming_angular_fail_OO':np.mean(diag['incoming_angular_fail']),
    }
    for lab in CLASSES:rows['p_'+lab]=float(np.mean(label==lab))
    # exclusive undercoordination decomposition as a conditional denominator
    for lab in ('D2A1','D1A2','D1A1','under_other'):
        rows['p_'+lab+'_given_under']=float(np.mean(label[n<4]==lab)) if np.any(n<4) else np.nan
    byclass=[]
    for lab in CLASSES:
        mask=label==lab
        byclass.append({'class':lab,'count':int(mask.sum()),'p':float(np.mean(mask)),
            'p_outOH_given_class':float(np.mean(with_outfree[mask])) if np.any(mask) else np.nan,
            'p_accproxy_out_given_class':float(np.mean(exposed_acceptor[mask])) if np.any(mask) else np.nan})
    # orientation histogram of -bisector grouped by acceptor status
    edges=np.linspace(-1,1,21);orient=[]
    for grp,mask in [('acceptor_deficient',A<2),('acceptor_normal',A>=2)]:
        counts=np.histogram(acceptor_cos[mask],bins=edges)[0]
        for j,v in enumerate(counts):orient.append({'group':grp,'cos_mid':float((edges[j]+edges[j+1])/2),'count':int(v),
          'prob':float(v/max(int(mask.sum()),1))})
    return rows,byclass,orient

def run(args):
    """Analyze one trajectory over the requested interface bands.

    Parameters
    ----------
    args : 
        Parsed command-line arguments.

    Returns
    -------
    result
        Analyze one trajectory over the requested interface bands.
    """
    u=mda.Universe(args.top,args.traj,format="NCDF");oi,hi=water_indices(u)
    rng=np.random.default_rng(args.seed);nframe=len(u.trajectory)
    frameids=np.unique(np.linspace(0,nframe-1,min(args.frames,nframe),dtype=int))
    rows=[];cls=[];orient=[];qcs=[]
    for frame in frameids:
        ts=u.trajectory[int(frame)];L=np.array(ts.dimensions[:3],float)
        if not np.allclose(ts.dimensions[3:],90,atol=.05):raise ValueError('Expected orthorhombic simulation box')
        O=np.array(ts.positions[oi],float)%L;H=np.array(ts.positions[hi],float)
        zr,depth,bounds,center=slab_geometry(O,L,args.dz,args.plateau_halfwidth)
        for inner,outer in args.bands:
            selected,qc=assign_regions(zr,depth,args.core_halfwidth,inner,outer,args.overlap_policy)
            for reg,eligible in selected.items():
                if len(eligible)<args.min_region_waters:
                    raise ValueError(f'Frame {frame}, {inner}-{outer} A, {reg}: {len(eligible)} waters')
                # Important: matched sampled refs per frame across depth bands are NOT
                # required for region comparison; independent replicate is uncertainty unit.
                refs=np.sort(rng.choice(eligible,size=min(args.nref,len(eligible)),replace=False))
                metrics,classes,angs=frame_metrics(O,H,L,refs,zr,depth,args.distance,args.angle,args.outward_cos,args.batch)
                basic={'frame':int(frame),'region':reg,'awi_inner_A':inner,'awi_outer_A':outer}
                rows.append({**basic,**metrics})
                cls.extend({**basic,**z} for z in classes)
                orient.extend({**basic,**z} for z in angs)
            qcs.append({'frame':int(frame),'awi_inner_A':inner,'awi_outer_A':outer,'half_density_lower_A':bounds[0],
                 'half_density_upper_A':bounds[1],**qc})
        print(f'frame={frame} of {nframe-1}',flush=True)
    out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    pd.DataFrame(rows).to_csv(out/'hypotheses_frame.csv',index=False)
    pd.DataFrame(cls).to_csv(out/'hypotheses_classes.csv',index=False)
    pd.DataFrame(orient).to_csv(out/'hypotheses_acceptor_orientation.csv',index=False)
    pd.DataFrame(qcs).to_csv(out/'hypotheses_qc.csv',index=False)
    md={k:(str(v) if isinstance(v,Path) else v) for k,v in vars(args).items()}
    md['bands']=[list(z) for z in args.bands]
    md['notes']=['Near-contact vs angle is an opportunity diagnostic; one water can have a close neighbor without a geometrically valid H-bond.',
      'Acceptor side is -OH bisector and NOT an explicitly simulated lone pair.',
      'p_defect_outOH_joint is normalized by all water molecules; p_outOH_given_defect uses defective-water denominator.',
      'Neither donor nor acceptor bond deficiency proves a missing unique electron-pair site.']
    (out/'hypotheses_metadata.json').write_text(json.dumps(md,indent=2))

def main():
    """Tell the caller that this analysis is written by the free-OH script.

    Returns
    -------
    None
        This entry point does not analyze a trajectory.
    """
    raise SystemExit(
        "Donor-acceptor and free-OH tables are written together by "
        "03_analyze_free_oh.py, in awi_0-6/ and awi_3-12/."
    )


if __name__=='__main__':
    main()
