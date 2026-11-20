#!/usr/bin/env python3
"""Density profile and sampled local order for one cooling-window trajectory.

Cooling windows are not stationary states. The core density is the mean of
the smoothed mass-density profile inside ``|z| <= core_halfwidth``. The
orientational order ``q``, the Chau-Hardwick translational order ``Sk``,
and the local structure index are the implementations in ``common.order``.
The Gaussian neighbor score is stored as ``radial_regularity``.
"""
import argparse
import json
import sys
from pathlib import Path

import MDAnalysis as mda
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.paths import ensure_analysis_on_path

ensure_analysis_on_path(__file__)
from common.constants import BULK_HALFWIDTH_A, PROFILE_BIN_A, PROFILE_SMOOTH_SIGMA_A, SEEDS
from common.constants import NUMBER_TO_MASS_DENSITY as NUMBER_TO_DENSITY
from common.order import (
    local_structure_index_batch,
    orientational_tetrahedral_q,
    radial_regularity,
    translational_tetrahedral_sk,
)
from common.slab import half_density_surfaces, region_masks, water_indices
from common.trajectory import frame_spacing_ps, require_film


def geometry(z, Lx, Ly, Lz, dz, core_halfwidth):
    """Mass-density profile of a recentered water slab.

    Parameters
    ----------
    z : array_like
        Oxygen ``z`` coordinates, in angstroms.
    Lx, Ly, Lz : float
        Orthorhombic box lengths, in angstroms.
    dz : float
        Requested bin width. The actual width divides ``Lz`` evenly.
    core_halfwidth : float
        Half-width of the bulk window about the slab center, in angstroms.

    Returns
    -------
    dict
        Slab-centered coordinates, the density histogram, the core density,
        half-density surfaces, 10-90 percent widths, and mass-conservation
        diagnostics.
    """
    z = np.asarray(z, dtype=float)

    # Locate the slab center using periodic coordinates.
    phase = 2.0 * np.pi * np.mod(z, Lz) / Lz
    resultant = np.mean(np.exp(1j * phase))

    if abs(resultant) < 0.05:
        raise RuntimeError(
            "Ambiguous periodic slab center; inspect z distribution."
        )

    center = (np.angle(resultant) % (2.0 * np.pi)) * Lz / (2.0 * np.pi)

    # Reconstruct a contiguous slab about its periodic center.
    zr = (z - center + 0.5 * Lz) % Lz - 0.5 * Lz

    # Use bins covering the periodic cell exactly.
    nbins = max(1, int(np.ceil(Lz / dz)))
    edges = np.linspace(-Lz / 2, Lz / 2, nbins + 1)
    binwidth = edges[1] - edges[0]

    counts, _ = np.histogram(zr, bins=edges)
    x = 0.5 * (edges[:-1] + edges[1:])

    rho = (
        counts * NUMBER_TO_DENSITY
        / (Lx * Ly * binwidth)
    )

    sm = gaussian_filter1d(
        rho,
        sigma=PROFILE_SMOOTH_SIGMA_A / binwidth,
    )

    surfaces = half_density_surfaces(
        x, sm, core_halfwidth, fractions=(0.1, 0.5, 0.9)
    )
    core_rho = surfaces["rho_bulk"]
    left = {fraction: surfaces["crossings"][fraction][0] for fraction in (0.1, 0.5, 0.9)}
    right = {fraction: surfaces["crossings"][fraction][1] for fraction in (0.1, 0.5, 0.9)}

    h_gds = right[0.5] - left[0.5]
    width_upper = right[0.1] - right[0.9]
    width_lower = left[0.9] - left[0.1]

    # Conserved water mass per unit lateral area.
    areal_mass = (
        len(z) * NUMBER_TO_DENSITY / (Lx * Ly)
    )

    # Independently reconstructed areal mass.
    integrated_mass = float(
        np.sum(rho) * binwidth
    )

    mass_error = abs(integrated_mass - areal_mass) / areal_mass

    if mass_error > 1e-8:
        raise RuntimeError(
            f"Density integral does not conserve mass: {mass_error:.3g}"
        )

    h_eq = areal_mass / core_rho

    # Diagnostic density associated with GDS thickness.
    rho_gds = (
        areal_mass / h_gds
        if np.isfinite(h_gds) and h_gds > 0
        else np.nan
    )

    return dict(
        z_relative=zr,
        x=x,
        rho=rho,
        rho_core=core_rho,
        gds_lower=left[0.5],
        gds_upper=right[0.5],
        h_gds=h_gds,
        width_lower_10_90=width_lower,
        width_upper_10_90=width_upper,
        h_eq=h_eq,
        rho_gds=rho_gds,
        areal_mass=areal_mass,
        mass_error=mass_error,
        z_center=center,
    )


def structure(positions, refs, Lx, Ly, lsi_cutoff=3.7):
    """Sampled orientational order, radial score, and local structure index.

    Neighbor searches are periodic in ``x`` and ``y`` only. The cell's
    ``z`` length is padded so the slab is not wrapped onto itself.

    Parameters
    ----------
    positions : numpy.ndarray, shape (n, 3)
        Oxygen coordinates.
    refs : array_like of int
        Waters to score.
    Lx, Ly : float
        Lateral box lengths, in angstroms.
    lsi_cutoff : float, optional
        Largest oxygen-oxygen distance included in the LSI, in angstroms.

    Returns
    -------
    q, sk, lsi, regularity, neighbor_distances : tuple of numpy.ndarray
        ``q``, Chau–Hardwick ``sk``, and the standard ``lsi`` come from
        ``common.order``. ``regularity`` is ``radial_regularity`` and is not
        ``Sk``. ``neighbor_distances`` excludes the molecule itself.
    """
    # Periodicity only in x/y. A padded z cell avoids wrapping the slab.
    # Query 17 nearest incl self; need > cutoff neighbors for accurate LSI.
    shifts=np.array([(a*Lx,b*Ly,0) for a in (-1,0,1) for b in (-1,0,1)],dtype=float)
    # Efficient: cKDTree periodic XY using large padded Z bounding interval
    pts=positions.copy(); pts[:,0]%=Lx; pts[:,1]%=Ly
    zmin=pts[:,2].min(); pts[:,2]-=zmin
    cell=np.array([Lx,Ly,max(pts[:,2].max()+10.0,10.0)])
    tree=cKDTree(pts,boxsize=cell)
    dist,idx=tree.query(pts[refs],k=20,workers=1)
    d4=dist[:,1:5]; idx4=idx[:,1:5]
    dv=pts[idx4]-pts[refs,None,:]; dv[:,:,:2]-=np.rint(dv[:,:,:2]/cell[:2])*cell[:2]
    norms=np.linalg.norm(dv,axis=2); uv=dv/np.maximum(norms[:,:,None],1e-8)
    q = orientational_tetrahedral_q(uv)
    sk = translational_tetrahedral_sk(d4)
    regularity = radial_regularity(d4)
    lsi = local_structure_index_batch(dist[:, 1:], lsi_cutoff)
    return q, sk, lsi, regularity, dist[:, 1:]

def main():
    """Analyze one trajectory and write per-chunk tables.

    Returns
    -------
    int
        Process status code.
    """
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--top',required=True);p.add_argument('--traj',action='append',required=True)
    p.add_argument('--out',required=True);p.add_argument('--stride',type=int,default=10)
    p.add_argument('--dz',type=float,default=PROFILE_BIN_A)
    p.add_argument('--core-halfwidth',type=float,default=BULK_HALFWIDTH_A)
    p.add_argument('--nref',type=int,default=4000);p.add_argument('--seed',type=int,default=SEEDS['cooling_chunk'])
    p.add_argument('--struct-every',type=int,default=5)
    p.add_argument('--start-frame', type=int, default=0,
                   help='First concatenated frame kept. A hold passes its manifest discard.')
    a=p.parse_args();out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    u=mda.Universe(a.top,a.traj)
    oxygen_index, _hydrogen = water_indices(u)
    ox=u.atoms[oxygen_index]
    stored_dt = frame_spacing_ps([float(ts.time) for ts in u.trajectory])
    require_film(len(ox), u.trajectory[0].dimensions[:3])
    rng=np.random.default_rng(a.seed); frames=[]; profiles=[];orders=[]
    sample_idx=np.sort(rng.choice(len(ox),min(len(ox),a.nref),replace=False))
    if a.start_frame < 0 or a.start_frame >= len(u.trajectory):
        raise SystemExit("The start frame removes every frame.")
    for n,ts in enumerate(u.trajectory[a.start_frame::a.stride]):
        Lx,Ly,Lz=map(float,ts.dimensions[:3]); xyz=ox.positions.astype('float64',copy=True)
        g=geometry(xyz[:,2],Lx,Ly,Lz,a.dz,a.core_halfwidth)
        frames.append({
            "frame": int(ts.frame),
            "time_ps": float(ts.time),
            "area_A2": Lx * Ly,
            "Lz_A": Lz,
            "rho_core_g_cm3": g["rho_core"],
            "rho_gds_g_cm3": g["rho_gds"],
            "h_gds_A": g["h_gds"],
            "h_eq_A": g["h_eq"],
            "width_lower_A": g["width_lower_10_90"],
            "width_upper_A": g["width_upper_10_90"],
            "gds_lower_A": g["gds_lower"],
            "gds_upper_A": g["gds_upper"],
            "z_center_A": g["z_center"],
            "areal_mass": g["areal_mass"],
            "mass_relative_error": g["mass_error"],
        })
        profiles.append(g['rho'])
        if n%a.struct_every==0:
            q, sk, lsi, regularity, dist = structure(xyz, sample_idx, Lx, Ly)
            masks = region_masks(
                g["z_relative"][sample_idx],
                np.minimum(
                    g["z_relative"][sample_idx] - g["gds_lower"],
                    g["gds_upper"] - g["z_relative"][sample_idx],
                ),
                a.core_halfwidth,
            )
            # Store binned distributions, not per-water CSV files that scale into many GB.
            # awi is the 3-12 A subsurface. awi_outer is the 0-6 A sensitivity band.
            for reg, ids in masks.items():
                if not np.any(ids): continue
                for key, vals, bins in [
                    ("q", q, np.linspace(-0.5, 1, 101)),
                    ("Sk", sk, np.linspace(0.99, 1.0, 101)),
                    ("LSI", lsi, np.linspace(0.0, 0.5, 101)),
                    ("radial_regularity", regularity, np.linspace(0, 1, 101)),
                ]:
                    good=ids&np.isfinite(vals)
                    hist,_=np.histogram(vals[good],bins=bins)
                    for j,cnt in enumerate(hist):
                        if cnt: orders.append((int(ts.frame),reg,key,float((bins[j]+bins[j+1])/2),int(cnt)))
    if not frames:
        raise SystemExit("No frames remain after the start frame and stride.")
    pd.DataFrame(frames).to_csv(out/'frame_geometry.csv',index=False)
    x=g['x']; pd.DataFrame({'z_relative_A':x,'rho_mean_g_cm3':np.mean(profiles,axis=0),
      'rho_sd_frames_g_cm3':np.std(profiles,axis=0,ddof=1) if len(profiles)>1 else np.zeros_like(x)}).to_csv(out/'density_profile.csv',index=False)
    pd.DataFrame(orders,columns=['frame','region','metric','bin_center','count']).to_csv(out/'structure_hist.csv',index=False)
    (out/'metadata.json').write_text(json.dumps(dict(top=a.top,traj=a.traj,stride=a.stride,start_frame=a.start_frame,n_frames=len(frames),
            original_dt_ps=stored_dt, analyzed_dt_ps=stored_dt * a.stride,
            n_waters=len(ox), nref=len(sample_idx),
            core_halfwidth_A=a.core_halfwidth,profile_bin_A=a.dz,smooth_sigma_A=PROFILE_SMOOTH_SIGMA_A,
            seed=a.seed,struct_every=a.struct_every,area_A2=float(Lx*Ly),
            lsi_cutoff_A=3.7,periodicity='xy for local order',
            warning='cooling chunks are nonequilibrium; the slab center is the circular mean of oxygen z'),indent=2))
    print('Complete:',a.traj,len(frames),'frames,',len(ox),'waters',flush=True)
    return 0
if __name__=='__main__':
    raise SystemExit(main())
