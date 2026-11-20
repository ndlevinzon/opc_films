#!/usr/bin/env python3
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.paths import ensure_analysis_on_path

ensure_analysis_on_path(__file__)
from common.constants import QSTAR_AINV, SEEDS, SQ_GRID_SPACING_A, adopted_qstar, fft_cell_counts
from opcfilm.config import CONFIG

def oxygen_indices(prmtop, water_names):
    """Oxygen atom indices of the named water residues.

    Parameters
    ----------
    prmtop : 
        Amber topology path.
    water_names : 
        Residue names treated as water.

    Returns
    -------
    result
        Oxygen atom indices of the named water residues.
    """
    import parmed as pmd
    top=pmd.load_file(prmtop)
    names={x.upper() for x in water_names}
    idx=[]
    for a in top.atoms:
        if a.residue.name.upper() in names and (a.atomic_number==8 or a.name.upper().startswith('O')):
            idx.append(a.idx)
    if not idx:
        raise RuntimeError('No water oxygens found; check --water-resnames')
    return np.array(idx,dtype=int)

def _sinc(value):
    """sin(x)/x, with the limit 1 at zero.

    Parameters
    ----------
    value : array_like
        Argument of the unnormalized sinc.

    Returns
    -------
    numpy.ndarray
        ``sin(x) / x``.
    """
    value = np.asarray(value, dtype=float)
    out = np.ones_like(value)
    nonzero = np.abs(value) > 1e-12
    out[nonzero] = np.sin(value[nonzero]) / value[nonzero]
    return out


def divide_histogram_form_factor(intensity, wavevector_x, wavevector_y, dx, dy):
    """Remove the histogram-bin form factor from a 2D structure factor.

    Parameters
    ----------
    intensity : numpy.ndarray
        ``|rho(q)|^2 / N`` on the FFT grid.
    wavevector_x, wavevector_y : numpy.ndarray
        Wavevector components in inverse angstroms, broadcast to ``intensity``.
    dx, dy : float
        Histogram cell sizes in angstroms.

    Returns
    -------
    numpy.ndarray
        ``intensity`` divided by ``sinc^2(q_x dx/2) sinc^2(q_y dy/2)``.
    """
    factor = _sinc(wavevector_x * dx / 2.0) ** 2 * _sinc(wavevector_y * dy / 2.0) ** 2
    return np.asarray(intensity, dtype=float) / np.maximum(factor, 1e-6)


def quadratic_peak(q, intensity, halfwidth=None, peak_min=None, peak_max=None):
    """Vertex of a quadratic fit around the maximum.

    Parameters
    ----------
    q : array_like
        Bin centers in inverse angstroms.
    intensity : array_like
        Structure factor on those bins.
    halfwidth : float, optional
        Half-width of the fit, in inverse angstroms.
    peak_min, peak_max : float, optional
        The maximum is sought only inside this interval. The quadratic uses
        every finite bin within ``halfwidth`` of that maximum.

    Returns
    -------
    float
        Fitted peak position. The bin of the maximum is returned when the
        curvature is not negative or the vertex leaves the window.
    """
    if halfwidth is None:
        halfwidth = CONFIG.qstar.peak_window_Ainv
    q = np.asarray(q, dtype=float)
    intensity = np.asarray(intensity, dtype=float)
    finite = np.isfinite(q) & np.isfinite(intensity)
    search = finite.copy()
    if peak_min is not None:
        search &= q >= peak_min
    if peak_max is not None:
        search &= q <= peak_max
    if int(search.sum()) < 5:
        raise ValueError("The peak window has fewer than five finite bins.")
    center = float(q[search][int(np.argmax(intensity[search]))])
    window = finite & (np.abs(q - center) <= halfwidth)
    if int(window.sum()) < 5:
        raise ValueError("Fewer than five bins lie within the quadratic window.")
    curvature, linear, _constant = np.polyfit(q[window], intensity[window], 2)
    if curvature >= 0.0:
        return center
    vertex = float(-linear / (2.0 * curvature))
    if vertex < float(q[window].min()) or vertex > float(q[window].max()):
        return center
    return vertex


def bootstrap_peak(frame_profiles, q, peak_min, peak_max, halfwidth, draws, rng):
    """Peak positions from resampled frames.

    Parameters
    ----------
    frame_profiles : array_like, shape (n_frames, n_bins)
        One radial structure factor per frame.
    q : array_like
        Bin centers.
    peak_min, peak_max : float
        Search window in inverse angstroms.
    halfwidth : float
        Quadratic half-width.
    draws : int
        Number of bootstrap draws.
    rng : numpy.random.Generator
        Random source.

    Returns
    -------
    numpy.ndarray
        One peak position per successful draw.
    """
    frame_profiles = np.asarray(frame_profiles, dtype=float)
    q = np.asarray(q, dtype=float)
    peaks = []
    count = len(frame_profiles)
    for _ in range(int(draws)):
        choice = rng.integers(0, count, count)
        mean = np.nanmean(frame_profiles[choice], axis=0)
        search = (q >= peak_min) & (q <= peak_max) & np.isfinite(mean)
        if int(search.sum()) < 5:
            continue
        peaks.append(quadratic_peak(q, mean, halfwidth, peak_min, peak_max))
    if len(peaks) < 2:
        raise RuntimeError("The bootstrap did not produce two peak positions.")
    return np.asarray(peaks, dtype=float)


def main():
    """Compute the in-plane oxygen structure factor of one or more trajectories.

    Returns
    -------
    result
        Compute the in-plane oxygen structure factor of one or more trajectories.
    """
    ap=argparse.ArgumentParser(description='2D in-plane oxygen static structure factor via density-grid FFT.')
    ap.add_argument('-p','--prmtop',required=True)
    ap.add_argument('-x','--traj',action='append',required=True, help='AMBER NetCDF; repeat for multiple files')
    ap.add_argument('-o','--out',required=True)
    ap.add_argument('--water-resnames',nargs='+',default=['WAT','HOH','OPC','TIP3','TIP4'])
    ap.add_argument('--grid-spacing', type=float, default=CONFIG.qstar.grid_spacing_A, help='Real-space FFT grid in angstroms.')
    ap.add_argument('--nx', type=int, default=None, help='Override the grid. Default follows --grid-spacing.')
    ap.add_argument('--ny', type=int, default=None, help='Override the grid. Default follows --grid-spacing.')
    ap.add_argument('--frame-stride',type=int,default=50)
    ap.add_argument('--discard-fraction', type=float, default=None,
                    help='Fraction of frames dropped at the start. Holds use the sheet value, 0.10.')
    ap.add_argument('--qmax',type=float,default=6)
    ap.add_argument('--dq',type=float,default=0.01, help='Radial bin width in A^-1.')
    ap.add_argument('--peak-min',type=float,default=CONFIG.qstar.peak_min_Ainv, help='Lower edge of the molecular peak search, in A^-1.')
    ap.add_argument('--peak-max',type=float,default=CONFIG.qstar.peak_max_Ainv, help='Upper edge of the molecular peak search, in A^-1.')
    ap.add_argument('--fit-halfwidth',type=float,default=CONFIG.qstar.peak_window_Ainv, help='Quadratic fit half-width around the maximum, in A^-1.')
    ap.add_argument('--bootstrap',type=int,default=200, help='Frame resamples used for the peak uncertainty.')
    ap.add_argument('--qstar-json', type=Path, default=None,
                    help='Per-row peak JSON. The name qstar.json is reserved for opcfilm qstar adopt.')
    args=ap.parse_args()

    from netCDF4 import Dataset
    from opcfilm.io.trajectory import discard_frame_count
    fraction = CONFIG.discard.hold_fraction if args.discard_fraction is None else args.discard_fraction
    indexed=[]
    for tr in args.traj:
        with Dataset(tr) as nc:
            n=nc.variables['coordinates'].shape[0]
            indexed.extend((tr, f) for f in range(0, n, args.frame_stride))
    start=discard_frame_count(len(indexed), 'hold', fraction)
    chosen=set(indexed[start:])
    if not chosen:
        raise SystemExit('No frames remain after the hold discard.')
    oi=oxygen_indices(args.prmtop,args.water_resnames)
    edges=np.arange(0,args.qmax+args.dq,args.dq)
    sumS=np.zeros(len(edges)-1); counts=np.zeros(len(edges)-1,dtype=np.int64)
    frame_profiles=[]
    nframes=0
    nx = args.nx
    ny = args.ny
    for tr in args.traj:
        with Dataset(tr) as nc:
            xyz=nc.variables['coordinates']
            cell=nc.variables.get('cell_lengths',None)
            if cell is None:
                raise RuntimeError(f'{tr}: no cell_lengths variable; needed for q grid')
            for f in range(0,xyz.shape[0],args.frame_stride):
                if (tr, f) not in chosen:
                    continue
                xy=np.asarray(xyz[f,oi,:2],dtype=float)
                Lx,Ly=np.asarray(cell[f,:2],dtype=float)
                if nx is None or ny is None:
                    nx = fft_cell_counts(Lx, args.grid_spacing)
                    ny = fft_cell_counts(Ly, args.grid_spacing)
                dx, dy = Lx / nx, Ly / ny
                xy[:,0]%=Lx; xy[:,1]%=Ly
                H,_,_=np.histogram2d(xy[:,0],xy[:,1],bins=[nx,ny],range=[[0,Lx],[0,Ly]])
                F=np.fft.fft2(H)
                S=(np.abs(F)**2)/len(oi)
                qx=2*np.pi*np.fft.fftfreq(nx,d=dx)
                qy=2*np.pi*np.fft.fftfreq(ny,d=dy)
                QX,QY=np.meshgrid(qx,qy,indexing='ij')
                S = divide_histogram_form_factor(S, QX, QY, dx, dy)
                Q=np.sqrt(QX*QX+QY*QY)
                mask=(Q>1e-12)&(Q<args.qmax)
                ib=np.digitize(Q[mask],edges)-1
                vals=S[mask]
                good=(ib>=0)&(ib<len(sumS))
                frame_sum=np.zeros_like(sumS)
                frame_count=np.zeros_like(counts)
                np.add.at(frame_sum,ib[good],vals[good]); np.add.at(frame_count,ib[good],1)
                np.add.at(sumS,ib[good],vals[good]); np.add.at(counts,ib[good],1)
                frame_profiles.append(np.divide(
                    frame_sum, frame_count,
                    out=np.full_like(frame_sum, np.nan),
                    where=frame_count > 0,
                ))
                nframes+=1
    q=0.5*(edges[:-1]+edges[1:])
    Sq=np.divide(sumS,counts,out=np.full_like(sumS,np.nan,dtype=float),where=counts>0)
    df=pd.DataFrame({'q_Ainv':q,'Sq2D':Sq,'samples':counts})
    df.to_csv(args.out,index=False)
    search = (q >= args.peak_min) & (q <= args.peak_max)
    if not np.any(np.isfinite(Sq[search])):
        raise RuntimeError(f'No finite S(q) between {args.peak_min:g} and {args.peak_max:g} A^-1.')
    measured = quadratic_peak(q, Sq, args.fit_halfwidth, args.peak_min, args.peak_max)
    peaks = bootstrap_peak(
        frame_profiles, q, args.peak_min, args.peak_max, args.fit_halfwidth,
        args.bootstrap, np.random.default_rng(SEEDS['local_order']),
    )
    adopted = adopted_qstar(measured)
    report = {
        'grid_spacing_A': args.grid_spacing,
        'grid_rule': 'round(L/spacing)',
        'nx': int(nx),
        'ny': int(ny),
        'dq_Ainv': args.dq,
        'sinc_corrected': True,
        'peak_min_Ainv': args.peak_min,
        'peak_max_Ainv': args.peak_max,
        'fit_halfwidth_Ainv': args.fit_halfwidth,
        'qstar_measured_Ainv': measured,
        'qstar_bootstrap_mean_Ainv': float(np.mean(peaks)),
        'qstar_bootstrap_sd_Ainv': float(np.std(peaks, ddof=1)),
        'qstar_bootstrap_p16_Ainv': float(np.percentile(peaks, 16)),
        'qstar_bootstrap_p84_Ainv': float(np.percentile(peaks, 84)),
        'qstar_row_candidate_Ainv': adopted,
        'qstar_reference_Ainv': QSTAR_AINV,
        'frames': nframes,
        'n_oxygen': int(len(oi)),
    }
    text = json.dumps(report, indent=2)
    row_json = Path(args.qstar_json) if args.qstar_json is not None else Path(args.out).with_name("qstar_row.json")
    if row_json.name == "qstar.json":
        raise SystemExit(
            f"{row_json} is the shared peak file. "
            "Pass a per-row path such as qstar_row.json. "
            "opcfilm qstar adopt writes outputs/qstar.json."
        )
    row_json.parent.mkdir(parents=True, exist_ok=True)
    row_json.write_text(text)
    print(f'Measured in-plane peak q* = {measured:.4f} ± {report["qstar_bootstrap_sd_Ainv"]:.4f} A^-1 on a {args.grid_spacing:g} A grid ({nx} x {ny}).')
    print(f'Row candidate q* = {adopted:.4f} A^-1. The reference value is {QSTAR_AINV:g} A^-1. The shared file is written by opcfilm qstar adopt.')
    print(f'Frames used: {nframes}; water oxygens: {len(oi)}')
    print(f'Wrote {args.out}')

if __name__=='__main__':
    main()
