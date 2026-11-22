#!/usr/bin/env python3
"""Potential energy per water along each cooling path.

Each manifest row is one cooling window. Amber mdout records are averaged
inside that window. Windows are combined within a replica at the same
nominal temperature, and the mean and standard deviation are then taken
across replicas. The temperature axis runs from hot to cold.

The mdout parser is ``common.mdout``. ``--parser`` can point at another
module that defines ``parse_dataframe`` or ``parse``.
"""
import argparse
import csv
import importlib.util
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from opcfilm.plotting.conventions import install_figure_conventions
install_figure_conventions()  # temperature axes hot to cold; legends outside panels

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.paths import ensure_analysis_on_path

ensure_analysis_on_path(__file__)
from common.manifest import load_manifest, trajectory_list
from common.mdout import nstep_span_matches, parse_dataframe
from common.rates import RATE_CODE_TO_K_PER_NS, RATE_COLOR_BY_VALUE
from common.trajectory import frame_spacing_ps, netcdf_time_ps, ntwx_from_frame_ps, resolve_n_water
from common.uncertainty import CURVE_BAND_SD
from opcfilm.config import CONFIG

RATE = RATE_CODE_TO_K_PER_NS
COLORS = RATE_COLOR_BY_VALUE

def read_tsv(path):
    """Read a tab-separated table with a header row.

    Parameters
    ----------
    path : path-like
        Input TSV.

    Returns
    -------
    list of dict
        One dictionary per row.
    """
    with Path(path).open(newline='') as fh:
        return list(csv.DictReader(fh, delimiter='\t'))

def load_parser(explicit):
    """Return a function that parses one mdout path into a DataFrame.

    Parameters
    ----------
    explicit : path-like or None
        Optional module path. The module must define ``parse_dataframe``
        or ``parse``. ``parse`` may return a DataFrame or a list of records.

    Returns
    -------
    parser : callable
        ``parser(path) -> pandas.DataFrame``.
    source : str
        ``"common.mdout"`` or the explicit module path.
    """
    if not explicit:
        return parse_dataframe, "common.mdout"
    path = Path(explicit)
    if not path.is_file():
        raise FileNotFoundError(f"MDOUT parser not found: {path}")
    spec = importlib.util.spec_from_file_location("opc_amber_mdout_parser", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if hasattr(module, "parse_dataframe"):
        return module.parse_dataframe, str(path)
    if not hasattr(module, "parse"):
        raise AttributeError(f"{path} defines neither parse_dataframe nor parse")

    def _as_frame(mdout_path):
        """DataFrame of thermodynamic records from an external parser.

        Parameters
        ----------
        mdout_path : 
            Amber mdout path.

        Returns
        -------
        result
            DataFrame of thermodynamic records from an external parser.
        """
        parsed = module.parse(mdout_path)
        if isinstance(parsed, pd.DataFrame):
            return parsed
        return pd.DataFrame(parsed)

    return _as_frame, str(path)

def build_tables(manifest, energy_map, parse_mdout, nwater, t_tol, min_records):
    """Average the potential energy inside each mapped cooling window.

    Parameters
    ----------
    manifest, energy_map : path-like
        Density manifest and a TSV with ``trajectory`` and ``mdout`` columns.
    parse_mdout : callable
        Returns a DataFrame with ``temp_K`` and ``EPtot``.
    nwater : int
        Number of water molecules used to convert the cell energy.
    t_tol : float
        Maximum allowed difference between the mean mdout temperature and
        the nominal window midpoint, in kelvin.
    min_records : int
        Minimum number of finite energy records.

    Returns
    -------
    samples, qc : tuple of pandas.DataFrame
        Accepted per-window energies, and one QC row for every ramp window.
    """
    maps = {}
    for row in read_tsv(energy_map):
        if 'trajectory' not in row or 'mdout' not in row:
            raise ValueError('Energy map must have trajectory and mdout TSV columns')
        traj = str(Path(row['trajectory']).expanduser().resolve())
        if traj in maps:
            raise ValueError(f'Duplicate trajectory in energy map: {traj}')
        maps[traj] = row['mdout'].strip()
    samples, qc = [], []
    for row in load_manifest(manifest):
        if row.get('kind') != 'ramp':
            continue
        paths = trajectory_list(row)
        if not paths:
            continue
        traj = str(paths[0].expanduser().resolve())
        rep = row['replica']
        try:
            rate = float(row['rate_K_per_ns']) if str(row.get('rate_K_per_ns', '')).strip() else RATE[int(row['rate_code'])]
            nominal = (float(row['temperature_start_K']) + float(row['temperature_end_K'])) / 2
        except (KeyError, ValueError, TypeError) as exc:
            qc.append(dict(trajectory=traj, status='INVALID_MANIFEST_T_OR_RATE', message=str(exc)))
            continue
        info = dict(trajectory=traj, replica=rep, rate_K_ns=rate, temperature_K=nominal,
                    mdout=maps.get(traj, ''), status='', message='')
        if not info['mdout']:
            info.update(status='MISSING_MAPPING')
            qc.append(info)
            continue
        path = Path(info['mdout']).expanduser()
        if not path.is_file():
            info.update(status='MISSING_MDOUT_FILE', message=str(path))
            qc.append(info)
            continue
        try:
            energies = parse_mdout(path)
            if not {'temp_K', 'EPtot'}.issubset(energies.columns):
                raise ValueError('Parser did not return temp_K and EPtot')
            t = pd.to_numeric(energies['temp_K'], errors='coerce')
            e = pd.to_numeric(energies['EPtot'], errors='coerce')
            keep = np.isfinite(t) & np.isfinite(e)
            if int(keep.sum()) < min_records:
                raise ValueError(f'Only {keep.sum()} records with finite T and EPtot')
            tmean = float(t[keep].mean())
            if abs(tmean - nominal) > t_tol:
                raise ValueError(f'Mean T {tmean:.2f} K vs nominal {nominal:.2f} K (tolerance {t_tol:g} K)')
            times = netcdf_time_ps(traj)
            ntwx = ntwx_from_frame_ps(frame_spacing_ps(times))
            if 'nstep' not in energies or not nstep_span_matches(
                energies.nstep.min(), energies.nstep.max(), len(times), ntwx
            ):
                raise ValueError('mdout nstep range does not match the trajectory frame count times ntwx')
            info.update(status='OK', n_energy_records=int(keep.sum()), mean_observed_T_K=tmean,
                        nstep_first=int(energies.nstep.min()) if 'nstep' in energies else np.nan,
                        nstep_last=int(energies.nstep.max()) if 'nstep' in energies else np.nan)
            samples.append(dict(replica=rep, rate_K_ns=rate, temperature_K=nominal,
                                trajectory=traj, mdout=str(path), n_energy_records=int(keep.sum()),
                                mean_observed_T_K=tmean,
                                EPtot_kcal_mol_cell=float(e[keep].mean()),
                                E_per_water_kcal_mol=float(e[keep].mean()/nwater)))
        except Exception as exc:
            info.update(status='REJECTED', message=f'{type(exc).__name__}: {exc}')
        qc.append(info)
    return pd.DataFrame(samples), pd.DataFrame(qc)

def summarize(chunks):
    """Average energy within each replica, then across replicas.

    Parameters
    ----------
    chunks : pandas.DataFrame
        Accepted windows from :func:`build_tables`.

    Returns
    -------
    replica, aggregate : tuple of pandas.DataFrame
        One row per replica and nominal temperature, then the across-replica
        mean and standard deviation.
    """
    # Prevent a repeated chunk at the same nominal T from masquerading as a replica.
    rep = chunks.groupby(['replica', 'rate_K_ns', 'temperature_K'], as_index=False).agg(
        E_per_water_kcal_mol=('E_per_water_kcal_mol', 'mean'),
        n_chunks=('trajectory', 'size'), n_energy_records=('n_energy_records', 'sum'),
        mean_observed_T_K=('mean_observed_T_K', 'mean'))
    agg = rep.groupby(['rate_K_ns', 'temperature_K'], as_index=False).agg(
        E_per_water_mean=('E_per_water_kcal_mol', 'mean'),
        E_per_water_sd=('E_per_water_kcal_mol', 'std'),
        n_replicas=('replica', 'nunique'))
    return rep, agg

def compare_lowT(rep, target):
    """Pair the slowest and fastest rates near one target temperature.

    Parameters
    ----------
    rep : pandas.DataFrame
        Per-replica energies from :func:`summarize`.
    target : float
        Requested temperature in kelvin. Each replica uses its nearest
        shared temperature.

    Returns
    -------
    pandas.DataFrame
        Fast-minus-slow energy difference for each replica. Empty when
        fewer than two rates are present.
    """
    rates = sorted(rep.rate_K_ns.unique())
    if len(rates) < 2:
        return pd.DataFrame()
    slow, fast = rates[0], rates[-1]
    rows=[]
    for rep_id,g in rep.groupby('replica'):
        a = g[g.rate_K_ns == slow].set_index('temperature_K')
        b = g[g.rate_K_ns == fast].set_index('temperature_K')
        shared = sorted(set(a.index) & set(b.index))
        if not shared:
            continue
        temperature = min(shared, key=lambda t: abs(t-target))
        es = float(a.loc[temperature, 'E_per_water_kcal_mol'])
        ef = float(b.loc[temperature, 'E_per_water_kcal_mol'])
        rows.append(dict(replica=rep_id, target_temperature_K=target,
                         matched_temperature_K=temperature,
                         slow_rate_K_ns=slow,fast_rate_K_ns=fast,
                         E_slow_kcal_mol_water=es,E_fast_kcal_mol_water=ef,
                         delta_E_fast_minus_slow_kcal_mol_water=ef-es))
    return pd.DataFrame(rows)

def draw(agg, prefix, tmin, tmax, band_sd, title):
    """Plot energy per water against the nominal window temperature.

    Parameters
    ----------
    agg : pandas.DataFrame
        Across-replica summary.
    prefix : path-like
        Output path without an extension. PNG and PDF files are written.
    tmin, tmax : float
        Axis limits. The axis is inverted so cooling runs left to right.
    band_sd : float
        Half-width of the shaded band in multiples of the replica standard
        deviation. Zero disables the band.
    title : str
        Axes title. An empty string leaves the title blank.

    Returns
    -------
    None
        PNG and PDF files are written from ``prefix``.
    """
    fig, ax = plt.subplots(figsize=(8.4, 5.1))
    for rate in sorted(agg.rate_K_ns.unique()):
        d=agg[agg.rate_K_ns == rate].sort_values('temperature_K')
        x=d.temperature_K.to_numpy(float)
        y=d.E_per_water_mean.to_numpy(float)
        sd=d.E_per_water_sd.to_numpy(float)
        color=COLORS.get(rate, '#555555')
        ax.plot(x,y, marker='o',ms=3.2,lw=1.7,color=color,label=f'{rate:g} K/ns')
        mask=np.isfinite(x)&np.isfinite(y)&np.isfinite(sd)
        if band_sd>0 and np.any(mask):
            ax.fill_between(x[mask],y[mask]-band_sd*sd[mask],y[mask]+band_sd*sd[mask],
                            color=color,alpha=.13,linewidth=0)
    ax.set_xlim(tmax,tmin)
    ax.set_xlabel('Nominal cooling-window midpoint (K)')
    ax.set_ylabel(r'Potential energy per water (kcal mol$^{-1}$)')
    if title:ax.set_title(title)
    ax.grid(alpha=.16)
    ax.legend(title='Cooling rate',loc='upper left',bbox_to_anchor=(1.02,1),frameon=False)
    fig.tight_layout()
    for ext in ('.png','.pdf'):
        fig.savefig(str(prefix)+ext,dpi=300 if ext=='.png' else None,bbox_inches='tight')
    plt.close(fig)

def main():
    """Read the manifest and energy map, then write tables and the figure.

    Returns
    -------
    int
        Process status code.
    """
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest',required=True,type=Path,help='config/manifest.tsv')
    p.add_argument('--energy-map',required=True,type=Path,help='Existing energy_map.tsv with trajectory,mdout columns')
    p.add_argument('--parser',type=Path,help='Optional module with parse_dataframe. The default is common.mdout.')
    p.add_argument('--out-prefix',required=True,type=Path)
    p.add_argument('--topology',type=Path,required=True,help='Amber prmtop. N is read from this file.')
    p.add_argument('--n-water',type=int,default=None,help='Optional cross-check against the topology count.')
    p.add_argument('--Tmin',type=float,default=CONFIG.energy.report_K)
    p.add_argument('--Tmax',type=float,default=CONFIG.energy.fit_max_K)
    p.add_argument('--lowT-report',type=float,default=CONFIG.energy.report_K)
    p.add_argument('--band-sd',type=float,default=CURVE_BAND_SD)
    p.add_argument('--max-temp-mismatch-K',type=float,default=3.)
    p.add_argument('--min-energy-records',type=int,default=2)
    p.add_argument('--title',default='Cooling-rate-dependent energetic stabilization')
    p.add_argument('--allow-incomplete',action='store_true',help='Permit missing/rejected mapped chunks; inspect QC carefully')
    args=p.parse_args()
    args.n_water = resolve_n_water(args.topology, args.n_water)
    if args.Tmin>=args.Tmax or args.band_sd<0:
        p.error('Invalid Tmin/Tmax or band-sd')
    args.out_prefix.parent.mkdir(parents=True,exist_ok=True)
    parse, parser_path = load_parser(args.parser)
    chunk, qc = build_tables(args.manifest,args.energy_map,parse,args.n_water,
                             args.max_temp_mismatch_K,args.min_energy_records)
    qc_file=Path(str(args.out_prefix)+'_mdout_qc.csv')
    qc.to_csv(qc_file,index=False)
    if chunk.empty:raise SystemExit(f'No accepted MDOUT chunks. Inspect {qc_file}')
    failures=qc[qc.status!='OK']
    if len(failures) and not args.allow_incomplete:
        raise SystemExit(f'{len(failures)} unmatched/rejected ramp chunks. Inspect {qc_file}; --allow-incomplete to plot partial data.')
    chunk=chunk[chunk.temperature_K.between(args.Tmin,args.Tmax)].copy()
    if chunk.empty:raise SystemExit('No chunks in selected T range')
    rep,agg=summarize(chunk)
    prefix=str(args.out_prefix)
    chunk.to_csv(prefix+'_chunk_energy.csv',index=False)
    rep.to_csv(prefix+'_replica_binned.csv',index=False)
    agg.to_csv(prefix+'_mean_sd_per_water.csv',index=False)
    low=compare_lowT(rep,args.lowT_report)
    low.to_csv(prefix+'_lowT_fast_vs_slow_by_replica.csv',index=False)
    if not low.empty:
        print('Low-temperature paired fast-minus-slow mean and SD:',
              low.delta_E_fast_minus_slow_kcal_mol_water.mean(),
              low.delta_E_fast_minus_slow_kcal_mol_water.std(ddof=1))
    draw(agg,args.out_prefix,args.Tmin,args.Tmax,args.band_sd,args.title)
    print(f'Parser: {parser_path}; accepted chunks: {len(chunk)}; replicas: {rep.replica.nunique()}')
    print(f'Saved {prefix}.png, {prefix}.pdf and QC/summary CSVs.')
    return 0

if __name__=='__main__':
    raise SystemExit(main())
