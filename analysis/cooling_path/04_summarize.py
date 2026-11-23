#!/usr/bin/env python3
"""Replica-aware density summaries and the cooling-rate response.

Reads ``density_manifest.tsv`` and the per-chunk ``frame_geometry.csv``
files. The air-water interface thickness runs from the last sustained
departure of the core plateau (0.98 times the core density) to the outer
edge of the liquid (0.05 times the core density). The response
``chi = Delta rho / Delta log10(rate)`` is computed inside each replica
before the replicas are averaged.
"""
import argparse
import sys
from pathlib import Path
import numpy as np
from scipy.ndimage import gaussian_filter1d
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from opcfilm.plotting.conventions import install_figure_conventions
install_figure_conventions()  # temperature axes hot to cold; legends outside panels
from scipy.stats import linregress, t

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.paths import ensure_analysis_on_path

ensure_analysis_on_path(__file__)
from common.rates import (
    INTERVAL_COLOR,
    RATE_CODES,
    RATE_CODE_TO_K_PER_NS as RATE_VALUE,
    RATE_LABEL,
    color_for,
    directory_code,
)
from common.constants import BULK_HALFWIDTH_A
from common.uncertainty import curve_caption

METRICS = ['rho_core_g_cm3', 'h_gds_A', 'h_eq_A', 'width_lower_A', 'width_upper_A', 'awi_full_thickness_mean_A', 'awi_full_thickness_upper_A', 'awi_full_thickness_lower_A', 'awi_inside_gds_mean_A', 'awi_outside_gds_mean_A']



def outer_liquid_extent(profile_path, floor_fraction=0.05, persistence_A=3,
                        smoothing_A=1.0, core_halfwidth_A=BULK_HALFWIDTH_A,
                        onset_fraction=0.98, onset_persistence_A=2.0):
    """Full liquid-density transition from the plateau onset to the vapor edge.

    The onset is the last sustained crossing of ``onset_fraction`` times the
    core plateau before the half-density surface. The outer edge is the first
    sustained crossing of ``floor_fraction`` times that plateau. Both
    crossings must persist for the requested distance so a local ripple is
    not treated as the interface. The profile is a time average; it does not
    measure molecular connectivity.

    Parameters
    ----------
    profile_path : path-like
        CSV with ``z_relative_A`` and ``rho_mean_g_cm3``.
    floor_fraction : float, optional
        Outer-edge density as a fraction of the core plateau.
    persistence_A : float, optional
        Distance the profile must stay below the outer threshold.
    smoothing_A : float, optional
        Gaussian smoothing length applied before the crossings are located.
    core_halfwidth_A : float, optional
        Half-width used to average the core plateau.
    onset_fraction : float, optional
        Plateau-departure density as a fraction of the core.
    onset_persistence_A : float, optional
        Distance the profile must stay below the onset threshold.

    Returns
    -------
    dict
        Upper, lower, and mean transition thicknesses, plus the mean
        distance of each edge inside and outside the half-density surface.
    """
    d = pd.read_csv(profile_path)
    z = d['z_relative_A'].to_numpy(float)
    raw = d['rho_mean_g_cm3'].to_numpy(float)
    if len(z) < 20 or not np.all(np.diff(z) > 0):
        raise ValueError(f'Invalid profile grid: {profile_path}')
    dz = float(np.median(np.diff(z)))
    y = gaussian_filter1d(raw, sigma=max(0.1, smoothing_A / dz), mode='nearest')
    plateau = float(np.mean(y[np.abs(z) < core_halfwidth_A]))
    if not np.isfinite(plateau) or plateau <= 0:
        raise ValueError(f'Invalid plateau density: {profile_path}')
    n_outer = max(2, int(np.ceil(persistence_A / dz)))
    n_onset = max(2, int(np.ceil(onset_persistence_A / dz)))
    metrics = {}
    diagnostic = {}
    for side, sign in [('upper', 1), ('lower', -1)]:
        r = sign * z
        order = np.argsort(r)
        r, v = r[order], y[order]
        keep = r >= 0
        r, v = r[keep], v[keep]

        crossings = np.flatnonzero((v[:-1] >= .5 * plateau) &
                                   (v[1:] < .5 * plateau))
        if not len(crossings):
            raise ValueError(f'Missing {side} 50% surface: {profile_path}')
        k = int(crossings[0])
        gds = float(np.interp(.5 * plateau, v[k:k+2][::-1], r[k:k+2][::-1]))

        # A plateau departure is operational, not an infinitely sharp onset.
        threshold_start = onset_fraction * plateau
        onset = np.nan
        # The LAST sustained downward crossing before the GDS rejects
        # short-lived depressions within the bulk plateau.
        for j in range(k-1, n_onset-2, -1):
            if (v[j] >= threshold_start and v[j+1] < threshold_start
                    and j+1+n_onset <= len(v)
                    and np.all(v[j+1:j+1+n_onset] < threshold_start)
                    and np.all(v[j+1:k+1] < threshold_start)):
                onset = float(np.interp(threshold_start, v[j:j+2][::-1],
                                        r[j:j+2][::-1]))
                break
        if not np.isfinite(onset):
            raise ValueError(f'No sustained plateau onset for {side}: {profile_path}')

        threshold_end = floor_fraction * plateau
        edge = np.nan
        for j in range(k+1, len(v)-n_outer):
            if (v[j] >= threshold_end and v[j+1] < threshold_end and
                    np.all(v[j+1:j+1+n_outer] < threshold_end)):
                edge = float(np.interp(threshold_end, v[j:j+2][::-1],
                                       r[j:j+2][::-1]))
                break
        if not np.isfinite(edge):
            raise ValueError(f'Missing vapor boundary for {side}: {profile_path}')
        metrics[side] = edge - onset
        diagnostic[side] = dict(onset=onset, gds=gds, edge=edge,
                                inside=gds-onset, outside=edge-gds)

    return {
        'awi_full_thickness_upper_A': metrics['upper'],
        'awi_full_thickness_lower_A': metrics['lower'],
        'awi_full_thickness_mean_A': (metrics['upper'] + metrics['lower']) / 2,
        'awi_inside_gds_mean_A': (diagnostic['upper']['inside'] + diagnostic['lower']['inside']) / 2,
        'awi_outside_gds_mean_A': (diagnostic['upper']['outside'] + diagnostic['lower']['outside']) / 2,
    }

def summary_stats(values):
    """Mean, sample standard deviation, and 95 percent half-width.

    Parameters
    ----------
    values : array_like
        Replica-level observations. Non-finite values are dropped.

    Returns
    -------
    mean, sd, halfwidth, n : tuple
        ``halfwidth`` is the Student t 95 percent half-width. It is NaN
        when fewer than two finite values are present.
    """
    a = np.asarray(values, dtype=float)
    a = a[np.isfinite(a)]
    n = len(a)
    mean = float(a.mean()) if n else np.nan
    sd = float(a.std(ddof=1)) if n > 1 else np.nan
    hw = float(t.ppf(.975, n-1) * sd / np.sqrt(n)) if n > 1 else np.nan
    return mean, sd, hw, n


def plotlines(agg, metric, ylabel, path, rate_map):
    """Plot one metric against temperature for every cooling rate.

    Parameters
    ----------
    agg : pandas.DataFrame
        Table with ``metric``, ``rate_K_ns``, ``temperature_K``, ``mean``,
        and ``sd``.
    metric : str
        Column value selected from ``agg.metric``.
    ylabel : str
        Vertical axis label.
    path : pathlib.Path
        Output path without an extension.
    rate_map : dict
        Maps a rate code to the legend label.

    Returns
    -------
    None
        PNG and PDF files are written at ``path``.
    """
    fig, ax = plt.subplots(figsize=(8.4, 5.1))
    for rate, g in agg[agg.metric == metric].groupby('rate_K_ns'):
        g = g.sort_values('temperature_K')
        ax.errorbar(g.temperature_K, g['mean'], yerr=g['sd'], marker='o', ms=4,
                    color=color_for(rate), lw=1.7, capsize=2,
                    label=f'{float(rate):g} K/ns')
    ax.invert_xaxis()
    ax.set(xlabel='Nominal cooling-window midpoint (K)', ylabel=ylabel)
    ax.grid(alpha=.16)
    ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1), frameon=False)
    fig.text(0.01, 0.01, curve_caption(), fontsize=8)
    fig.tight_layout()
    fig.savefig(path.with_suffix('.png'), dpi=300, bbox_inches='tight')
    fig.savefig(path.with_suffix('.pdf'), bbox_inches='tight')
    plt.close(fig)


def main():
    """Aggregate chunk tables and write the cooling-rate response.

    Returns
    -------
    int
        Process status code.
    """
    ap = argparse.ArgumentParser()
    ap.add_argument('--manifest', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--min-rates', type=int, default=3)
    ap.add_argument('--awi-floor-fraction', type=float, default=0.05,
                    help='Fraction of core density defining the outer edge (default 0.05)')
    ap.add_argument('--awi-persistence-A', type=float, default=3,
                    help='Below-threshold distance required beyond outer edge (default 3 Å)')
    ap.add_argument('--awi-smoothing-A', type=float, default=1.0)
    ap.add_argument('--awi-onset-fraction', type=float, default=0.98)
    ap.add_argument('--awi-onset-persistence-A', type=float, default=2.0)
    args = ap.parse_args()
    if not 0.0 < args.awi_floor_fraction < 0.5:
        ap.error('--awi-floor-fraction must lie between 0 and 0.5')
    if not 0.5 < args.awi_onset_fraction < 1.0:
        ap.error('--awi-onset-fraction must be between 0.5 and 1.0')
    if args.awi_onset_persistence_A <= 0:
        ap.error('--awi-onset-persistence-A must be positive')
    if args.awi_persistence_A <= 0 or args.awi_smoothing_A < 0:
        ap.error('AWI persistence must be positive and smoothing nonnegative')
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    from common.manifest import load_manifest, result_directory
    rows = load_manifest(args.manifest)
    repo = Path(__file__).resolve().parents[2]
    for row in rows:
        row["outdir"] = str(result_directory(repo, "profiles", row))
        # Hold profiles are written by the capillary rule (03_run_row.py
        # --capwave), which runs the same density pass in its own directory.
        if row.get("kind") == "hold" and not (Path(row["outdir"]) / "frame_geometry.csv").exists():
            capillary = result_directory(repo, "capillary", row)
            if (capillary / "frame_geometry.csv").exists():
                row["outdir"] = str(capillary)
    data, holds, missing = [], [], []
    for row in rows:
        path = Path(row['outdir']) / 'frame_geometry.csv'
        if not path.exists():
            missing.append(str(path))
            continue
        frame = pd.read_csv(path)
        if frame.empty:
            missing.append(f'EMPTY: {path}')
            continue
        # An AWI outer shoulder is a per-chunk *profile* property, not a
        # per-frame 10–90% width. Keep it at one observation per chunk.
        profile_path = Path(row['outdir']) / 'density_profile.csv'
        awi_metrics = None
        if profile_path.is_file():
            try:
                awi_metrics = outer_liquid_extent(
                    profile_path, floor_fraction=args.awi_floor_fraction,
                    persistence_A=args.awi_persistence_A,
                    smoothing_A=args.awi_smoothing_A,
                    onset_fraction=args.awi_onset_fraction,
                    onset_persistence_A=args.awi_onset_persistence_A)
            except (ValueError, KeyError) as exc:
                missing.append(f'AWI PROFILE ERROR: {profile_path}: {exc}')
        else:
            missing.append(f'MISSING AWI PROFILE: {profile_path}')
        start = pd.to_numeric(row.get('temperature_start_K'), errors='coerce')
        end = pd.to_numeric(row.get('temperature_end_K'), errors='coerce')
        temperature = float((start + end)/2) if np.isfinite(start) and np.isfinite(end) else np.nan
        for col in METRICS:
            if col not in frame:
                missing.append(f'MISSING COLUMN {col}: {path}')
                continue
            x = pd.to_numeric(frame[col], errors='coerce').to_numpy(float)
            good = x[np.isfinite(x)]
            if not len(good):
                continue
            data.append(dict(kind=row['kind'], replica=row['replica'], rate_code=row['rate_code'],
                             start_K=start, end_K=end, temperature_K=temperature, label=row['label'],
                             metric=col, value=float(good.mean()),
                             sd_frames=float(good.std(ddof=1)) if len(good)>1 else np.nan,
                             n_frames=len(good)))
        if awi_metrics is not None:
            for col, value in awi_metrics.items():
                data.append(dict(kind=row['kind'], replica=row['replica'],
                    rate_code=row['rate_code'],start_K=start,end_K=end,
                    temperature_K=temperature,label=row['label'],metric=col,
                    value=float(value),sd_frames=np.nan,n_frames=1))
        if row['kind'] == 'hold':
            for col in ('rho_core_g_cm3', 'h_gds_A', 'h_eq_A'):
                if col not in frame or 'time_ps' not in frame:
                    continue
                x = pd.to_numeric(frame[col], errors='coerce').to_numpy(float)
                time_ns = (pd.to_numeric(frame.time_ps, errors='coerce').to_numpy(float) - float(frame.time_ps.iloc[0]))/1000
                good = np.isfinite(x) & np.isfinite(time_ns)
                fit = linregress(time_ns[good], x[good]) if good.sum()>5 and np.ptp(time_ns[good])>0 else None
                q = max(1, len(x)//4)
                holds.append(dict(replica=row['replica'], label=row['label'], metric=col,
                                  slope_per_ns=fit.slope if fit else np.nan,
                                  pvalue_naive=fit.pvalue if fit else np.nan,
                                  first_quarter=np.nanmean(x[:q]), last_quarter=np.nanmean(x[-q:])))
    if not data:
        raise SystemExit('No density measurements found. Check manifest output paths.')
    df = pd.DataFrame(data)
    df.to_csv(out/'chunk_metrics.csv', index=False)
    pd.DataFrame(holds).to_csv(out/'hold_drift_diagnostics.csv', index=False)
    (out/'missing_outputs.txt').write_text('\n'.join(missing)+'\n')

    ramp = df[df.kind=='ramp'].copy()
    ramp['rate_code'] = pd.to_numeric(ramp.rate_code, errors='coerce')
    ramp['rate_K_ns'] = ramp.rate_code.map(RATE_VALUE)
    per = ramp.groupby(['replica','rate_K_ns','temperature_K','metric'],as_index=False).agg(
        value=('value','mean'), n_chunks=('value','size'))
    per.to_csv(out/'replica_temperature_metrics.csv', index=False)
    agg = per.groupby(['rate_K_ns','temperature_K','metric'],as_index=False).agg(
        mean=('value','mean'), sd=('value','std'), n_rep=('replica','nunique'))
    agg['sem'] = agg.sd / np.sqrt(agg.n_rep)
    agg['ci95_halfwidth'] = (
      t.ppf(0.975, agg['n_rep'] - 1)
      * agg['sem']
    )
    agg.loc[agg.n_rep<2, 'ci95_halfwidth'] = np.nan
    agg.to_csv(out/'rate_temperature_mean_sd.csv', index=False)

    # Preserve historical pooled exploratory slopes for comparison only.
    pooled = []
    for (temp, metric), group in per.groupby(['temperature_K','metric']):
        counts = group.groupby('rate_K_ns').replica.nunique()
        sample = group[group.rate_K_ns.isin(counts[counts>=2].index)]
        if sample.rate_K_ns.nunique()<args.min_rates:
            continue
        fit = linregress(np.log10(sample.rate_K_ns.to_numpy(float)), sample.value.to_numpy(float))
        pooled.append(dict(temperature_K=temp, metric=metric, n_rates=sample.rate_K_ns.nunique(),
                           n_replicas=len(sample), chi_per_decade=fit.slope, intercept=fit.intercept,
                           r2=fit.rvalue**2, pvalue_naive=fit.pvalue, fit_stderr_naive=fit.stderr))
    pooled = pd.DataFrame(pooled)
    pooled.to_csv(out/'rate_response_exploratory.csv', index=False)

    # Corrected independent-unit analysis: one slope per replica and temperature.
    fits = []
    for (rep, temp, metric), group in per.groupby(['replica','temperature_K','metric']):
        g = group[np.isfinite(group.value) & (group.rate_K_ns>0)].drop_duplicates('rate_K_ns')
        if len(g)<args.min_rates:
            continue
        fit = linregress(np.log10(g.rate_K_ns.to_numpy(float)), g.value.to_numpy(float))
        fits.append(dict(replica=rep, temperature_K=temp, metric=metric, n_rates=len(g),
                         rate_min=float(g.rate_K_ns.min()), rate_max=float(g.rate_K_ns.max()),
                         chi_per_decade=float(fit.slope), intercept=float(fit.intercept),
                         r2=float(fit.rvalue**2)))
    fits = pd.DataFrame(fits)
    fits.to_csv(out/'rate_response_by_replica.csv', index=False)
    ss = []
    for (temp, metric), g in fits.groupby(['temperature_K','metric']):
        mu, sd, ci, n = summary_stats(g.chi_per_decade)
        ss.append(dict(temperature_K=temp, metric=metric, chi_mean=mu, chi_sd=sd,
                       chi_ci95_low=mu-ci if np.isfinite(ci) else np.nan,
                       chi_ci95_high=mu+ci if np.isfinite(ci) else np.nan,
                       ci95_halfwidth=ci, n_rep=n, n_rates_min=int(g.n_rates.min())))
    response = pd.DataFrame(ss)
    response.to_csv(out/'rate_response_replica_mean.csv', index=False)

    # Adjacent-protocol finite differences, computed within each independent replica.
    # They are interval-specific secant slopes, not derivatives at a single rate.
    interval_rows = []
    for (rep, temp, metric), group in per.groupby(['replica', 'temperature_K', 'metric']):
        values = {directory_code(rate): float(val) for rate, val in
                  zip(group['rate_K_ns'], group['value']) if np.isfinite(val)}
        for lo, hi in zip(RATE_CODES[:-1], RATE_CODES[1:]):
            if lo not in values or hi not in values:
                continue
            logspan = np.log10(RATE_VALUE[hi] / RATE_VALUE[lo])
            interval_rows.append(dict(
                replica=rep, temperature_K=temp, metric=metric,
                lower_rate_K_ns=RATE_VALUE[lo], upper_rate_K_ns=RATE_VALUE[hi],
                lower_rate_code=lo, upper_rate_code=hi,
                interval_label=f'{RATE_VALUE[lo]:g}–{RATE_VALUE[hi]:g} K/ns',
                chi_per_decade=(values[hi] - values[lo]) / logspan))
    intervals = pd.DataFrame(interval_rows, columns=[
        'replica', 'temperature_K', 'metric', 'lower_rate_K_ns',
        'upper_rate_K_ns', 'lower_rate_code', 'upper_rate_code',
        'interval_label', 'chi_per_decade'])
    intervals.to_csv(out/'rate_response_intervals_by_replica.csv', index=False)
    interval_stats = []
    for (temp, metric, lo, hi), g in intervals.groupby(
            ['temperature_K','metric','lower_rate_code','upper_rate_code']):
        mu, sd, ci, n = summary_stats(g['chi_per_decade'])
        interval_stats.append(dict(
            temperature_K=temp, metric=metric,
            lower_rate_code=lo, upper_rate_code=hi,
            lower_rate_K_ns=RATE_VALUE[int(lo)], upper_rate_K_ns=RATE_VALUE[int(hi)],
            interval_label=f'{RATE_VALUE[int(lo)]:g}–{RATE_VALUE[int(hi)]:g} K/ns',
            chi_mean=mu, chi_sd=sd, ci95_halfwidth=ci, n_rep=n))
    interval_summary = pd.DataFrame(interval_stats, columns=[
        'temperature_K','metric','lower_rate_code','upper_rate_code',
        'lower_rate_K_ns','upper_rate_K_ns','interval_label',
        'chi_mean','chi_sd','ci95_halfwidth','n_rep'])
    interval_summary.to_csv(out/'rate_response_intervals_summary.csv', index=False)

    fig, ax = plt.subplots(figsize=(8.4, 5.1))
    local = interval_summary[interval_summary['metric'] == 'rho_core_g_cm3']
    for lo, hi in zip(RATE_CODES[:-1], RATE_CODES[1:]):
        g = local[(local['lower_rate_code'] == lo) &
                  (local['upper_rate_code'] == hi)].sort_values('temperature_K')
        if g.empty:
            continue
        ax.errorbar(g['temperature_K'], g['chi_mean'], yerr=g['chi_sd'],
                    color=INTERVAL_COLOR[(lo,hi)], marker='o', ms=4, lw=1.7,
                    capsize=2, label=f'{RATE_VALUE[lo]:g}–{RATE_VALUE[hi]:g} K/ns')
    ax.axhline(0, color='gray', ls='--', lw=1)
    ax.invert_xaxis()
    ax.grid(alpha=.16)
    ax.set(xlabel='Nominal cooling-window midpoint (K)',
           ylabel='Interval density response (g/cm³ per decade)')
    if len(local):
        ax.legend(loc='upper left', bbox_to_anchor=(1.02,1), frameon=False)
    fig.tight_layout()
    fig.savefig(out/'chi_density_rate_intervals_vs_T.png', dpi=300, bbox_inches='tight')
    fig.savefig(out/'chi_density_rate_intervals_vs_T.pdf', bbox_inches='tight')
    plt.close(fig)

    plotlines(agg,'rho_core_g_cm3','Core density (g/cm³)',out/'density_vs_T',RATE_LABEL)
    plotlines(agg,'h_gds_A','GDS slab thickness (Å)',out/'slab_thickness_gds_vs_T',RATE_LABEL)
    plotlines(agg,'awi_full_thickness_mean_A','Full AWI transition thickness (Å)',out/'thickness_vs_T',RATE_LABEL)
    plotlines(agg,'awi_full_thickness_mean_A','Full AWI transition thickness (Å)',out/'interface_thickness_vs_T',RATE_LABEL)
    plotlines(agg,'awi_full_thickness_lower_A','Lower AWI transition thickness (Å)',out/'awi_full_lower_vs_T',RATE_LABEL)
    plotlines(agg,'awi_full_thickness_upper_A','Upper AWI transition thickness (Å)',out/'awi_full_upper_vs_T',RATE_LABEL)
    plotlines(agg,'awi_inside_gds_mean_A','AWI extent inside GDS (Å)',out/'awi_inside_gds_vs_T',RATE_LABEL)
    plotlines(agg,'awi_outside_gds_mean_A','AWI extent outside GDS (Å)',out/'awi_outside_gds_vs_T',RATE_LABEL)
    plotlines(agg,'width_lower_A','Lower 10–90% width (Å)',out/'width_vs_T',RATE_LABEL)
    plotlines(agg,'width_upper_A','Upper 10–90% width (Å)',out/'width_upper_vs_T',RATE_LABEL)

    fig, ax = plt.subplots(figsize=(8.4, 5.1))
    d = response[response.metric=='rho_core_g_cm3'].sort_values('temperature_K')
    if len(d):
        ax.errorbar(d.temperature_K, d.chi_mean, yerr=d.chi_sd, fmt='o-', ms=4,
                    lw=1.8, capsize=2, label='Mean ± SD (independent replicas)')
        ax.legend(loc='upper left',bbox_to_anchor=(1.02,1),frameon=False)
    ax.axhline(0, color='gray',ls='--',lw=1)
    ax.invert_xaxis()
    ax.grid(alpha=.16)
    ax.set(xlabel='Nominal cooling-window midpoint (K)',
           ylabel='Density cooling-rate response (g/cm³ per decade)')
    fig.tight_layout()
    fig.savefig(out/'chi_density_vs_T.png', dpi=300, bbox_inches='tight')
    fig.savefig(out/'chi_density_vs_T.pdf', bbox_inches='tight')
    plt.close(fig)

    # Discrete minima: report location of lowest *observed* density in a supplied
    # temperature window, not a fitted transition or a statistically independent point.
    minima = []
    density = per[per.metric=='rho_core_g_cm3'].copy()
    for (rep, rate), g in density.groupby(['replica','rate_K_ns']):
        gg = g.sort_values('temperature_K')
        gg = gg[np.isfinite(gg.temperature_K) & np.isfinite(gg.value)]
        if gg.empty:
            continue
        k = gg.value.idxmin()
        row = gg.loc[k]
        cold = gg.loc[gg.temperature_K.idxmin()]
        minima.append(dict(replica=rep,rate_K_ns=rate, temperature_min_K=float(row.temperature_K),
                           density_min_g_cm3=float(row.value), coldest_T_K=float(cold.temperature_K),
                           coldest_rho_g_cm3=float(cold.value),
                           cold_minus_min_g_cm3=float(cold.value-row.value), n_temperatures=len(gg)))
    minima = pd.DataFrame(minima)
    minima.to_csv(out/'density_minimum_by_replica.csv',index=False)
    minstats = []
    for rate,g in minima.groupby('rate_K_ns'):
        tmu,tsd,tci,tn=summary_stats(g.temperature_min_K)
        rmu,rsd,rci,rn=summary_stats(g.density_min_g_cm3)
        dmu,dsd,dci,dn=summary_stats(g.cold_minus_min_g_cm3)
        minstats.append(dict(rate_K_ns=rate, n_rep=tn, minimum_T_mean_K=tmu,
                             minimum_T_sd_K=tsd, minimum_T_ci95_halfwidth_K=tci,
                             minimum_density_mean_g_cm3=rmu,minimum_density_sd_g_cm3=rsd,
                             cold_minus_min_mean_g_cm3=dmu,cold_minus_min_sd_g_cm3=dsd))
    minstats = pd.DataFrame(minstats)
    minstats.to_csv(out/'density_minimum_summary.csv',index=False)
    fig, ax = plt.subplots(figsize=(8.4,5.1))
    for rate,g in agg[agg.metric=='rho_core_g_cm3'].groupby('rate_K_ns'):
        g = g.sort_values('temperature_K')
        (line,) = ax.plot(g.temperature_K,g['mean'],marker='o',ms=3.50,lw=1.50,
                          color=color_for(rate),
                           label=f'{float(rate):g} K/ns')
        minrow = g.loc[g['mean'].idxmin()]
        ax.scatter([minrow.temperature_K],[minrow['mean']],marker='D',s=75,
                   color=line.get_color(),edgecolors='black',linewidths=.6,zorder=5)
    ax.invert_xaxis();ax.grid(alpha=.16)
    ax.set(xlabel='Nominal cooling-window midpoint (K)',ylabel='Core density (g/cm³)')
    ax.legend(loc='upper left',bbox_to_anchor=(1.02,1),frameon=False)
    fig.tight_layout()
    fig.savefig(out/'density_minima_vs_T.png',dpi=300,bbox_inches='tight')
    fig.savefig(out/'density_minima_vs_T.pdf',bbox_inches='tight')
    plt.close(fig)

    print(f'Analyzed {len(df)} metric entries; {len(missing)} missing outputs; results in {out}')
    print('Replica-level response:',out/'rate_response_by_replica.csv')
    print('Response with independent-replica uncertainty:',out/'rate_response_replica_mean.csv')
    print('Density minimum summary:',out/'density_minimum_summary.csv')
    print('Adjacent-rate responses:',out/'rate_response_intervals_summary.csv')

if __name__=='__main__':
    main()
