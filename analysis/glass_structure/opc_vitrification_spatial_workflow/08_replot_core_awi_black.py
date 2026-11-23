#!/usr/bin/env python3
"""Replot the normalized bulk and interface order parameter, and find where it reaches 0.5.

The claim is whether f~ reaches 0.5 at a higher temperature in the bulk.
Raw f_lda is not plotted. This script does not refit the classifier.

The classifier scores the cooling windows of every rate as well as the holds
and endpoints. Only the holds (and the two endpoints that anchor f~ at 0 and
1) are plotted, so cooling windows of different rates are not drawn as one
series. When ``frame_region_metrics.tsv`` is present the replica means are
rebuilt from it with the cooling rate in the key, which also corrects a
``region_summary.tsv`` written before the rate was part of its key.

Outputs in ``--outdir``:

* ``f_normalized_bulk_vs_awi.{png,pdf}`` and
  ``free_volume_normalized_bulk_vs_awi.{png,pdf}``;
* ``hold_region_replica_means.csv``: replica means on the holds and endpoints;
* ``half_point_by_replica.csv``: temperature at which each replica's curve
  crosses 0.5, per region and metric (linear interpolation between the two
  bracketing temperatures, scanning from hot to cold);
* ``half_point_summary.csv``: mean and SD over replicas, and the paired
  bulk minus band difference with a Student-t 95% interval (SI table
  tab:si_ftilde_half).
"""
import argparse
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from opcfilm.plotting.conventions import install_figure_conventions
install_figure_conventions()  # temperature axes hot to cold; legends outside panels
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from common.classifier import CLASSIFIER_ID, write_panel_record

METRICS = {
    'f_normalized': ('Normalized glass-like fraction, f~', 'f_normalized_bulk_vs_awi'),
    'free_volume_normalized': ('Normalized geometric free-volume fraction', 'free_volume_normalized_bulk_vs_awi'),
}
REGIONS = (('interior', '-', 'Bulk'), ('awi_3_12', '--', 'AWI 3–12 Å'))
BULK = 'interior'
BANDS = ('awi_3_12', 'awi_2_10', 'awi_4_14')
DEFAULT_KINDS = ('hold', 'endpoint')


def replica_means(frames, kinds):
    """Replica means per kind, rate, temperature, and region from frame rows."""
    frames = frames[frames['kind'].isin(kinds)].copy()
    if 'rate_K_ns' not in frames:
        frames['rate_K_ns'] = ''
    frames['rate_K_ns'] = frames['rate_K_ns'].fillna('').astype(str)
    keys = ['kind', 'rate_K_ns', 'temperature_K', 'region', 'replica']
    columns = [m for m in METRICS if m in frames]
    for column in columns:
        frames[column] = pd.to_numeric(frames[column], errors='coerce')
    return frames.groupby(keys, as_index=False)[columns].mean()


def summarize(per_replica):
    """Mean, SD and replica count per temperature and region."""
    rows = []
    for (temperature, region), part in per_replica.groupby(['temperature_K', 'region']):
        record = {'temperature_K': float(temperature), 'region': region, 'n_replicas': part['replica'].nunique()}
        for metric in METRICS:
            values = part[metric].to_numpy(float)
            values = values[np.isfinite(values)]
            record[f'{metric}_mean'] = values.mean() if values.size else np.nan
            record[f'{metric}_sd'] = values.std(ddof=1) if values.size > 1 else np.nan
        rows.append(record)
    return pd.DataFrame(rows)


def replot(df, metric, destination, band_sd=1.0):
    """Plot one normalized metric against temperature for bulk and the 3-12 Å band.

    Parameters
    ----------
    df : pandas.DataFrame
        Summary with ``temperature_K``, ``region``, ``<metric>_mean`` and ``<metric>_sd``,
        restricted to the holds and endpoints.
    metric : str
        ``f_normalized`` or ``free_volume_normalized``.
    destination : pathlib.Path
        Figure path without a suffix.
    band_sd : float
        Width of the replica standard-deviation band. Zero hides it.
    """
    fig, ax = plt.subplots(figsize=(8.4, 5.1))
    for region, linestyle, label in REGIONS:
        d = df[df['region'] == region].sort_values('temperature_K')
        if d.empty:
            raise ValueError(f'No {region} values in the summary')
        x = d['temperature_K'].to_numpy(float)
        y = d[f'{metric}_mean'].to_numpy(float)
        sd = d[f'{metric}_sd'].to_numpy(float)
        n = d['n_replicas'].dropna().unique() if 'n_replicas' in d else []
        nlabel = f' (n={int(n[0])} replicas)' if len(n) == 1 else ''
        ax.plot(x, y, color='black', linestyle=linestyle, marker='o',
                markersize=4, linewidth=1.9, label=label+nlabel)
        if band_sd > 0 and np.isfinite(sd).any():
            good = np.isfinite(sd)
            ax.fill_between(x, y - band_sd*np.nan_to_num(sd), y + band_sd*np.nan_to_num(sd),
                            where=good, color='black',
                            alpha=0.09 if region == BULK else 0.17, linewidth=0)
    ax.axhline(0.5, color='black', linewidth=0.80)
    ax.set_xlim(df.temperature_K.max()+2, df.temperature_K.min()-2)
    ax.set_xlabel('Temperature (K)')
    ax.set_ylabel(METRICS[metric][0])
    ax.grid(alpha=0.16)
    ax.legend(title='Spatial population', loc='upper left',
              bbox_to_anchor=(1.02, 1), frameon=False)
    fig.text(0.0, 0.0, "Holds and the 300/78 K endpoints. Shaded band: replica standard deviation.", fontsize=8)
    fig.tight_layout()
    for suffix in ('.png', '.pdf'):
        fig.savefig(destination.with_suffix(suffix),
                    dpi=300 if suffix == '.png' else None, bbox_inches='tight')
    plt.close(fig)
    print(f'Wrote {destination}.png and {destination}.pdf')


def half_point(temperature, value, level=0.5):
    """First temperature, scanning from hot to cold, at which ``value`` reaches ``level``.

    Linear interpolation between the bracketing points. NaN if the curve never
    reaches the level, or already exceeds it at the hottest point.
    """
    order = np.argsort(-np.asarray(temperature, float))
    t = np.asarray(temperature, float)[order]
    v = np.asarray(value, float)[order]
    good = np.isfinite(t) & np.isfinite(v)
    t, v = t[good], v[good]
    if t.size < 2 or v[0] >= level:
        return np.nan
    for i in range(1, t.size):
        if v[i] >= level:
            return float(np.interp(level, [v[i - 1], v[i]], [t[i - 1], t[i]]))
    return np.nan


def half_points(per_replica):
    """Half-point temperature of every replica, region, and metric, plus the summary."""
    rows = []
    for (replica, region), part in per_replica.groupby(['replica', 'region']):
        for metric in METRICS:
            rows.append({'replica': replica, 'region': region, 'metric': metric,
                         'T_half_K': half_point(part['temperature_K'], part[metric])})
    by_replica = pd.DataFrame(rows)
    summary = []
    for (metric, region), part in by_replica.groupby(['metric', 'region']):
        values = part['T_half_K'].dropna().to_numpy(float)
        record = {'metric': metric, 'region': region, 'n_replicas': values.size,
                  'T_half_mean_K': values.mean() if values.size else np.nan,
                  'T_half_sd_K': values.std(ddof=1) if values.size > 1 else np.nan}
        if region != BULK:
            bulk = by_replica[(by_replica.metric == metric) & (by_replica.region == BULK)].set_index('replica')['T_half_K']
            band = part.set_index('replica')['T_half_K']
            paired = (bulk - band).dropna()
            record['bulk_minus_band_mean_K'] = paired.mean() if len(paired) else np.nan
            if len(paired) > 1:
                half = stats.t.ppf(0.975, len(paired) - 1) * paired.std(ddof=1) / np.sqrt(len(paired))
                record['bulk_minus_band_ci95_low_K'] = paired.mean() - half
                record['bulk_minus_band_ci95_high_K'] = paired.mean() + half
            record['n_paired'] = len(paired)
        summary.append(record)
    return by_replica, pd.DataFrame(summary)


def main():
    """Read the classifier outputs and write the normalized figures and half points."""
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--outdir', type=Path, default=Path('outputs/core_awi_lda_free_volume'),
                   help='Folder containing region_summary.tsv; output plots are written here')
    p.add_argument('--summary', type=Path, default=None, help='Override summary TSV')
    p.add_argument('--frames', type=Path, default=None,
                   help='frame_region_metrics.tsv (default: next to the summary). Used when present.')
    p.add_argument('--kinds', nargs='+', default=list(DEFAULT_KINDS),
                   help='Manifest kinds to plot (default: hold endpoint).')
    p.add_argument('--band-sd', type=float, default=1., help='SD multiplier (0 hides bands)')
    a = p.parse_args()
    summary_path = a.summary or a.outdir/'region_summary.tsv'
    frames_path = a.frames or summary_path.with_name('frame_region_metrics.tsv')
    a.outdir.mkdir(parents=True, exist_ok=True)
    if frames_path.is_file():
        frames = pd.read_csv(frames_path, sep='\t', dtype={'rate_K_ns': str})
        per_replica = replica_means(frames, a.kinds)
        source = frames_path
    else:
        if not summary_path.is_file():
            p.error(f'Neither {frames_path} nor {summary_path} exists')
        per_replica = None
        source = summary_path
    if per_replica is not None:
        per_replica.to_csv(a.outdir / 'hold_region_replica_means.csv', index=False)
        df = summarize(per_replica)
        by_replica, half = half_points(per_replica)
        by_replica.to_csv(a.outdir / 'half_point_by_replica.csv', index=False)
        half.to_csv(a.outdir / 'half_point_summary.csv', index=False)
        print(half.to_string(index=False))
    else:
        df = pd.read_csv(summary_path, sep='\t')
        if 'kind' in df:
            df = df[df['kind'].isin(a.kinds)]
        print(f'No {frames_path}: plotted {summary_path} without half points.', file=sys.stderr)
    missing = {f'{metric}_{suffix}' for metric in METRICS for suffix in ('mean', 'sd')} - set(df.columns)
    if missing:
        p.error(f'Missing columns in {source}: {sorted(missing)}')
    write_panel_record(
        a.outdir / 'classifier_panel.json',
        CLASSIFIER_ID,
        'quantitative',
        '300 K versus 78 K, |z| <= 15 Å, leave one replica out',
        'The drawn curves are f~ and the normalized free volume on the holds. Raw f_lda is not the bulk-first claim.',
    )
    for metric, (_, stem) in METRICS.items():
        replot(df, metric, a.outdir/stem, band_sd=a.band_sd)


if __name__ == '__main__':
    main()
