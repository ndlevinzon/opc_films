#!/usr/bin/env python3
"""Replot the normalized bulk and interface order parameter.

The claim is whether f~ reaches 0.5 at a higher temperature in the bulk.
Raw f_lda is not plotted. This script does not refit the classifier.
"""
import argparse
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from common.classifier import CLASSIFIER_ID, write_panel_record

METRICS = {
    'f_normalized': ('Normalized glass-like fraction, f~', 'f_normalized_bulk_vs_awi'),
    'free_volume_normalized': ('Normalized geometric free-volume fraction', 'free_volume_normalized_bulk_vs_awi'),
}
REGIONS = (('interior', '-', 'Bulk'), ('awi_3_12', '--', 'AWI 3–12 Å'))


def replot(df, metric, destination, band_sd=1.0):
    """Plot one normalized metric against temperature.

    Parameters
    ----------
    df : pandas.DataFrame
        ``region_summary.tsv`` from the quantitative classifier.
    metric : str
        Column prefix, ``f_normalized`` or ``free_volume_normalized``.
    destination : pathlib.Path
        Figure path without a suffix.
    band_sd : float
        Width of the replica standard-deviation band. Zero hides it.

    Returns
    -------
    None
        Writes a PNG and a PDF.
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
                            alpha=0.09 if region == 'interior' else 0.17, linewidth=0)
    ax.axhline(0.5, color='black', linewidth=0.80)
    ax.set_xlim(df.temperature_K.max()+2, df.temperature_K.min()-2)
    ax.set_xlabel('Temperature (K)')
    ax.set_ylabel(METRICS[metric][0])
    ax.grid(alpha=0.16)
    ax.legend(title='Spatial population', loc='upper left',
              bbox_to_anchor=(1.02, 1), frameon=False)
    fig.text(0.0, 0.0, "Shaded band: replica standard deviation. Frame-to-frame spread is a table column, not this band.", fontsize=8)
    fig.tight_layout()
    for suffix in ('.png', '.pdf'):
        fig.savefig(destination.with_suffix(suffix),
                    dpi=300 if suffix == '.png' else None, bbox_inches='tight')
    plt.close(fig)
    print(f'Wrote {destination}.png and {destination}.pdf')


def main():
    """Read ``region_summary.tsv`` and write the normalized figures.

    Returns
    -------
    None
        Writes the figures and ``classifier_panel.json``.
    """
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--outdir', type=Path, default=Path('outputs/core_awi_lda_free_volume'),
                   help='Folder containing region_summary.tsv; output plots are written here')
    p.add_argument('--summary', type=Path, default=None, help='Override summary TSV')
    p.add_argument('--band-sd', type=float, default=1., help='SD multiplier (0 hides bands)')
    a = p.parse_args()
    summary_path = a.summary or a.outdir/'region_summary.tsv'
    if not summary_path.is_file():
        p.error(f'Summary not found: {summary_path}')
    df = pd.read_csv(summary_path, sep='\t')
    required = {'temperature_K', 'region', 'n_replicas'} | {
        f'{metric}_{suffix}' for metric in METRICS for suffix in ('mean','sd')
    }
    missing = required - set(df.columns)
    if missing:
        p.error(f'Missing columns in {summary_path}: {sorted(missing)}')
    a.outdir.mkdir(parents=True, exist_ok=True)
    write_panel_record(
        a.outdir / 'classifier_panel.json',
        CLASSIFIER_ID,
        'quantitative',
        '300 K versus 78 K, |z| <= 15 Å, leave one replica out',
        'The drawn curves are f~ and the normalized free volume. Raw f_lda is not the bulk-first claim.',
    )
    for metric, (_, stem) in METRICS.items():
        replot(df, metric, a.outdir/stem, band_sd=a.band_sd)


if __name__ == '__main__':
    main()
