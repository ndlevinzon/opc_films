#!/bin/bash
# Phase A figures. Each command is an existing plotting script.
set -euo pipefail
cd "$(dirname "$0")/../.."
mkdir -p results/figures
top=$(python -c "import pandas as pd; m=pd.read_csv('config/manifest.tsv', sep='\t', dtype=str, keep_default_na=False); print(next(value for value in m.topology if value))")
python analysis/interface_dynamics/figure1_from_v2.py \
  --manifest config/manifest.tsv --out results/figures/fig1 --rate 4.44
python analysis/interface_dynamics/figure2_from_v3.py \
  --manifest config/manifest.tsv --out results/figures/fig2 --rate 4.44
python analysis/cooling_path/06_plot_energetic_stabilization.py \
  --manifest config/manifest.tsv \
  --energy-map results/energy/energy_map.tsv \
  --topology "$top" \
  --out-prefix results/figures/fig3
python analysis/glass_transition/05b_plot_tg_vs_cooling_rate.py \
  results/tg/*_fit_summary.csv -o results/figures/fig4
python analysis/glass_structure/opc_vitrification_spatial_workflow/08_replot_core_awi_black.py \
  --outdir results/classifier
python analysis/interface_dynamics/hydrogen_bonds/06_plot_hypothesis.py \
  --manifest config/manifest.tsv --outroot results/defects --out results/figures/fig7
python analysis/interface_dynamics/hydrogen_bonds/05_plot_defect_fraction.py \
  --replica-means results/figures/fig7/replica_means.csv \
  --out results/figures/fig7_defects
