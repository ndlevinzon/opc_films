#!/bin/bash
# All figures, rebuilt from the reduced results into results/figures.
#
# 1. Main figures. A failure stops the rule.
# 2. SI figures drawn from pipeline outputs. A failure is logged to
#    results/figures/si/FAILED.txt and the rule continues.
# 3. Plots made by earlier rules (T_g, energy, cooling summary, classifier)
#    are copied in, so every figure of the run is in one directory.
#
# The directory is cleared first, so no file from an earlier run remains.
set -euo pipefail
cd "$(dirname "$0")/../.."
rm -rf results/figures
mkdir -p results/figures/si
top=$(python -c "import pandas as pd; m=pd.read_csv('config/manifest.tsv', sep='\t', dtype=str, keep_default_na=False); print(next(value for value in m.topology if value))")

# ---- 1. Main figures -------------------------------------------------------
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

# ---- 2. SI figures from pipeline outputs -----------------------------------
SI=results/figures/si
optional() {
  local name=$1; shift
  if ! "$@" > "$SI/${name}.log" 2>&1; then
    echo "$name: $*" >> "$SI/FAILED.txt"
    echo "SI figure $name failed; see $SI/${name}.log" >&2
  fi
}
optional sq_compare python analysis/glass_transition/02b_plot_sq.py \
  results/qstar_row/*/sq.csv -o "$SI/Sq2D_compare.png"
optional fs_curves python analysis/glass_transition/03d_plot_fs_curves.py \
  --fs-glob 'results/fs_tau/*/fs_curve.csv' --out-prefix "$SI/fs_curves"
optional tau_replicas python analysis/glass_transition/04b_plot_tau.py \
  -i results/tg/tau_all.csv --out-prefix "$SI/tau"
optional defects_3to12 python analysis/interface_dynamics/hydrogen_bonds/05_plot_defect_fraction.py \
  --replica-means results/figures/fig7/replica_means.csv \
  --awi-inner 3 --awi-outer 12 --panel-label "" \
  --out results/figures/fig7_defects_3to12A
optional fig1c_contrast python analysis/interface_dynamics/plot_figure1c.py \
  --tau results/figures/fig1/tau_censoring_corrected.csv --out "$SI/fig1c_contrast"
optional hb_onset python analysis/interface_dynamics/close_hb_onset.py \
  --hb-summary results/figures/fig2/hb_intermittent_replica.csv \
  --tau results/figures/fig1/tau_censoring_corrected.csv --out "$SI/hb_onset"
optional regional_summary python analysis/interface_dynamics/plot_all.py \
  --corrected-dir results/figures/fig1 --hb-dir results/figures/fig2 \
  --onset-dir "$SI/hb_onset" --out "$SI/regional_summary" --no-strict

# ---- 3. Plots made by earlier rules -----------------------------------------
collect() {
  local source=$1 name=$2
  [ -d "$source" ] || return 0
  mkdir -p "results/figures/$name"
  find "$source" -maxdepth 1 -type f \( -name '*.png' -o -name '*.pdf' \) \
    -exec cp -p {} "results/figures/$name/" \;
}
collect results/tg tg
collect results/energy energy
collect results/cooling_summary cooling_summary
collect results/classifier classifier

find results/figures -type f \( -name '*.png' -o -name '*.pdf' \) | sort > results/figures/INVENTORY.txt
echo "Figures: $(grep -c . results/figures/INVENTORY.txt) files listed in results/figures/INVENTORY.txt"
[ -s "$SI/FAILED.txt" ] && echo "Some SI figures failed: see $SI/FAILED.txt" >&2
exit 0
