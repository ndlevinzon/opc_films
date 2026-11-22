#!/bin/bash
# Every figure of the paper, rebuilt from the reduced results.
#
#   results/figures/main/   manuscript panels, named Fig<N><panel>_<what>
#   results/figures/si/     SI panels
#   results/figures/work/   tables and diagnostic plots of the figure scripts
#   results/figures/INVENTORY.txt   every PNG and PDF, and what is missing
#
# The directory is cleared first, so no file from an earlier run remains.
# The work steps that later panels read must succeed. Every other panel is
# optional: a failure is logged to results/figures/FAILED.txt and the rest
# are still drawn.
set -euo pipefail
cd "$(dirname "$0")/../.."
R=results
FIG=$R/figures
rm -rf "$FIG"
mkdir -p "$FIG/main" "$FIG/si" "$FIG/work"
mkdir -p "$FIG/work/fs_curves" "$FIG/work/energetic_stabilization"
W=$FIG/work
top=$(python -c "import pandas as pd; m=pd.read_csv('config/manifest.tsv', sep='\t', dtype=str, keep_default_na=False); print(next(value for value in m.topology if value))")

optional() {
  local name=$1; shift
  mkdir -p "$W/logs"
  if ! "$@" > "$W/logs/${name}.log" 2>&1; then
    echo "$name: $*" >> "$FIG/FAILED.txt"
    echo "Figure step $name failed; see $W/logs/${name}.log" >&2
  fi
}
# copy <source without extension> <destination stem>: PNG and PDF if present.
copy() {
  local found=0
  for ext in png pdf; do
    if [ -f "$1.$ext" ]; then cp -p "$1.$ext" "$2.$ext"; found=1; fi
  done
  [ $found = 1 ] || echo "missing: $1.{png,pdf} for $(basename "$2")" >> "$FIG/FAILED.txt"
}

# ---- Work steps whose tables later panels read (required) -------------------
python analysis/interface_dynamics/figure1_from_v2.py \
  --manifest config/manifest.tsv --out "$W/regional_dynamics" --rate 4.44 --temps 235,215,195
python analysis/interface_dynamics/figure2_from_v3.py \
  --manifest config/manifest.tsv --out "$W/hbond" --rate 4.44 --temps 215,78
python analysis/interface_dynamics/hydrogen_bonds/06_plot_hypothesis.py \
  --manifest config/manifest.tsv --outroot $R/defects --out "$W/defects"

# ---- Main text --------------------------------------------------------------
M=$FIG/main
# Fig. 1b, 4, 5, 6, 7a-b, 7e
optional panels python analysis/figures/panels.py --results $R --out "$W/panels"
for f in "$W"/panels/Fig*.png "$W"/panels/Fig*.pdf; do [ -e "$f" ] && cp -p "$f" "$M/"; done
# Fig. 2
copy $R/cooling_summary/density_vs_T "$M/Fig2a_bulk_density"
copy $R/cooling_summary/chi_density_vs_T "$M/Fig2b_density_rate_response"
copy $R/cooling_summary/interface_thickness_vs_T "$M/Fig2c_AWI_thickness"
optional fig2d python analysis/cooling_path/06_plot_energetic_stabilization.py \
  --manifest config/manifest.tsv --energy-map $R/energy/energy_map.tsv \
  --topology "$top" --out-prefix "$W/energetic_stabilization/fig2d"
copy "$W/energetic_stabilization/fig2d" "$M/Fig2d_potential_energy"
copy $R/thermo/Cv_apparent_vs_T "$M/Fig2e_Cv_apparent"
copy $R/thermo/alpha_core_vs_T "$M/Fig2f_alpha_core"
# Fig. 3
optional fig3a python analysis/glass_transition/03d_plot_fs_curves.py \
  --fs-glob "$R/fs_tau/hold-*-4.44-*/fs_curve.csv" --temps 250 240 230 220 210 200 190 180 170 160 150 \
  --xscale log --out-prefix "$W/fs_curves/fig3a"
copy "$W/fs_curves/fig3a" "$M/Fig3a_Fs_curves"
copy $R/tg/4.44_arrhenius "$M/Fig3b_arrhenius"
copy $R/energy/compare "$M/Fig3c_Tg_vs_rate"
# Fig. 7c-d
copy "$W/defects/paired_p_defect_bulk_minus_AWI" "$M/Fig7c_defect_fraction_contrast"
copy "$W/defects/paired_exclusive_defect_decomposition_bulk_minus_AWI" "$M/Fig7d_defect_decomposition"

# ---- SI ---------------------------------------------------------------------
S=$FIG/si
for f in "$W"/panels/SI_*.png "$W"/panels/SI_*.pdf; do [ -e "$f" ] && cp -p "$f" "$S/"; done
optional defects_0to6 python analysis/interface_dynamics/hydrogen_bonds/05_plot_defect_fraction.py \
  --replica-means "$W/defects/replica_means.csv" --out "$S/SI_defect_fraction_0to6A"
optional defects_3to12 python analysis/interface_dynamics/hydrogen_bonds/05_plot_defect_fraction.py \
  --replica-means "$W/defects/replica_means.csv" --awi-inner 3 --awi-outer 12 --panel-label "" \
  --out "$S/SI_defect_fraction_3to12A"
optional sq_compare python analysis/glass_transition/02b_plot_sq.py \
  $R/qstar_row/*/sq.csv -o "$S/SI_Sq2D_compare.png"
optional fs_curves_all python analysis/glass_transition/03d_plot_fs_curves.py \
  --fs-glob "$R/fs_tau/*/fs_curve.csv" --out-prefix "$S/SI_fs_curves_all"
optional tau_replicas python analysis/glass_transition/04b_plot_tau.py \
  -i $R/tg/tau_all.csv --out-prefix "$S/SI_tau"
optional fig1c_contrast python analysis/interface_dynamics/plot_figure1c.py \
  --tau "$W/regional_dynamics/tau_censoring_corrected.csv" --out "$W/regional_contrast"
optional hb_onset python analysis/interface_dynamics/close_hb_onset.py \
  --hb-summary "$W/hbond/hb_intermittent_replica.csv" \
  --tau "$W/regional_dynamics/tau_censoring_corrected.csv" --out "$W/hb_onset"
optional regional_summary python analysis/interface_dynamics/plot_all.py \
  --corrected-dir "$W/regional_dynamics" --hb-dir "$W/hbond" \
  --onset-dir "$W/hb_onset" --out "$W/regional_summary" --no-strict
optional ftilde python analysis/glass_structure/opc_vitrification_spatial_workflow/08_replot_core_awi_black.py \
  --outdir $R/classifier
# SI copies of the plots above and of the plots earlier rules made.
for pair in \
  "$W/regional_dynamics/Fig1C_tau:SI_regional_tau" \
  "$W/regional_dynamics/Fig1D_decoupling:SI_regional_decoupling" \
  "$W/regional_contrast/Fig1C_region_relaxation_contrast:SI_regional_contrast" \
  "$W/hb_onset/FIG_HB_temperature_lag:SI_HB_retention_contrast_by_lag" \
  "$W/hbond/Fig2A_hbond_distribution_T215K:SI_HB_coordination_215K" \
  "$W/hbond/Fig2A_hbond_distribution_T79K:SI_HB_coordination_78K" \
  "$W/defects/depth_sensitivity_p_outOH_given_defect:SI_defect_depth_sensitivity" \
  "$R/energy/ramp:SI_ramp_energy" \
  "$R/classifier/f_normalized_bulk_vs_awi:SI_ftilde_bands" \
  "$R/classifier/free_volume_normalized_bulk_vs_awi:SI_free_volume_normalized" \
  "$R/thermo/alpha_thickness_vs_T:SI_alpha_thickness" \
  "$R/cooling_summary/chi_density_rate_intervals_vs_T:SI_chi_intervals" \
  "$R/cooling_summary/density_minima_vs_T:SI_density_minima" \
  "$R/cooling_summary/slab_thickness_gds_vs_T:SI_slab_thickness" \
  "$R/cooling_summary/width_vs_T:SI_interface_width"; do
  copy "${pair%%:*}" "$S/${pair##*:}"
done
for f in "$W"/regional_summary/V4_*.png; do
  [ -e "$f" ] && cp -p "$f" "$S/SI_$(basename "$f")"
done

# Every other plot the run made, so nothing is left outside results/figures.
for folder in tg energy thermo cooling_summary classifier; do
  [ -d "$R/$folder" ] || continue
  mkdir -p "$S/other/$folder"
  find "$R/$folder" -maxdepth 1 -type f \( -name '*.png' -o -name '*.pdf' \) -exec cp -p {} "$S/other/$folder/" \;
done

# ---- Inventory --------------------------------------------------------------
{
  echo "# Main text"; find "$M" -name '*.png' | sort | sed "s|$FIG/||"
  echo; echo "# SI"; find "$S" -name '*.png' | sort | sed "s|$FIG/||"
  if [ -f "$FIG/FAILED.txt" ]; then echo; echo "# Missing or failed"; cat "$FIG/FAILED.txt"; fi
  if [ -f "$W/panels/FAILED_panels.txt" ]; then echo; echo "# Panels skipped"; cat "$W/panels/FAILED_panels.txt"; fi
} > "$FIG/INVENTORY.txt"
echo "Main: $(find "$M" -name '*.png' | wc -l) panels. SI: $(find "$S" -name '*.png' | wc -l) panels. See $FIG/INVENTORY.txt"
exit 0
