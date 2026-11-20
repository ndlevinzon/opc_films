# Figures

The manuscript panel letters are not in this repository. This table maps each figure module to the existing plotting script and to the reduced tables that script's numbers come from. A panel letter is filled in when the manuscript assigns one.

Phase A runs the script. Phase B reads only `summary_replicas.csv` and `contrast.csv` for that stage. The temperature axis runs hot to cold. Bands are the replica mean ± 1 SD. Contrasts are Student-t 95% intervals.

| Figure | Phase A script | Phase A output | Table | Metric | Region | Rows |
| --- | --- | --- | --- | --- | --- | --- |
| fig1 | `analysis/interface_dynamics/figure1_from_v2.py` | `results/figures/fig1` | `results/regional/summary_replicas.csv`, `results/regional/contrast.csv` | regional relaxation time | interior, awi | ramp and hold |
| fig2 | `analysis/interface_dynamics/figure2_from_v3.py` | `results/figures/fig2` | `results/hbond/summary_replicas.csv`, `results/hbond/contrast.csv` | hydrogen-bond retention | interior, awi | ramp and hold |
| fig3 | `analysis/cooling_path/06_plot_energetic_stabilization.py` | `results/figures/fig3` | `results/energy/summary_replicas.csv`, `results/energy/contrast.csv` | potential energy and energetic arrest | film | ramp |
| fig4 | `analysis/glass_transition/05b_plot_tg_vs_cooling_rate.py` | `results/figures/fig4.csv` | `results/tg/summary_replicas.csv`, `results/tg/contrast.csv` | dynamic glass temperature | film | hold and endpoint, one row per rate |
| fig5 | `analysis/glass_structure/opc_vitrification_spatial_workflow/08_replot_core_awi_black.py` | `results/classifier` | `results/classifier/summary_replicas.csv`, `results/classifier/contrast.csv` | LDA score and free volume | interior, awi | six endpoints, then holds and ramps |
| fig6 | manual: `analysis/glass_structure/plot_local_structure.py` | script `--out` | `results/local_order/summary_replicas.csv`, `results/local_order/contrast.csv` | q, S_k, LSI, CHILL+ | interior, awi | hold and endpoint |
| fig7 | `analysis/interface_dynamics/hydrogen_bonds/06_plot_hypothesis.py` | `results/figures/fig7` | `results/defects/summary_replicas.csv`, `results/defects/contrast.csv` | free OH and defect fraction | interior, awi | ramp and hold |

Fig. 6 is a manual step. `plot_local_structure.py` and `dipole_profile.py` take `--topology` and `--trajectory`, not a manifest row, and `figures.sh` does not call them. `05_plot_defect_fraction.py` reads `replica_means.csv` written by `06_plot_hypothesis.py`.
