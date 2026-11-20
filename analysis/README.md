# Analysis

Python 3.12. The numbered scripts are executed, not imported. Each one adds `analysis/` to `sys.path` and then imports `common`.

Shared code lives in `common/`:

| Module | Contents |
| --- | --- |
| `constants.py` | Cross-check film size, `q*`, the 15 Å bulk, and the two interface bands. |
| `rates.py` | Physical rates in K/ns, and the only map onto directory codes and file tags. |
| `order.py` | `q`, Chau–Hardwick `Sk`, LSI, and `radial_regularity`. |
| `trajectory.py` | NetCDF frame spacing, water count, and the film-size check. |
| `contrast.py` | Bulk minus interface. |
| `uncertainty.py` | Replica mean ± 1 SD, and paired Student-t 95% intervals. |
| `mdout.py` | Amber `mdout` parser (`parse_file`, `parse_dataframe`). |
| `slab.py` | Half-density surfaces, bulk and interface masks, and hydrogen-bond counts. |
| `classifier.py` | The one 300 K versus 78 K bulk score, leave-one-replica-out, and `f~`. |
| `manifest.py` | One table for ramps, holds, and endpoints. Output paths encode `kind`. |
| `paths.py` | `ensure_analysis_on_path`. |

## Directories

| Directory | Methods section |
| --- | --- |
| `preprocessing/` | Center the slab and write density profiles with cpptraj. |
| `cooling_path/` | Density, Gibbs surfaces, interfacial thickness, and potential energy along the ramp. |
| `glass_transition/` | `S_parallel(q)`, `F_s`, relaxation times, `T_g^dyn`, and the energetic arrest temperature. |
| `glass_structure/` | `q`, Chau–Hardwick `Sk`, LSI, CHILL+, dipole orientation, and depth-resolved structure factors. The classifier and the geometric probe are in `opc_vitrification_spatial_workflow/`. Cross-section coloring is in `render_nvt_temp_x_sections/` and is display-only. |
| `interface_dynamics/` | Regional translation, rotation, and translation–rotation decoupling. |
| `interface_dynamics/hydrogen_bonds/` | Coordination defects, donor–acceptor states, and intermittent partner retention. |

Set `OPC_DATA_ROOT` to the directory that contains `opc/` and `build/opc_slab.prmtop`. Array jobs take `--account` and `--partition` on the `sbatch` command and `CONDA_ENV` when a named conda environment should be activated.

## Names from the manuscript review

| Reviewed name | Where it is |
| --- | --- |
| `analyze_v2.py` | `interface_dynamics/analyze_v2.py`. `F_s`, `C_1`, and `C_2` use the normalized sum of the two O–H vectors, the bond bisector, not the OPC electrostatic dipole. |
| `v5_geometry.py` | Retired as a driver. `interface_dynamics/v5_geometry.py` only points at `hydrogen_bonds/03_analyze_free_oh.py`. |
| `check_frame_timing.py` | `interface_dynamics/check_frame_timing.py`. |
| `validate_and_plot.py` | `interface_dynamics/figure1_from_v2.py`. It checks coverage, stationarity, and temperature order, then writes the regional figures. |
| `plot_figure1c.py` | `interface_dynamics/plot_figure1c.py`. |
| `hbond_v3.py`, `figure2_from_v3.py`, `01_hb_temperature.py` | Same names in `interface_dynamics/`. |
| Onset, matched exchange, predictive tables | `close_hb_onset.py`, `02_matched_exchange.py`, and `03_nested_predictive.py`. `plot_all.py` reads them when `--onset-dir`, `--matched-dir`, or `--predictive-csv` is set. |
| `run_row_v6.py` | Retired. `hydrogen_bonds/02_run_row.py` runs `03_analyze_free_oh.py`, which writes the donor–acceptor tables. `04_analyze_hypothesis.py` is not a second trajectory pass. |
| `V6_bulk_minus_AWI_defect_contrast.py` | `hydrogen_bonds/06_plot_hypothesis.py` writes `paired_p_defect_bulk_minus_AWI`. `05_plot_defect_fraction.py` draws the two curves. |
| LDA classifier, 0.80 Å probe free volume | `common/classifier.py` and `glass_structure/opc_vitrification_spatial_workflow/07_core_awi_lda_and_free_volume.py`. The render scripts are display-only. |
| `C_V,app` and `α_app` | `cooling_path/opc_thermo_response/02_thermo_response.py`. |
| Main-text Fig. 2 assembly | No single assembly script. The panels are `glass_transition/01b_plot_temp_vs_cooling_rate.py`, `cooling_path/06_plot_energetic_stabilization.py`, `cooling_path/04_summarize.py`, and the derivative script above. |
