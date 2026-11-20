# Locating the glass transition

Two arrest temperatures are computed independently. `T_g^dyn` comes from the in-plane relaxation time. The energetic arrest temperature comes from the change in slope of the potential energy.

| Script | Role |
| --- | --- |
| `01_parse_mdout.py` | Amber `mdout` to a table of temperature and energy. |
| `01b_plot_temp_vs_cooling_rate.py` | Temperature against time for each rate and replica. |
| `02_sq2_fft.py` | In-plane oxygen structure factor. The FFT grid is `round(L/0.5)`. The histogram form factor is divided out, and the peak is a quadratic fit within 0.15 Å^-1 of the maximum inside `--peak-min` and `--peak-max`. A frame bootstrap is written with the peak. The adopted value is `outputs/qstar.json`. |
| `02b_plot_sq.py` | Averages and plots those curves. |
| `03_fs_alpha.py` | In-plane self-intermediate scattering function at the adopted `q*` in `outputs/qstar.json`. A `-q` that disagrees with that file is an error. Reads `--manifest` and `--row`. A hand-entered path requires `--allow-unlisted`. Writes `tau_status` and `frame_ps_selected`. |
| `03b_plot_relaxation.py` | Relaxation curves and `_classified.csv`. |
| `03d_plot_fs_curves.py` | Additional `F_s` figures. |
| `04_collect_tau.py` | Collects the `03` summaries. Copies `tau_status` to `status`, or merges `--classified` from `03b`. |
| `04b_plot_tau.py` | Relaxation time against temperature. |
| `05_fit_relaxation_tg.py` | Arrhenius fit of the resolved times from 210 to 230 K, then the cooling-rate condition `τ(T_g) = T_g^2 / (R_c B)`. Each temperature keeps its own `frame_ps_selected`. The 230 K half-frame shift uses that temperature's spacing. It requires the `status` column written by `04`. |
| `05b_plot_tg_vs_cooling_rate.py` | Reads those fit tables and plots `T = a + b log10(R)`. It does not contain a hardcoded list of transition temperatures. |
| `06_plot_ramp_energy.py` | Potential energy along the ramp and the smooth change in slope. The arrest temperature is the 10% departure from the high-temperature slope. |
| `06b_fit_energy_tg.py` | Intersection of a straight glass line and a straight liquid line. |
| `06d_compare_dynamic_energetic.py` | `ΔT(R_c) = T_arrest - T_g^dyn` and the logarithmic slope of each temperature. |

Run the relaxation pipeline in this order: `03_fs_alpha.py`, then `04_collect_tau.py`, then `05_fit_relaxation_tg.py`. `03b_plot_relaxation.py` draws the curves and can supply `--classified` to `04`. `04` copies `tau_status` onto `status` when that column is already in the `03` summaries, so `05` still receives a status column if `03b` was not run. `03d_plot_fs_curves.py` and `04b_plot_tau.py` are figures and are not required for the fit.

`q*` is the adopted value in `outputs/qstar.json` for every temperature, rate, and region. `03_fs_alpha.py` and `analyze_v2.py` read that file and stop if it is missing. Run `03_fs_alpha.py` once on the hold trajectories. The in-plane displacement uses the minimum image, so a re-imaged trajectory is acceptable. Pass `--stride 1` on a file that still contains every frame. A file that was already thinned must not be strided again. It reads the frame spacing from the NetCDF `time` variable, and the block rows come from that same run. Do not merge a second high-temperature series. The 230 K time is about two hold frames. `05_fit_relaxation_tg.py` reads `frame_ps_selected` per temperature. A 1 ps recomputation at 220–230 K can sit in the same table as 10 ps holds. The half-frame shift uses the spacing on the 230 K rows. Holds and endpoints drop the first 10% unless `--discard-ns` is set. The dropped time is `discard_ps` in the summary. `--cooling-rate` must be a physical rate. Recomputing `F_s` at 220–230 K from 1 ps output, or from the 2 ps cooling-window frames, is still required. Censored relaxation times stay in the tables as bounds and are excluded from the fits. Curves show the replica mean ± 1 standard deviation. Contrasts use a paired Student-t 95% interval.

With three replicas the bootstrap has ten distinct resamples at each temperature, so its interval measures replica agreement. `05_fit_relaxation_tg.py` writes `_arrhenius_window_sensitivity.csv` with `T_g^dyn` and `E_a` for windows starting at 210, 200, 190, and 180 K, and only at temperatures where all three replicas are resolved. That window choice is the larger uncertainty. `03b_plot_relaxation.py` does not write a second window table.

The smooth energetic fit is only `06_plot_ramp_energy.py`. It uses a soft-L1 loss with `f_scale = 0.001` kcal mol^-1. Each replica row records `RMSE / f_scale` and the name of any parameter sitting on a bound. `06b_fit_energy_tg.py` is the separate two-line intersection and is not that smooth fit. Energy scripts read the water count from `--topology`. Uncertainties on the energetic arrest temperature are a Monte Carlo over the replica fits.

Replica energy tables read by `01b_plot_temp_vs_cooling_rate.py` are named `ramp_{111,44p4,22p2,11p1,4p44}_r{n}.csv`.
