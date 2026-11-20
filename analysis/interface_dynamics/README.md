# Molecular motion in the bulk and at the interface

`plot_all.py` draws the regional comparison from tables that have already been reduced. It does not read a trajectory and it does not extrapolate a relaxation time past the last stored lag. A time that never crossed `1/e` is plotted as a bound. Paired bulk-minus-interface differences use the same replica and the same temperature. The bulk is a solid line and the interface is dashed.

`figure1_from_v2.py` writes `replica_curves.csv` (with `temperature_K`), `tau_censoring_corrected.csv`, `tau_summary_corrected.csv`, and `decoupling_paired_resolved.csv` into its `--out` directory. Point `--corrected-dir` at that directory. The censoring labels are `resolved`, `fast_censored`, and `slow_censored`. `decoupling_paired_resolved.csv` contains `Fs`, `C1`, and `C2` in picoseconds, and both `D_TR,1 = log10(C1/Fs)` and `D_TR,2 = log10(C2/Fs)`, only for rows in which all three times are resolved. `plot_all.py` stops when one of those tables is missing. `--no-strict` turns that check off. Hydrogen-bond panels read `hb_coord_distributions.csv` and `hb_intermittent_replica.csv` when `--hb-dir` is set. Optional onset and matched-exchange tables are read from `--onset-dir` and `--matched-dir`. Missing optional files are reported and left out of the figure.

The decoupling parameter is `log10(τ_rot / τ_alpha)` and is computed only when both times are resolved. Regional differences are bulk minus interface. `--rate` is the physical rate in K/ns. Output names keep the `V4_` prefix, with paired contrasts named `*_bulk_minus_AWI*`.

`check_frame_timing.py` reads the NetCDF `time` variable and rejects a nonuniform step. `figure1_from_v2.py` is the validation and figure step for those tables. `plot_figure1c.py` is the paired bulk-over-interface time.

`analyze_v2.py` assigns regions from the frame's half-density depth: bulk `|z| <= 15` Å, subsurface 3–12 Å (`awi`), and outermost 0–6 Å (`awi_outer`). The frame spacing is the NetCDF `time` step. A cooling window drops about 10 K. `analyze_v2.py` and `hbond_v3.py` keep lags that mix at most 2 K, which is 450 ps at 4.44 K/ns. `close_hb_onset.py` drops requested lags above that cap. `--hold` turns the cap off and, unless `--lags-ps` is set, uses lags out to 25% of the hold, the same fraction as `03_fs_alpha.py`. `hbond_v3.py` keeps lags through 640 ps because retention is compared at 320 ps. The 150–250 K holds are the runs that can resolve a bulk `C_1` time of about 0.5 ns. Cooling windows remain appropriate at 230 K and above, where the relaxation time is much shorter than 450 ps. Hold frames are 10 ps apart, so lags of 2–8 ps are absent. The cap is the same in both regions. `plot_all.py --hb-temps` draws the coordination histogram at the requested temperatures; the default is 215 K and 78 K, each matched to the nearest stored temperature and named with that temperature.

```bash
python plot_all.py --corrected-dir corrected/ --hb-dir hbonds/ --out figures --rate 4.44
```

Hydrogen-bond lifetimes, defect fractions, and donor–acceptor states are computed in `hydrogen_bonds/`.
