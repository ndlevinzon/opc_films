# Thermodynamic signatures along the cooling path

Each cooling window is one trajectory. The scripts label a window by the temperatures in its filename (`300_to_290K` and the same pattern for the other windows) and summarize the midpoint.

| Script | Role |
| --- | --- |
| `02_build_manifest.py` | Writes `outputs/manifest.tsv` and the same rows to `outputs/density_manifest.tsv`. |
| `03_run_row.py` | One manifest row: calls `01_analyze_chunk.py`. `--capwave` runs cpptraj `surftension` on a `kind=hold` row only. cpptraj is `--cpptraj` or the `cpptraj` on `PATH`. |
| `01_analyze_chunk.py` | Density profile, core density, Gibbs surfaces, and sampled `q`, `Sk`, and LSI. |
| `04_summarize.py` | Replica means, the rate sensitivity `chi`, and the interface thickness. |
| `05_capillary_modes.py` | Height of each interface and the autocorrelation of its Fourier modes. |
| `06_plot_energetic_stabilization.py` | Potential energy per molecule from Amber `mdout`, using `common.mdout`. |
| `opc_thermo_response/02_thermo_response.py` | Apparent `C_V` and the core-density and thickness expansion coefficients along the ramp. |
| `run_workflow.slurm` | Array job over the manifest. |

## Definitions used here

The core density is the mean mass density for `|z| <= 15` Å after the slab is recentered. The half-density surfaces are the interpolated crossings of half that density, from `common.slab.half_density_surfaces`. The interface thickness runs from `0.98 ρ_core` to `0.05 ρ_core` and is averaged over the two surfaces. `chi` is the change in core density between adjacent rates, divided by the change in `log10` of the cooling rate, inside each replica before averaging. `--rates` takes physical rates in K/ns.

`q`, Chau–Hardwick `Sk`, and LSI come from `common.order`. The Gaussian neighbor score is written as `radial_regularity`, not as `Sk`.

`03_run_row.py --capwave` runs cpptraj `surftension` only for a manifest row with `kind` equal to `hold`. A cooling window is rejected. `dt` is the spacing of the frames selected by `--stride`, read from the NetCDF `time` variable, and `bulkhalfwidth` is 15 Å. `blocktime` is one fifth of the hold length. The input is `capwave.cpptraj.in` in the chunk directory. The holds are added by `02_build_manifest.py --holds-tsv`, and endpoints by `--endpoints-tsv`. Either TSV has columns `label`, `replica`, `trajectory`, and `temperature_K`. Optional columns are `kind`, `rate_K_ns`, `source_restart`, `frame_ps`, and `mdout`. A finished `mdout` supplies `cut_A`. Directories `opc/<replica>/hold/<temperature>K/` and `opc/<replica>/endpoint/<temperature>K/` are discovered without a TSV. This archive has neither, so the cutoff check is not completed. The output directory is `outputs/<kind>/<replica>/<rate>Kns/<temperature>K/<label>`. `duration_ps` is the finished mdout `TIME(PS)` span. `discard_fraction` is 0.10 for a hold or an endpoint and 0 for a ramp. No finished hold mdout is in this archive, so `duration_ps` is blank until those logs are added. `05_capillary_modes.py` builds the same height field on a 2.5 Å grid, with 2.5 Å lateral smoothing, 1.5 Å vertical smoothing, and modes out to 0.30 Å^-1. The slab center is the circular mean from `slab_geometry`, and the frame spacing is the NetCDF `time` step. It does not convert the spectrum to a surface tension.

Centered trajectories are not pre-thinned. `preprocessing/extract_profiles.sh` keeps every frame unless `STRIDE` is set.

## Running

```bash
export OPC_DATA_ROOT=/path/to/data
python 02_build_manifest.py --root "$OPC_DATA_ROOT"
sbatch --account=ACCOUNT --partition=PARTITION --array=1-N run_workflow.slurm
```

Set `CONDA_ENV` to a conda prefix when `python` on the compute node is not the analysis environment. `CAPWAVE=1` turns on the interface-mode analysis. `OVERWRITE=1` recomputes finished rows.
