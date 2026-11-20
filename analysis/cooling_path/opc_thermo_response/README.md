# OPC cooling-path heat capacity and thermal expansion

Reuse existing **corrected density outputs** and `density_manifest.tsv`. No MD re-runs for alpha. Heat capacity **requires AMBER MDOUT energy logs** mapped one-to-one to the exact trajectory chunks; no attempt to guess log paths.

## Install and run

```bash
python 02_thermo_response.py \
  --density-manifest outputs/density_manifest.tsv \
  --out outputs/thermo_alpha --window 3
```

For heat capacity, create tab-separated `energy_map.tsv` with exact headers `trajectory<TAB>mdout`, mapping each full absolute trajectory filename in the density manifest to its corresponding **actual AMBER MDOUT output file**. Do not use slurm stdout unless it includes AMBER energy records; an `.out` name is not proof of format. An MDOUT from an entire run must not be assigned to several 10-K chunks.

Example (replace both paths with the real existing paths):

```
trajectory	mdout
/absolute/path/to/cool_50ns_chunk_00_300_to_290K.nc	/absolute/path/to/the_matching_amber_run.out
```

Test an MDOUT parser on one known file:

Energies are read by `common.mdout`. That parser stops at the Amber `AVERAGES`, `RMS FLUCTUATIONS`, and `FINAL RESULTS` blocks.

Then run:

```bash
python 02_thermo_response.py \
  --density-manifest outputs/density_manifest.tsv \
  --energy-map energy_map.tsv \
  --topology "$OPC_DATA_ROOT/build/opc_slab.prmtop" \
  --out outputs/thermo_response --window 3
```

Inspect `input_qc.csv` and reject missing/mismatched energy chunks. The mean mdout temperature must be within 3 K of the window midpoint, and the `nstep` span must match the trajectory frame count times `ntwx`. No files are overwritten in the density workflow.

## Outputs

- `thermo_chunk_means.csv`: per-chunk density, GDS thickness, and (if mapped) potential energy
- `thermo_derivatives_by_replica.csv`: per-replica derivatives for the main window (default 3 points, 20 K)
- `thermo_derivatives_window5_by_replica.csv`: the same derivatives on a 5-point, 40 K window
- `thermo_response_replica_summary.csv`: between-replica means, SD, t-CI of the main window
- `alpha_core_vs_T.png/pdf`: alpha_core,app = -(1/rho_core) d rho_core/dT
- `alpha_thickness_vs_T.png/pdf`: alpha_h,app = d ln h_GDS/dT (valid for fixed lateral area)
- `Cv_apparent_vs_T.png/pdf`: configurational (1/N) dEPtot/dT, in kcal mol^-1 K^-1. Add 3R for the total heat capacity of rigid water.
- `alpha_consistency_by_replica.csv`: compare density and thickness estimates
- `input_qc.csv`: checked inputs and mapping status

Plots have **hot -> cold (left to right)**, rate colors, SD across independent replicas, and external legends. The plotted derivative uses `--window` (default 3). Window 5 is written beside it because a 40 K slope is as wide as the glass transition and flattens the peak.

## Crucial physical limitations

- All ramp derivatives are **apparent path derivatives**, not stationary equilibrium Cv or alpha. No NPT expansion coefficient is inferred from an NVT simulation.
- For a fixed-area film, `alpha_thickness` is the film thickness response, not a bulk NPT volumetric thermal-expansion coefficient. `alpha_core` is an effective core-density derivative at fixed lateral dimensions, strongly dependent on interface geometry and choice of core.
- Each reported replica estimate is independent; adjacent temperature chunks from the same replica are serially correlated. Confidence intervals only describe between-replica variation, not all protocol/systematic errors.
- `C_V,app` is `(1/N) dEPtot/dT`, the same potential energy used for the energetic arrest. It is configurational. Add `3R` for the total heat capacity of rigid water. The unit on the figure is kcal mol^-1 K^-1.
- The 4/11/22/44/111 directory codes correspond to **actual rates 4.44/11.1/22.2/44.4/111 K/ns**.
- Do not compare per-water classical rigid-model heat capacity uncritically with experimental Cp.

To create a safe blank mapping for all ramp chunks:

```bash
python 03_make_energy_map_template.py \
  --density-manifest "$PWD/outputs/density_manifest.tsv" \
  --out "$PWD/energy_map.tsv"
```

Fill the `mdout` column with the exact log for each chunk. Blank rows remain density-only and are explicitly flagged in `input_qc.csv`; they cannot contribute to a heat-capacity derivative. If a single MDOUT file covers multiple temperature windows, split it by verified time/step ranges first rather than mapping the full file to each window.

## File naming for this project's cooling runs

Actual energy files are `OUT/cool_5ns_chunk_00_300_to_290K.mdout`, while
trajectories may be `TRAJ/cool_50ns_chunk_00_300_to_290K.nc`. The mapper
accepts `*.mdout` and `*.out` and prefers **chunk number plus start/end
temperatures**, so the differing `5ns`/`50ns` prefix does not prevent matching.

Regenerate the map:

```bash
python 03_make_energy_map_template.py \
 --density-manifest "$PWD/outputs/density_manifest.tsv" \
 --out "$PWD/energy_map.tsv"
awk -F'\t' 'NR>1{n[$5]++} END{for(s in n)print s,n[s]}' energy_map_review.tsv
```

**Check actual MDOUT temperature and step coverage:** the name match is not
proof that the energy log and NetCDF cover the same physical interval.
