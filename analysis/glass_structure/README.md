# Identifying the glass and where it forms

These scripts characterize the 300 K liquid and the 78 K glass, including the constant-temperature continuations, and the depth dependence of the order parameters. The quantitative discriminant and the 0.80 Å geometric probe are `opc_vitrification_spatial_workflow/07_core_awi_lda_and_free_volume.py`, which calls `common.classifier`. `render_nvt_temp_x_sections/` colors cross-sections and is display-only.

| Script | Role |
| --- | --- |
| `local_order.py` | Orientational tetrahedrality `q`, Chau–Hardwick translational tetrahedrality `Sk`, the local structure index, CHILL+ through OVITO, and depth-resolved in-plane structure factors. |
| `plot_local_structure.py` | Figures for one or both of `--condition cool` and `--condition warm`. The default is both. |
| `dipole_profile.py` | `<cos θ>` of the molecular dipole relative to the surface normal. |
| `plot_density_orientation.py` | Mass-density and orientation profiles on one set of axes. |

## Running `local_order.py`

`--topology` and `--trajectory` are required. Outputs are written after the process changes to `--out`. The distribution files are `q_per_water_distribution.csv`, `Sk_per_water_distribution.csv`, and `LSI_per_water_distribution.csv`. `plot_local_structure.py` also accepts the older `water_` and `warm_` names. Box lengths and the half-density surface are computed from the trajectory. A box that is not the OPC film is an error. `--stride` defaults to 50. On a 10 ps hold that is one configuration per 0.5 ns. The stride is written to `metadata.json`. `band_order_summary.csv` is the mean of `q`, `Sk`, LSI, and the CHILL+ fractions in the bulk, the 0–6 Å band, and the 3–12 Å band. The 10 Å z profiles are still written. CHILL+ is recentered on the circular-mean slab center. `--no-sq` and `--no-bragg` skip the structure-factor and Bragg diagnostics.

`q`, `Sk`, and LSI are the functions in `common.order`. They use the nearest neighbors with no distance cutoff, so a surface molecule is scored with distant neighbors, and LSI is missing when fewer than two neighbors lie within 3.7 Å. Interfacial order is low partly for that reason. The shaded band on a depth-resolved `S(q)` curve is the frame-to-frame width, not a replica uncertainty. The per-molecule spread on a z profile is labeled as a distribution width.

The joint `(q, Sk)` bars are normalized to the molecules inside `q >= 0.45` and `Sk >= 0.9965`. Each temperature writes its excluded fraction to `joint_window_exclusion.csv`. The 78 K angular Bragg figure is drawn only with the 300 K baseline on the same axes.

## Density plot

`plot_density_orientation.py` reads a cpptraj `density ... number` profile. The second column is a number density in Å^-3, and the mass density is that value times `M/N_A * 10^24`. A 300 K profile (`--temperature 300`) must give a bulk density within 0.10 g cm^-3 of 1. Displayed curves and the half-density surface use one Gaussian of width 1 Å. The equimolar surface is drawn and labeled equimolar.

```bash
python plot_density_orientation.py --density density.dat --cosine cos_theta.csv --out figure.png
python dipole_profile.py --topology opc_slab.prmtop --trajectory glass.nc --out dipole/
python plot_local_structure.py --data-dir /path/to/local_order_outputs --out figures
```
