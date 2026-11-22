# OPC free-standing film

Analysis and simulation inputs for a free-standing film of 139,000 OPC water molecules, cooled at constant volume from 300 K to 78 K.

The five cooling times are 2, 5, 10, 20, and 50 ns (111, 44.4, 22.2, 11.1, and 4.44 K/ns), with three replicas. Directory codes for those rates are `111`, `44`, `22`, `11`, and `4`.

## Layout

| Path | Contents |
| --- | --- |
| `MD/` | Minimization, equilibration, cooling, and constant-temperature continuation. |
| `analysis/preprocessing/` | cpptraj centering and density profiles. |
| `analysis/cooling_path/` | Density, interfacial width, and energy along each cooling window. |
| `analysis/glass_transition/` | In-plane structure factor, relaxation times, and both arrest temperatures. |
| `analysis/glass_structure/` | Tetrahedral order, CHILL+, dipole orientation, and depth-resolved structure factors. |
| `analysis/glass_structure/opc_vitrification_spatial_workflow/` | The 300 K versus 78 K classifier, `f~`, and the geometric free-volume probe. |
| `analysis/glass_structure/render_nvt_temp_x_sections/` | Display-only cross-sections. They call `common.classifier` and are not a second training protocol. |
| `analysis/interface_dynamics/` | Bulk versus interface relaxation, hydrogen bonds, and defects. |
| `src/opcfilm/` | Installable package. `config/project.yaml` is the parameter sheet. `analysis/common/` re-exports it. |
| `config/` | `project.yaml` and `manifest.tsv`. |
| `workflow/` | Snakemake stage graph and the `pipeline.py` fallback. |
| `tests/` | Synthetic tests. `pytest.ini` points them at `src/` and `analysis/`. |

## Data

`config/manifest.tsv` is the only manifest. A row's identity is `row_id` (`hold-r2-4.44-250`). Holds keep `rate_K_per_ns` empty and record the cooling rate they continue as `source_rate_K_per_ns`. Per-row outputs are `results/<stage>/<row_id>/`.

The three replicas start cooling from the same `min/step9.rst7` and differ by `ig = -1`. At 300 K that shared start decorrelates within picoseconds.

`s05` waits for `outputs/qstar.json`. `s02` reduces the relaxation times and is not a per-row stage. The classifier and the free-volume probe are one group job on all six endpoint rows.

Trajectory discovery uses `OPC_DATA_ROOT` or `--root`:

```text
$OPC_DATA_ROOT/opc/<r1|r2|r3>/ramp_nvt/<4|11|22|44|111>K_ns/TRAJ/
$OPC_DATA_ROOT/build/opc_slab.prmtop
```

Amber executables come from `AMBERHOME`. Cluster account, partition, and module names are passed on the `sbatch` command line rather than stored in the scripts. See `MD/README.md` and `analysis/README.md`.

`environment.txt` is the package list from the environment used to check this archive. It includes AmberTools, OVITO, netCDF4, ParmEd, and scikit-learn. Production `pmemd` is still the binary named by `AMBERHOME`. The package is released under the MIT License; see `LICENSE`.

## Shared definitions

The quantitative structural score is `analysis/common/classifier.py`: 300 K versus 78 K, every water inside `|z| <= 15` Å, 20 frames per replica, leaving one replica out. `f~` is that score scaled by the same region's own 300 K and 78 K values. `plot_color_LDA.py` (250 K versus 150 K) and `phase_maps_with_300_78.py` (one medoid, `|z| <= 45` Å) write `classifier_panel.json` with `role = display_only`. The 0.80 Å quantity is the geometric probe-center fraction in `07_core_awi_lda_and_free_volume.py`, not a pore volume. There is no `pLDA`.

One bulk is used for every regional comparison: `|z| <= 15` Å (`BULK_HALFWIDTH_A`). The two interface bands are depth below that frame's half-density surface: 0–6 Å for defects and free OH, and 3–12 Å for structure, dynamics, and the classifier. A contrast is bulk minus interface. Curves are the replica mean ± 1 standard deviation. Paired contrasts use a Student-t 95% interval. Command lines take the physical cooling rate (4.44, 11.1, 22.2, 44.4, or 111 K/ns).

`q`, Chau–Hardwick `Sk`, and the local structure index are computed in `src/opcfilm/structure/order.py`. The Gaussian neighbor score is named `radial_regularity` and is not `Sk`. Water counts and box lengths are read from the topology and the frame. The numbers in `config/project.yaml` are the parameter sheet. `common/constants.py` loads that file. The scripts stop if a trajectory disagrees with the film size in the sheet.

`opcfilm synthetic` runs every stage on a 2,000-water film. `opcfilm catalog build --root $OPC_DATA_ROOT` writes `config/manifest.tsv`. OVITO is an optional dependency (`opcfilm[chill]`) because the pinned build is the conda package.
