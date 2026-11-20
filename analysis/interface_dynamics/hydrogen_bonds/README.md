# Hydrogen-bond rearrangement and network defects

A hydrogen bond is an oxygen–oxygen pair closer than 3.5 Å with an O–H···O angle greater than 150°. The geometry is `common.slab.hydrogen_bond_counts`. This directory counts free OH groups and donor–acceptor defects. It does not compute intermittent retention. That correlation is `../hbond_v3.py`.

The defect fraction is the fraction of molecules whose total coordination is not 4. Donor–acceptor states are labeled `DmAn` from the number of bonds donated and accepted. `06_plot_hypothesis.py` subtracts the interface from the bulk (`bulk_minus_awi`). A negative value is an excess at the interface.

| Script | Role |
| --- | --- |
| `01_build_manifest.py` | Tab-separated manifest from `$OPC_DATA_ROOT` or `--root`. |
| `02_run_row.py` | One manifest row. This replaces `run_row_v6.py`. Skips a chunk when both band directories already have `free_oh_metadata.json`, unless `--overwrite` is set. |
| `03_analyze_free_oh.py` | One pass. Writes free-OH tables and donor–acceptor tables for both bands. |
| `04_analyze_hypothesis.py` | Functions used by `03`. Running it directly only points back at `03`. |
| `05_plot_defect_fraction.py` | Defect-fraction figures. The main band is 0–6 Å. |
| `06_plot_hypothesis.py` | Donor–acceptor figures. Contrasts are bulk minus interface, including `paired_p_defect_bulk_minus_AWI`. This replaces `V6_bulk_minus_AWI_defect_contrast.py`. |
| `run_array.sh` | Array job. It does not override the band, the bulk, or the overlap rule. |

The bulk is `|z| <= 15` Å. Overlap with either interface band is an error. Each band is written under its own directory, `awi_0-6/` and `awi_3-12/`. The 0–6 Å band is the main defect and free-OH comparison. The 3–12 Å band is the sensitivity check for those quantities and the main band for structure. `--rates` on the manifest builder is a physical rate in K/ns.

Each band directory contains `free_oh_frame_summary.csv`, `free_oh_joint_coordination.csv`, `free_oh_orientation.csv`, `free_oh_qc.csv`, `free_oh_metadata.json`, `hypotheses_frame.csv`, `hypotheses_classes.csv`, and `hypotheses_acceptor_orientation.csv`.

```bash
export OPC_DATA_ROOT=/path/to/data
python 01_build_manifest.py --root "$OPC_DATA_ROOT" --out outputs/trajectory_manifest.tsv
sbatch --account=ACCOUNT --partition=PARTITION --array=1-N run_array.sh
```
