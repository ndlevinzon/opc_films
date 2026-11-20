# Bulk versus interface order parameter

The quantitative classifier is `common.classifier`. Its id is
`qsk_lsi_shared_covariance_300_vs_78_bulk15A_leave_one_replica`.

`plot_color_LDA.py` trains on one 250 K frame and one 150 K frame inside
`|z| <= 45` Å. `phase_maps_with_300_78.py` trains on one medoid from each
concatenated endpoint, also inside `|z| <= 45` Å. Both write
`classifier_panel.json` with `role = display_only`. Neither number is a
transition temperature.

## What is fit

Liquid is 300 K and glass is 78 K. Training waters are every oxygen with
`|z| <= 15` Å (`BULK_HALFWIDTH_A`), on 20 evenly spaced frames of a
per-replica endpoint. A concatenated `300K_prop_analysis` file has no
replica label and is refused. Each replica is scored by the model trained
on the other two. `leave_one_replica_out.tsv` reports the score
distribution and the misclassification rate at both endpoints.

Features are `q`, Chau–Hardwick `Sk`, and the local structure index from
`common.order`. Neighbors have no distance cutoff, so a molecule under the
surface has a thinned outer shell and scores more liquid-like. The box is
the frame's own box. The center is the circular mean from `slab_geometry`.

The score is the equal-prior shared-covariance discriminant. A high score
is glass-like. Endpoint scores are not forced to 0 or 1.

## What is reported

`f` is the fraction of waters with score at least 0.5. For each region,

```text
f~(T) = [f(T) - f(300 K)] / [f(78 K) - f(300 K)]
```

The same normalization is applied to the geometric free-volume fraction.
A bulk-first claim is that `f~` reaches 0.5 at a higher temperature in the
bulk than in the 3–12 Å band, with the replica standard deviation. It is
not that raw `f` is larger in the bulk.

`08_replot_core_awi_black.py` plots `f~` and the normalized free volume
from `region_summary.tsv` and draws the 0.5 line. The sensitivity bands
2–10 Å and 4–14 Å are extra rows in the same table. They do not overlap
the 15 Å bulk.

The probe radius is 0.80 Å and the oxygen exclusion radius is 1.52 Å. The
fraction is a geometric insertion fraction, not a pore volume and not SURFNET.
The same insertion is also written at probe radii 0, 0.5, and 1.2 Å. Those
columns are a sensitivity check. The reported fraction is the 0.80 Å value.

## Supporting information

The free-volume number is the fraction of uniformly placed probe centers whose
nearest oxygen, under the minimum image, is at least `r_O + r_probe` away.
`r_O = 1.52` Å and `r_probe = 0.80` Å. Only oxygen centers are excluded.
Hydrogens, cavity connectivity, and percolation are not part of the test.
Label it geometric probe-center insertion free volume. Do not call it porosity
and do not cite it as SURFNET.

Holds and endpoints drop the first 10% of frames (`HOLD_DISCARD_FRACTION`)
before training or scoring. That discard is recorded as `discard_fraction` in
the manifest. `duration_ps` is the span of `TIME(PS)` in the finished mdout.
This archive has no finished hold mdout, so those lengths are blank until the
logs are present.

A missing replica is fatal unless `--allow-missing` is set. At least 20 frames
are scored. The shaded band on the figure is the replica standard deviation.
`f_normalized_frame_width` is the frame-to-frame spread and is not that band.

## How to run

Endpoints and holds have to be rows of `outputs/manifest.tsv`, one
trajectory per replica. This archive does not contain those files.

```bash
py -3 07_core_awi_lda_and_free_volume.py \
  --manifest outputs/manifest.tsv \
  --outdir outputs/classifier \
  --frames 20
```

The manifest is written by `cooling_path/02_build_manifest.py`. Holds and
endpoints are discovered under `opc/<replica>/hold/<temperature>K/` and
`opc/<replica>/endpoint/<temperature>K/`, or passed with `--holds-tsv` and
`--endpoints-tsv`.

## Cutoff

`cut_A` is read from a finished mdout when one sits next to the trajectory.
No finished hold mdout is in this archive, so that check is not completed.
The archived `prop_trajectory.mdin` says `cut = 8.0`. Cooling inputs and
`hold_from_cooling.py` say `cut = 9.0`. If a finished hold log says 8 Å,
the 78 K and 150–250 K data are at a different cutoff from the ramps.
