# Preprocessing

`extract_profiles.sh` runs cpptraj on every NetCDF trajectory in `TRAJ_DIR` (default `TRAJ`). For each file it centers the water at the origin, images it into one slab, and writes the centered trajectory plus oxygen-number, mass, and charge density profiles along z. The bin width is 0.25 Å. The centered trajectory keeps every frame (`STRIDE=1`). Set `STRIDE` only when the centered file itself should be thinned. Downstream scripts then apply their own stride, and they read the frame spacing from the NetCDF `time` variable.

cpptraj is taken from `PATH`. The topology is required:

```bash
PRMTOP=/path/to/build/opc_slab.prmtop TRAJ_DIR=TRAJ ANALYSIS_DIR=analysis ./extract_profiles.sh
```

`ANALYSIS_DIR` is the output directory. There is no default topology path.
