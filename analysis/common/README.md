# Shared analysis code

These modules are imported by the scripts under the other analysis directories. They contain no trajectory loops of their own.

`constants.py` holds the film size used as a cross-check: 139,000 waters, lateral lengths 194.567 and 188.781 Å, and a 300 Å cell along z. Scripts read the count and the box from the files they analyze and stop on a mismatch. `BULK_HALFWIDTH_A` is 15 Å. `AWI_OUTER_A` is 0–6 Å and `AWI_SUB_A` is 3–12 Å, both measured inward from the half-density surface. `QSTAR_AINV` is 1.925 Å^-1. A peak measured on the 0.5 Å structure-factor grid replaces it only when the two differ by more than 0.025 Å^-1 (`adopted_qstar`). `NUMBER_TO_MASS_DENSITY` converts a number density in molecules per Å^3 to g/cm^3. `CONTRAST` is `bulk_minus_awi`. Curve bands use 1 standard deviation. Density profiles use 0.25 Å bins and a Gaussian of width 1 Å. Region labels may use 0.5 Å bins, and that width is written into the run metadata. A hydrogen bond is 3.5 Å and 150°. `SEEDS` lists the sampler seeds; each script writes the seed it used.

`rates.py` is the only place that turns a physical rate (4.44, 11.1, 22.2, 44.4, 111.0 K/ns) into a directory code, a file tag, or a path. Command lines reject directory codes. `as_physical_rate` is the reader for tables that still store those codes.

`order.py` computes orientational `q`, Chau–Hardwick `Sk`, the standard local structure index, and `radial_regularity`.

`trajectory.py` reads NetCDF `time`, counts waters in a topology, and checks the film size.

`contrast.py` forms every regional difference as bulk minus interface.

`uncertainty.py` builds replica mean ± `band` standard deviations and paired Student-t intervals.

`mdout.py` reads Amber energy files into a list of records or a pandas DataFrame. Fortran `D` exponents are accepted.

`slab.py` recenters a slab, places the half-density surfaces with `half_density_surfaces`, assigns the bulk and both interface bands, and counts hydrogen bonds. `hydrogen_bond_counts` and `directed_hydrogen_bonds` share the 3.5 Å / 150° geometry. The bulk and either interface band must not overlap. The default policy is `error`. Region labels use 0.5 Å bins. Profile smoothing is a Gaussian of width 1 Å.

`paths.py` walks upward from a script until it finds `common/__init__.py`, so a script can be moved within `analysis/` without a new relative import.
