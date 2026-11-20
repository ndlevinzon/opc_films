"""SI parameter table.

The table is ``docs/methods_parameters.md``, generated from
``config/project.yaml``. This module rewrites that page. It does not plot
a metric from a stage table.
"""

from opcfilm.config import config_path


def draw(argv=None):
    """Rewrite the SI parameter page from the parameter sheet.

    Parameters
    ----------
    argv : sequence of str, optional
        Ignored. The page has one source.

    Returns
    -------
    pathlib.Path
        ``docs/methods_parameters.md``.
    """
    del argv
    source = config_path()
    destination = source.parents[1] / "docs" / "methods_parameters.md"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        "# Methods parameters\n\n"
        "Generated from `config/project.yaml`.\n\n"
        "```yaml\n" + source.read_text(encoding="utf-8") + "```\n\n"
        "## Software recorded in environment.txt\n\n"
        "AmberTools 26.0 and OVITO 3.16.1 are in that conda list. "
        "cpptraj is the AmberTools binary. `pmemd.cuda` is the `AMBERHOME` "
        "binary that ran the trajectories; its version is not in "
        "`environment.txt`, so it is not stated here. Snakemake is "
        "`snakemake>=9.6,<10`. The SLURM executor plugin version is not in "
        "`environment.txt`.\n",
        encoding="utf-8",
    )
    return destination
