"""Probe free volume in each region.

The production script is
``analysis/glass_structure/opc_vitrification_spatial_workflow/07_core_awi_lda_and_free_volume.py``.
The reported radius is the sheet value, and the sensitivity radii are stored
beside it.
"""

from opcfilm.stages import run as _run


def run(row, root):
    """Run the free-volume stage for one manifest row.

    Parameters
    ----------
    row : dict
        Manifest row.
    root : path-like
        Directory that contains ``results/``.

    Returns
    -------
    None
        This function does not return.

    Raises
    ------
    NotImplementedError
        The stage is the free-volume script, not a package stand-in.
    """
    _run("s07", row, root)
