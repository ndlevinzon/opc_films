"""Figure 5, the normalized classifier and free volume.

Drawn by ``analysis/glass_structure/opc_vitrification_spatial_workflow/08_replot_core_awi_black.py``.
"""

from opcfilm.plotting.figures.manuscript import draw as _draw

SCRIPT = "analysis/glass_structure/opc_vitrification_spatial_workflow/08_replot_core_awi_black.py"
INPUTS = {
    "summary": "results/classifier/summary_replicas.csv",
    "contrast": "results/classifier/contrast.csv",
}


def draw(argv):
    """Draw figure 5.

    Parameters
    ----------
    argv : sequence of str
        Arguments of ``08_replot_core_awi_black.py``.

    Returns
    -------
    int
        The script's exit status.
    """
    return _draw(SCRIPT, argv, INPUTS)
