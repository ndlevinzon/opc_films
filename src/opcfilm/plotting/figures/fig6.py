"""Figure 6, local structure.

Drawn by ``analysis/glass_structure/plot_local_structure.py``.
"""

from opcfilm.plotting.figures.manuscript import draw as _draw

SCRIPT = "analysis/glass_structure/plot_local_structure.py"
INPUTS = {
    "summary": "results/local_order/summary_replicas.csv",
    "contrast": "results/local_order/contrast.csv",
}


def draw(argv):
    """Draw figure 6.

    Parameters
    ----------
    argv : sequence of str
        Arguments of ``plot_local_structure.py``.

    Returns
    -------
    int
        The script's exit status.
    """
    return _draw(SCRIPT, argv, INPUTS)
