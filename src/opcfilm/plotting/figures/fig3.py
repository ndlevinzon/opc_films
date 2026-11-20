"""Figure 3, energy along the cooling path.

Drawn by ``analysis/cooling_path/06_plot_energetic_stabilization.py``.
"""

from opcfilm.plotting.figures.manuscript import draw as _draw

SCRIPT = "analysis/cooling_path/06_plot_energetic_stabilization.py"
INPUTS = {
    "summary": "results/energy/summary_replicas.csv",
    "contrast": "results/energy/contrast.csv",
}


def draw(argv):
    """Draw figure 3.

    Parameters
    ----------
    argv : sequence of str
        Arguments of ``06_plot_energetic_stabilization.py``.

    Returns
    -------
    int
        The script's exit status.
    """
    return _draw(SCRIPT, argv, INPUTS)
