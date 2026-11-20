"""Figure 7, coordination defects.

Drawn by ``analysis/interface_dynamics/hydrogen_bonds/05_plot_defect_fraction.py``
and ``06_plot_hypothesis.py``.
"""

from opcfilm.plotting.figures.manuscript import draw as _draw

SCRIPT = "analysis/interface_dynamics/hydrogen_bonds/06_plot_hypothesis.py"
INPUTS = {
    "summary": "results/defects/summary_replicas.csv",
    "contrast": "results/defects/contrast.csv",
}


def draw(argv):
    """Draw figure 7.

    Parameters
    ----------
    argv : sequence of str
        Arguments of ``06_plot_hypothesis.py``.

    Returns
    -------
    int
        The script's exit status.
    """
    return _draw(SCRIPT, argv, INPUTS)
