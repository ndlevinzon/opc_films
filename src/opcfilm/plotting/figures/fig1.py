"""Figure 1, regional relaxation.

Drawn by ``analysis/interface_dynamics/figure1_from_v2.py``.
"""

from opcfilm.plotting.figures.manuscript import draw as _draw

SCRIPT = "analysis/interface_dynamics/figure1_from_v2.py"
INPUTS = {
    "summary": "results/regional/summary_replicas.csv",
    "contrast": "results/regional/contrast.csv",
}


def draw(argv):
    """Draw figure 1.

    Parameters
    ----------
    argv : sequence of str
        Arguments of ``figure1_from_v2.py``.

    Returns
    -------
    int
        The script's exit status.
    """
    return _draw(SCRIPT, argv, INPUTS)
