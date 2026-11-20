"""Figure 2, hydrogen-bond retention.

Drawn by ``analysis/interface_dynamics/figure2_from_v3.py``.
"""

from opcfilm.plotting.figures.manuscript import draw as _draw

SCRIPT = "analysis/interface_dynamics/figure2_from_v3.py"
INPUTS = {
    "summary": "results/hbond/summary_replicas.csv",
    "contrast": "results/hbond/contrast.csv",
}


def draw(argv):
    """Draw figure 2.

    Parameters
    ----------
    argv : sequence of str
        Arguments of ``figure2_from_v3.py``.

    Returns
    -------
    int
        The script's exit status.
    """
    return _draw(SCRIPT, argv, INPUTS)
