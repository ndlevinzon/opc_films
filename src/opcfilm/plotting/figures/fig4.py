"""Figure 4, glass temperature against cooling rate.

Drawn by ``analysis/glass_transition/05b_plot_tg_vs_cooling_rate.py``.
"""

from opcfilm.plotting.figures.manuscript import draw as _draw

SCRIPT = "analysis/glass_transition/05b_plot_tg_vs_cooling_rate.py"
INPUTS = {
    "summary": "results/tg/summary_replicas.csv",
    "contrast": "results/tg/contrast.csv",
}


def draw(argv):
    """Draw figure 4.

    Parameters
    ----------
    argv : sequence of str
        Arguments of ``05b_plot_tg_vs_cooling_rate.py``.

    Returns
    -------
    int
        The script's exit status.
    """
    return _draw(SCRIPT, argv, INPUTS)
