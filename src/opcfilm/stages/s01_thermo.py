"""Per-window energies, the smooth arrest fit, and the response functions.

The production scripts are ``analysis/glass_transition/06_plot_ramp_energy.py``,
``06b_fit_energy_tg.py``, and ``cooling_path/opc_thermo_response/02_thermo_response.py``.
"""

from opcfilm.stages import run as _run


def run(row, root):
    """Run the thermo stage for one manifest row.

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
        The stage is the energy script, not a package stand-in.
    """
    _run("s01", row, root)
