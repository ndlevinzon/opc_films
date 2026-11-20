"""Render a cooling or hold input from ``MD/templates`` and the parameter sheet.

The templates are the source for a new input. The heredocs inside the GPU
scripts are the inputs that were submitted for minimization and cooling.
"""

from __future__ import annotations

from pathlib import Path

from opcfilm.config import CONFIG, DT_PS


def _repository():
    for parent in Path(__file__).resolve().parents:
        if (parent / "MD" / "templates").is_dir():
            return parent
    raise FileNotFoundError("MD/templates was not found.")


def _fill(kind, values):
    """Substitute ``{{ name }}`` placeholders.

    Parameters
    ----------
    kind : str
        ``cool`` or ``hold``.
    values : dict
        Placeholder names and the text that replaces them.

    Returns
    -------
    str
        Rendered namelist.
    """
    text = (_repository() / "MD" / "templates" / f"{kind}.mdin.j2").read_text(encoding="utf-8")
    for name, value in values.items():
        text = text.replace("{{ " + name + " }}", str(value))
        text = text.replace("{{" + name + "}}", str(value))
    return text


def _shared():
    return {
        "dt": f"{DT_PS:g}",
        "cut": f"{CONFIG.trajectory.production_cut_A:.1f}",
        "gamma_ln": f"{CONFIG.trajectory.gamma_ln_per_ps:g}",
    }


def render_text(kind, temperature_K, length_ns, temperature_end_K=None):
    """Render one namelist.

    Parameters
    ----------
    kind : str
        ``cool`` or ``hold``.
    temperature_K : float
        Start temperature for cooling, or the hold setpoint.
    length_ns : float
        Requested length. ``nstlim`` uses the sheet time step.
    temperature_end_K : float, optional
        Cooling end temperature. A hold ignores it.

    Returns
    -------
    str
        The namelist.

    Raises
    ------
    ValueError
        If the kind is unknown or the length is not positive.
    """
    nstlim = int(round(float(length_ns) * 1e6 / (DT_PS * 1000.0)))
    if nstlim < 1:
        raise ValueError("The length must be positive.")
    values = _shared()
    values["nstlim"] = nstlim
    if kind == "hold":
        values["temperature_K"] = f"{float(temperature_K):g}"
        values["ntwx"] = f"{CONFIG.trajectory.hold_ntwx:g}"
    elif kind == "cool":
        if temperature_end_K is None:
            raise ValueError("Cooling needs --temperature-end.")
        values["temperature_start_K"] = f"{float(temperature_K):g}"
        values["temperature_end_K"] = f"{float(temperature_end_K):g}"
        values["ntwx"] = f"{CONFIG.trajectory.cooling_ntwx:g}"
    else:
        raise ValueError("kind must be cool or hold.")
    return _fill(kind, values)


def render(kind, temperature_K, length_ns, destination, temperature_end_K=None):
    """Write a rendered namelist.

    Parameters
    ----------
    kind : str
        ``cool`` or ``hold``.
    temperature_K : float
        Start or setpoint temperature.
    length_ns : float
        Requested length in nanoseconds.
    destination : path-like
        Output path.
    temperature_end_K : float, optional
        Cooling end temperature.

    Returns
    -------
    pathlib.Path
        The file that was written.
    """
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        render_text(kind, temperature_K, length_ns, temperature_end_K),
        encoding="utf-8",
    )
    return destination
