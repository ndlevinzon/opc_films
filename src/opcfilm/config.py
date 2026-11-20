"""Load ``config/project.yaml`` into frozen dataclasses.

Numeric values live in that file. The module attributes below are the loaded
values, so existing imports such as ``BULK_HALFWIDTH_A`` keep working.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, fields, replace
from pathlib import Path


def _parse_scalar(text):
    text = text.strip()
    if text.startswith("[") and text.endswith("]"):
        return _parse_flow(text)
    if text in ("true", "false"):
        return text == "true"
    if text in ("", "~", "null"):
        return None
    try:
        if any(mark in text.lower() for mark in (".", "e")):
            return float(text)
        return int(text)
    except ValueError:
        if (text.startswith('"') and text.endswith('"')) or (
            text.startswith("'") and text.endswith("'")
        ):
            return text[1:-1]
        return text


def _parse_flow(text):
    inner = text.strip()[1:-1].strip()
    if inner == "":
        return []
    items = []
    depth = 0
    start = 0
    for index, char in enumerate(inner):
        if char == "[":
            depth += 1
        elif char == "]":
            depth -= 1
        elif char == "," and depth == 0:
            items.append(_parse_scalar(inner[start:index]))
            start = index + 1
    items.append(_parse_scalar(inner[start:]))
    return items


def _load_yaml(text):
    try:
        import yaml
    except ImportError:
        return _load_yaml_fallback(text)
    loaded = yaml.safe_load(text)
    if not isinstance(loaded, dict):
        raise ValueError("project.yaml must be a mapping.")
    return loaded


def _load_yaml_fallback(text):
    """Indentation parser used when PyYAML is not installed.

    Parameters
    ----------
    text : str
        YAML text. Supports maps, nested maps, and flow sequences.

    Returns
    -------
    dict
        Parsed mapping.
    """
    root = {}
    stack = [(-1, root)]
    for raw in text.splitlines():
        if not raw.strip() or raw.lstrip().startswith("#"):
            continue
        indent = len(raw) - len(raw.lstrip(" "))
        line = raw.strip()
        key, _, value = line.partition(":")
        if not key or line.startswith("-"):
            raise ValueError(f"Unsupported YAML line: {raw}")
        while stack and indent <= stack[-1][0]:
            stack.pop()
        parent = stack[-1][1]
        if value.strip() == "":
            child = {}
            parent[key] = child
            stack.append((indent, child))
        else:
            parent[key] = _parse_scalar(value)
    return root


def config_path():
    """Path of the parameter sheet.

    Returns
    -------
    pathlib.Path
        ``OPCFILM_CONFIG`` when that variable is set, otherwise
        ``config/project.yaml`` found by walking up from this file.

    Raises
    ------
    FileNotFoundError
        If neither location exists.
    """
    override = os.environ.get("OPCFILM_CONFIG")
    if override:
        path = Path(override)
        if not path.is_file():
            raise FileNotFoundError(path)
        return path
    for parent in Path(__file__).resolve().parents:
        candidate = parent / "config" / "project.yaml"
        if candidate.is_file():
            return candidate
    raise FileNotFoundError("config/project.yaml was not found.")


def config_sha256():
    """SHA-256 of the parameter sheet bytes.

    Returns
    -------
    str
        Hex digest. A run records this so the numbers can be recovered.
    """
    return hashlib.sha256(config_path().read_bytes()).hexdigest()


def _tuple(value):
    if isinstance(value, list):
        return tuple(_tuple(item) if isinstance(item, list) else item for item in value)
    return value


@dataclass(frozen=True)
class Film:
    """Expected film and the number-density conversion.

    Parameters
    ----------
    n_water : int
        Cross-check water count.
    box_A : tuple of float
        Cross-check box lengths, in angstroms.
    water_molar_mass_g_mol : float
        Molar mass used in the density conversion.
    number_density_divisor : float
        Divisor that converts Å^-3 to g cm^-3.
    avogadro_per_mol : float
        Avogadro constant, per mole.
    dt_ps : float
        Integration step, in picoseconds.
    replicas : tuple of str
        Directory names under ``opc/``, for example ``r1``.
    """

    n_water: int
    box_A: tuple
    water_molar_mass_g_mol: float
    number_density_divisor: float
    avogadro_per_mol: float
    dt_ps: float
    replicas: tuple


@dataclass(frozen=True)
class Regions:
    """Bulk half-width and the two interface bands.

    Parameters
    ----------
    bulk_halfwidth_A : float
        Half-width of the interior, in angstroms.
    awi_A : tuple of float
        Main interface band, depth below the half-density surface.
    awi_outer_A : tuple of float
        Outer interface band.
    sensitivity_bands_A : tuple
        Extra bands used only for the classifier sensitivity.
    names : tuple of str
        Region names written to tables.
    """

    bulk_halfwidth_A: float
    awi_A: tuple
    awi_outer_A: tuple
    sensitivity_bands_A: tuple
    names: tuple


@dataclass(frozen=True)
class Profile:
    """Density-profile binning.

    Parameters
    ----------
    bin_A : float
        Profile bin width, in angstroms.
    region_bin_A : float
        Bin width used when a region label is drawn from a profile.
    smooth_sigma_A : float
        Gaussian smoothing length, in angstroms.
    """

    bin_A: float
    region_bin_A: float
    smooth_sigma_A: float


@dataclass(frozen=True)
class HBond:
    """Geometric hydrogen-bond definition and the retention lags.

    Parameters
    ----------
    oo_cutoff_A : float
        Maximum oxygen-oxygen distance, in angstroms.
    angle_deg : float
        Minimum O-H···O angle, in degrees.
    lags_ps : tuple of float
        Retention lag list. The comparison lag is inside this list.
    comparison_lag_ps : float
        Lag at which retention is compared, in picoseconds.
    origins : int
        Time origins used by the hydrogen-bond script. Regional dynamics
        uses a separate origin count.
    """

    oo_cutoff_A: float
    angle_deg: float
    lags_ps: tuple
    comparison_lag_ps: float
    origins: int


@dataclass(frozen=True)
class QStar:
    """Adopted in-plane peak and the structure-factor grid.

    Parameters
    ----------
    json : str
        Path of the one ``q*`` file, relative to the repository.
    adopted_Ainv : float
        Adopted peak, in inverse angstroms.
    tolerance_Ainv : float
        A recomputed peak replaces the adopted value only beyond this gap.
    grid_spacing_A : float
        Real-space FFT cell size, in angstroms.
    peak_min_Ainv : float
        Lower edge of the peak search.
    peak_max_Ainv : float
        Upper edge of the peak search.
    peak_window_Ainv : float
        Half-width of the quadratic fit, in inverse angstroms.
    adopt_from : dict
        Which manifest rows may contribute a measured peak to the one
        adopted ``q*`` file.
    """

    json: str
    adopted_Ainv: float
    tolerance_Ainv: float
    grid_spacing_A: float
    peak_min_Ainv: float
    peak_max_Ainv: float
    peak_window_Ainv: float
    adopt_from: dict


@dataclass(frozen=True)
class Dynamics:
    """Lag lists, origin counts, and the cooling-window cap.

    Parameters
    ----------
    cooling_lag_cap_K : float
        Maximum temperature change covered by a cooling-window lag.
    hold_lag_fraction : float
        Longest hold lag, as a fraction of the hold.
    cooling_lags_ps : tuple of float
        Default cooling lag list, in picoseconds.
    origins : int
        Number of time origins.
    references_per_region : int
        Reference waters drawn in each region.
    event_window_ps : float
        Event interval, in picoseconds.
    """

    cooling_lag_cap_K: float
    hold_lag_fraction: float
    cooling_lags_ps: tuple
    origins: int
    references_per_region: int
    event_window_ps: float


@dataclass(frozen=True)
class Discard:
    """Equilibration discard.

    Parameters
    ----------
    hold_fraction : float
        Fraction dropped from a hold or an endpoint.
    ramp_fraction : float
        Fraction dropped from a cooling window.
    """

    hold_fraction: float
    ramp_fraction: float


@dataclass(frozen=True)
class Classifier:
    """Quantitative LDA settings.

    Parameters
    ----------
    liquid_K : float
        Liquid reference temperature.
    glass_K : float
        Glass reference temperature.
    bulk_halfwidth_A : float
        Training region half-width, in angstroms.
    frames : int
        Frames per replica endpoint.
    threshold : float
        Score at which a water is counted as glass.
    scheme : str
        Validation scheme. The quantitative run is leave-one-replica-out.
    scale_mad : float
        Median absolute deviation scale.
    covariance_jitter : float
        Diagonal added to the pooled covariance.
    """

    liquid_K: float
    glass_K: float
    bulk_halfwidth_A: float
    frames: int
    threshold: float
    scheme: str
    scale_mad: float
    covariance_jitter: float


@dataclass(frozen=True)
class Probe:
    """Geometric free-volume probe.

    Parameters
    ----------
    radius_A : float
        Reported probe radius, in angstroms.
    sensitivity_radii_A : tuple of float
        Extra radii stored beside the reported value.
    oxygen_exclusion_A : float
        Oxygen exclusion radius, in angstroms.
    n_probes : int
        Probe centers inserted in one region.
    """

    radius_A: float
    sensitivity_radii_A: tuple
    oxygen_exclusion_A: float
    n_probes: int


@dataclass(frozen=True)
class Arrhenius:
    """Dynamic glass-transition fit window.

    Parameters
    ----------
    window_K : tuple of float
        Reported fit window, in kelvin.
    sensitivity_lower_K : tuple of float
        Lower edges used only as a sensitivity check.
    gas_constant_J_mol_K : float
        Gas constant, in J mol^-1 K^-1.
    """

    window_K: tuple
    sensitivity_lower_K: tuple
    gas_constant_J_mol_K: float


@dataclass(frozen=True)
class Energy:
    """Energetic-arrest fit bounds.

    Parameters
    ----------
    bin_K : float
        Energy bin width, in kelvin.
    f_scale_kcal_mol_per_water : float
        Softplus scale used to report the fit residual.
    fit_min_K : float
        Lower temperature of the smooth fit.
    fit_max_K : float
        Upper temperature of the smooth fit.
    glass_min_K : float
        Lower edge of the glass line.
    glass_max_K : float
        Upper edge of the glass line.
    liquid_min_K : float
        Lower edge of the liquid line.
    liquid_max_K : float
        Upper edge of the liquid line.
    report_K : float
        Temperature at which the energy gap is reported.
    """

    bin_K: float
    f_scale_kcal_mol_per_water: float
    fit_min_K: float
    fit_max_K: float
    glass_min_K: float
    glass_max_K: float
    liquid_min_K: float
    liquid_max_K: float
    report_K: float


@dataclass(frozen=True)
class Uncertainty:
    """Error-bar convention.

    Parameters
    ----------
    curve_band_sd : float
        Half-width of a curve band, in replica standard deviations.
    contrast : str
        Contrast direction. The reported contrast is bulk minus interface.
    paired_confidence : float
        Confidence level of a paired Student-t interval.
    """

    curve_band_sd: float
    contrast: str
    paired_confidence: float


@dataclass(frozen=True)
class Trajectory:
    """Saved-frame spacing and the cutoffs that were run.

    Parameters
    ----------
    cooling_ntwx : int
        Cooling save interval, in steps.
    hold_ntwx : int
        Hold save interval, in steps.
    cooling_frame_ps : float
        Cooling frame spacing, in picoseconds.
    hold_frame_ps : float
        Hold frame spacing, in picoseconds.
    production_cut_A : float
        Real-space cutoff written by the cooling input, in angstroms.
    equilibration_cut_A : float
        Real-space cutoff of equilibration and the archived prop input.
    gamma_ln_per_ps : float
        Langevin collision frequency, per picosecond.
    """

    cooling_ntwx: int
    hold_ntwx: int
    cooling_frame_ps: float
    hold_frame_ps: float
    production_cut_A: float
    equilibration_cut_A: float
    gamma_ln_per_ps: float


@dataclass(frozen=True)
class ProjectConfig:
    """The whole parameter sheet.

    Parameters
    ----------
    film, regions, profile, hbond, qstar, dynamics, discard, classifier, probe, arrhenius, energy, uncertainty, trajectory :
        Nested sections of ``project.yaml``.
    seeds : dict
        Sampler seeds.
    censoring : tuple of str
        Allowed relaxation statuses.
    overrides : tuple
        Command-line overrides applied to this object. Empty for the sheet itself.
    """

    film: Film
    regions: Regions
    profile: Profile
    hbond: HBond
    qstar: QStar
    dynamics: Dynamics
    discard: Discard
    classifier: Classifier
    probe: Probe
    arrhenius: Arrhenius
    energy: Energy
    uncertainty: Uncertainty
    trajectory: Trajectory
    seeds: dict
    censoring: tuple
    overrides: tuple = ()


_SECTIONS = {
    "film": Film,
    "regions": Regions,
    "profile": Profile,
    "hbond": HBond,
    "qstar": QStar,
    "dynamics": Dynamics,
    "discard": Discard,
    "classifier": Classifier,
    "probe": Probe,
    "arrhenius": Arrhenius,
    "energy": Energy,
    "uncertainty": Uncertainty,
    "trajectory": Trajectory,
}


def _section(cls, payload):
    converted = {}
    for item in fields(cls):
        if item.name not in payload:
            raise ValueError(f"project.yaml is missing {cls.__name__}.{item.name}.")
        value = payload[item.name]
        if isinstance(value, list):
            value = _tuple(value)
        converted[item.name] = value
    return cls(**converted)


def load_config(path=None):
    """Load the parameter sheet.

    Parameters
    ----------
    path : path-like, optional
        Sheet to read. The default is :func:`config_path`.

    Returns
    -------
    ProjectConfig
        Frozen configuration. Nested sections are frozen as well.

    Raises
    ------
    ValueError
        If a section or a field is missing.
    """
    path = Path(path) if path is not None else config_path()
    payload = _load_yaml(path.read_text(encoding="utf-8"))
    sections = {
        name: _section(cls, payload[name])
        for name, cls in _SECTIONS.items()
        if name in payload
    }
    missing = [name for name in _SECTIONS if name not in sections]
    if missing:
        raise ValueError(f"project.yaml is missing {missing}.")
    if "seeds" not in payload or "censoring" not in payload:
        raise ValueError("project.yaml is missing seeds or censoring.")
    return ProjectConfig(
        seeds=dict(payload["seeds"]),
        censoring=_tuple(payload["censoring"]),
        **sections,
    )


def apply_cli_overrides(items):
    """Apply ``--set section.field=value`` and publish the result.

    Parameters
    ----------
    items : sequence of str
        Each item is ``section.field=value``.

    Returns
    -------
    ProjectConfig
        The overridden sheet. ``overrides`` lists every replacement.

    Raises
    ------
    ValueError
        If an item has no ``=`` or names an unknown field.
    """
    updates = {}
    for item in items:
        key, separator, raw = str(item).partition("=")
        if not separator:
            raise ValueError(f"Override {item!r} must be section.field=value.")
        updates[key] = _parse_scalar(raw)
    updated = with_overrides(CONFIG, updates)
    _bind(updated)
    return updated


def with_overrides(config, updates):
    """Return a copy with sensitivity overrides applied.

    Parameters
    ----------
    config : ProjectConfig
        Loaded sheet.
    updates : dict
        Keys are ``section.field``. Values replace that field.

    Returns
    -------
    ProjectConfig
        Copy whose ``overrides`` records every replacement.

    Raises
    ------
    ValueError
        If a key does not name a field.
    """
    current = config
    applied = list(config.overrides)
    for key, value in updates.items():
        section_name, _, field_name = str(key).partition(".")
        if not field_name or not hasattr(current, section_name):
            raise ValueError(f"Unknown override {key}.")
        section = getattr(current, section_name)
        if field_name not in {item.name for item in fields(section)}:
            raise ValueError(f"Unknown override {key}.")
        current = replace(current, **{section_name: replace(section, **{field_name: value})})
        applied.append((str(key), value))
    return replace(current, overrides=tuple(applied))


def adopted_qstar(measured, reference=None, tolerance=None):
    """Choose the in-plane peak used for relaxation times.

    Parameters
    ----------
    measured : float
        Peak position in inverse angstroms.
    reference : float, optional
        Adopted peak. The sheet value is used when this is omitted.
    tolerance : float, optional
        Replacement tolerance. The sheet value is used when this is omitted.

    Returns
    -------
    float
        The reference when the measured peak lies inside the tolerance,
        otherwise the measured peak.
    """
    reference = QSTAR_AINV if reference is None else float(reference)
    tolerance = QSTAR_RECOMPUTE_TOLERANCE_AINV if tolerance is None else float(tolerance)
    measured = float(measured)
    if abs(measured - reference) > tolerance:
        return measured
    return float(reference)


def fft_cell_counts(length, spacing=None):
    """FFT cells that make each cell as close as possible to ``spacing``.

    Parameters
    ----------
    length : float
        Box length in angstroms.
    spacing : float, optional
        Requested cell size in angstroms. The sheet value is the default.

    Returns
    -------
    int
        ``round(length / spacing)``, at least one.

    Raises
    ------
    ValueError
        If the grid would have no cells.
    """
    if spacing is None:
        spacing = SQ_GRID_SPACING_A
    count = int(round(float(length) / float(spacing)))
    if count < 1:
        raise ValueError("The FFT grid would have no cells.")
    return count


def load_adopted_qstar(path, override=None, tolerance=1e-4):
    """Read the adopted in-plane peak from the structure-factor JSON.

    Parameters
    ----------
    path : path-like
        JSON written by the structure-factor stage. It must contain
        ``qstar_adopted_Ainv``.
    override : float, optional
        Value passed as ``-q``. It must agree with the file.
    tolerance : float, optional
        Allowed absolute difference, in inverse angstroms.

    Returns
    -------
    float
        Adopted ``q*`` in inverse angstroms.

    Raises
    ------
    FileNotFoundError
        If ``path`` does not exist.
    ValueError
        If ``override`` disagrees with the file, or the key is missing.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(
            f"{path} is missing. Run the structure-factor stage and pass that JSON."
        )
    payload = json.loads(path.read_text(encoding="utf-8"))
    if "qstar_adopted_Ainv" not in payload:
        raise ValueError(f"{path} has no qstar_adopted_Ainv.")
    adopted = float(payload["qstar_adopted_Ainv"])
    if override is not None and abs(float(override) - adopted) > float(tolerance):
        raise ValueError(
            f"-q {float(override):.6f} disagrees with {path} "
            f"(qstar_adopted_Ainv = {adopted:.6f})."
        )
    return adopted


def _bind(config):
    """Publish the sheet as the names existing scripts already import.

    Parameters
    ----------
    config : ProjectConfig
        Loaded sheet.

    Returns
    -------
    None
        Names are written onto this module.
    """
    film = config.film
    module = globals()
    module["CONFIG"] = config
    module["N_WATER"] = int(film.n_water)
    module["DEFAULT_LX_A"] = float(film.box_A[0])
    module["DEFAULT_LY_A"] = float(film.box_A[1])
    module["DEFAULT_LZ_A"] = float(film.box_A[2])
    module["WATER_MOLAR_MASS_G_MOL"] = float(film.water_molar_mass_g_mol)
    module["NUMBER_DENSITY_DIVISOR"] = float(film.number_density_divisor)
    module["NUMBER_TO_MASS_DENSITY"] = (
        float(film.water_molar_mass_g_mol) / float(film.number_density_divisor)
    )
    module["AVOGADRO"] = float(film.avogadro_per_mol)
    module["DT_PS"] = float(film.dt_ps)
    module["BULK_HALFWIDTH_A"] = float(config.regions.bulk_halfwidth_A)
    module["AWI_SUB_A"] = tuple(float(item) for item in config.regions.awi_A)
    module["AWI_OUTER_A"] = tuple(float(item) for item in config.regions.awi_outer_A)
    module["PROFILE_BIN_A"] = float(config.profile.bin_A)
    module["REGION_BIN_A"] = float(config.profile.region_bin_A)
    module["PROFILE_SMOOTH_SIGMA_A"] = float(config.profile.smooth_sigma_A)
    module["HBOND_OO_CUTOFF_A"] = float(config.hbond.oo_cutoff_A)
    module["HBOND_ANGLE_DEG"] = float(config.hbond.angle_deg)
    module["HBOND_LAGS_PS"] = tuple(float(item) for item in config.hbond.lags_ps)
    module["HBOND_COMPARISON_LAG_PS"] = float(config.hbond.comparison_lag_ps)
    module["QSTAR_AINV"] = float(config.qstar.adopted_Ainv)
    module["QSTAR_RECOMPUTE_TOLERANCE_AINV"] = float(config.qstar.tolerance_Ainv)
    module["SQ_GRID_SPACING_A"] = float(config.qstar.grid_spacing_A)
    module["QSTAR_JSON"] = config.qstar.json
    module["HOLD_DISCARD_FRACTION"] = float(config.discard.hold_fraction)
    module["CURVE_BAND_SD"] = float(config.uncertainty.curve_band_sd)
    module["CONTRAST"] = str(config.uncertainty.contrast)
    module["SEEDS"] = dict(config.seeds)
    module["CENSORING"] = tuple(config.censoring)
    module["R_GAS"] = float(config.arrhenius.gas_constant_J_mol_K)


CONFIG: ProjectConfig
N_WATER: int
DEFAULT_LX_A: float
DEFAULT_LY_A: float
DEFAULT_LZ_A: float
WATER_MOLAR_MASS_G_MOL: float
NUMBER_DENSITY_DIVISOR: float
NUMBER_TO_MASS_DENSITY: float
AVOGADRO: float
DT_PS: float
BULK_HALFWIDTH_A: float
AWI_SUB_A: tuple
AWI_OUTER_A: tuple
PROFILE_BIN_A: float
REGION_BIN_A: float
PROFILE_SMOOTH_SIGMA_A: float
HBOND_OO_CUTOFF_A: float
HBOND_ANGLE_DEG: float
HBOND_LAGS_PS: tuple
HBOND_COMPARISON_LAG_PS: float
QSTAR_AINV: float
QSTAR_RECOMPUTE_TOLERANCE_AINV: float
SQ_GRID_SPACING_A: float
QSTAR_JSON: str
HOLD_DISCARD_FRACTION: float
CURVE_BAND_SD: float
CONTRAST: str
SEEDS: dict
CENSORING: tuple
R_GAS: float

_bind(load_config())
