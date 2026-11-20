"""Production stages.

The Phase A runner is ``workflow/Snakefile``. It calls the analysis scripts
directly. ``opcfilm s00`` and the other numbered commands refuse to run, so
a placeholder cannot be written into a figure under the same stage name.

The calculations live in the analysis scripts named here. The package command
refuses to write a stand-in number for them.

The graph follows the data, not the stage numbers:

- ``s00``, ``s01``, ``sq``, ``s04``, and ``s06`` are per row and do not wait
  on another stage. ``sq`` writes that row's ``qstar_row.json``.
- ``qstar adopt`` reads those row files and is the only writer of
  ``outputs/qstar.json``.
- ``s05`` is per row and waits for that adopted peak.
- ``s02`` is not per row. It reduces τ over every hold temperature of a rate.
- ``s03`` and ``s07`` are one group job. The classifier is trained on all six
  endpoint rows together, then scores every hold. Free volume is that same job.
"""

PER_ROW = ("s00", "s01", "sq", "s04", "s06")
AFTER_QSTAR = ("s05",)
REDUCTIONS = {"s02": "s05"}
GROUP = ("s03", "s07")

SCRIPTS = {
    "s00": "analysis/cooling_path/01_analyze_chunk.py and 04_summarize.py",
    "s01": "analysis/glass_transition/06_plot_ramp_energy.py and cooling_path/opc_thermo_response/02_thermo_response.py",
    "sq": "analysis/glass_transition/02_sq2_fft.py",
    "s04": "analysis/cooling_path/05_capillary_modes.py",
    "s06": "analysis/interface_dynamics/hbond_v3.py and analysis/interface_dynamics/hydrogen_bonds/03_analyze_free_oh.py",
    "s05": "analysis/glass_transition/03_fs_alpha.py and analysis/interface_dynamics/analyze_v2.py",
    "s02": "analysis/glass_transition/05_fit_relaxation_tg.py",
    "s03": "analysis/glass_structure/opc_vitrification_spatial_workflow/07_core_awi_lda_and_free_volume.py",
    "s07": "analysis/glass_structure/opc_vitrification_spatial_workflow/07_core_awi_lda_and_free_volume.py",
}


def run(stage, row, root):
    """Refuse a production stage that is still the analysis script.

    Parameters
    ----------
    stage : str
        Stage name, for example ``s00``.
    row : dict
        Manifest row. It is named in the error so the command that was
        requested is visible. ``s02``, ``s03``, and ``s07`` are reductions
        or a group job, so ``row`` may be ``None``.
    root : path-like
        Unused. The analysis script chooses its own output directory.

    Returns
    -------
    None
        This function does not return.

    Raises
    ------
    NotImplementedError
        Always. The message names the script that computes the stage.
    """
    del row, root
    script = SCRIPTS[stage]
    if stage in REDUCTIONS:
        detail = f"{stage} reduces {REDUCTIONS[stage]} and is not a per-row command."
    elif stage in GROUP:
        detail = f"{stage} is the endpoint group job (leave-one-replica-out), not one row."
    elif stage in AFTER_QSTAR:
        detail = f"{stage} waits for outputs/qstar.json."
    else:
        detail = f"{stage} is per row."
    raise NotImplementedError(f"{detail} Run {script}.")
