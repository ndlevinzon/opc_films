"""Figures read hot to cold and keep legends outside the panels."""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest

from opcfilm.plotting.conventions import (
    apply_figure_conventions,
    install_figure_conventions,
    is_temperature_label,
)


@pytest.mark.parametrize("label, expected", [
    ("Temperature (K)", True),
    ("Cooling-window midpoint temperature (K)", True),
    ("Window midpoint (K)", True),
    ("$T$ (K)", True),
    ("$1000/T$ (K$^{-1}$)", False),
    (r"$\log_{10} R$ (R in K/ns)", False),
    (r"$\Delta T$ (K)", False),
    ("Lag (ps)", False),
    ("", False),
])
def test_temperature_labels(label, expected):
    assert is_temperature_label(label) is expected


def test_single_panel(tmp_path):
    install_figure_conventions()
    figure, axis = plt.subplots()
    axis.plot([78, 300], [0, 1], label="Bulk")
    axis.set_xlabel("Temperature (K)")
    axis.legend()
    figure.savefig(tmp_path / "a.png")
    figure.savefig(tmp_path / "a.pdf")
    assert axis.xaxis_inverted()
    legend = axis.get_legend()
    figure.canvas.draw()
    box = legend.get_window_extent().transformed(axis.transAxes.inverted())
    assert box.x0 >= 1.0
    plt.close(figure)


def test_grid_merges_legends_and_leaves_inverse_temperature():
    figure, axes = plt.subplots(1, 2)
    for axis in axes:
        axis.plot([3, 13], [0, 1], label="Bulk")
        axis.set_xlabel("$1000/T$ (K$^{-1}$)")
        axis.legend()
    top = axes[0].twiny()
    top.set_xlabel("Temperature (K)")
    apply_figure_conventions(figure)
    assert not axes[0].xaxis_inverted()
    assert not top.xaxis_inverted()
    assert all(axis.get_legend() is None for axis in axes)
    assert len(figure.legends) == 1
    assert [t.get_text() for t in figure.legends[0].get_texts()] == ["Bulk"]
    plt.close(figure)


def test_already_inverted_axis_is_kept():
    figure, axis = plt.subplots()
    axis.plot([78, 300], [0, 1])
    axis.set_xlim(300, 78)
    axis.set_xlabel("Temperature (K)")
    apply_figure_conventions(figure)
    assert axis.get_xlim() == (300, 78)
    plt.close(figure)
