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


@pytest.mark.parametrize("label, region", [
    ("Bulk", "bulk"),
    ("interior Fs", "bulk"),
    ("275 K AWI", "awi"),
    ("AWI outer", "awi_outer"),
    ("0–6 Å AWI", "awi_outer"),
    ("3–12 Å AWI", "awi"),
    ("Bulk − AWI", None),
    ("Replica mean ± 1 SD", None),
])
def test_region_of_label(label, region):
    from opcfilm.plotting.conventions import region_of

    assert region_of(label) == region


def test_two_series_are_black_with_region_styles():
    figure, axis = plt.subplots()
    bulk, = axis.plot([1, 2], [1, 2], "o-", color="C0", label="interior")
    awi, = axis.plot([1, 2], [2, 3], "o-", color="C1", label="awi")
    axis.fill_between([1, 2], [0, 1], [2, 3], color="C0", alpha=0.2)
    axis.legend()
    apply_figure_conventions(figure)
    assert matplotlib.colors.same_color(bulk.get_color(), "black")
    assert matplotlib.colors.same_color(awi.get_color(), "black")
    assert bulk.get_linestyle() == "-"
    assert awi.get_linestyle() == "--"
    assert [t.get_text() for t in axis.get_legend().get_texts()] == ["Bulk", "AWI"]
    plt.close(figure)


def test_outer_awi_is_dropped_and_panel_is_black():
    figure, axis = plt.subplots()
    lines = [axis.plot([1, 2, 3], [k, k + 1, k + 2], color=f"C{k}", label=name)[0]
             for k, name in enumerate(["Bulk", "AWI", "AWI outer"])]
    axis.fill_between([1, 2, 3], [1.8, 2.8, 3.8], [2.2, 3.2, 4.2], color="C2", alpha=0.2)
    axis.legend()
    apply_figure_conventions(figure)
    assert lines[2].axes is None
    assert len(axis.collections) == 0
    assert matplotlib.colors.same_color(lines[1].get_color(), "black")
    assert [line.get_linestyle() for line in lines[:2]] == ["-", "--"]
    assert [t.get_text() for t in axis.get_legend().get_texts()] == ["Bulk", "AWI"]
    plt.close(figure)


def test_temperature_series_keep_color():
    figure, axis = plt.subplots()
    lines = [axis.plot([1, 2], [k, k + 1], color=f"C{k}", label=f"{t} K, {r}")[0]
             for k, (t, r) in enumerate([(235, "Bulk"), (235, "AWI"), (195, "Bulk"), (195, "AWI")])]
    apply_figure_conventions(figure)
    assert matplotlib.colors.same_color(lines[2].get_color(), "C2")
    assert [line.get_linestyle() for line in lines] == ["-", "--", "-", "--"]
    plt.close(figure)
