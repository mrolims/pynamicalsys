"""
PlotStyler: aliases of the line and marker parameters, the tick direction,
tick padding on one or several Axes, and apply_style changing only the
global settings, never the figures.
"""

import warnings

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pytest

from pynamicalsys import PlotStyler


@pytest.fixture(autouse=True)
def restore_rcparams():
    """apply_style changes global state, so restore it after each test."""
    with matplotlib.rc_context():
        yield
    plt.close("all")


def test_defaults():
    ps = PlotStyler()
    assert ps.linewidth == 1.0
    assert ps.markersize == 5.0
    assert ps.markeredgewidth == 1.0
    assert ps.markeredgecolor == "black"


def test_aliases_set_rcparams():
    PlotStyler(lw=2.5, ms=3.0, mew=0.0, mec="red").apply_style()
    assert plt.rcParams["lines.linewidth"] == 2.5
    assert plt.rcParams["lines.markersize"] == 3.0
    assert plt.rcParams["lines.markeredgewidth"] == 0.0
    assert plt.rcParams["lines.markeredgecolor"] == "red"


def test_full_names_still_work():
    ps = PlotStyler(linewidth=2.5, markersize=3.0, markeredgewidth=0.0)
    assert (ps.linewidth, ps.markersize, ps.markeredgewidth) == (2.5, 3.0, 0.0)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"linewidth": 1.0, "lw": 2.0},
        {"markersize": 1.0, "ms": 2.0},
        {"markeredgewidth": 1.0, "mew": 2.0},
        {"markeredgecolor": "k", "mec": "r"},
    ],
)
def test_parameter_and_alias_together(kwargs):
    with pytest.raises(TypeError):
        PlotStyler(**kwargs)


@pytest.mark.parametrize("inward, expected", [(True, "in"), (False, "out")])
def test_tick_direction(inward, expected):
    PlotStyler(tick_direction_in=inward).apply_style()
    assert plt.rcParams["xtick.direction"] == expected
    assert plt.rcParams["ytick.direction"] == expected


def _major_pad(axis):
    return axis.get_major_ticks()[0].get_pad()


def test_tick_padding_single_axes():
    fig, ax = plt.subplots()
    PlotStyler().set_tick_padding(ax, pad_x=11, pad_y=13)
    assert _major_pad(ax.xaxis) == 11
    assert _major_pad(ax.yaxis) == 13


@pytest.mark.parametrize("shape", [(1, 3), (2, 2)])
def test_tick_padding_array_of_axes(shape):
    fig, ax = plt.subplots(*shape)
    PlotStyler().set_tick_padding(ax, pad_x=11)
    for axis in ax.ravel():
        assert _major_pad(axis.xaxis) == 11


def test_tick_padding_list_of_axes():
    fig, ax = plt.subplots(1, 2)
    PlotStyler().set_tick_padding(list(ax), pad_y=9)
    assert all(_major_pad(axis.yaxis) == 9 for axis in ax)


def test_apply_style_creates_no_figure():
    PlotStyler().apply_style()
    assert plt.get_fignums() == []


def test_apply_style_leaves_the_current_figure_alone():
    fig, ax = plt.subplots()
    ax.plot([1, 2, 3])
    PlotStyler().apply_style()
    assert fig.axes == [ax]
    assert len(ax.lines) == 1


@pytest.mark.parametrize("shared", [dict(sharex=True), dict(sharey=True)])
def test_apply_style_after_shared_log_axes_emits_no_warning(shared):
    _, ax = plt.subplots(1, 2, **shared)
    for axis in ax:
        axis.plot([1, 10, 100], [1e-3, 1e-1, 1])
        axis.set_xscale("log")
        axis.set_yscale("log")
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        PlotStyler().apply_style()
