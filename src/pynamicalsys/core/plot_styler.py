# plot_styler.py

# Copyright (C) 2025-2026 Matheus Rolim Sales
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/>.

from typing import Any, Iterable, Optional, Union

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes


def _resolve_alias(name: str, value: Any, alias: str, alias_value: Any, default: Any) -> Any:
    """Return the value of a parameter that can also be given by an alias."""
    if value is not None and alias_value is not None:
        raise TypeError(f"Got both '{name}' and '{alias}', which are aliases of one another")
    if alias_value is not None:
        return alias_value
    if value is not None:
        return value
    return default


class PlotStyler:
    """
    A utility class to globally configure and apply consistent styling for Matplotlib plots.

    This class sets default plot aesthetics such as font sizes, line widths, marker styles,
    tick behavior, and more, using Matplotlib's rcParams. It ensures consistent visual style
    across multiple figures and subplots with minimal repetitive code.

    Parameters
    ----------
    fontsize : int, default=20
        Base font size for labels, titles, and text. Tick labels use ``fontsize - 3``.
    legend_fontsize : int, default=14
        Font size for the legend.
    axes_linewidth : float, default=1.3
        Width of the axes borders.
    font_family : str, default="STIXGeneral"
        Font family used for text and math rendering.
    tick_direction_in : bool, default=True
        Whether ticks point inward (``True``) or outward (``False``).
    minor_ticks_visible : bool, default=False
        Whether minor ticks are visible.
    linewidth : float, default=1.0
        Default line width for plot lines. Alias: ``lw``.
    markersize : float, default=5.0
        Default marker size for lines with markers. Alias: ``ms``.
    markeredgewidth : float, default=1.0
        Edge width for plot markers. Alias: ``mew``.
    markeredgecolor : str, default="black"
        Edge color for plot markers. Alias: ``mec``.
    ticks_on_all_sides : bool, default=True
        Whether ticks are also drawn on the top and right axes.
    lw, ms, mew, mec : optional
        Aliases of ``linewidth``, ``markersize``, ``markeredgewidth``, and
        ``markeredgecolor``, as in Matplotlib. They are keyword-only.

    Raises
    ------
    TypeError
        If a parameter and its alias are both given.

    Methods
    -------
    apply_style()
        Applies the defined styling parameters globally using Matplotlib's rcParams.
    set_tick_padding(ax, pad_x=None, pad_y=None)
        Sets custom padding between axis ticks and labels for one or several Axes.

    Examples
    --------
    >>> ps = PlotStyler(fontsize=15, lw=1.5, ms=2, mew=0)
    >>> ps.apply_style()
    """

    def __init__(
        self,
        fontsize: int = 20,
        legend_fontsize: int = 14,
        axes_linewidth: float = 1.3,
        font_family: str = "STIXGeneral",
        tick_direction_in: bool = True,
        minor_ticks_visible: bool = False,
        linewidth: Optional[float] = None,
        markersize: Optional[float] = None,
        markeredgewidth: Optional[float] = None,
        markeredgecolor: Optional[str] = None,
        ticks_on_all_sides: bool = True,
        *,
        lw: Optional[float] = None,
        ms: Optional[float] = None,
        mew: Optional[float] = None,
        mec: Optional[str] = None,
    ) -> None:
        self.fontsize = fontsize
        self.legend_fontsize = legend_fontsize
        self.axes_linewidth = axes_linewidth
        self.font_family = font_family
        self.tick_labelsize = fontsize - 3
        self.tick_direction_in = tick_direction_in
        self.minor_ticks_visible = minor_ticks_visible
        self.linewidth = _resolve_alias("linewidth", linewidth, "lw", lw, 1.0)
        self.markersize = _resolve_alias("markersize", markersize, "ms", ms, 5.0)
        self.markeredgewidth = _resolve_alias(
            "markeredgewidth", markeredgewidth, "mew", mew, 1.0
        )
        self.markeredgecolor = _resolve_alias(
            "markeredgecolor", markeredgecolor, "mec", mec, "black"
        )
        self.ticks_on_all_sides = ticks_on_all_sides

    def apply_style(self) -> None:
        """
        Apply global matplotlib styling.

        All Matplotlib rcParams are first reset to their defaults, so settings
        made before this call are discarded. Only these global settings
        change: existing figures are left as they are, and the style applies to
        the figures created afterwards.
        """
        self._reset_style()  # Reset any previous styles
        plt.rc("font", size=self.fontsize)
        plt.rc("xtick", labelsize=self.tick_labelsize)
        plt.rc("ytick", labelsize=self.tick_labelsize)
        plt.rc("legend", fontsize=self.legend_fontsize)

        # Font and math rendering
        plt.rc("font", family=self.font_family)
        plt.rcParams["mathtext.fontset"] = "stix"

        # Axes linewidth
        mpl.rcParams["axes.linewidth"] = self.axes_linewidth

        # Global tick settings
        if self.tick_direction_in:
            plt.rcParams["xtick.direction"] = "in"
            plt.rcParams["ytick.direction"] = "in"

        if self.minor_ticks_visible:
            plt.rcParams["xtick.minor.visible"] = True
            plt.rcParams["ytick.minor.visible"] = True

        # Ticks on all sides (requires setting this per axis after creation)
        if self.ticks_on_all_sides:
            mpl.rcParams["xtick.top"] = True
            mpl.rcParams["ytick.right"] = True

        # Line and marker styling
        plt.rcParams["lines.linewidth"] = self.linewidth
        plt.rcParams["lines.markersize"] = self.markersize
        plt.rcParams["lines.markeredgewidth"] = self.markeredgewidth
        plt.rcParams["lines.markeredgecolor"] = self.markeredgecolor

    def set_tick_padding(
        self,
        ax: Union[Axes, Iterable[Axes], np.ndarray],
        pad_x: Optional[int] = None,
        pad_y: Optional[int] = None,
    ) -> None:
        """
        Set custom tick padding for one or several Axes.

        Parameters
        ----------
        ax : matplotlib Axes or array-like of Axes
            The axes to modify. It can be a single Axes or any array-like of
            Axes, such as the array returned by ``plt.subplots``, in which case
            the padding is applied to every Axes.
        pad_x : int, optional
            Padding for x-axis major ticks.
        pad_y : int, optional
            Padding for y-axis major ticks.

        Examples
        --------
        >>> fig, ax = plt.subplots(2, 2)
        >>> ps.set_tick_padding(ax, pad_x=8, pad_y=6)
        """
        for axis in np.asarray(ax, dtype=object).ravel():
            if pad_x is not None:
                axis.tick_params(axis="x", which="major", pad=pad_x)
            if pad_y is not None:
                axis.tick_params(axis="y", which="major", pad=pad_y)

    def _reset_style(self) -> None:
        """
        Reset all matplotlib settings to default values.

        This method undoes any customizations made by `apply_style`
        or other changes to `matplotlib.rcParams`.
        """
        mpl.rcdefaults()
        plt.rcdefaults()
