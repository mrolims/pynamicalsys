# display.py

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

"""
Optional rich display helpers.

IPython is an optional dependency of `pynamicalsys`. It is used only to render
the governing equations of the built-in models as typeset mathematics inside a
notebook. Importing it eagerly would make a notebook-only convenience a hard
requirement of a numerical library, so the import is deferred to the moment a
model's `info` property is actually accessed.
"""

from typing import Any


def render_latex(latex: str) -> Any:
    """
    Wrap a LaTeX string in an object that renders as typeset mathematics.

    Parameters
    ----------
    latex : str
        LaTeX source describing the governing equations of a model.

    Returns
    -------
    IPython.display.Math or str
        An `IPython.display.Math` instance when IPython is installed, so that
        the equation renders in a notebook. If IPython is not available, the
        LaTeX source is returned unchanged, so that `info` remains usable in a
        plain interpreter or a headless script.

    Notes
    -----
    Install the optional dependency with `pip install pynamicalsys[notebook]`
    to get rendered output. A plain-text form of every equation is always
    available via `info["equation_readable"]`, regardless of whether IPython
    is installed.
    """
    if not isinstance(latex, str):
        return latex

    try:
        from IPython.display import Math
    except ImportError:
        return latex

    return Math(latex)
