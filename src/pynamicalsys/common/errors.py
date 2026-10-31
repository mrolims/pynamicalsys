# errors.py

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


class MissingFunctionError(ValueError, RuntimeError):
    """
    A function that a method needs, such as the backward map, was not given.

    It is a `ValueError`, as the other errors about missing or invalid
    arguments, and also a `RuntimeError`, which some methods raised for this
    before, so that code catching either keeps working.
    """
