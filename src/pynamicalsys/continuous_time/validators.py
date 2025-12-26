# validators.py

# Copyright (C) 2025 Matheus Rolim Sales
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

from numbers import Integral, Real


def validate_times(transient_time, total_time, type_=Real) -> tuple[float, float]:

    if isinstance(total_time, (Integral, Real)):
        total_time = float(total_time)
    else:
        raise ValueError("total_time must be a valid number")
    if total_time < 0:
        raise ValueError("total_time must be non-negative")

    if transient_time is not None:

        if isinstance(transient_time, (Integral, Real)):
            transient_time = float(transient_time)
        else:
            raise ValueError("transient_time must be a valid number")
        if transient_time < 0:
            raise ValueError("transient_time must be non-negative")

        if transient_time >= total_time:
            raise ValueError("transient_time must be less than total_time")

    return transient_time, total_time
