# basins.py

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

from numpy.typing import NDArray
import numpy as np
from sklearn.cluster import DBSCAN

from pynamicalsys.common.types import flow_t
from pynamicalsys.continuous_time.poincare import ensemble_poincare_section
from pynamicalsys.continuous_time.stroboscopic import ensemble_stroboscopic_map


def basin_of_attraction(
    u: NDArray[np.float64],
    parameters: NDArray[np.float64],
    num_intersections: int,
    equations_of_motion: flow_t,
    transient_time: np.float64 | None,
    time_step: np.float64,
    atol: np.float64,
    rtol: np.float64,
    integrator,
    select_map: str,
    section_index: int | None = None,
    section_value: np.float64 | None = None,
    crossing: int | None = None,
    sampling_time: np.float64 | None = None,
    eps: np.float64 = np.float64(0.05),
    min_samples: int = 1,
) -> NDArray[np.int32]:
    """
    Identify attraction basins from reduced trajectory data using clustering.

    Parameters
    ----------
    u : NDArray[np.float64]
        Array of initial conditions with shape `(num_ic, neq)`.
    parameters : NDArray[np.float64]
        System parameters passed to the equations of motion.
    num_intersections : int
        Number of map points recorded for each trajectory.
    equations_of_motion : flow_t
        Continuous-time vector field with signature
        `(time, u, parameters) -> dudt`.
    transient_time : np.float64 | None
        Initial integration time discarded before recording map points.
    time_step : np.float64
        Initial integration step size.
    atol : np.float64
        Absolute tolerance used by adaptive integrators.
    rtol : np.float64
        Relative tolerance used by adaptive integrators.
    integrator : callable
        Step method with unified signature returning
        `(u_next, time_next, time_step_next, accept)`.
    select_map : str
        Map used to construct the reduced trajectory data:
        - `"PS"` for Poincaré section
        - `"SM"` for stroboscopic map
    section_index : int | None, optional
        Coordinate index defining the Poincaré section when `select_map="PS"`.
    section_value : np.float64 | None, optional
        Section value for the Poincaré section when `select_map="PS"`.
    crossing : int | None, optional
        Crossing orientation selector for the Poincaré section when
        `select_map="PS"`.
    sampling_time : np.float64 | None, optional
        Sampling interval for the stroboscopic map when `select_map="SM"`.
    eps : np.float64, optional
        Neighborhood radius used by DBSCAN.
    min_samples : int, optional
        Minimum number of samples required by DBSCAN to form a cluster.

    Returns
    -------
    NDArray[np.int32]
        Cluster labels associated with each initial condition. Equal labels are
        interpreted as belonging to the same basin, and `-1` denotes noise as
        assigned by DBSCAN.

    Notes
    -----
    The clustering is performed on trajectory centroids computed from the map
    points after removing the time column. This function does not integrate the
    full basin boundary directly; it groups trajectories according to the
    geometry of their reduced asymptotic data.
    """
    if select_map == "PS":
        if section_index is None or section_value is None or crossing is None:
            raise ValueError(
                "section_index, section_value, and crossing must be provided when select_map='PS'"
            )

        data = ensemble_poincare_section(
            u=u,
            parameters=parameters,
            num_intersections=num_intersections,
            equations_of_motion=equations_of_motion,
            transient_time=transient_time,
            time_step=time_step,
            atol=atol,
            rtol=rtol,
            integrator=integrator,
            section_index=section_index,
            section_value=section_value,
            crossing=crossing,
        )

    elif select_map == "SM":
        if sampling_time is None:
            raise ValueError("sampling_time must be provided when select_map='SM'")

        data = ensemble_stroboscopic_map(
            u=u,
            parameters=parameters,
            num_intersections=num_intersections,
            sampling_time=sampling_time,
            equations_of_motion=equations_of_motion,
            transient_time=transient_time,
            time_step=time_step,
            atol=atol,
            rtol=rtol,
            integrator=integrator,
        )

    else:
        raise ValueError("select_map must be either 'PS' or 'SM'")

    traj_data = data[:, :, 1:]
    trajectory_centroids = traj_data.mean(axis=1)

    db = DBSCAN(eps=float(eps), min_samples=min_samples, n_jobs=-1).fit(
        trajectory_centroids
    )

    return np.asarray(db.labels_, dtype=np.int32)
