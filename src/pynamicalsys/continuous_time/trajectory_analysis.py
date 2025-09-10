# trajectory_analysis.py

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

from typing import Callable, Dict, List, Optional, Sequence, Tuple, Union

from joblib import Parallel, delayed
import numpy as np
from numba import njit, prange
from numpy.typing import NDArray
from sklearn.cluster import DBSCAN

from pynamicalsys.continuous_time.numerical_integrators import rk4_step_wrapped


@njit
def step(
    time: np.float64,
    u: NDArray[np.float64],
    parameters: NDArray[np.float64],
    equations_of_motion: Callable[
        [NDArray[np.float64], NDArray[np.float64]], NDArray[np.float64]
    ],
    jacobian: Optional[
        Callable[
            [np.float64, NDArray[np.float64], NDArray[np.float64]], NDArray[np.float64]
        ]
    ] = None,
    time_step: float = 0.01,
    atol: float = 1e-6,
    rtol: float = 1e-3,
    integrator=rk4_step_wrapped,
    number_of_deviation_vectors: Optional[int] = None,
) -> NDArray[np.float64]:

    u = u.copy()
    accept = False

    while not accept:
        u_new, time_new, time_step_new, accept = integrator(
            time,
            u,
            parameters,
            equations_of_motion,
            jacobian=jacobian,
            time_step=time_step,
            atol=atol,
            rtol=rtol,
            number_of_deviation_vectors=number_of_deviation_vectors,
        )
        if accept:
            time = time_new
            u = u_new.copy()

        time_step = time_step_new

    return u_new, time_new, time_step_new


@njit
def evolve_system(
    u: NDArray[np.float64],
    parameters: NDArray[np.float64],
    total_time: float,
    equations_of_motion: Callable[
        [NDArray[np.float64], NDArray[np.float64]], NDArray[np.float64]
    ],
    time_step: float = 0.01,
    atol: float = 1e-6,
    rtol: float = 1e-3,
    integrator=rk4_step_wrapped,
) -> NDArray[np.float64]:

    u = u.copy()

    time = 0
    while time < total_time:
        u, time, time_step = step(
            time,
            u,
            parameters,
            equations_of_motion,
            time_step=time_step,
            atol=atol,
            rtol=rtol,
            integrator=integrator,
        )
        if time + time_step > total_time:
            time_step = total_time - time

    return u


@njit
def generate_trajectory(
    u: NDArray[np.float64],
    parameters: NDArray[np.float64],
    total_time: float,
    equations_of_motion: Callable[
        [np.float64, NDArray[np.float64], NDArray[np.float64]], NDArray[np.float64]
    ],
    transient_time: Optional[float] = None,
    time_step: float = 0.01,
    atol: float = 1e-6,
    rtol: float = 1e-3,
    integrator=rk4_step_wrapped,
) -> NDArray[np.float64]:

    u = u.copy()
    if transient_time is not None:
        u = evolve_system(
            u,
            parameters,
            transient_time,
            equations_of_motion,
            time_step=time_step,
            atol=atol,
            rtol=rtol,
            integrator=integrator,
        )
        time = transient_time
    else:
        time = 0

    neq = len(u)
    result = np.zeros(neq + 1)
    trajectory = []

    while time < total_time:
        if time + time_step > total_time:
            time_step = total_time - time

        u, time, time_step = step(
            time,
            u,
            parameters,
            equations_of_motion,
            time_step=time_step,
            atol=atol,
            rtol=rtol,
            integrator=integrator,
        )

        result = [time]
        for i in range(neq):
            result.append(u[i])
        trajectory.append(result)

    return trajectory


def ensemble_trajectories(
    u: NDArray[np.float64],
    parameters: NDArray[np.float64],
    total_time: float,
    equations_of_motion: Callable[
        [np.float64, NDArray[np.float64], NDArray[np.float64]], NDArray[np.float64]
    ],
    transient_time: Optional[float] = None,
    time_step: float = 0.01,
    atol: float = 1e-6,
    rtol: float = 1e-3,
    integrator=rk4_step_wrapped,
) -> NDArray[np.float64]:

    def run_one(u_i, parameters, total_time, equations_of_motion, **kwargs):
        result = generate_trajectory(
            u_i, parameters, total_time, equations_of_motion, **kwargs
        )

        return np.array(result)

    results = Parallel(n_jobs=-1)(  # -1 = use all cores
        delayed(run_one)(
            u[i],
            parameters,
            total_time,
            equations_of_motion,
            transient_time=transient_time,
            time_step=time_step,
            atol=atol,
            rtol=rtol,
            integrator=integrator,
        )
        for i in range(len(u))
    )

    return results


@njit
def generate_poincare_section(
    u: NDArray[np.float64],
    parameters: NDArray[np.float64],
    num_intersections: int,
    equations_of_motion: Callable[
        [np.float64, NDArray[np.float64], NDArray[np.float64]], NDArray[np.float64]
    ],
    transient_time: float,
    time_step: float,
    atol: float,
    rtol: float,
    integrator,
    section_index: int,
    section_value: float,
    crossing: int,
) -> NDArray[np.float64]:
    neq = len(u)
    section_points = np.zeros((num_intersections, neq + 1))
    count = 0

    u = u.copy()
    if transient_time is not None:
        u = evolve_system(
            u,
            parameters,
            transient_time,
            equations_of_motion,
            time_step=time_step,
            atol=atol,
            rtol=rtol,
            integrator=integrator,
        )
        time = transient_time
    else:
        time = 0

    time_step_prev = time_step
    time_prev = time
    u_prev = u.copy()
    while count < num_intersections:
        u_new, time_new, time_step_new = step(
            time_prev,
            u_prev,
            parameters,
            equations_of_motion,
            time_step=time_step_prev,
            atol=atol,
            rtol=rtol,
            integrator=integrator,
        )

        # Check for crossings
        if (u_prev[section_index] - section_value) * (
            u_new[section_index] - section_value
        ) < 0.0:
            lam = (section_value - u_prev[section_index]) / (
                u_new[section_index] - u_prev[section_index]
            )

            t_cross = time_new - time_step_prev + lam * time_step_prev
            u_cross = (1 - lam) * u_prev + lam * u_new
            velocity = equations_of_motion(time, u_cross, parameters)[section_index]

            if crossing == 0 or np.sign(velocity) == crossing:
                section_points[count, 0] = t_cross
                section_points[count, 1:] = u_cross
                count += 1

        time_prev = time_new
        time_step_prev = time_step_new
        u_prev = u_new

    return section_points


@njit(parallel=True)
def ensemble_poincare_section(
    u: NDArray[np.float64],
    parameters: NDArray[np.float64],
    num_intersections: int,
    equations_of_motion: Callable[
        [np.float64, NDArray[np.float64], NDArray[np.float64]], NDArray[np.float64]
    ],
    transient_time: float,
    time_step: float,
    atol: float,
    rtol: float,
    integrator,
    section_index: int,
    section_value: float,
    crossing: int,
) -> NDArray[np.float64]:
    num_ic, neq = u.shape
    section_points = np.zeros((num_ic, num_intersections, neq + 1))
    for i in prange(num_ic):
        section_points[i] = generate_poincare_section(
            u[i],
            parameters,
            num_intersections,
            equations_of_motion,
            transient_time,
            time_step,
            atol,
            rtol,
            integrator,
            section_index,
            section_value,
            crossing,
        )

    return section_points


@njit
def generate_stroboscopic_map(
    u: NDArray[np.float64],
    parameters: NDArray[np.float64],
    num_samples: int,
    sampling_time: float,
    equations_of_motion: Callable[
        [np.float64, NDArray[np.float64], NDArray[np.float64]], NDArray[np.float64]
    ],
    transient_time: float,
    time_step: float,
    atol: float,
    rtol: float,
    integrator,
) -> NDArray[np.float64]:

    u = np.asarray(u)
    neq = len(u)
    strobe_points = np.zeros((num_samples, neq + 1))
    if transient_time is not None:
        u_curr = evolve_system(
            u,
            parameters,
            transient_time,
            equations_of_motion,
            time_step=time_step,
            atol=atol,
            rtol=rtol,
            integrator=integrator,
        )
        time_curr = transient_time
    else:
        u_curr = u.copy()
        time_curr = 0

    time_target = time_curr + sampling_time
    count = 0
    while count < num_samples:
        u_prev = u_curr.copy()
        time_prev = time_curr
        # Integrate until we reach or surpass the target strobe time
        while time_curr < time_target:
            u_curr, time_curr, time_step = step(
                time_curr,
                u_curr,
                parameters,
                equations_of_motion,
                time_step=time_step,
                atol=atol,
                rtol=rtol,
                integrator=integrator,
            )

        # Linear interpolation to exactly hit time_target
        lam = (time_target - time_prev) / (time_curr - time_prev)
        strobe_points[count, 0] = time_target
        strobe_points[count, 1:] = (1 - lam) * u_prev + lam * u_curr

        count += 1
        time_target += sampling_time

    return strobe_points


@njit(parallel=True)
def ensemble_stroboscopic_map(
    u: NDArray[np.float64],
    parameters: NDArray[np.float64],
    num_samples: int,
    sampling_time: float,
    equations_of_motion: Callable[
        [np.float64, NDArray[np.float64], NDArray[np.float64]], NDArray[np.float64]
    ],
    transient_time: float,
    time_step: float,
    atol: float,
    rtol: float,
    integrator,
) -> NDArray[np.float64]:
    num_ic, neq = u.shape
    strobe_points = np.zeros((num_ic, num_samples, neq + 1))

    for i in prange(num_ic):
        strobe_points[i] = generate_stroboscopic_map(
            u[i],
            parameters,
            num_samples,
            sampling_time,
            equations_of_motion,
            transient_time,
            time_step,
            atol,
            rtol,
            integrator,
        )

    return strobe_points


def basin_of_attraction(
    u: NDArray[np.float64],
    parameters: NDArray[np.float64],
    equations_of_motion: Callable[
        [np.float64, NDArray[np.float64], NDArray[np.float64]], NDArray[np.float64]
    ],
    transient_time: float,
    time_step: float,
    atol: float,
    rtol: float,
    integrator: Callable,
    select_map: str,
    total_time: float = None,
    num_intersections: int = None,
    section_index: int = None,
    section_value: float = None,
    crossing: int = None,
    sampling_time: float = None,
    eps: float = 0.05,
    min_samples: int = 1,
) -> NDArray[np.int32]:

    if select_map == "PS":
        if (
            num_intersections is None
            or section_index is None
            or section_value is None
            or crossing is None
        ):
            raise ValueError(
                "You must provide num_intersections, section_index, section_value, and crossing"
            )
        data = ensemble_poincare_section(
            u,
            parameters,
            num_intersections,
            equations_of_motion,
            transient_time,
            time_step,
            atol,
            rtol,
            integrator,
            section_index,
            section_value,
            crossing,
        )

        traj_data = data[:, :, 1:]
        trajectory_centroids = traj_data.mean(axis=1)  # shape (num_ic, 2)
    elif select_map == "SM":
        if num_intersections is None or sampling_time is None:
            raise ValueError("You must provide num_intersections and sampling_time")

        data = ensemble_stroboscopic_map(
            u,
            parameters,
            num_intersections,
            sampling_time,
            equations_of_motion,
            transient_time,
            time_step,
            atol,
            rtol,
            integrator,
        )
        traj_data = data[:, :, 1:]
        trajectory_centroids = traj_data.mean(axis=1)  # shape (num_ic, 2)
    else:
        if total_time is None:
            raise ValueError("You must provide total_time")

        trajectories = ensemble_trajectories(
            u,
            parameters,
            total_time,
            equations_of_motion,
            transient_time,
            time_step,
            atol,
            rtol,
            integrator,
        )

        for i, traj in enumerate(trajectories):
            x = traj[:, 1]  # x coordinate
            v = traj[:, 2]  # v coordinate
            trajectory_centroids[i, 0] = np.mean(x)
            trajectory_centroids[i, 1] = np.mean(v)

    db = DBSCAN(eps=eps, min_samples=min_samples, n_jobs=-1).fit(trajectory_centroids)
    labels = db.labels_  # shape (num_ic,)

    return labels
