import numpy as np
from numpy.typing import NDArray

from pynamicalsys.common.types import jacobian_t, map_t, numeric_t


def eigenvalues_and_eigenvectors(
    u: NDArray[np.float64],
    parameters: NDArray[np.float64],
    mapping: map_t,
    jacobian: jacobian_t,
    period: int,
    normalize: bool = True,
    sort_by_magnitude: bool = True,
) -> tuple[NDArray[np.complex128], NDArray[np.complex128]]:
    """
    Compute the eigenvalues and eigenvectors of the monodromy matrix of a
    discrete-time periodic orbit.

    The monodromy matrix is the Jacobian of the `period`-times iterated map
    evaluated along the orbit starting from `u`. Its eigenvalues are the
    Floquet multipliers of the orbit.

    Parameters
    ----------
    u : NDArray[np.float64]
        Initial condition of shape `(system_dimension,)`.
    parameters : NDArray[np.float64]
        System parameters.
    mapping : map_t
        Discrete-time map.
    jacobian : jacobian_t
        Jacobian of the map.
    period : int
        Period of the orbit.
    normalize : bool, optional
        If True, normalize the returned eigenvectors to unit Euclidean norm.
    sort_by_magnitude : bool, optional
        If True, sort the eigenpairs by decreasing eigenvalue magnitude.

    Returns
    -------
    tuple[NDArray[np.complex128], NDArray[np.complex128]]
        A tuple `(eigenvalues, eigenvectors)` where

        - `eigenvalues` has shape `(system_dimension,)`
        - `eigenvectors` has shape `(system_dimension, system_dimension)`

        Each column of `eigenvectors` is an eigenvector associated with the
        eigenvalue in the same position.

    Notes
    -----
    The monodromy matrix is constructed as

    `M = J(u_{p-1}) @ ... @ J(u_1) @ J(u_0)`

    where `u_{n+1} = mapping(u_n, parameters)`.
    """
    current_u = np.asarray(u, dtype=np.float64).copy()
    dim = current_u.size

    monodromy = np.eye(dim, dtype=np.complex128)

    for _ in range(period):
        J = np.asarray(
            jacobian(current_u, parameters, mapping),
            dtype=np.complex128,
        )
        monodromy = J @ monodromy
        current_u = mapping(current_u, parameters)

    eigenvalues, eigenvectors = np.linalg.eig(monodromy)

    if normalize:
        for i in range(dim):
            norm = np.linalg.norm(eigenvectors[:, i])
            if norm > 0.0:
                eigenvectors[:, i] /= norm

    if sort_by_magnitude:
        order = np.argsort(np.abs(eigenvalues))[::-1]
        eigenvalues = eigenvalues[order]
        eigenvectors = eigenvectors[:, order]

    return eigenvalues, eigenvectors


def classify_stability(
    u: NDArray[np.float64],
    parameters: NDArray[np.float64],
    mapping: map_t,
    jacobian: jacobian_t,
    period: int,
    threshold: numeric_t = 1.0,
    tol: numeric_t = 1e-8,
) -> dict[str, str | NDArray[np.complex128]]:
    """
    Classify the local linear stability of a 2D periodic orbit of a
    discrete-time map.

    The classification is based on the Floquet multipliers, i.e., the
    eigenvalues of the monodromy matrix.

    Parameters
    ----------
    u : NDArray[np.float64]
        Initial condition of shape `(2,)`.
    parameters : NDArray[np.float64]
        System parameters.
    mapping : map_t
        Discrete-time map.
    jacobian : jacobian_t
        Jacobian of the map.
    period : int
        Period of the orbit.
    threshold : numeric_t, optional
        Reference radius used to separate contracting and expanding
        multipliers. For standard discrete-time stability analysis this should
        remain equal to `1.0`.
    tol : numeric_t, optional
        Numerical tolerance used when deciding whether a multiplier lies on
        the threshold.

    Returns
    -------
    dict[str, str | NDArray[np.complex128]]
        Dictionary with keys

        - `"classification"` : stability label
        - `"eigenvalues"` : Floquet multipliers
        - `"eigenvectors"` : corresponding eigenvectors

    Notes
    -----
    The returned classification follows this convention:

    - `"stable node"`
    - `"stable spiral"`
    - `"unstable node"`
    - `"unstable spiral"`
    - `"saddle"`
    - `"center"`
    - `"elliptic (quasi-periodic)"`
    - `"marginal or degenerate"`
    """
    eigenvalues, eigenvectors = eigenvalues_and_eigenvectors(
        u=u,
        parameters=parameters,
        mapping=mapping,
        jacobian=jacobian,
        period=period,
        normalize=True,
        sort_by_magnitude=True,
    )

    lam1, lam2 = eigenvalues
    abs_lam1 = np.abs(lam1)
    abs_lam2 = np.abs(lam2)

    is_real = np.isreal(lam1) and np.isreal(lam2)

    if abs_lam1 < threshold - tol and abs_lam2 < threshold - tol:
        classification = "stable node" if is_real else "stable spiral"
    elif abs_lam1 > threshold + tol and abs_lam2 > threshold + tol:
        classification = "unstable node" if is_real else "unstable spiral"
    elif (abs_lam1 < threshold - tol and abs_lam2 > threshold + tol) or (
        abs_lam2 < threshold - tol and abs_lam1 > threshold + tol
    ):
        classification = "saddle"
    elif abs(abs_lam1 - threshold) <= tol and abs(abs_lam2 - threshold) <= tol:
        classification = "center" if is_real else "elliptic (quasi-periodic)"
    else:
        classification = "marginal or degenerate"

    return {
        "classification": classification,
        "eigenvalues": eigenvalues,
        "eigenvectors": eigenvectors,
    }
