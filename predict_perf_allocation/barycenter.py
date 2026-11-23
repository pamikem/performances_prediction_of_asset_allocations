import logging

import numpy as np
from scipy.optimize import minimize

from predict_perf_allocation.soft_dtw import SoftDTW

logger = logging.getLogger(__name__)


def _as_series_collection(X):
    """Convert a collection of sequences to 2D float arrays."""
    if isinstance(X, np.ndarray):
        if X.ndim == 2:
            sequences = [row[:, np.newaxis] for row in X]
        elif X.ndim == 3:
            sequences = [X[index] for index in range(X.shape[0])]
        else:
            raise ValueError(
                "X must have shape [n_series, n_timestamps] or "
                "[n_series, n_timestamps, n_features], or be a sequence of 1D/2D arrays."
            )
    else:
        try:
            sequences = list(X)
        except TypeError as error:
            raise ValueError("X must be a non-empty collection of time series.") from error

    if not sequences:
        raise ValueError("X must contain at least one time series.")

    normalized = []
    n_features = None
    for index, sequence in enumerate(sequences):
        sequence = np.asarray(sequence, dtype=np.float64)
        if sequence.ndim == 1:
            sequence = sequence[:, np.newaxis]
        if sequence.ndim != 2 or 0 in sequence.shape:
            raise ValueError(
                f"X[{index}] must have shape [n_timestamps, n_features] "
                "with non-empty dimensions."
            )
        if n_features is None:
            n_features = sequence.shape[1]
        elif sequence.shape[1] != n_features:
            raise ValueError("All time series in X must have the same feature count.")
        normalized.append(sequence)

    return normalized


def sdtw_barycenter(
    X,
    barycenter_init=None,
    gamma=1.0,
    weights=None,
    lambda_=0.0,
    rho=0.0,
    method="L-BFGS-B",
    tol=1e-3,
    max_iter=50,
    return_result=False,
):
    """Compute a Soft-DTW barycenter of a collection of time series.

    The barycenter minimizes the weighted sum of Soft-DTW costs between itself
    and each input series. Inputs may have different lengths, but must have the
    same number of features.

    Parameters
    ----------
    X : array-like
        A collection of sequences. A 2D array is interpreted as
        ``[n_series, n_timestamps]`` univariate data; a 3D array uses shape
        ``[n_series, n_timestamps, n_features]``. A sequence of 1D or 2D arrays
        also supports series with different lengths.
    barycenter_init : array-like, optional
        Initial barycenter with shape ``[n_timestamps, n_features]``. If None,
        the arithmetic mean initializes the optimizer when all input lengths
        match; otherwise an explicit initialization is required.
    gamma : float, default=1.0
        Soft-DTW smoothing parameter. Smaller values approach hard DTW.
    weights : array-like, optional
        Non-negative weights, one per input sequence. They are normalized to
        sum to one. Equal weights are used by default.
    lambda_ : float, default=0.0
        Timestamp mismatch penalty, matching the project's SoftDTW interface.
    rho : float, default=0.0
        Penalty for invalid pairwise point distances, matching the project's
        SoftDTW interface. Project input data is expected to be finite.
    method : str, default="L-BFGS-B"
        Optimization method passed to ``scipy.optimize.minimize``.
    tol : float, default=1e-3
        Optimizer tolerance.
    max_iter : int, default=50
        Maximum optimizer iterations.
    return_result : bool, default=False
        If True, return SciPy's optimization result with its ``x`` field reshaped
        to the barycenter dimensions. Otherwise return only the barycenter array.

    Returns
    -------
    ndarray or OptimizeResult
        The barycenter, shaped ``[n_timestamps, n_features]``, or the optimizer
        result when ``return_result=True``.
    """
    sequences = _as_series_collection(X)

    if barycenter_init is None:
        lengths = {sequence.shape[0] for sequence in sequences}
        if len(lengths) != 1:
            raise ValueError(
                "barycenter_init is required when input time series have different lengths."
            )
        barycenter_init = np.mean(np.stack(sequences), axis=0)
    else:
        barycenter_init = np.asarray(barycenter_init, dtype=np.float64)
        if barycenter_init.ndim == 1:
            barycenter_init = barycenter_init[:, np.newaxis]
        if barycenter_init.ndim != 2 or 0 in barycenter_init.shape:
            raise ValueError("barycenter_init must have shape [length, n_features].")
    if barycenter_init.shape[1] != sequences[0].shape[1]:
        raise ValueError("barycenter_init must have the same feature count as X.")
    if not np.isfinite(gamma) or gamma <= 0:
        raise ValueError("gamma must be a finite positive value.")
    if not np.isfinite(lambda_) or lambda_ < 0:
        raise ValueError("lambda_ must be a finite non-negative value.")
    if not np.isfinite(rho) or rho < 0:
        raise ValueError("rho must be a finite non-negative value.")
    if not np.isfinite(tol) or tol < 0:
        raise ValueError("tol must be a finite non-negative value.")
    if not isinstance(max_iter, int) or max_iter <= 0:
        raise ValueError("max_iter must be a positive integer.")

    if weights is None:
        weights = np.ones(len(sequences), dtype=np.float64)
    else:
        weights = np.asarray(weights, dtype=np.float64)
    if weights.ndim != 1 or weights.shape[0] != len(sequences):
        raise ValueError("weights must contain one value per input time series.")
    if not np.isfinite(weights).all() or np.any(weights < 0):
        raise ValueError("weights must be finite and non-negative.")
    weight_sum = weights.sum()
    if weight_sum <= 0:
        raise ValueError("weights must have a positive sum.")
    weights = weights / weight_sum

    barycenter_shape = barycenter_init.shape

    def objective(flat_barycenter):
        barycenter = flat_barycenter.reshape(barycenter_shape)
        gradient = np.zeros_like(barycenter)
        value = 0.0

        for sequence, weight in zip(sequences, weights, strict=True):
            soft_dtw = SoftDTW.from_timeseries(
                barycenter,
                sequence,
                gamma=gamma,
                lambda_=lambda_,
                rho=rho,
            )
            value += weight * soft_dtw.compute()
            gradient += weight * soft_dtw.grad_x()

        return value, gradient.ravel()

    result = minimize(
        objective,
        barycenter_init.ravel(),
        method=method,
        jac=True,
        tol=tol,
        options={"maxiter": max_iter},
    )
    if not result.success:
        logger.warning(
            "Soft-DTW barycenter optimizer did not converge (status %s): %s",
            result.status,
            result.message,
        )
    result.x = result.x.reshape(barycenter_shape)
    result.barycenter = result.x
    if return_result:
        return result
    return result.x
