from math import exp, log

from numba import njit, prange, set_num_threads
import numpy as np

set_num_threads(4)
DBL_MAX = np.finfo(np.float64).max


@njit
def _softmin3(a, b, c, gamma):
    a /= -gamma
    b /= -gamma
    c /= -gamma

    max_val = max(a, max(b, c))

    tmp = exp(a - max_val) + exp(b - max_val) + exp(c - max_val)

    return -gamma * (log(tmp) + max_val)


@njit
def _soft_dtw(D, R, gamma):
    m = D.shape[0]
    n = D.shape[1]

    # Initialization: fill with DBL_MAX, then set R[0,0] = 0
    for i in range(m + 1):
        for j in range(n + 1):
            R[i, j] = 1.7976931348623157e308  # DBL_MAX

    R[0, 0] = 0.0

    # DP recursion (D indexed from 0, R from 1)
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            R[i, j] = D[i - 1, j - 1] + _softmin3(R[i - 1, j], R[i - 1, j - 1], R[i, j - 1], gamma)


@njit
def _squared_euclidean_distances(X, Y, D, lambda_, rho):
    m = X.shape[0]
    n = Y.shape[0]
    d = X.shape[1]

    for i in range(m):
        for j in range(n):
            dist = 0.0
            valid = True
            for k in range(d):
                diff = X[i, k] - Y[j, k]
                sq_dist = diff * diff
                if not np.isfinite(sq_dist):
                    valid = False
                    break
                dist += sq_dist

            if valid and np.isfinite(dist):
                point_dist = dist
            else:
                point_dist = rho

            time_diff = i - j
            D[i, j] = point_dist + lambda_ * time_diff * time_diff


@njit(parallel=True)
def _pairwise_soft_dtw(X, Y, out, gamma, lambda_, rho):
    n_x = X.shape[0]
    n_y = Y.shape[0]
    m = X.shape[1]
    n = Y.shape[1]

    for pair_idx in prange(n_x * n_y):
        i = pair_idx // n_y
        j = pair_idx % n_y

        D = np.empty((m, n), dtype=np.float64)
        R = np.zeros((m + 2, n + 2), dtype=np.float64)

        _squared_euclidean_distances(X[i], Y[j], D, lambda_, rho)
        _soft_dtw(D, R, gamma)

        out[i, j] = R[m, n]


@njit(parallel=True)
def _pairwise_soft_dtw_symmetric(X, out, gamma, lambda_, rho):
    n_series = X.shape[0]
    m = X.shape[1]

    for pair_idx in prange(n_series * n_series):
        i = pair_idx // n_series
        j = pair_idx % n_series

        if j < i:
            continue

        D = np.empty((m, m), dtype=np.float64)
        R = np.zeros((m + 2, m + 2), dtype=np.float64)

        _squared_euclidean_distances(X[i], X[j], D, lambda_, rho)
        _soft_dtw(D, R, gamma)

        out[i, j] = R[m, m]
        out[j, i] = out[i, j]


@njit
def _soft_dtw_grad(D, R, E, gamma):
    m = D.shape[0] - 1
    n = D.shape[1] - 1

    # Zero out E
    for i in range(m + 2):
        for j in range(n + 2):
            E[i, j] = 0.0

    # Set boundary conditions (matching Cython memset + loop logic)
    for i in range(1, m + 1):
        D[i - 1, n] = 0.0
        R[i, n + 1] = -1.7976931348623157e308  # -DBL_MAX

    for j in range(1, n + 1):
        D[m, j - 1] = 0.0
        R[m + 1, j] = -1.7976931348623157e308  # -DBL_MAX

    E[m + 1, n + 1] = 1.0
    R[m + 1, n + 1] = R[m, n]
    D[m, n] = 0.0

    # DP backward recursion
    for j in range(n, 0, -1):
        for i in range(m, 0, -1):
            a = exp((R[i + 1, j] - R[i, j] - D[i, j - 1]) / gamma)
            b = exp((R[i, j + 1] - R[i, j] - D[i - 1, j]) / gamma)
            c = exp((R[i + 1, j + 1] - R[i, j] - D[i, j]) / gamma)
            E[i, j] = E[i + 1, j] * a + E[i, j + 1] * b + E[i + 1, j + 1] * c


@njit
def _jacobian_product_sq_euc(X, Y, E, G):
    """
    Jacobian-vector product for squared Euclidean distance.
    Used to compute gradients w.r.t. X when D_ij = ||X_i - Y_j||^2.

    Parameters
    ----------
    X : array, shape = [m, d]  (barycenter candidate)
    Y : array, shape = [n, d]  (target time series)
    E : array, shape = [m, n]  (soft-DTW gradient w.r.t. D)
    G : array, shape = [m, d]  (output gradient w.r.t. X, accumulated in-place)
    """
    m = X.shape[0]
    n = Y.shape[0]
    d = X.shape[1]

    for i in range(m):
        for j in range(n):
            for k in range(d):
                G[i, k] += E[i, j] * 2 * (X[i, k] - Y[j, k])


def _as_timeseries(X):
    X = np.asarray(X, dtype=np.float64)

    if X.ndim == 1:
        X = X.reshape(-1, 1)

    elif X.ndim != 2:
        raise ValueError("A single time series must be a 1D or 2D array.")

    _check_timeseries(X)
    return X


def _as_collection(X):
    X = np.asarray(X, dtype=np.float64)

    if X.ndim == 2:
        X = X[:, :, np.newaxis]

    elif X.ndim != 3:
        raise ValueError(
            "A collection of time series must be a 2D array [n_series, n_timestamps] "
            "or a 3D array [n_series, n_timestamps, n_features]."
        )

    _check_collection(X)
    return X


def _check_timeseries(X):
    if 0 in X.shape:
        raise ValueError("A time series must have at least one timestamp and one feature.")


def _check_collection(X):
    if 0 in X.shape:
        raise ValueError(
            "A collection must have at least one sequence, one timestamp, and one feature."
        )


def _check_collections_compatible(X, Y):
    if X.shape[2] != Y.shape[2]:
        raise ValueError("X and Y must have the same number of features.")


def _check_lambda(lambda_):
    if not np.isfinite(lambda_) or lambda_ < 0:
        raise ValueError("lambda_ must be a finite non-negative value.")


def _check_rho(rho):
    if not np.isfinite(rho) or rho < 0:
        raise ValueError("rho must be a finite non-negative value.")


def _check_gamma(gamma):
    if not np.isfinite(gamma) or gamma <= 0:
        raise ValueError("gamma must be a finite positive value.")
    return float(gamma)


def _as_distance_matrix(D, lambda_=0.0, rho=0.0):
    D = np.asarray(D, dtype=np.float64)

    if D.ndim != 2:
        raise ValueError("A distance matrix must be a 2D array.")

    if 0 in D.shape:
        raise ValueError("A distance matrix must have at least one row and one column.")

    D = D.copy()
    D[~np.isfinite(D)] = rho

    if lambda_ > 0:
        t = np.arange(D.shape[0], dtype=np.float64)
        s = np.arange(D.shape[1], dtype=np.float64)
        D += lambda_ * (t[:, np.newaxis] - s[np.newaxis, :]) ** 2
        D[~np.isfinite(D)] = rho

    return D


class SoftDTW:
    def __init__(self, D, gamma=1.0, lambda_=0.0, rho=0.0):
        """
        Parameters
        ----------
        D : array, shape = [m, n]
            Distance matrix between elements of two time series.
        gamma : float
            Regularization parameter.
            Lower is less smoothed (closer to true DTW).
        lambda_ : float
            Non-negative timestamp mismatch weight. Adds
            lambda_ * (i - j) ** 2 to each distance matrix entry.
        rho : float
            Finite non-negative penalty used for invalid or infinite distance
            entries.

        Attributes
        ----------
        self.R_ : array, shape = [m + 2, n + 2]
            Accumulated cost matrix (stored after calling `compute`).
        """
        gamma = _check_gamma(gamma)
        _check_lambda(lambda_)
        _check_rho(rho)

        if hasattr(D, "compute"):
            self.D = D.compute()
        else:
            self.D = D

        self.D = _as_distance_matrix(self.D, lambda_, rho)
        self.gamma = gamma
        self.lambda_ = lambda_
        self.rho = rho

    def compute(self):
        """
        Compute soft-DTW by dynamic programming.

        Returns
        -------
        sdtw : float
            soft-DTW discrepancy.
        """
        m, n = self.D.shape

        # +2: indices start from 1, and edge cases in backward recursion
        self.R_ = np.zeros((m + 2, n + 2), dtype=np.float64)

        _soft_dtw(self.D, self.R_, self.gamma)

        return self.R_[m, n]

    def grad(self):
        """
        Compute gradient of soft-DTW w.r.t. D by dynamic programming.

        Returns
        -------
        grad : array, shape = [m, n]
            Gradient w.r.t. D.
        """
        if not hasattr(self, "R_"):
            raise ValueError("Needs to call compute() first.")

        m, n = self.D.shape

        # Pad D with an extra row and column (edge cases in backward pass)
        D = np.vstack((self.D, np.zeros(n)))
        D = np.hstack((D, np.zeros((m + 1, 1))))

        # +2: indices start from 1, edge cases in recursion
        E = np.zeros((m + 2, n + 2), dtype=np.float64)

        _soft_dtw_grad(D, self.R_, E, self.gamma)

        return E[1:-1, 1:-1]

    @classmethod
    def from_timeseries(cls, X, Y, gamma=1.0, lambda_=0.0, rho=0.0):
        gamma = _check_gamma(gamma)
        dist = SquaredEuclidean(X, Y, lambda_=lambda_, rho=rho)
        obj = cls(dist.compute(), gamma=gamma)
        obj.distance_ = dist
        obj.lambda_ = lambda_
        obj.rho = rho
        return obj

    @classmethod
    def pairwise(cls, X, Y=None, gamma=1.0, lambda_=1.0, rho=0.0):
        """
        Compute soft-DTW between all pairs in one or two collections of sequences.

        Parameters
        ----------
        X : array
            If 2D, interpreted as univariate sequences with shape
            [n_sequences, n_timestamps]. If 3D, interpreted as multivariate
            sequences with shape [n_sequences, n_timestamps, n_features].
        Y : array, optional
            Optional second collection with the same feature dimension. If omitted,
            a symmetric pairwise matrix between sequences in X is returned.
        gamma : float
            Regularization parameter. Must be strictly positive.
        lambda_ : float
            Non-negative timestamp mismatch weight. Adds
            lambda_ * (i - j) ** 2 to each timestamp-pair distance.
        rho : float
            Finite non-negative penalty used for timestamp-pair feature
            distances that contain NaN or infinite values.
        Returns
        -------
        distances : array, shape = [n_x, n_y]
            Pairwise raw soft-DTW values.
        """
        _check_lambda(lambda_)
        _check_rho(rho)
        gamma = _check_gamma(gamma)

        X = _as_collection(X)

        if Y is None:
            out = np.empty((X.shape[0], X.shape[0]), dtype=np.float64)
            _pairwise_soft_dtw_symmetric(X, out, gamma, lambda_, rho)
            return out

        Y = _as_collection(Y)
        _check_collections_compatible(X, Y)

        out = np.empty((X.shape[0], Y.shape[0]), dtype=np.float64)
        _pairwise_soft_dtw(X, Y, out, gamma, lambda_, rho)

        return out

    def grad_x(self):
        if not hasattr(self, "distance_"):
            raise ValueError("grad_x() is only available when built from time series.")
        return self.distance_.jacobian_product(self.grad())


class SquaredEuclidean:
    def __init__(self, X, Y, lambda_=0.0, rho=0.0):
        """
        Parameters
        ----------
        X : array, shape = [m, d]
            First time series.
        Y : array, shape = [n, d]
            Second time series.
        lambda_ : float
            Non-negative timestamp mismatch weight. Adds
            lambda_ * (i - j) ** 2 to each distance matrix entry.
        rho : float
            Finite non-negative penalty used for invalid or infinite distance
            entries.
        """
        _check_lambda(lambda_)
        _check_rho(rho)
        self.X = _as_timeseries(X)
        self.Y = _as_timeseries(Y)
        self.lambda_ = lambda_
        self.rho = rho

    def compute(self):
        """
        Compute squared Euclidean distance matrix.

        Returns
        -------
        D : array, shape = [m, n]
        """
        D = np.empty((self.X.shape[0], self.Y.shape[0]), dtype=np.float64)
        _squared_euclidean_distances(self.X, self.Y, D, self.lambda_, self.rho)
        return D

    def jacobian_product(self, E):
        """
        Compute the product between the Jacobian and E.

        G[i, k] = sum_j E[i, j] * 2 * (X[i, k] - Y[j, k])

        Parameters
        ----------
        E : array, shape = [m, n]
            Gradient w.r.t. D (from soft-DTW backward pass).

        Returns
        -------
        G : array, shape = [m, d]
            Gradient w.r.t. X.
        """
        G = np.zeros_like(self.X)
        _jacobian_product_sq_euc(self.X, self.Y, E, G)
        return G
