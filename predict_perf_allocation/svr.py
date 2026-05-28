import logging
import os

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.metrics import r2_score
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.svm import SVR
from sklearn.utils.validation import (
    check_array,
    check_consistent_length,
    check_is_fitted,
    column_or_1d,
)

from predict_perf_allocation.soft_dtw import (
    SoftDTW,
    _as_collection,
    _check_collections_compatible,
)

logger = logging.getLogger(__name__)


class SVMRegTS(RegressorMixin, BaseEstimator):
    """Support vector machine regressor for equal-length time series.

    The estimator follows the scikit-learn API and delegates the optimization to
    :class:`sklearn.svm.SVR` with a precomputed kernel. The final kernel is a
    additive or multiplicative combination of a Soft-DTW time-series kernel and
    an RBF kernel built from tabular features Xf.
    """

    def __init__(
        self,
        C=1.0,
        epsilon=0.1,
        soft_dtw_gamma=1.0,
        lambda_=1.0,
        rho=0.0,
        rbf_gamma=1.0,
        alpha=0.5,
        kernel_combination="additive",
        max_kernel_memory_bytes=None,
        max_kernel_memory_fraction=None,
        max_memory_usage_fraction=None,
    ):
        self.C = C
        self.epsilon = epsilon
        self.soft_dtw_gamma = soft_dtw_gamma
        self.lambda_ = lambda_
        self.rho = rho
        self.rbf_gamma = rbf_gamma
        self.alpha = alpha
        self.kernel_combination = kernel_combination
        self.max_kernel_memory_bytes = max_kernel_memory_bytes
        self.max_kernel_memory_fraction = max_kernel_memory_fraction
        self.max_memory_usage_fraction = max_memory_usage_fraction

    def fit(self, X, y, Xf=None, sample_weight=None):
        """Fit the SVM regressor.

        Parameters
        ----------
        X : array-like, shape = [n_samples, n_timestamps] or
            [n_samples, n_timestamps, n_features]
            Training time series.
        y : array-like, shape = [n_samples]
            Regression targets.
        Xf : array-like, shape = [n_samples, n_features]
            Tabular features characterizing each time series.
        sample_weight : array-like, shape = [n_samples], optional
            Per-sample weights forwarded to :class:`sklearn.svm.SVR`.

        Returns
        -------
        self : object
            Fitted estimator.
        """
        X = self._validate_X(X)
        y = column_or_1d(y)
        Xf = self._validate_Xf_fit(Xf, X.shape[0])
        check_consistent_length(X, y, Xf)

        self.X_fit_ = X
        self.Xf_fit_ = Xf
        self.n_features_in_ = X.shape[2]
        self.n_timestamps_in_ = X.shape[1]
        self.n_features_Xf_in_ = Xf.shape[1]
        self.alpha_ = self._validate_alpha()
        self.kernel_combination_ = self._validate_kernel_combination()
        self.soft_dtw_gamma_ = self._validate_kernel_gamma(
            self.soft_dtw_gamma, "soft_dtw_gamma"
        )
        if self.kernel_combination_ == "additive" and self.alpha_ == 0.0:
            self.rbf_gamma_ = None
        else:
            self.rbf_gamma_ = self._resolve_rbf_gamma(Xf)

        K = self._kernel(X, Xf=Xf)
        self.svr_ = self._make_svr()
        self.svr_.fit(K, y, sample_weight=sample_weight)

        return self

    def predict(self, X, Xf=None):
        """Predict regression targets for time series in X."""
        check_is_fitted(self, "svr_")
        X = self._validate_X_predict(X)
        Xf = self._validate_Xf_predict(Xf, X.shape[0])
        K = self._kernel(X, self.X_fit_, Xf, self.Xf_fit_)
        return self.svr_.predict(K)

    def score(self, X, y, Xf=None, sample_weight=None):
        """Return the coefficient of determination R^2 on the test data."""
        return r2_score(y, self.predict(X, Xf), sample_weight=sample_weight)

    @property
    def n_support_(self):
        """Number of support vectors."""
        check_is_fitted(self, "svr_")
        return self.svr_.n_support_

    @property
    def support_(self):
        """Indices of support vectors."""
        check_is_fitted(self, "svr_")
        return self.svr_.support_

    @property
    def support_vectors_(self):
        """Support vectors in the original time-series representation."""
        check_is_fitted(self, "svr_")
        return self.X_fit_[self.support_]

    @property
    def dual_coef_(self):
        """Dual coefficients of support vectors."""
        check_is_fitted(self, "svr_")
        return self.svr_.dual_coef_

    @property
    def intercept_(self):
        """Constants in the regression function."""
        check_is_fitted(self, "svr_")
        return self.svr_.intercept_

    def _make_svr(self):
        return SVR(
            C=self.C,
            epsilon=self.epsilon,
            kernel="precomputed",
        )

    def _kernel(self, X, Y=None, Xf=None, Yf=None):
        n_y = X.shape[0] if Y is None else Y.shape[0]
        self._check_kernel_memory_budget(X.shape[0], n_y)
        distances = self._distance(X, Y)
        gak = np.exp(-self.soft_dtw_gamma_ * distances)
        gak_max = np.max(gak)
        if gak_max > 2:
            logger.warning(
                "GAK kernel output contains values exceeding 2 "
                "(soft_dtw_gamma=%s, max=%s).",
                self.soft_dtw_gamma_,
                gak_max,
            )

        if self.kernel_combination_ == "additive" and self.alpha_ == 0.0:
            kernel = gak
        else:
            rbf = rbf_kernel(Xf, Yf, gamma=self.rbf_gamma_)
            if self.kernel_combination_ == "additive":
                kernel = (1.0 - self.alpha_) * gak + self.alpha_ * rbf
            else:
                kernel = gak * rbf

        if Y is None:
            np.fill_diagonal(kernel, 1.0)
        return check_array(kernel, ensure_all_finite=True)

    def _distance(self, X, Y=None):
        return SoftDTW.pairwise(
            X,
            Y,
            gamma=self.soft_dtw_gamma_,
            lambda_=self.lambda_,
            rho=self.rho,
        )

    def _validate_X(self, X):
        return _as_collection(X)

    def _validate_Xf_fit(self, Xf, n_samples):
        if Xf is None:
            raise ValueError("Xf is required and must have the same length as X.")

        Xf = check_array(Xf, dtype=np.float64, ensure_2d=True, ensure_all_finite=True)
        if Xf.shape[0] != n_samples:
            raise ValueError("Xf must have the same number of samples as X.")
        return Xf

    def _validate_X_predict(self, X):
        X = _as_collection(X)
        _check_collections_compatible(X, self.X_fit_)
        if X.shape[1] != self.n_timestamps_in_:
            raise ValueError(
                "X must have the same number of timestamps as the fitted data "
                f"({self.n_timestamps_in_})."
            )
        return X

    def _validate_Xf_predict(self, Xf, n_samples):
        if Xf is None:
            raise ValueError("Xf is required and must have the same length as X.")

        Xf = check_array(Xf, dtype=np.float64, ensure_2d=True, ensure_all_finite=True)
        if Xf.shape[0] != n_samples:
            raise ValueError("Xf must have the same number of samples as X.")
        if Xf.shape[1] != self.n_features_Xf_in_:
            raise ValueError(
                "Xf must have the same number of features as the fitted Xf "
                f"({self.n_features_Xf_in_})."
            )
        return Xf

    def _resolve_rbf_gamma(self, Xf):
        if self.rbf_gamma == "auto":
            variance = Xf.var()
            if variance == 0:
                raise ValueError("Cannot set rbf_gamma='auto' when Xf has zero variance.")
            return 1.0 / (Xf.shape[1] * variance)

        return self._validate_kernel_gamma(self.rbf_gamma, "rbf_gamma")

    def _validate_alpha(self):
        if not np.isfinite(self.alpha) or not 0 <= self.alpha <= 1:
            raise ValueError("alpha must be a finite value between 0 and 1.")
        return float(self.alpha)

    def _validate_kernel_combination(self):
        if self.kernel_combination not in {"additive", "multiplicative"}:
            raise ValueError(
                "kernel_combination must be either 'additive' or 'multiplicative'."
            )
        return self.kernel_combination

    def _validate_kernel_gamma(self, gamma, name):
        if not np.isfinite(gamma) or gamma < 0:
            raise ValueError(f"{name} must be a finite non-negative value.")
        return float(gamma)

    def _check_kernel_memory_budget(self, n_x, n_y):
        estimated_bytes = self._estimate_kernel_memory_bytes(n_x, n_y)
        total_memory, available_memory = _system_memory()

        max_kernel_memory_bytes = self._resolve_memory_limit(
            self.max_kernel_memory_bytes,
            "max_kernel_memory_bytes",
        )
        if max_kernel_memory_bytes is not None and estimated_bytes > max_kernel_memory_bytes:
            raise MemoryError(
                "SVMRegTS kernel computation would allocate about "
                f"{_format_bytes(estimated_bytes)} for shape ({n_x}, {n_y}), exceeding "
                f"max_kernel_memory_bytes={_format_bytes(max_kernel_memory_bytes)}."
            )

        max_kernel_memory_fraction = self._validate_optional_fraction(
            self.max_kernel_memory_fraction,
            "max_kernel_memory_fraction",
        )
        if (
            max_kernel_memory_fraction is not None
            and total_memory is not None
            and estimated_bytes > max_kernel_memory_fraction * total_memory
        ):
            raise MemoryError(
                "SVMRegTS kernel computation would allocate about "
                f"{_format_bytes(estimated_bytes)} for shape ({n_x}, {n_y}), exceeding "
                f"{max_kernel_memory_fraction:.0%} of system memory "
                f"({_format_bytes(total_memory)})."
            )

        max_memory_usage_fraction = self._validate_optional_fraction(
            self.max_memory_usage_fraction,
            "max_memory_usage_fraction",
        )
        if (
            max_memory_usage_fraction is not None
            and total_memory is not None
            and available_memory is not None
        ):
            projected_available_memory = available_memory - estimated_bytes
            projected_usage_fraction = 1.0 - (projected_available_memory / total_memory)
            if projected_usage_fraction > max_memory_usage_fraction:
                raise MemoryError(
                    "SVMRegTS kernel computation would allocate about "
                    f"{_format_bytes(estimated_bytes)} for shape ({n_x}, {n_y}); projected "
                    f"system memory usage is {projected_usage_fraction:.0%}, exceeding "
                    f"max_memory_usage_fraction={max_memory_usage_fraction:.0%}."
                )

    def _estimate_kernel_memory_bytes(self, n_x, n_y):
        n_pairs = n_x * n_y
        # distances, GAK, RBF, and final precomputed kernel are dense float64 matrices.
        return n_pairs * np.dtype(np.float64).itemsize * 4

    def _resolve_memory_limit(self, value, name):
        if value is None:
            return None
        if not np.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be a positive finite value or None.")
        return int(value)

    def _validate_optional_fraction(self, value, name):
        if value is None:
            return None
        if not np.isfinite(value) or not 0 < value <= 1:
            raise ValueError(f"{name} must be a finite value in (0, 1] or None.")
        return float(value)


def _system_memory():
    meminfo = _linux_memory_info()
    if meminfo is not None:
        return meminfo

    if hasattr(os, "sysconf"):
        try:
            page_size = os.sysconf("SC_PAGE_SIZE")
            total_pages = os.sysconf("SC_PHYS_PAGES")
            available_pages = os.sysconf("SC_AVPHYS_PAGES")
        except (OSError, ValueError):
            return None, None

        return page_size * total_pages, page_size * available_pages

    return None, None


def _linux_memory_info():
    try:
        with open("/proc/meminfo", encoding="utf-8") as file:
            values = {}
            for line in file:
                key, raw_value = line.split(":", 1)
                values[key] = int(raw_value.strip().split()[0]) * 1024
    except (FileNotFoundError, OSError, ValueError):
        return None

    total = values.get("MemTotal")
    available = values.get("MemAvailable")
    if total is None or available is None:
        return None
    return total, available


def _format_bytes(value):
    value = float(value)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if value < 1024 or unit == "TiB":
            return f"{value:.1f} {unit}"
        value /= 1024
