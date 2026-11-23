import json
import logging
import math
from pathlib import Path
from time import perf_counter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numba import get_num_threads
from sklearn.decomposition import KernelPCA

from predict_perf_allocation.barycenter import sdtw_barycenter
from predict_perf_allocation.clustering import (
    ALLOCATION_COL,
    GROUP_COL,
    MODELS_DIR,
    RAW_DATA_DIR,
    RET_COLS,
    TS_COL,
    VOL_COLS,
    _check_columns,
    _check_gak_memory_budget,
    _date_number_from_ts,
    _drop_rows_with_nans,
    _filter_rows_by_allocation,
    _filter_rows_by_timestamp_window,
    _sample_rows_per_group,
    _time_series_from_x_train,
    _time_series_feature_columns,
)
from predict_perf_allocation.soft_dtw import SoftDTW

logger = logging.getLogger(__name__)
PCA_RUN_PREFIX = "pca_run_"
TURNOVER_COL = "MEDIAN_DAILY_TURNOVER"
PCA_SOFT_DTW_PARAMS = {
    "soft_dtw_gamma": 0.1,
    "lambda_": 1.0,
    "rho": 0.0,
}
BARYCENTER_PARAMS = {
    "method": "L-BFGS-B",
    "tol": 1e-3,
    "max_iter": 50,
}


def _create_pca_run_dir():
    """Create the next incrementally numbered PCA output directory."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    run_ids = []
    for path in MODELS_DIR.glob(f"{PCA_RUN_PREFIX}*"):
        suffix = path.name.removeprefix(PCA_RUN_PREFIX)
        if path.is_dir() and suffix.isdigit():
            run_ids.append(int(suffix))

    run_dir = MODELS_DIR / f"{PCA_RUN_PREFIX}{max(run_ids, default=0) + 1}"
    run_dir.mkdir()
    return run_dir


def _save_run_config(run_dir, run_args):
    config = {
        "run_kernel_pca_args": run_args,
        "kernel_pca_params": {"kernel": "precomputed"},
        "soft_dtw_params": PCA_SOFT_DTW_PARAMS,
        "barycenter_params": BARYCENTER_PARAMS,
        "barycenter_csv_volume_units": "raw",
    }
    config_path = run_dir / "run_config.json"
    with config_path.open("w", encoding="utf-8") as file:
        json.dump(config, file, indent=2)
    logger.info("Saved run configuration to %s.", config_path)


def _barycenters_by_timestep_and_group(X_train, X_ts, time_series_features):
    """Compute one Soft-DTW barycenter per timestep and group."""
    started_at = perf_counter()
    barycenters = []
    metadata = []

    for (timestamp, group), positions in X_train.groupby(
        [TS_COL, GROUP_COL],
        sort=True,
        dropna=False,
    ).indices.items():
        allocation_series = X_ts[positions]
        if len(positions) == 1:
            barycenter = allocation_series[0].copy()
        else:
            barycenter_init = np.mean(allocation_series, axis=0)
            barycenter = sdtw_barycenter(
                allocation_series,
                barycenter_init=barycenter_init,
                gamma=PCA_SOFT_DTW_PARAMS["soft_dtw_gamma"],
                lambda_=PCA_SOFT_DTW_PARAMS["lambda_"],
                rho=PCA_SOFT_DTW_PARAMS["rho"],
                **BARYCENTER_PARAMS,
            )

        barycenters.append(barycenter)
        allocation_rows = X_train.iloc[positions]
        allocations = allocation_rows[ALLOCATION_COL].astype(str).tolist()
        if time_series_features in {"returns", "both"}:
            last_return = barycenter[RET_COLS.index("RET_1"), 0]
        else:
            last_return = allocation_rows["RET_1"].mean()
        metadata.append(
            {
                TS_COL: timestamp,
                GROUP_COL: group,
                "RET_1": last_return,
                TURNOVER_COL: allocation_rows[TURNOVER_COL].mean(),
                "n_allocations": len(positions),
                "allocations": ", ".join(allocations),
            }
        )

    if not barycenters:
        raise ValueError("Cannot compute barycenters from an empty dataset.")
    logger.info(
        "Soft-DTW barycenter computation took %.3f seconds for %s TS/GROUP inputs.",
        perf_counter() - started_at,
        len(barycenters),
    )
    return np.stack(barycenters), pd.DataFrame(metadata)


def _save_barycenters_csv(
    run_dir,
    barycenters,
    metadata,
    X_train,
    time_series_features,
):
    """Save barycenters with named features, metadata, and raw volume units."""
    barycenters_df = metadata.copy()
    channel = 0
    feature_groups = []
    if time_series_features in {"returns", "both"}:
        feature_groups.append((RET_COLS, channel))
        channel += 1
    if time_series_features in {"volumes", "both"}:
        feature_groups.append((VOL_COLS, channel))

    for feature_cols, channel_index in feature_groups:
        feature_values = barycenters[:, :, channel_index]
        if feature_cols == VOL_COLS:
            volume_values = X_train[VOL_COLS].to_numpy(dtype=float)
            finite_values = volume_values[np.isfinite(volume_values)]
            if finite_values.size > 0:
                volume_min = finite_values.min()
                volume_max = finite_values.max()
                feature_values = feature_values * (volume_max - volume_min) + volume_min
        for timestep, column in enumerate(feature_cols):
            barycenters_df[column] = feature_values[:, timestep]

    csv_path = run_dir / "barycenters.csv"
    barycenters_df.to_csv(csv_path, index=False)
    logger.info("Saved barycenter time series to %s.", csv_path)


def _impute_from_saved_barycenters(X_train, time_series_features):
    """Fill missing allocation values from the saved TS/GROUP barycenters."""
    feature_cols = list(
        dict.fromkeys(
            _time_series_feature_columns(time_series_features)
            + ["RET_1", TURNOVER_COL]
        )
    )
    _check_columns(X_train, feature_cols)
    missing_masks = {
        column: X_train[column].isna()
        for column in feature_cols
        if X_train[column].isna().any()
    }
    if not missing_masks:
        return X_train

    barycenters_path = MODELS_DIR / "pca_run_bary" / "barycenters.csv"
    if not barycenters_path.exists():
        raise FileNotFoundError(
            f"Cannot impute missing values: barycenter file not found at {barycenters_path}."
        )

    barycenters = pd.read_csv(barycenters_path)
    join_cols = [TS_COL, GROUP_COL]
    _check_columns(barycenters, join_cols)
    if barycenters.duplicated(join_cols).any():
        raise ValueError(
            f"{barycenters_path} must contain at most one barycenter per TS/GROUP pair."
        )

    volume_units_validated = False
    for column, missing in missing_masks.items():
        _check_columns(barycenters, [column])
        lookup = barycenters[join_cols + [column]]
        imputed_values = X_train.loc[missing, join_cols].merge(
            lookup,
            on=join_cols,
            how="left",
            validate="many_to_one",
        )[column]
        if imputed_values.isna().any():
            raise ValueError(
                f"Could not impute {column}: missing or null barycenter values for "
                "one or more TS/GROUP pairs."
            )
        if column in VOL_COLS and not volume_units_validated:
            config_path = MODELS_DIR / "pca_run_bary" / "run_config.json"
            if not config_path.exists():
                raise FileNotFoundError(
                    f"Cannot verify volume units: {config_path} was not found."
                )
            with config_path.open(encoding="utf-8") as file:
                volume_units = json.load(file).get("barycenter_csv_volume_units")
            if volume_units != "raw":
                raise ValueError(
                    "Saved barycenter volumes are scaled. Rerun barycenter PCA "
                    "to save barycenters.csv in raw volume units before imputation."
                )
            volume_units_validated = True
        X_train.loc[missing, column] = imputed_values.to_numpy()
        logger.info(
            "Imputed %s missing %s values from saved barycenters.",
            missing.sum(),
            column,
        )

    return X_train


def _save_embedding_subplots(group_embeddings, run_dir):
    """Save one first-two-components scatter subplot per GROUP."""
    n_groups = len(group_embeddings)
    if n_groups == 0:
        raise ValueError("Cannot plot PCA results because no groups were fitted.")

    n_columns = min(3, n_groups)
    n_rows = math.ceil(n_groups / n_columns)
    fig, axes = plt.subplots(
        n_rows,
        n_columns,
        figsize=(6 * n_columns, 5 * n_rows),
        squeeze=False,
    )
    for group_index, (group, embedding_df) in enumerate(group_embeddings):
        ax = axes.flat[group_index]
        plot_df = embedding_df.copy()
        plot_df["timestamp_number"] = _date_number_from_ts(plot_df[TS_COL])
        plot_df = plot_df.sort_values("timestamp_number", kind="stable")
        scatter = ax.scatter(
            plot_df["component_1"],
            plot_df["component_2"],
            c=plot_df["timestamp_number"],
            cmap="viridis",
        )
        ax.set_title(f"{GROUP_COL} = {group}")
        ax.set_xlabel("Component 1")
        ax.set_ylabel("Component 2")
        fig.colorbar(scatter, ax=ax, label="Timestamp number")

    for ax in axes.flat[n_groups:]:
        ax.set_visible(False)

    fig.suptitle("Kernel PCA embeddings by group")
    fig.tight_layout(rect=(0, 0, 0.96, 0.96))
    plot_path = run_dir / "embedding_subplots.png"
    fig.savefig(plot_path, dpi=200)
    plt.close(fig)
    logger.info("Saved group PCA subplots to %s.", plot_path)


def _fit_kernel_pca_by_group(
    X_ts,
    embedding_metadata,
    n_components,
    random_state,
    run_dir,
):
    """Fit and save a separate precomputed-kernel PCA for each GROUP."""
    grouped_positions = embedding_metadata.groupby(
        GROUP_COL,
        sort=True,
        dropna=False,
    ).indices
    group_embeddings = []

    for group_number, (group, positions) in enumerate(grouped_positions.items(), start=1):
        group_ts = X_ts[positions]
        if len(positions) <= n_components:
            raise ValueError(
                f"GROUP {group!r} has {len(positions)} PCA inputs; "
                f"n_components={n_components} requires more inputs."
            )

        _check_gak_memory_budget(len(positions))
        soft_dtw_started_at = perf_counter()
        distances = SoftDTW.pairwise(
            group_ts,
            gamma=PCA_SOFT_DTW_PARAMS["soft_dtw_gamma"],
            lambda_=PCA_SOFT_DTW_PARAMS["lambda_"],
            rho=PCA_SOFT_DTW_PARAMS["rho"],
        )
        logger.info(
            "GROUP %r Soft-DTW pairwise computation took %.3f seconds "
            "for %s sequences using %s Numba threads.",
            group,
            perf_counter() - soft_dtw_started_at,
            len(positions),
            get_num_threads(),
        )
        affinity = np.exp(- distances / PCA_SOFT_DTW_PARAMS["soft_dtw_gamma"])
        logger.info(
            "GROUP %r affinity range: min=%.8g, max=%.8g",
            group,
            np.min(affinity),
            np.max(affinity),
        )
        model = KernelPCA(
            n_components=n_components,
            kernel="precomputed",
            random_state=random_state,
            n_jobs=8,
        )
        kernel_pca_started_at = perf_counter()
        embedding = model.fit_transform(affinity)
        for component, eigenvalue in enumerate(model.eigenvalues_, start=1):
            logger.info(
                "GROUP %r PCA component %d eigenvalue: %.8g",
                group,
                component,
                eigenvalue,
            )
        logger.info(
            "GROUP %r kernel PCA fit_transform took %.3f seconds for %s inputs.",
            group,
            perf_counter() - kernel_pca_started_at,
            len(positions),
        )

        group_dir = run_dir / f"group_{group_number:03d}"
        group_dir.mkdir()
        embedding_df = embedding_metadata.iloc[positions].copy()
        for component in range(n_components):
            embedding_df[f"component_{component + 1}"] = embedding[:, component]

        embedding_path = group_dir / "embedding.csv"
        embedding_df.to_csv(embedding_path, index=True, index_label="INPUT_ID")
        logger.info("Saved GROUP %r embedding to %s.", group, embedding_path)

        group_embeddings.append((group, embedding_df))

    _save_embedding_subplots(group_embeddings, run_dir)


def run_kernel_pca(
    n_components=2,
    drop_na=True,
    random_state=42,
    n_rows_per_group=2000,
    T0=None,
    T1=None,
    allocation_id=None,
    time_series_features="both",
    compute_group_barycenters=False,
):
    """Run kernel PCA on the selected financial time-series data.

    Kernel PCA uses the precomputed Soft-DTW affinity matrix from the clustering
    workflow. PCA is fit independently for each ``GROUP``. Optionally, one
    Soft-DTW barycenter is computed per ``TS`` and ``GROUP`` before fitting.
    Per-group embeddings, a subplot figure colored by timestamp, and the run
    configuration are saved under ``models/pca_run_*``. Barycenter runs also
    save the row-aligned time series in ``barycenters.csv``.

    Parameters
    ----------
    n_components : int, default=2
        Number of embedding dimensions.
    drop_na : bool, default=True
        Drop rows with missing values in the selected time-series columns. If
        False in regular mode, fill missing values from the saved barycenter for
        the same ``TS`` and ``GROUP`` in ``models/pca_run_bary/barycenters.csv``.
    random_state : int, default=42
        Random seed forwarded to scikit-learn's KernelPCA.
    n_rows_per_group : int or None, default=2000
        Maximum number of rows sampled per ``GROUP``. If None, use all rows.
        Ignored when ``compute_group_barycenters=True``.
    T0, T1 : int or None, default=None
        Inclusive lower and upper bounds for ``TS`` values formatted as
        ``DATE_XXXX``. Ignored when ``compute_group_barycenters=True``.
    allocation_id : int or None, default=None
        If set, keep only the matching ``ALLOCATION_XX`` rows. Ignored when
        ``compute_group_barycenters=True``.
    time_series_features : {"returns", "volumes", "both"}, default="both"
        Choose which time-series columns are used for the Soft-DTW affinity.
    compute_group_barycenters : bool, default=False
        If True, average allocations within each ``TS``/``GROUP`` pair using
        Soft-DTW barycenters, then fit kernel PCA on those barycenters.
    """
    if not isinstance(n_components, int) or n_components < 2:
        raise ValueError("n_components must be an integer of at least 2.")
    _time_series_feature_columns(time_series_features)

    run_dir = _create_pca_run_dir()
    run_args = {
        "n_components": n_components,
        "drop_na": drop_na,
        "random_state": random_state,
        "n_rows_per_group": n_rows_per_group,
        "T0": T0,
        "T1": T1,
        "allocation_id": allocation_id,
        "time_series_features": time_series_features,
        "compute_group_barycenters": compute_group_barycenters,
        "ignored_arguments": (
            ["n_rows_per_group", "T0", "T1", "allocation_id"]
            if compute_group_barycenters
            else []
        ),
        "impute_missing_from_saved_barycenters": (
            not drop_na and not compute_group_barycenters
        ),
    }
    _save_run_config(run_dir, run_args)

    X_train = pd.read_csv(RAW_DATA_DIR / "X_train.csv", index_col="ROW_ID")
    _check_columns(
        X_train,
        [TS_COL, ALLOCATION_COL, GROUP_COL, "RET_1", TURNOVER_COL],
    )
    if compute_group_barycenters:
        logger.info(
            "Barycenter mode ignores n_rows_per_group, T0, T1, and allocation_id."
        )
    else:
        X_train = _filter_rows_by_timestamp_window(X_train, T0, T1)
        X_train = _filter_rows_by_allocation(X_train, allocation_id, n_components)

        if allocation_id is None:
            X_train = _sample_rows_per_group(
                X_train,
                n_rows_per_group=n_rows_per_group,
                random_state=1984,
            )
        else:
            logger.info("Skipped per-group sampling because allocation_id is set.")

    if drop_na:
        X_train = _drop_rows_with_nans(X_train, time_series_features)
    elif not compute_group_barycenters:
        X_train = _impute_from_saved_barycenters(X_train, time_series_features)

    if compute_group_barycenters:
        X_ts = _time_series_from_x_train(X_train, time_series_features)
        X_ts, embedding_metadata = _barycenters_by_timestep_and_group(
            X_train,
            X_ts,
            time_series_features,
        )
        _save_barycenters_csv(
            run_dir,
            X_ts,
            embedding_metadata,
            X_train,
            time_series_features,
        )
    else:
        X_ts = _time_series_from_x_train(X_train, time_series_features)
        embedding_metadata = X_train[[TS_COL, ALLOCATION_COL, GROUP_COL]].copy()
        embedding_metadata["RET_1"] = X_train["RET_1"]
        embedding_metadata[TURNOVER_COL] = X_train[TURNOVER_COL]

    _fit_kernel_pca_by_group(
        X_ts,
        embedding_metadata,
        n_components,
        random_state,
        run_dir,
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
    run_kernel_pca(
        n_components=5,
        drop_na=False,
        random_state=42,
        n_rows_per_group=10000,
        T0=None,
        T1=None,
        allocation_id=1,
        time_series_features="volumes",
        compute_group_barycenters=False,
    )
