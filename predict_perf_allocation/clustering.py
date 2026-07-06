import json
import logging
import os
import pickle
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.cluster import SpectralClustering
from sklearn.preprocessing import MinMaxScaler

from predict_perf_allocation.soft_dtw import SoftDTW

logger = logging.getLogger(__name__)

RET_COLS = [f"RET_{i}" for i in range(20, 0, -1)]
VOL_COLS = [f"SIGNED_VOLUME_{i}" for i in range(20, 0, -1)]
GROUP_COL = "GROUP"
TS_COL = "TS"
ALLOCATION_COL = "ALLOCATION"
TARGET_COL = "target"
SOFT_DTW_PARAMS = {
    "soft_dtw_gamma": 0.1,
    "lambda_": 1.0,
    "rho": 0.1,
}
MEMORY_PARAMS = {
    "max_memory_usage_fraction": 0.85,
}
GRAPH_PARAMS = {
    "n_neighbors": 2,
    "show_node_labels": False,
}

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
MODELS_DIR = PROJECT_ROOT / "models"
CLUSTERING_MODEL_DIR = MODELS_DIR / "clustering"


def _check_columns(X_train, columns):
    missing_columns = [column for column in columns if column not in X_train.columns]
    if missing_columns:
        raise ValueError(f"X_train is missing required columns: {missing_columns}")


def _scale_volumes(X_train, vol_cols):
    scaled_volumes = X_train[vol_cols].to_numpy(dtype=float).copy()
    scaler = MinMaxScaler()
    finite_values = np.isfinite(scaled_volumes)

    if finite_values.any():
        scaled_volumes[finite_values] = scaler.fit_transform(
            scaled_volumes[finite_values].reshape(-1, 1)
        ).ravel()

    return scaled_volumes


def _time_series_from_x_train(X_train):
    returns = X_train[RET_COLS].to_numpy(dtype=float)
    volumes = _scale_volumes(X_train, VOL_COLS)

    return np.stack((returns, volumes), axis=2)


def _save_run_config(model_dir, run_args):
    config = {
        "run_clustering_args": run_args,
        "soft_dtw_params": SOFT_DTW_PARAMS,
        "memory_params": MEMORY_PARAMS,
        "graph_params": GRAPH_PARAMS,
    }
    config_path = model_dir / "run_config.json"
    with config_path.open("w", encoding="utf-8") as file:
        json.dump(config, file, indent=2)
    logging.info("Saved run configuration to %s.", config_path)


def _save_clustering_outputs(model_dir, X_train, labels, affinity):
    _check_columns(X_train, [TS_COL, ALLOCATION_COL])

    labels_df = X_train[[TS_COL, ALLOCATION_COL]].copy()
    labels_df.loc[:, "cluster"] = labels

    labels_path = model_dir / "labels.csv"
    labels_df.to_csv(labels_path, index=True)
    logging.info("Saved clustering labels to %s.", labels_path)

    affinity_path = model_dir / "affinity_matrix.pkl"
    with affinity_path.open("wb") as file:
        pickle.dump(affinity, file)
    logging.info("Saved affinity matrix to %s.", affinity_path)

    _save_affinity_heatmap(model_dir, affinity, labels)


def _save_affinity_heatmap(model_dir, affinity, labels):
    sorted_indexes = np.argsort(labels, kind="stable")
    sorted_labels = labels[sorted_indexes]
    sorted_affinity = affinity[np.ix_(sorted_indexes, sorted_indexes)]
    cluster_boundaries = np.flatnonzero(sorted_labels[1:] != sorted_labels[:-1]) + 1

    n_samples = affinity.shape[0]
    figure_size = min(max(n_samples / 50, 8), 18)
    fig, ax = plt.subplots(figsize=(figure_size, figure_size))
    sns.heatmap(
        sorted_affinity,
        ax=ax,
        cmap="Blues",
        xticklabels=False,
        yticklabels=False,
        cbar_kws={"label": "GAK affinity"},
    )
    for boundary in cluster_boundaries:
        ax.axhline(boundary, color="red", linewidth=1.0)
        ax.axvline(boundary, color="red", linewidth=1.0)

    ax.set_title("GAK affinity matrix ordered by cluster")
    ax.set_xlabel("Rows ordered by cluster")
    ax.set_ylabel("Rows ordered by cluster")
    fig.tight_layout()

    heatmap_path = model_dir / "affinity_matrix_heatmap.png"
    fig.savefig(heatmap_path, dpi=200)
    plt.close(fig)
    logging.info("Saved affinity matrix heatmap to %s.", heatmap_path)


def compute_adjacency_matrix(affinity):
    """Build an unweighted top-k neighborhood adjacency matrix from affinities."""
    affinity = np.asarray(affinity, dtype=float)
    if affinity.ndim != 2 or affinity.shape[0] != affinity.shape[1]:
        raise ValueError("affinity must be a square 2D matrix.")

    n_samples = affinity.shape[0]
    adjacency = np.zeros_like(affinity, dtype=float)
    if n_samples <= 1:
        return adjacency

    n_neighbors = GRAPH_PARAMS["n_neighbors"]
    if n_neighbors is None:
        n_neighbors = int(np.log(n_samples)) + 1
    elif not isinstance(n_neighbors, int) or n_neighbors <= 0:
        raise ValueError("GRAPH_PARAMS n_neighbors must be a positive integer or None.")

    n_neighbors = min(n_neighbors, n_samples - 1)
    affinity_without_self = affinity.copy()
    np.fill_diagonal(affinity_without_self, -np.inf)

    neighbor_indexes = np.argpartition(
        affinity_without_self,
        kth=n_samples - n_neighbors - 1,
        axis=1,
    )[:, -n_neighbors:]
    row_indexes = np.arange(n_samples)[:, np.newaxis]
    adjacency[row_indexes, neighbor_indexes] = 1.0
    adjacency = np.maximum(adjacency, adjacency.T)
    np.fill_diagonal(adjacency, 0.0)

    return adjacency


def _save_neighborhood_graph(model_dir, adjacency, target_values, ts_values, random_state):
    graph = nx.from_numpy_array(adjacency)
    target_values = np.asarray(target_values, dtype=float)
    node_colors = np.where(target_values > 0, "green", "red")
    node_labels = _timestamp_node_labels(ts_values)

    fig, ax = plt.subplots(figsize=(10, 10))
    positions = nx.spring_layout(graph, seed=random_state)
    nx.draw_networkx_edges(
        graph,
        positions,
        ax=ax,
        edge_color="#b0b0b0",
        alpha=0.35,
        width=0.8,
    )
    nx.draw_networkx_nodes(
        graph,
        positions,
        ax=ax,
        node_color=node_colors,
        node_size=120 if GRAPH_PARAMS["show_node_labels"] else 40,
        linewidths=0.2,
        edgecolors="white",
    )
    if GRAPH_PARAMS["show_node_labels"]:
        nx.draw_networkx_labels(
            graph,
            positions,
            labels=node_labels,
            ax=ax,
            font_size=7,
            font_color="black",
        )
    ax.legend(
        handles=[
            mpatches.Patch(color="green", label="target > 0"),
            mpatches.Patch(color="red", label="target <= 0"),
        ],
        loc="best",
    )
    ax.set_title("Neighborhood graph from GAK affinities")
    ax.set_axis_off()
    fig.tight_layout()

    graph_path = model_dir / "neighborhood_graph.png"
    fig.savefig(graph_path, dpi=200)
    plt.close(fig)
    logging.info("Saved neighborhood graph to %s.", graph_path)


def _target_values_for_rows(y_train, row_index):
    _check_columns(y_train, [TARGET_COL])
    missing_rows = row_index.difference(y_train.index)
    if len(missing_rows) > 0:
        raise ValueError("y_train is missing targets for selected X_train rows.")
    return y_train.loc[row_index, TARGET_COL].to_numpy(dtype=float)


def _timestamp_node_labels(ts_values):
    ts_values = pd.Series(ts_values).astype("string")
    extracted_numbers = ts_values.str.extract(r"^DATE_(\d+)$", expand=False)
    labels = extracted_numbers.astype("Int64").astype("string")
    labels = labels.fillna(ts_values)
    return dict(enumerate(labels.astype(str)))


def _drop_rows_with_nans(X_train):
    feature_cols = RET_COLS + VOL_COLS
    rows_not_na = X_train[feature_cols].notna().all(axis=1)
    dropped_rows = len(X_train) - rows_not_na.sum()
    logging.info("Dropped %s X_train rows with NaNs.", dropped_rows)
    return X_train.loc[rows_not_na].copy()


def _sample_rows_per_group(X_train, n_rows_per_group, random_state):
    if n_rows_per_group is not None:
        if not isinstance(n_rows_per_group, int):
            raise ValueError("n_rows_per_group must be a positive integer or None.")
        if n_rows_per_group <= 0:
            raise ValueError("n_rows_per_group must be a positive integer or None.")

    _check_columns(X_train, [GROUP_COL])

    sampled_indexes = []
    row_counts = []
    for group, group_data in X_train.groupby(GROUP_COL, sort=True):
        n_selected = len(group_data)
        if n_rows_per_group is not None:
            n_selected = min(n_rows_per_group, len(group_data))
            sampled_index = group_data.sample(n=n_selected, random_state=random_state).index
        else:
            sampled_index = group_data.index

        sampled_indexes.append(sampled_index)
        row_counts.append(
            {
                GROUP_COL: group,
                "available_rows": len(group_data),
                "selected_rows": n_selected,
            }
        )

    if not sampled_indexes:
        raise ValueError("Cannot sample rows because X_train is empty.")

    sampled_index = sampled_indexes[0].append(sampled_indexes[1:])
    X_train_sampled = X_train.loc[sampled_index].copy()
    logging.info(
        "Sampled up to %s rows per %s: %s -> %s rows.",
        n_rows_per_group,
        GROUP_COL,
        len(X_train),
        len(X_train_sampled),
    )
    for counts in row_counts:
        logging.info(
            "Group %s rows: selected %s/%s.",
            counts[GROUP_COL],
            counts["selected_rows"],
            counts["available_rows"],
        )
    return X_train_sampled


def _date_number_from_ts(ts_values):
    ts_numbers = ts_values.astype("string").str.extract(r"^DATE_(\d+)$", expand=False)
    if ts_numbers.isna().any():
        invalid_values = sorted(ts_values.loc[ts_numbers.isna()].astype(str).unique())
        invalid_preview = ", ".join(invalid_values[:5])
        raise ValueError(
            f"{TS_COL} values must have format DATE_XXXX. "
            f"Invalid values include: {invalid_preview}"
        )
    return ts_numbers.astype(int)


def _validate_timestamp_bound(bound, bound_name):
    if bound is None:
        return
    if not isinstance(bound, int):
        raise ValueError(f"{bound_name} must be an integer or None.")
    if bound < 0:
        raise ValueError(f"{bound_name} must be a non-negative integer or None.")


def _allocation_name_from_id(allocation_id):
    if allocation_id is None:
        return None
    if not isinstance(allocation_id, int):
        raise ValueError("allocation_id must be an integer or None.")
    if allocation_id < 0:
        raise ValueError("allocation_id must be a non-negative integer or None.")
    return f"ALLOCATION_{allocation_id:02d}"


def _filter_rows_by_allocation(X_train, allocation_id, n_clusters):
    allocation_name = _allocation_name_from_id(allocation_id)
    if allocation_name is None:
        return X_train

    _check_columns(X_train, [ALLOCATION_COL])
    rows_for_allocation = X_train[ALLOCATION_COL] == allocation_name
    n_rows = rows_for_allocation.sum()
    if n_rows < n_clusters:
        logger.warning(
            "Allocation %s has only %s rows, fewer than n_clusters=%s. Stopping clustering.",
            allocation_name,
            n_rows,
            n_clusters,
        )
        raise ValueError(
            f"Allocation {allocation_name} has only {n_rows} rows, "
            f"fewer than n_clusters={n_clusters}."
        )

    logging.info(
        "Kept %s/%s X_train rows for %s=%s.",
        n_rows,
        len(X_train),
        ALLOCATION_COL,
        allocation_name,
    )
    return X_train.loc[rows_for_allocation].copy()


def _filter_rows_by_timestamp_window(X_train, T0, T1):
    if T0 is None and T1 is None:
        return X_train

    _check_columns(X_train, [TS_COL])
    _validate_timestamp_bound(T0, "T0")
    _validate_timestamp_bound(T1, "T1")
    if T0 is not None and T1 is not None and T0 > T1:
        raise ValueError("T0 must be less than or equal to T1.")

    ts_numbers = _date_number_from_ts(X_train[TS_COL])
    rows_in_window = pd.Series(True, index=X_train.index)
    if T0 is not None:
        rows_in_window &= ts_numbers >= T0
    if T1 is not None:
        rows_in_window &= ts_numbers <= T1

    lower_bound = "-inf" if T0 is None else f"DATE_{T0:04d}"
    upper_bound = "+inf" if T1 is None else f"DATE_{T1:04d}"
    logging.info(
        "Kept %s/%s X_train rows with %s between %s and %s.",
        rows_in_window.sum(),
        len(X_train),
        TS_COL,
        lower_bound,
        upper_bound,
    )
    if not rows_in_window.any():
        raise ValueError(
            f"No X_train rows found with {TS_COL} between {lower_bound} and {upper_bound}."
        )
    return X_train.loc[rows_in_window].copy()


def _check_gak_memory_budget(n_samples):
    estimated_bytes = _estimate_gak_memory_bytes(n_samples)
    total_memory, available_memory = _system_memory()
    max_memory_usage_fraction = MEMORY_PARAMS["max_memory_usage_fraction"]

    if total_memory is None or available_memory is None:
        return

    projected_available_memory = available_memory - estimated_bytes
    projected_usage_fraction = 1.0 - (projected_available_memory / total_memory)
    if projected_usage_fraction > max_memory_usage_fraction:
        logger.warning(
            "GAK affinity computation would allocate about %s for shape (%s, %s); "
            "projected system memory usage is %.0f%%, exceeding "
            "max_memory_usage_fraction=%.0f%%.",
            _format_bytes(estimated_bytes),
            n_samples,
            n_samples,
            projected_usage_fraction * 100,
            max_memory_usage_fraction * 100,
        )


def _estimate_gak_memory_bytes(n_samples):
    n_pairs = n_samples * n_samples
    # Soft-DTW distances and the final GAK affinity are dense float64 matrices.
    return n_pairs * np.dtype(np.float64).itemsize * 2


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


def compute_gak_affinity(X_train):
    """Compute a GAK affinity matrix between rows of ``X_train``.

    Rows are represented as bivariate time series built from all matching
    ``RET_*`` and scaled ``SIGNED_VOLUME_*`` columns.
    """
    _check_columns(X_train, RET_COLS + VOL_COLS)

    X_ts = _time_series_from_x_train(X_train)
    _check_gak_memory_budget(X_ts.shape[0])
    soft_dtw_gamma = SOFT_DTW_PARAMS["soft_dtw_gamma"]
    distances = SoftDTW.pairwise(
        X_ts,
        gamma=soft_dtw_gamma,
        lambda_=SOFT_DTW_PARAMS["lambda_"],
        rho=SOFT_DTW_PARAMS["rho"],
    )
    affinity = np.exp(-soft_dtw_gamma * distances)
    affinity_max = np.max(affinity)
    if affinity_max > 1.0:
        logger.warning(
            "GAK affinity matrix contains values exceeding 1.0 "
            "(soft_dtw_gamma=%s, max=%s).",
            soft_dtw_gamma,
            affinity_max,
        )

    return affinity


def run_clustering(
    n_clusters,
    drop_na=True,
    random_state=42,
    n_rows_per_group=2000,
    T0=None,
    T1=None,
    allocation_id=None,
    plot_graph=False,
):
    """Cluster rows in ``data/raw/X_train.csv`` with GAK spectral clustering.

    Parameters
    ----------
    n_clusters : int
        Number of clusters to fit.
    drop_na : bool, default=True
        Drop rows with NaNs in return or volume columns before clustering.
    random_state : int, default=42
        Random seed passed to sampling, ``SpectralClustering``, and graph layout.
    n_rows_per_group : int or None, default=2000
        Maximum number of rows to sample from each ``GROUP``. If None, use all rows.
    T0 : int or None, default=None
        Lower inclusive timestamp bound for ``TS`` values formatted as ``DATE_XXXX``.
    T1 : int or None, default=None
        Upper inclusive timestamp bound for ``TS`` values formatted as ``DATE_XXXX``.
    allocation_id : int or None, default=None
        If set, keep only rows where ``ALLOCATION`` equals ``ALLOCATION_XX``.
    plot_graph : bool, default=False
        If True, compute an unweighted neighborhood graph and save it as a PNG.

    Results are saved under ``models/clustering``.
    """
    if not isinstance(n_clusters, int) or n_clusters <= 0:
        raise ValueError("n_clusters must be a positive integer.")

    CLUSTERING_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    _save_run_config(
        CLUSTERING_MODEL_DIR,
        {
            "n_clusters": n_clusters,
            "drop_na": drop_na,
            "random_state": random_state,
            "n_rows_per_group": n_rows_per_group,
            "T0": T0,
            "T1": T1,
            "allocation_id": allocation_id,
            "plot_graph": plot_graph,
        },
    )

    X_train = pd.read_csv(RAW_DATA_DIR / "X_train.csv", index_col="ROW_ID")
    y_train = pd.read_csv(RAW_DATA_DIR / "y_train.csv", index_col="ROW_ID")
    _check_columns(X_train, [TS_COL, ALLOCATION_COL])
    X_train = _filter_rows_by_timestamp_window(X_train, T0, T1)
    X_train = _filter_rows_by_allocation(X_train, allocation_id, n_clusters)

    if drop_na:
        X_train = _drop_rows_with_nans(X_train)

    if allocation_id is None:
        X_train = _sample_rows_per_group(
            X_train,
            n_rows_per_group=n_rows_per_group,
            random_state=1984,
        )
    else:
        logging.info("Skipped per-group sampling because allocation_id is set.")

    if n_clusters > len(X_train):
        raise ValueError("n_clusters cannot exceed the number of rows in X_train.")

    affinity = compute_gak_affinity(X_train)

    model = SpectralClustering(
        n_clusters=n_clusters,
        eigen_solver=None,
        affinity="precomputed",
        random_state=random_state,
        assign_labels="cluster_qr",
    )
    labels = model.fit_predict(affinity)
    _save_clustering_outputs(CLUSTERING_MODEL_DIR, X_train, labels, affinity)

    if plot_graph:
        adjacency = compute_adjacency_matrix(affinity)
        target_values = _target_values_for_rows(y_train, X_train.index)
        _save_neighborhood_graph(
            CLUSTERING_MODEL_DIR,
            adjacency,
            target_values,
            X_train[TS_COL].to_numpy(),
            random_state,
        )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
    run_clustering(
        n_clusters=3,
        drop_na=False,
        random_state=42,
        n_rows_per_group=500,
        T0=None,
        T1=None,
        allocation_id=94,
        plot_graph=False
    )
