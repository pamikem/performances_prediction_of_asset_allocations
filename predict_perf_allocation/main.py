import gc
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, r2_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import MinMaxScaler

from predict_perf_allocation.svc import SVMTS
from predict_perf_allocation.svr import SVMRegTS

RET_COLS = [f"RET_{i}" for i in range(20, 0, -1)]
VOL_COLS = [f"SIGNED_VOLUME_{i}" for i in range(20, 0, -1)]
TURNOVER_COL = "MEDIAN_DAILY_TURNOVER"
GROUP_COL = "GROUP"
TARGET_COL = "target"
REQUIRED_FEATURE_COLS = RET_COLS + VOL_COLS + [TURNOVER_COL]
SVM_PARAMS = {
    "C": 1.0,
    "soft_dtw_gamma": 0.1,
    "lambda_": 1.0,
    "rbf_gamma": "auto",
    "alpha": 0.0,
    "random_state": 42,
    "max_memory_usage_fraction": 0.85,
}

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
MODELS_DIR = PROJECT_ROOT / "models"


def _fit_transform_flattened_columns(df, cols):
    scaler = MinMaxScaler()
    values = df[cols].to_numpy(dtype=float)
    scaled_values = scaler.fit_transform(values.reshape(-1, 1)).reshape(values.shape)
    return scaled_values, scaler


def _transform_flattened_columns(df, cols, scaler):
    values = df[cols].to_numpy(dtype=float)
    return scaler.transform(values.reshape(-1, 1)).reshape(values.shape)


def _fit_transform_column(df, col):
    scaler = MinMaxScaler()
    values = df[[col]].to_numpy(dtype=float)
    return scaler.fit_transform(values), scaler


def preprocess_data(X_train, y_train, X_test=None, drop_na=True, use_svc=True):
    """Scale model inputs and prepare the training target."""
    y_train = y_train.copy()
    X_train = X_train.copy()
    X_test = None if X_test is None else X_test.copy()

    if drop_na:
        rows_not_na = X_train[REQUIRED_FEATURE_COLS].notna().all(axis=1)
        dropped_rows = len(X_train) - rows_not_na.sum()
        y_train = y_train.loc[rows_not_na, TARGET_COL]
        X_train = X_train.loc[rows_not_na].copy()
        logging.info("Dropped %s X_train rows with NaNs.", dropped_rows)
    else:
        y_train = y_train.loc[X_train.index, TARGET_COL]

    X_train.loc[:, VOL_COLS], vol_scaler = _fit_transform_flattened_columns(X_train, VOL_COLS)
    X_train.loc[:, [TURNOVER_COL]], turnover_scaler = _fit_transform_column(X_train, TURNOVER_COL)

    if X_test is not None:
        X_test.loc[:, VOL_COLS] = _transform_flattened_columns(X_test, VOL_COLS, vol_scaler)
        X_test.loc[:, [TURNOVER_COL]] = turnover_scaler.transform(
            X_test[[TURNOVER_COL]].to_numpy(dtype=float)
        )

    X_train_ret = X_train[RET_COLS].to_numpy(dtype=float)[:, :, np.newaxis]
    X_train_vol = X_train[VOL_COLS].to_numpy(dtype=float)[:, :, np.newaxis]
    X_train_ts = np.concatenate((X_train_ret, X_train_vol), axis=2)
    X_train_features = X_train[[TURNOVER_COL]].to_numpy(dtype=float)
    train_groups = X_train[GROUP_COL].to_numpy()
    y_train_raw = y_train.to_numpy(dtype=float).ravel()
    y_train_binary = _sign_labels(y_train_raw)
    y_train_model = y_train_binary if use_svc else y_train_raw

    X_test_ts = None
    X_test_features = None
    test_groups = None
    if X_test is not None:
        X_test_ret = X_test[RET_COLS].to_numpy(dtype=float)[:, :, np.newaxis]
        X_test_vol = X_test[VOL_COLS].to_numpy(dtype=float)[:, :, np.newaxis]
        X_test_ts = np.concatenate((X_test_ret, X_test_vol), axis=2)
        X_test_features = X_test[[TURNOVER_COL]].to_numpy(dtype=float)
        test_groups = X_test[GROUP_COL].to_numpy()

    scalers = {
        "volumes": vol_scaler,
        "turnover": turnover_scaler,
    }
    return {
        "X_train_ts": X_train_ts,
        "X_train_features": X_train_features,
        "train_groups": train_groups,
        "y_train": y_train_model,
        "y_train_raw": y_train_raw,
        "X_test_ts": X_test_ts,
        "X_test_features": X_test_features,
        "test_groups": test_groups,
        "scalers": scalers,
    }


def _model_params(use_svc):
    if use_svc:
        return SVM_PARAMS
    return {key: value for key, value in SVM_PARAMS.items() if key != "random_state"}


def fit_group_svms(X_train_ts, X_train_features, y_train, train_groups, use_svc=True):
    models = {}
    for group in sorted(np.unique(train_groups)):
        group_mask = train_groups == group
        group_y = y_train[group_mask]

        if use_svc and np.unique(group_y).size < 2:
            logging.warning("Skipping group %s because it contains only one class.", group)
            continue

        group_X_ts = X_train_ts[group_mask]
        group_X_features = X_train_features[group_mask]
        model_class = SVMTS if use_svc else SVMRegTS
        model = model_class(**_model_params(use_svc))
        logging.info(
            "Fitting %s for group %s on %s rows.",
            model_class.__name__,
            group,
            group_X_ts.shape[0],
        )
        model.fit(group_X_ts, group_y, Xf=group_X_features)
        models[group] = model

        del group_X_ts, group_X_features, group_y, group_mask, model
        gc.collect()

    return models


def predict_group_svms(models, X_ts, X_features, groups):
    y_pred = np.empty(len(groups), dtype=float)

    for group in sorted(np.unique(groups)):
        if group not in models:
            raise ValueError(f"No fitted SVM found for group {group}.")

        group_mask = groups == group
        y_pred[group_mask] = models[group].predict(
            X_ts[group_mask],
            Xf=X_features[group_mask],
        )

        del group_mask
        gc.collect()

    return y_pred


def _drop_rows_with_nans(X_train, y_train):
    rows_not_na = X_train[REQUIRED_FEATURE_COLS].notna().all(axis=1)
    dropped_rows = len(X_train) - rows_not_na.sum()
    logging.info("Dropped %s X_train rows with NaNs.", dropped_rows)
    return X_train.loc[rows_not_na].copy(), y_train.loc[rows_not_na].copy()


def _sign_labels(values):
    return (np.sign(np.asarray(values).ravel()) == 1).astype(int)


def _binary_labels(y):
    return _sign_labels(y[TARGET_COL].to_numpy(dtype=float))


def _sign_accuracy(y_true, y_pred):
    return accuracy_score(_sign_labels(y_true), _sign_labels(y_pred))


def _cv_stratification_labels(X_train, y_binary):
    return X_train[GROUP_COL].astype(str).to_numpy() + "_" + y_binary.astype(str)


def _sample_rows_per_group(X_train, y_train, n_rows_per_group, random_state):
    if n_rows_per_group is None:
        return X_train, y_train
    if not isinstance(n_rows_per_group, int):
        raise ValueError("n_rows_per_group must be a positive integer or None.")
    if n_rows_per_group <= 0:
        raise ValueError("n_rows_per_group must be a positive integer or None.")

    sampled_indexes = []
    row_counts = []
    for group, group_data in X_train.groupby(GROUP_COL, sort=True):
        n_selected = min(n_rows_per_group, len(group_data))
        sampled_index = group_data.sample(n=n_selected, random_state=random_state).index
        sampled_indexes.append(sampled_index)
        row_counts.append(
            {
                GROUP_COL: group,
                "available_rows": len(group_data),
                "selected_rows": n_selected,
            }
        )

    sampled_index = sampled_indexes[0].append(sampled_indexes[1:])

    X_train_sampled = X_train.loc[sampled_index].copy()
    y_train_sampled = y_train.loc[sampled_index].copy()
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
    return X_train_sampled, y_train_sampled


def _group_accuracy_scores(y_true, y_pred, groups, prefix):
    scores = {}
    for group in sorted(np.unique(groups)):
        group_mask = groups == group
        scores[f"{prefix}_accuracy_group_{group}"] = _sign_accuracy(
            y_true[group_mask],
            y_pred[group_mask],
        )
    return scores


def _save_run_config(model_dir, run_args):
    config = {
        "run_svm_args": run_args,
        "svm_params": SVM_PARAMS,
    }
    config_path = model_dir / "run_config.json"
    with config_path.open("w", encoding="utf-8") as file:
        json.dump(config, file, indent=2)
    logging.info("Saved run configuration to %s.", config_path)


def run_svm(
    model_name="svm_group_cv",
    use_svc=True,
    drop_na=True,
    n_splits=2,
    random_state=42,
    n_rows_per_group=2000,
):
    model_dir = MODELS_DIR / model_name
    model_dir.mkdir(parents=True, exist_ok=True)
    _save_run_config(
        model_dir,
        {
            "model_name": model_name,
            "use_svc": use_svc,
            "drop_na": drop_na,
            "n_splits": n_splits,
            "random_state": random_state,
            "n_rows_per_group": n_rows_per_group,
        },
    )

    X_train = pd.read_csv(RAW_DATA_DIR / "X_train.csv", index_col="ROW_ID")
    y_train = pd.read_csv(RAW_DATA_DIR / "y_train.csv", index_col="ROW_ID")

    if drop_na:
        X_train, y_train = _drop_rows_with_nans(X_train, y_train)

    X_train, y_train = _sample_rows_per_group(
        X_train,
        y_train,
        n_rows_per_group=n_rows_per_group,
        random_state=random_state,
    )

    y_binary = _binary_labels(y_train)
    cv_stratification = _cv_stratification_labels(X_train, y_binary)
    splitter = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=random_state,
    )
    cv_scores = []

    for fold, (train_idx, valid_idx) in enumerate(
        splitter.split(X_train, cv_stratification),
        start=1,
    ):
        logging.info("Starting fold %s/%s.", fold, n_splits)

        fold_X_train = X_train.iloc[train_idx].copy()
        fold_y_train = y_train.iloc[train_idx].copy()
        fold_X_valid = X_train.iloc[valid_idx].copy()
        fold_y_valid = y_train.iloc[valid_idx][TARGET_COL].to_numpy(dtype=float)

        data = preprocess_data(
            fold_X_train,
            fold_y_train,
            X_test=fold_X_valid,
            drop_na=drop_na,
            use_svc=use_svc,
        )
        del fold_X_train, fold_y_train, fold_X_valid
        gc.collect()

        models = fit_group_svms(
            data["X_train_ts"],
            data["X_train_features"],
            data["y_train"],
            data["train_groups"],
            use_svc=use_svc,
        )
        train_pred = predict_group_svms(
            models,
            data["X_train_ts"],
            data["X_train_features"],
            data["train_groups"],
        )
        y_pred = predict_group_svms(
            models,
            data["X_test_ts"],
            data["X_test_features"],
            data["test_groups"],
        )
        train_score = _sign_accuracy(data["y_train_raw"], train_pred)
        valid_score = _sign_accuracy(fold_y_valid, y_pred)
        train_group_scores = _group_accuracy_scores(
            data["y_train_raw"],
            train_pred,
            data["train_groups"],
            prefix="train",
        )
        valid_group_scores = _group_accuracy_scores(
            fold_y_valid,
            y_pred,
            data["test_groups"],
            prefix="valid",
        )
        logging.info(
            "Fold %s train accuracy: %.6f, validation accuracy: %.6f",
            fold,
            train_score,
            valid_score,
        )
        fold_scores = {
            "fold": fold,
            "train_accuracy": train_score,
            "valid_accuracy": valid_score,
            "n_train": len(train_idx),
            "n_valid": len(valid_idx),
            "n_models": len(models),
        }
        if not use_svc:
            train_r2 = r2_score(data["y_train_raw"], train_pred)
            valid_r2 = r2_score(fold_y_valid, y_pred)
            logging.info(
                "Fold %s train R2: %.6f, validation R2: %.6f",
                fold,
                train_r2,
                valid_r2,
            )
            fold_scores |= {
                "train_r2": train_r2,
                "valid_r2": valid_r2,
            }

        cv_scores.append(fold_scores | train_group_scores | valid_group_scores)

        del (
            data,
            models,
            train_pred,
            y_pred,
            fold_y_valid,
            train_idx,
            valid_idx,
            fold_scores,
            train_group_scores,
            valid_group_scores,
        )
        gc.collect()

    scores = pd.DataFrame(cv_scores)
    scores_path = model_dir / "svm_group_cv_scores.csv"
    scores.to_csv(scores_path, index=False)
    logging.info("Saved cross-validation scores to %s.", scores_path)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
    run_svm(n_rows_per_group=4000, use_svc=True, n_splits=2)
