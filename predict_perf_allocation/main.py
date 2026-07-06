import gc
import json
import logging
from pathlib import Path
import pickle

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
TS_COL = "TS"
TARGET_COL = "target"
SCORE_DECIMALS = 3
SVM_PARAMS = {
    "C": 0.1,
    "soft_dtw_gamma": 0.1,
    "lambda_": 1.0,
    "rbf_gamma": 1.0,
    "alpha": 0.0,
    "kernel_combination": "multiplicative",
    "random_state": 1984,
    "max_memory_usage_fraction": 0.85
}

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
MODELS_DIR = PROJECT_ROOT / "models"
SAMPLE_SUBMISSION_PATH = RAW_DATA_DIR / "sample_submission.csv"


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


def _turnover_imputation_values(X_train):
    group_means = X_train.groupby(GROUP_COL)[TURNOVER_COL].mean()
    global_mean = X_train[TURNOVER_COL].mean()
    if pd.isna(global_mean):
        raise ValueError(f"Cannot impute {TURNOVER_COL}: all training values are NaN.")
    return {
        "group_means": group_means,
        "global_mean": global_mean,
    }


def _impute_turnover_by_group(df, imputation_values, dataset_name):
    missing_turnover = df[TURNOVER_COL].isna()
    if not missing_turnover.any():
        return df

    group_means = imputation_values["group_means"]
    global_mean = imputation_values["global_mean"]
    imputed_values = df.loc[missing_turnover, GROUP_COL].map(group_means).fillna(global_mean)
    df.loc[missing_turnover, TURNOVER_COL] = imputed_values.to_numpy(dtype=float)
    logging.info(
        "Imputed %s %s NaNs with %s means.",
        missing_turnover.sum(),
        dataset_name,
        GROUP_COL,
    )
    return df


def _feature_columns(k_last_days):
    if k_last_days is None:
        return RET_COLS, VOL_COLS
    if not isinstance(k_last_days, int):
        raise ValueError("k_last_days must be an integer between 1 and 20 or None.")
    if not 1 <= k_last_days <= len(RET_COLS):
        raise ValueError("k_last_days must be an integer between 1 and 20 or None.")
    return RET_COLS[-k_last_days:], VOL_COLS[-k_last_days:]


def _required_feature_cols(ret_cols, vol_cols):
    return ret_cols + vol_cols + [TURNOVER_COL]


def preprocess_data(
    X_train,
    y_train,
    X_test=None,
    drop_na=True,
    use_svc=True,
    ret_cols=None,
    vol_cols=None,
):
    """Scale model inputs and prepare the training target."""
    ret_cols = RET_COLS if ret_cols is None else ret_cols
    vol_cols = VOL_COLS if vol_cols is None else vol_cols
    y_train = y_train.copy()
    X_train = X_train.copy()
    X_test = None if X_test is None else X_test.copy()

    if drop_na:
        rows_not_na = X_train[_required_feature_cols(ret_cols, vol_cols)].notna().all(axis=1)
        dropped_rows = len(X_train) - rows_not_na.sum()
        y_train = y_train.loc[rows_not_na, TARGET_COL]
        X_train = X_train.loc[rows_not_na].copy()
        logging.info("Dropped %s X_train rows with NaNs.", dropped_rows)
        turnover_imputation_values = None
    else:
        y_train = y_train.loc[X_train.index, TARGET_COL]
        turnover_imputation_values = _turnover_imputation_values(X_train)
        X_train = _impute_turnover_by_group(
            X_train,
            turnover_imputation_values,
            "X_train",
        )
        if X_test is not None:
            X_test = _impute_turnover_by_group(
                X_test,
                turnover_imputation_values,
                "X_test",
            )

    X_train.loc[:, vol_cols], vol_scaler = _fit_transform_flattened_columns(X_train, vol_cols)
    X_train.loc[:, [TURNOVER_COL]], turnover_scaler = _fit_transform_column(X_train, TURNOVER_COL)

    if X_test is not None:
        X_test.loc[:, vol_cols] = _transform_flattened_columns(X_test, vol_cols, vol_scaler)
        X_test.loc[:, [TURNOVER_COL]] = turnover_scaler.transform(
            X_test[[TURNOVER_COL]].to_numpy(dtype=float)
        )

    X_train_ret = X_train[ret_cols].to_numpy(dtype=float)[:, :, np.newaxis]
    X_train_vol = X_train[vol_cols].to_numpy(dtype=float)[:, :, np.newaxis]
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
        X_test_ret = X_test[ret_cols].to_numpy(dtype=float)[:, :, np.newaxis]
        X_test_vol = X_test[vol_cols].to_numpy(dtype=float)[:, :, np.newaxis]
        X_test_ts = np.concatenate((X_test_ret, X_test_vol), axis=2)
        X_test_features = X_test[[TURNOVER_COL]].to_numpy(dtype=float)
        test_groups = X_test[GROUP_COL].to_numpy()

    scalers = {
        "volumes": vol_scaler,
        "turnover": turnover_scaler,
        "turnover_imputation": turnover_imputation_values,
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


def preprocess_test_data(X_test, scalers, ret_cols, vol_cols, drop_na=True):
    X_test = X_test.copy()
    if drop_na:
        rows_not_na = X_test[_required_feature_cols(ret_cols, vol_cols)].notna().all(axis=1)
        dropped_rows = len(X_test) - rows_not_na.sum()
        X_test = X_test.loc[rows_not_na].copy()
        logging.info("Dropped %s X_test rows with NaNs.", dropped_rows)
    else:
        X_test = _impute_turnover_by_group(
            X_test,
            scalers["turnover_imputation"],
            "X_test",
        )

    X_test.loc[:, vol_cols] = _transform_flattened_columns(
        X_test,
        vol_cols,
        scalers["volumes"],
    )
    X_test.loc[:, [TURNOVER_COL]] = scalers["turnover"].transform(
        X_test[[TURNOVER_COL]].to_numpy(dtype=float)
    )

    X_test_ret = X_test[ret_cols].to_numpy(dtype=float)[:, :, np.newaxis]
    X_test_vol = X_test[vol_cols].to_numpy(dtype=float)[:, :, np.newaxis]
    return {
        "row_ids": X_test.index.to_numpy(),
        "X_test_ts": np.concatenate((X_test_ret, X_test_vol), axis=2),
        "X_test_features": X_test[[TURNOVER_COL]].to_numpy(dtype=float),
        "test_groups": X_test[GROUP_COL].to_numpy(),
    }


def _drop_rows_with_nans(X_train, y_train, ret_cols, vol_cols):
    rows_not_na = X_train[_required_feature_cols(ret_cols, vol_cols)].notna().all(axis=1)
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
    if n_rows_per_group is not None:
        if not isinstance(n_rows_per_group, int):
            raise ValueError("n_rows_per_group must be a positive integer or None.")
        if n_rows_per_group <= 0:
            raise ValueError("n_rows_per_group must be a positive integer or None.")

    sampled_indexes = []
    row_counts = []
    for group, group_data in X_train.groupby(GROUP_COL, sort=True):
        n_selected = len(group_data)
        if n_rows_per_group is not None:
            n_selected = min(n_rows_per_group, len(group_data))
            sampled_index = group_data.sample(n=n_selected, random_state=random_state).index
        else:
            sampled_index = group_data.index

        selected_targets = y_train.loc[sampled_index, TARGET_COL]
        sampled_indexes.append(sampled_index)
        row_counts.append(
            {
                GROUP_COL: group,
                "available_rows": len(group_data),
                "selected_rows": n_selected,
                "positive_targets": (selected_targets > 0).sum(),
                "negative_targets": (selected_targets <= 0).sum(),
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
            "Group %s rows: selected %s/%s, positive targets: %s, negative targets: %s.",
            counts[GROUP_COL],
            counts["selected_rows"],
            counts["available_rows"],
            counts["positive_targets"],
            counts["negative_targets"],
        )
    return X_train_sampled, y_train_sampled


def _date_number_from_ts(ts_values):
    ts_numbers = ts_values.astype("string").str.extract(r"^DATE_(\d+)$", expand=False)
    if ts_numbers.isna().any():
        invalid_values = sorted(ts_values.loc[ts_numbers.isna()].astype(str).unique())
        invalid_preview = ", ".join(invalid_values[:5])
        raise ValueError(
            f"{TS_COL} values must have format 'DATE_XXXX'. "
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


def _filter_rows_by_timestamp_window(X_train, y_train, T0, T1):
    if T0 is None and T1 is None:
        return X_train, y_train

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
    return X_train.loc[rows_in_window].copy(), y_train.loc[rows_in_window].copy()


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


def save_best_models_and_predict_test(
    best_bundle,
    model_dir,
    use_svc,
    drop_na,
    ret_cols,
    vol_cols,
):
    best_models_dir = model_dir / "best_models"
    best_models_dir.mkdir(parents=True, exist_ok=True)

    for group, model in best_bundle["models"].items():
        model_path = best_models_dir / f"group_{group}.pkl"
        with model_path.open("wb") as file:
            pickle.dump(model, file)

    scalers_path = best_models_dir / "scalers.pkl"
    with scalers_path.open("wb") as file:
        pickle.dump(best_bundle["scalers"], file)

    metadata_path = best_models_dir / "metadata.json"
    with metadata_path.open("w", encoding="utf-8") as file:
        json.dump(
            {
                "best_fold": best_bundle["fold"],
                "best_valid_accuracy": round(best_bundle["valid_accuracy"], SCORE_DECIMALS),
                "use_svc": use_svc,
                "ret_cols": ret_cols,
                "vol_cols": vol_cols,
            },
            file,
            indent=2,
        )
    logging.info(
        "Saved best fold %s models to %s.",
        best_bundle["fold"],
        best_models_dir,
    )

    X_test = pd.read_csv(RAW_DATA_DIR / "X_test.csv", index_col="ROW_ID")
    test_data = preprocess_test_data(
        X_test,
        best_bundle["scalers"],
        ret_cols,
        vol_cols,
        drop_na=drop_na,
    )
    del X_test
    gc.collect()

    y_pred = predict_group_svms(
        best_bundle["models"],
        test_data["X_test_ts"],
        test_data["X_test_features"],
        test_data["test_groups"],
    )
    submission = pd.read_csv(SAMPLE_SUBMISSION_PATH)
    predictions = pd.Series(_sign_labels(y_pred), index=test_data["row_ids"])
    submission.loc[:, "prediction"] = (
        submission["ROW_ID"].map(predictions).fillna(submission["prediction"]).astype(int)
    )
    submission.to_csv(SAMPLE_SUBMISSION_PATH, index=False)
    logging.info("Saved X_test predictions to %s.", SAMPLE_SUBMISSION_PATH)


def run_svm(
    model_name="svm_group_cv",
    use_svc=True,
    drop_na=True,
    n_splits=2,
    random_state=42,
    n_rows_per_group=2000,
    k_last_days=None,
    predict_test=False,
    T0=None,
    T1=None,
):
    ret_cols, vol_cols = _feature_columns(k_last_days)
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
            "k_last_days": k_last_days,
            "predict_test": predict_test,
            "T0": T0,
            "T1": T1,
        },
    )

    X_train = pd.read_csv(RAW_DATA_DIR / "X_train.csv", index_col="ROW_ID")
    y_train = pd.read_csv(RAW_DATA_DIR / "y_train.csv", index_col="ROW_ID")

    X_train, y_train = _filter_rows_by_timestamp_window(X_train, y_train, T0, T1)

    if drop_na:
        X_train, y_train = _drop_rows_with_nans(X_train, y_train, ret_cols, vol_cols)

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
    train_on_smaller_cv_fold = n_splits > 2
    if train_on_smaller_cv_fold:
        logging.warning(
            "n_splits > 2: training on the smaller fold and validating on the larger split."
        )

    cv_scores = []
    best_bundle = None
    best_valid_score = -np.inf

    for fold, (train_idx, valid_idx) in enumerate(
        splitter.split(X_train, cv_stratification),
        start=1,
    ):
        logging.info("Starting fold %s/%s.", fold, n_splits)

        if train_on_smaller_cv_fold:
            train_idx, valid_idx = valid_idx, train_idx

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
            ret_cols=ret_cols,
            vol_cols=vol_cols,
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
            "Fold %s train accuracy: %.3f, validation accuracy: %.3f",
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
                "Fold %s train R2: %.3f, validation R2: %.3f",
                fold,
                train_r2,
                valid_r2,
            )
            fold_scores |= {
                "train_r2": train_r2,
                "valid_r2": valid_r2,
            }

        cv_scores.append(fold_scores | train_group_scores | valid_group_scores)

        if valid_score > best_valid_score:
            if best_bundle is not None:
                del best_bundle
                gc.collect()
            best_valid_score = valid_score
            best_bundle = {
                "fold": fold,
                "valid_accuracy": valid_score,
                "models": models,
                "scalers": data["scalers"],
            }
            models = None
            logging.info("Fold %s is the new best validation fold.", fold)

        if models is not None:
            del models

        del (
            data,
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

    scores = pd.DataFrame(cv_scores).round(SCORE_DECIMALS)
    scores_path = model_dir / "scores.csv"
    scores.to_csv(scores_path, index=False, float_format="%.3f")
    logging.info("Saved cross-validation scores to %s.", scores_path)

    if predict_test:
        if best_bundle is None:
            raise ValueError("Cannot predict X_test because no fold model was fitted.")
        save_best_models_and_predict_test(
            best_bundle,
            model_dir,
            use_svc=use_svc,
            drop_na=drop_na,
            ret_cols=ret_cols,
            vol_cols=vol_cols,
        )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
    run_svm(
        model_name="svc_prod_kernel",
        use_svc=True,
        drop_na=False,
        n_splits=2,
        random_state=42,
        n_rows_per_group=10000,
        k_last_days=None,
        predict_test=False,
        T0=1,
        T1=2,
    )
