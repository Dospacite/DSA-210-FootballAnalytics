from __future__ import annotations

import argparse
import json
import math
import os
import warnings
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, log_loss
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from dataset import (
    DEFAULT_CUTOFF,
    DEFAULT_ALLOW_FALLBACK_EVENTS,
    OUTPUT_DIR,
    VALIDATION_FRACTION_WITHIN_TRAIN,
    goal_count_bucket_label,
    load_prepared_splits,
)

try:
    from catboost import CatBoostClassifier
except ImportError:  # pragma: no cover - exercised via runtime failure path
    CatBoostClassifier = None

try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover - optional dependency
    tqdm = None

SEED = int(os.getenv("FOOTBALL_PREDICTOR_SEED", "42"))
DEFAULT_ARTIFACTS_DIR = os.path.join(OUTPUT_DIR, "predictor_artifacts")
RESULT_LABELS = ["1", "X", "2"]
GOAL_BUCKET_LABELS = ["0", "1", "2", "3", "4+"]
DEFAULT_ROLLING_BACKTEST_WINDOWS = 3
LOGISTIC_MAX_ITER = 150


@dataclass(frozen=True)
class TaskConfig:
    name: str
    target_column: str
    labels: list[str]


@dataclass(frozen=True)
class ModelSpec:
    name: str
    family: str
    feature_view: str


TASKS = (
    TaskConfig(name="1x2", target_column="target_1x2", labels=RESULT_LABELS),
    TaskConfig(name="goal_bucket", target_column="target_goal_bucket", labels=GOAL_BUCKET_LABELS),
)

MODEL_SPECS = (
    ModelSpec(name="frequency_prior", family="prior", feature_view="none"),
    ModelSpec(name="market_prior", family="market", feature_view="prematch_only"),
    ModelSpec(name="logistic_regression", family="logistic", feature_view="full_hybrid"),
    ModelSpec(name="catboost_prematch_only", family="catboost", feature_view="prematch_only"),
    ModelSpec(name="catboost_full_hybrid", family="catboost", feature_view="full_hybrid"),
    ModelSpec(name="catboost_odds_dropped", family="catboost", feature_view="odds_dropped"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train/evaluate Mackolik 1X2 and goal-bucket predictors.")
    parser.add_argument("--cutoff", type=int, default=DEFAULT_CUTOFF)
    parser.add_argument("--artifacts-dir", default=DEFAULT_ARTIFACTS_DIR)
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--force-rebuild", action="store_true")
    parser.add_argument("--load-prepared", action="store_true", default=True)
    parser.add_argument("--no-load-prepared", action="store_false", dest="load_prepared")
    parser.add_argument("--rolling-backtest-windows", type=int, default=DEFAULT_ROLLING_BACKTEST_WINDOWS)
    parser.add_argument("--allow-fallback-events", action="store_true", default=DEFAULT_ALLOW_FALLBACK_EVENTS)
    parser.add_argument("--no-allow-fallback-events", action="store_false", dest="allow_fallback_events")
    return parser.parse_args()


def progress_iterable(
    iterable: Any,
    *,
    total: int | None = None,
    desc: str,
    unit: str,
    leave: bool = True,
) -> Any:
    if tqdm is None:
        return iterable
    return tqdm(iterable, total=total, desc=desc, unit=unit, leave=leave)


def progress_message(message: str) -> None:
    if tqdm is not None:
        tqdm.write(message)
        return
    print(message)


def training_progress_bar(task_name: str, model_name: str, total: int) -> Any:
    if tqdm is None:
        return nullcontext(None)
    return tqdm(total=total, desc=f"{task_name}:{model_name}", unit="phase", leave=False)


def phase_progress_bar(desc: str, total: int, *, unit: str = "phase", leave: bool = False) -> Any:
    if tqdm is None:
        return nullcontext(None)
    return tqdm(total=total, desc=desc, unit=unit, leave=leave)


class CatBoostTqdmCallback:
    def __init__(self, progress_bar: Any) -> None:
        self.progress_bar = progress_bar
        self.last_iteration = 0

    def after_iteration(self, info: Any) -> bool:
        if self.progress_bar is None:
            return True
        iteration = max(int(getattr(info, "iteration", 0) or 0), 0)
        delta = iteration - self.last_iteration
        if delta > 0:
            self.progress_bar.update(delta)
            self.last_iteration = iteration
        return True


def flatten_feature_column(frame: pd.DataFrame) -> pd.DataFrame:
    features = pd.json_normalize(frame["features"]).copy()
    features.index = frame.index
    return features


def infer_goal_bucket(frame: pd.DataFrame) -> pd.Series:
    if "target_goal_bucket" in frame.columns and frame["target_goal_bucket"].notna().any():
        return frame["target_goal_bucket"].astype(str)
    if "target_total_goals" in frame.columns and frame["target_total_goals"].notna().any():
        return frame["target_total_goals"].fillna(0).astype(int).map(goal_count_bucket_label)
    return frame.apply(
        lambda row: goal_count_bucket_label(int(row.get("target_total_goals", 0) or 0)),
        axis=1,
    )


def infer_event_source(frame: pd.DataFrame, features: pd.DataFrame) -> pd.Series:
    if "event_source" in frame.columns:
        return frame["event_source"].fillna("unknown").astype(str)
    if {"live_15_source_is_f24", "live_15_source_is_match_data"}.issubset(features.columns):
        source = pd.Series("unknown", index=features.index, dtype="object")
        source.loc[pd.to_numeric(features["live_15_source_is_f24"], errors="coerce").fillna(0.0) > 0.5] = "f24"
        source.loc[pd.to_numeric(features["live_15_source_is_match_data"], errors="coerce").fillna(0.0) > 0.5] = "match_data"
        return source.astype(str)
    return pd.Series("unknown", index=features.index, dtype="object")


def build_modeling_frame(frame: pd.DataFrame) -> pd.DataFrame:
    features = flatten_feature_column(frame)
    modeling = features.copy()
    modeling["match_id"] = frame["match_id"].astype(str)
    modeling["match_timestamp"] = pd.to_datetime(frame["match_timestamp"], utc=True, errors="coerce")
    modeling["event_source"] = infer_event_source(frame, features)
    modeling["target_1x2"] = frame["target_1x2"].astype(str)
    modeling["target_goal_bucket"] = infer_goal_bucket(frame).astype(str)
    return modeling.sort_values(["match_timestamp", "match_id"]).reset_index(drop=True)


def chronological_train_validation_split(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    ordered = frame.sort_values(["match_timestamp", "match_id"]).reset_index(drop=True)
    if len(ordered) < 2:
        raise RuntimeError("Need at least two training rows to create a validation split.")
    split_index = int(len(ordered) * (1.0 - VALIDATION_FRACTION_WITHIN_TRAIN))
    split_index = max(1, min(split_index, len(ordered) - 1))
    return ordered.iloc[:split_index].copy(), ordered.iloc[split_index:].copy()


def feature_group_name(feature_name: str) -> str:
    if feature_name.startswith("odds_"):
        return "odds"
    if feature_name.startswith("history_"):
        return "history"
    if feature_name.startswith("embedded_"):
        return "embedded"
    if feature_name.startswith("squad_"):
        return "squad"
    if feature_name.startswith("live_"):
        return "live"
    if feature_name.startswith("context_"):
        return "context"
    if feature_name.startswith("interaction_"):
        return "interaction"
    return "other"


def feature_dependencies(feature_name: str) -> set[str]:
    group = feature_group_name(feature_name)
    if group != "interaction":
        return {group}

    deps: set[str] = set()
    if "favorite_gap" in feature_name:
        deps.add("odds")
    if "history_" in feature_name:
        deps.add("history")
    if "embedded" in feature_name:
        deps.add("embedded")
    if "squad_" in feature_name:
        deps.add("squad")
    if "matchday" in feature_name:
        deps.add("context")
    if "live_" in feature_name:
        deps.add("live")
    return deps or {"interaction"}


def protected_columns() -> set[str]:
    return {"match_id", "match_timestamp", "event_source", "target_1x2", "target_goal_bucket"}


def feature_columns(frame: pd.DataFrame) -> list[str]:
    return [column for column in frame.columns if column not in protected_columns()]


def select_feature_columns(frame: pd.DataFrame, view_name: str) -> list[str]:
    columns = feature_columns(frame)
    if view_name == "full_hybrid":
        return sorted(columns)
    if view_name == "prematch_only":
        allowed = {"context", "odds", "embedded", "history", "squad"}
        return sorted(
            column
            for column in columns
            if feature_dependencies(column).issubset(allowed)
        )
    if view_name == "odds_dropped":
        excluded = {"odds"}
        return sorted(
            column
            for column in columns
            if not feature_dependencies(column).intersection(excluded)
        )
    if view_name == "live_dropped":
        excluded = {"live"}
        return sorted(
            column
            for column in columns
            if not feature_dependencies(column).intersection(excluded)
        )
    raise ValueError(f"Unsupported feature view: {view_name}")


def split_feature_types(frame: pd.DataFrame, columns: list[str]) -> tuple[list[str], list[str]]:
    categorical_columns: list[str] = []
    numeric_columns: list[str] = []
    for column in columns:
        series = frame[column]
        if pd.api.types.is_numeric_dtype(series):
            numeric_columns.append(column)
        else:
            categorical_columns.append(column)
    return sorted(numeric_columns), sorted(categorical_columns)


def ensure_minimum_class_support(frame: pd.DataFrame, task: TaskConfig, split_name: str) -> None:
    values = frame[task.target_column].astype(str)
    observed = sorted(values.dropna().unique().tolist())
    if len(observed) < 2:
        raise RuntimeError(
            f"Insufficient class support for task '{task.name}' in {split_name}: observed {observed or ['<none>']}."
        )


def label_to_index_map(labels: list[str]) -> dict[str, int]:
    return {label: index for index, label in enumerate(labels)}


def encode_target(values: pd.Series, labels: list[str]) -> np.ndarray:
    mapping = label_to_index_map(labels)
    encoded = values.astype(str).map(mapping)
    if encoded.isna().any():
        unknown = sorted(values[encoded.isna()].astype(str).unique().tolist())
        raise RuntimeError(f"Unknown labels encountered: {unknown}")
    return encoded.astype(int).to_numpy()


def make_logistic_pipeline(numeric_columns: list[str], categorical_columns: list[str]) -> Pipeline:
    transformers: list[tuple[str, Any, list[str]]] = []
    if numeric_columns:
        transformers.append(
            (
                "numeric",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numeric_columns,
            )
        )
    if categorical_columns:
        transformers.append(
            (
                "categorical",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical_columns,
            )
        )
    preprocessor = ColumnTransformer(transformers=transformers, remainder="drop")
    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "model",
                LogisticRegression(
                    max_iter=LOGISTIC_MAX_ITER,
                    class_weight="balanced",
                    random_state=SEED,
                    solver="saga",
                ),
            ),
        ]
    )


def fit_logistic_with_progress(
    train_frame: pd.DataFrame,
    predict_frame: pd.DataFrame,
    columns: list[str],
    numeric_columns: list[str],
    categorical_columns: list[str],
    train_target: np.ndarray,
    *,
    task_name: str,
    model_name: str,
) -> tuple[Pipeline, np.ndarray]:
    pipeline = make_logistic_pipeline(numeric_columns, categorical_columns)
    preprocessor = pipeline.named_steps["preprocessor"]
    model = pipeline.named_steps["model"]

    train_features = train_frame.reindex(columns=columns)
    predict_features = predict_frame.reindex(columns=columns)
    transformed_train = preprocessor.fit_transform(train_features)
    transformed_predict = preprocessor.transform(predict_features)

    with training_progress_bar(task_name, model_name, LOGISTIC_MAX_ITER) as progress:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=ConvergenceWarning)
            for _ in range(LOGISTIC_MAX_ITER):
                model.max_iter = 1
                model.warm_start = True
                model.fit(transformed_train, train_target)
                if progress is not None:
                    progress.update(1)

    probabilities = np.asarray(model.predict_proba(transformed_predict), dtype=float)
    return pipeline, probabilities


def prepare_catboost_frame(frame: pd.DataFrame, columns: list[str], categorical_columns: list[str]) -> pd.DataFrame:
    prepared = frame.reindex(columns=columns).copy()
    for column in categorical_columns:
        prepared[column] = prepared[column].fillna("__missing__").astype(str)
    return prepared


def fit_catboost_classifier(
    train_frame: pd.DataFrame,
    validation_frame: pd.DataFrame,
    target_column: str,
    labels: list[str],
    columns: list[str],
    task_name: str,
    model_name: str,
) -> tuple[Any, Any]:
    if CatBoostClassifier is None:
        raise RuntimeError("catboost is required for predictor.py. Install dependencies from requirements.txt first.")

    numeric_columns, categorical_columns = split_feature_types(train_frame, columns)
    train_x = prepare_catboost_frame(train_frame, columns, categorical_columns)
    val_x = prepare_catboost_frame(validation_frame, columns, categorical_columns)
    train_y = encode_target(train_frame[target_column], labels)
    val_y = encode_target(validation_frame[target_column], labels)
    cat_features = [train_x.columns.get_loc(column) for column in categorical_columns]

    tuning_model = CatBoostClassifier(
        loss_function="MultiClass",
        eval_metric="MultiClass",
        auto_class_weights="Balanced",
        random_seed=SEED,
        iterations=400,
        learning_rate=0.05,
        depth=6,
        l2_leaf_reg=4.0,
        early_stopping_rounds=40,
        verbose=False,
        allow_writing_files=False,
    )
    tuning_callbacks = []
    with training_progress_bar(task_name, f"{model_name}:tune", 400) as tuning_progress:
        if tuning_progress is not None:
            tuning_callbacks.append(CatBoostTqdmCallback(tuning_progress))
        tuning_model.fit(
            train_x,
            train_y,
            eval_set=(val_x, val_y),
            cat_features=cat_features,
            use_best_model=True,
            callbacks=tuning_callbacks or None,
        )
        if tuning_progress is not None and tuning_progress.n < tuning_progress.total:
            tuning_progress.update(tuning_progress.total - tuning_progress.n)

    best_iterations = tuning_model.get_best_iteration()
    final_iterations = max(int(best_iterations) + 1 if best_iterations is not None and best_iterations >= 0 else 150, 50)
    final_train = pd.concat([train_frame, validation_frame], ignore_index=True)
    final_x = prepare_catboost_frame(final_train, columns, categorical_columns)
    final_y = encode_target(final_train[target_column], labels)
    final_model = CatBoostClassifier(
        loss_function="MultiClass",
        eval_metric="MultiClass",
        auto_class_weights="Balanced",
        random_seed=SEED,
        iterations=final_iterations,
        learning_rate=0.05,
        depth=6,
        l2_leaf_reg=4.0,
        verbose=False,
        allow_writing_files=False,
    )
    final_callbacks = []
    with training_progress_bar(task_name, f"{model_name}:final", final_iterations) as final_progress:
        if final_progress is not None:
            final_callbacks.append(CatBoostTqdmCallback(final_progress))
        final_model.fit(final_x, final_y, cat_features=cat_features, callbacks=final_callbacks or None)
        if final_progress is not None and final_progress.n < final_progress.total:
            final_progress.update(final_progress.total - final_progress.n)
    return tuning_model, final_model


def class_metrics_from_report(report: dict[str, Any], labels: list[str]) -> dict[str, dict[str, float]]:
    metrics: dict[str, dict[str, float]] = {}
    for label in labels:
        label_report = report.get(str(label), {})
        metrics[str(label)] = {
            "precision": float(label_report.get("precision", 0.0)),
            "recall": float(label_report.get("recall", 0.0)),
            "f1_score": float(label_report.get("f1-score", 0.0)),
            "support": float(label_report.get("support", 0.0)),
        }
    return metrics


def calibration_summary(y_true: np.ndarray, probabilities: np.ndarray, labels: list[str], bins: int = 10) -> dict[str, Any]:
    pred_indices = probabilities.argmax(axis=1)
    confidences = probabilities.max(axis=1)
    correctness = (pred_indices == y_true).astype(float)
    ece = 0.0
    bin_rows: list[dict[str, float]] = []
    for bin_index in range(bins):
        left = bin_index / bins
        right = (bin_index + 1) / bins
        if bin_index == bins - 1:
            mask = (confidences >= left) & (confidences <= right)
        else:
            mask = (confidences >= left) & (confidences < right)
        if not mask.any():
            continue
        avg_confidence = float(confidences[mask].mean())
        avg_accuracy = float(correctness[mask].mean())
        weight = float(mask.mean())
        ece += abs(avg_confidence - avg_accuracy) * weight
        bin_rows.append(
            {
                "bin_left": left,
                "bin_right": right,
                "count": int(mask.sum()),
                "avg_confidence": avg_confidence,
                "avg_accuracy": avg_accuracy,
                "gap": avg_confidence - avg_accuracy,
            }
        )

    per_class_brier: dict[str, float] = {}
    for label_index, label in enumerate(labels):
        one_vs_rest = (y_true == label_index).astype(float)
        per_class_brier[str(label)] = float(np.mean((probabilities[:, label_index] - one_vs_rest) ** 2))

    return {
        "ece": float(ece),
        "mean_confidence": float(confidences.mean()) if len(confidences) else 0.0,
        "mean_accuracy": float(correctness.mean()) if len(correctness) else 0.0,
        "per_class_brier": per_class_brier,
        "bins": bin_rows,
    }


def rank_probability_score(y_true: np.ndarray, probabilities: np.ndarray) -> float:
    class_count = probabilities.shape[1]
    if class_count <= 1:
        return 0.0
    truth = np.eye(class_count)[y_true]
    cumulative_truth = np.cumsum(truth, axis=1)
    cumulative_probabilities = np.cumsum(probabilities, axis=1)
    squared_error = np.square(cumulative_probabilities - cumulative_truth).sum(axis=1)
    return float(np.mean(squared_error / (class_count - 1)))


def evaluate_probabilities(y_true: np.ndarray, probabilities: np.ndarray, labels: list[str]) -> dict[str, Any]:
    pred_indices = probabilities.argmax(axis=1)
    report = classification_report(
        y_true,
        pred_indices,
        labels=list(range(len(labels))),
        target_names=labels,
        zero_division=0,
        output_dict=True,
    )
    return {
        "accuracy": float(accuracy_score(y_true, pred_indices)),
        "macro_f1": float(f1_score(y_true, pred_indices, average="macro", zero_division=0)),
        "log_loss": float(log_loss(y_true, probabilities, labels=list(range(len(labels))))),
        "rank_probability_score": rank_probability_score(y_true, probabilities),
        "confusion_matrix": {
            "labels": labels,
            "matrix": confusion_matrix(y_true, pred_indices, labels=list(range(len(labels)))).tolist(),
        },
        "per_class_metrics": class_metrics_from_report(report, labels),
        "calibration": calibration_summary(y_true, probabilities, labels),
    }


def evaluate_source_cohorts(
    frame: pd.DataFrame,
    y_true: np.ndarray,
    probabilities: np.ndarray,
    labels: list[str],
) -> dict[str, Any]:
    if "event_source" not in frame.columns:
        return {}
    cohorts: dict[str, Any] = {}
    event_source = frame["event_source"].fillna("unknown").astype(str).reset_index(drop=True)
    for source_name in sorted(event_source.unique().tolist()):
        mask = (event_source == source_name).to_numpy()
        if int(mask.sum()) < 1:
            continue
        cohorts[source_name] = {
            "rows": int(mask.sum()),
            **evaluate_probabilities(y_true[mask], probabilities[mask], labels),
        }
    return cohorts


def feature_importance_summary(model: Any, feature_names: list[str]) -> dict[str, Any]:
    if not hasattr(model, "get_feature_importance"):
        return {"by_feature": [], "by_group": {}}
    importances = model.get_feature_importance()
    rows = []
    grouped: dict[str, float] = {}
    for name, score in zip(feature_names, importances, strict=True):
        numeric_score = float(score)
        rows.append({"feature": name, "importance": numeric_score, "group": feature_group_name(name)})
        grouped[feature_group_name(name)] = grouped.get(feature_group_name(name), 0.0) + numeric_score
    rows.sort(key=lambda row: row["importance"], reverse=True)
    grouped = dict(sorted(grouped.items(), key=lambda item: item[1], reverse=True))
    return {"by_feature": rows[:40], "by_group": grouped}


def predict_proba(model: Any, frame: pd.DataFrame, columns: list[str], categorical_columns: list[str]) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        if CatBoostClassifier is not None and isinstance(model, CatBoostClassifier):
            prepared = prepare_catboost_frame(frame, columns, categorical_columns)
            return np.asarray(model.predict_proba(prepared), dtype=float)
        return np.asarray(model.predict_proba(frame.reindex(columns=columns)), dtype=float)
    raise RuntimeError(f"Model {type(model).__name__} does not support predict_proba.")


def frequency_prior_probabilities(train_target: pd.Series, labels: list[str], rows: int) -> np.ndarray:
    encoded = encode_target(train_target, labels)
    counts = np.bincount(encoded, minlength=len(labels)).astype(float)
    probabilities = counts / counts.sum()
    return np.tile(probabilities, (rows, 1))


def market_prior_probabilities(frame: pd.DataFrame, task: TaskConfig) -> np.ndarray | None:
    if task.name != "1x2":
        return None
    required_columns = [
        "odds_match_result_implied_home",
        "odds_match_result_implied_draw",
        "odds_match_result_implied_away",
    ]
    if any(column not in frame.columns for column in required_columns):
        return None
    probabilities = frame[required_columns].apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy(dtype=float)
    row_sums = probabilities.sum(axis=1, keepdims=True)
    valid = row_sums[:, 0] > 0
    if not valid.any():
        return None
    fallback = probabilities[valid].mean(axis=0)
    fallback_sum = fallback.sum()
    if fallback_sum <= 0:
        return None
    fallback = fallback / fallback_sum
    row_sums[row_sums == 0.0] = 1.0
    normalized = probabilities / row_sums
    missing_mask = probabilities.sum(axis=1) <= 0
    if missing_mask.any():
        normalized[missing_mask] = fallback
    return normalized


def rolling_origin_splits(frame: pd.DataFrame, windows: int) -> list[tuple[pd.DataFrame, pd.DataFrame]]:
    ordered = frame.sort_values(["match_timestamp", "match_id"]).reset_index(drop=True)
    if windows < 1 or len(ordered) < 6:
        return []
    holdout_size = max(1, len(ordered) // (windows + 2))
    splits: list[tuple[pd.DataFrame, pd.DataFrame]] = []
    for index in range(windows):
        train_end = holdout_size * (index + 2)
        test_end = min(train_end + holdout_size, len(ordered))
        if test_end - train_end < 1 or train_end < 4:
            continue
        train_slice = ordered.iloc[:train_end].copy()
        test_slice = ordered.iloc[train_end:test_end].copy()
        if not test_slice.empty:
            splits.append((train_slice, test_slice))
    return splits


def benchmark_task(
    train_frame: pd.DataFrame,
    test_frame: pd.DataFrame,
    task: TaskConfig,
) -> dict[str, Any]:
    ensure_minimum_class_support(train_frame, task, "train split")
    dev_train, dev_validation = chronological_train_validation_split(train_frame)
    ensure_minimum_class_support(dev_train, task, "train-development split")
    ensure_minimum_class_support(dev_validation, task, "validation split")

    y_validation = encode_target(dev_validation[task.target_column], task.labels)
    y_test = encode_target(test_frame[task.target_column], task.labels)

    task_results: dict[str, Any] = {}
    for spec in progress_iterable(
        MODEL_SPECS,
        total=len(MODEL_SPECS),
        desc=f"{task.name} models",
        unit="model",
        leave=False,
    ):
        phase_total = 4 if spec.family in {"prior", "market"} else 1
        progress_context = (
            nullcontext(None)
            if spec.family in {"logistic", "catboost"}
            else training_progress_bar(task.name, spec.name, phase_total)
        )
        with progress_context as model_progress:
            def advance_phase() -> None:
                if model_progress is not None:
                    model_progress.update(1)

            if spec.family == "prior":
                val_probs = frequency_prior_probabilities(dev_train[task.target_column], task.labels, len(dev_validation))
                advance_phase()
                test_probs = frequency_prior_probabilities(train_frame[task.target_column], task.labels, len(test_frame))
                advance_phase()
                validation_metrics = evaluate_probabilities(y_validation, val_probs, task.labels)
                advance_phase()
                test_metrics = evaluate_probabilities(y_test, test_probs, task.labels)
                advance_phase()
                task_results[spec.name] = {
                    "family": spec.family,
                    "feature_view": spec.feature_view,
                    "feature_count": 0,
                    "validation": validation_metrics,
                    "validation_by_source": evaluate_source_cohorts(dev_validation, y_validation, val_probs, task.labels),
                    "test": test_metrics,
                    "test_by_source": evaluate_source_cohorts(test_frame, y_test, test_probs, task.labels),
                }
                continue

            if spec.family == "market":
                market_val_probs = market_prior_probabilities(dev_validation, task)
                advance_phase()
                market_test_probs = market_prior_probabilities(test_frame, task)
                advance_phase()
                if market_val_probs is None or market_test_probs is None:
                    advance_phase()
                    advance_phase()
                    continue
                validation_metrics = evaluate_probabilities(y_validation, market_val_probs, task.labels)
                advance_phase()
                test_metrics = evaluate_probabilities(y_test, market_test_probs, task.labels)
                advance_phase()
                task_results[spec.name] = {
                    "family": spec.family,
                    "feature_view": spec.feature_view,
                    "feature_count": 3,
                    "validation": validation_metrics,
                    "validation_by_source": evaluate_source_cohorts(dev_validation, y_validation, market_val_probs, task.labels),
                    "test": test_metrics,
                    "test_by_source": evaluate_source_cohorts(test_frame, y_test, market_test_probs, task.labels),
                }
                continue

            columns = select_feature_columns(train_frame, spec.feature_view)
            if not columns:
                raise RuntimeError(f"No features selected for model '{spec.name}'.")
            numeric_columns, categorical_columns = split_feature_types(train_frame, columns)

            if spec.family == "logistic":
                model, validation_probs = fit_logistic_with_progress(
                    dev_train,
                    dev_validation,
                    columns,
                    numeric_columns,
                    categorical_columns,
                    encode_target(dev_train[task.target_column], task.labels),
                    task_name=task.name,
                    model_name=f"{spec.name}:dev",
                )

                final_model, test_probs = fit_logistic_with_progress(
                    train_frame,
                    test_frame,
                    columns,
                    numeric_columns,
                    categorical_columns,
                    encode_target(train_frame[task.target_column], task.labels),
                    task_name=task.name,
                    model_name=f"{spec.name}:final",
                )
                task_results[spec.name] = {
                    "family": spec.family,
                    "feature_view": spec.feature_view,
                    "feature_count": len(columns),
                    "feature_groups": sorted({feature_group_name(column) for column in columns}),
                    "validation": evaluate_probabilities(y_validation, validation_probs, task.labels),
                    "validation_by_source": evaluate_source_cohorts(dev_validation, y_validation, validation_probs, task.labels),
                    "test": evaluate_probabilities(y_test, test_probs, task.labels),
                    "test_by_source": evaluate_source_cohorts(test_frame, y_test, test_probs, task.labels),
                }
                continue

            tuning_model, final_model = fit_catboost_classifier(
                dev_train,
                dev_validation,
                task.target_column,
                task.labels,
                columns,
                task.name,
                spec.name,
            )
            validation_probs = predict_proba(tuning_model, dev_validation, columns, categorical_columns)
            test_probs = predict_proba(final_model, test_frame, columns, categorical_columns)
            task_results[spec.name] = {
                "family": spec.family,
                "feature_view": spec.feature_view,
                "feature_count": len(columns),
                "feature_groups": sorted({feature_group_name(column) for column in columns}),
                "validation": evaluate_probabilities(y_validation, validation_probs, task.labels),
                "validation_by_source": evaluate_source_cohorts(dev_validation, y_validation, validation_probs, task.labels),
                "test": evaluate_probabilities(y_test, test_probs, task.labels),
                "test_by_source": evaluate_source_cohorts(test_frame, y_test, test_probs, task.labels),
                "feature_importance": feature_importance_summary(final_model, columns),
            }

    return task_results


def rolling_backtest(
    train_frame: pd.DataFrame,
    *,
    windows: int,
) -> dict[str, Any]:
    splits = rolling_origin_splits(train_frame, windows)
    if not splits:
        return {"windows_requested": int(windows), "folds": [], "summary": []}

    fold_rows: list[dict[str, Any]] = []
    for fold_index, (fold_train, fold_test) in enumerate(
        progress_iterable(splits, total=len(splits), desc="Rolling backtest", unit="fold"),
        start=1,
    ):
        with phase_progress_bar(f"Backtest fold {fold_index}", len(TASKS), unit="task", leave=False) as fold_progress:
            for task in TASKS:
                if fold_train[task.target_column].astype(str).nunique() < 2 or fold_test[task.target_column].astype(str).nunique() < 1:
                    if fold_progress is not None:
                        fold_progress.update(1)
                    continue
                task_results = benchmark_task(fold_train, fold_test, task)
                for model_name, model_result in task_results.items():
                    metrics = model_result["test"]
                    fold_rows.append(
                        {
                            "fold": fold_index,
                            "task": task.name,
                            "model": model_name,
                            "family": model_result["family"],
                            "feature_view": model_result["feature_view"],
                            "train_rows": int(len(fold_train)),
                            "test_rows": int(len(fold_test)),
                            "accuracy": metrics["accuracy"],
                            "macro_f1": metrics["macro_f1"],
                            "log_loss": metrics["log_loss"],
                            "rank_probability_score": metrics["rank_probability_score"],
                        }
                    )
                if fold_progress is not None:
                    fold_progress.update(1)

    if not fold_rows:
        return {"windows_requested": int(windows), "folds": [], "summary": []}

    fold_frame = pd.DataFrame(fold_rows)
    summary = (
        fold_frame.groupby(["task", "model", "family", "feature_view"], as_index=False)
        .agg(
            folds=("fold", "count"),
            mean_accuracy=("accuracy", "mean"),
            mean_macro_f1=("macro_f1", "mean"),
            mean_log_loss=("log_loss", "mean"),
            mean_rank_probability_score=("rank_probability_score", "mean"),
        )
        .sort_values(["task", "mean_log_loss", "mean_rank_probability_score", "mean_accuracy"], ascending=[True, True, True, False])
    )
    return {
        "windows_requested": int(windows),
        "folds": fold_rows,
        "summary": summary.to_dict(orient="records"),
    }


def weakest_classes(task_result: dict[str, Any], labels: list[str]) -> list[tuple[str, float]]:
    metrics = task_result["test"]["per_class_metrics"]
    items = [(label, float(metrics[label]["f1_score"])) for label in labels]
    return sorted(items, key=lambda item: item[1])


def analysis_lines(results: dict[str, Any]) -> list[str]:
    lines: list[str] = []
    for task in TASKS:
        benchmarks = results[task.name]
        champion = benchmarks["catboost_full_hybrid"]["test"]
        prematch = benchmarks["catboost_prematch_only"]["test"]
        no_odds = benchmarks["catboost_odds_dropped"]["test"]
        market = benchmarks.get("market_prior", {}).get("test")

        lines.append(f"[{task.name}] Champion test accuracy={champion['accuracy']:.4f}, macro_f1={champion['macro_f1']:.4f}, log_loss={champion['log_loss']:.4f}.")
        lines.append(f"[{task.name}] Champion test RPS={champion['rank_probability_score']:.4f}.")
        if market is not None:
            lines.append(
                f"[{task.name}] Full-hybrid vs market delta: accuracy={champion['accuracy'] - market['accuracy']:+.4f}, "
                f"macro_f1={champion['macro_f1'] - market['macro_f1']:+.4f}, log_loss={market['log_loss'] - champion['log_loss']:+.4f}, "
                f"RPS={market['rank_probability_score'] - champion['rank_probability_score']:+.4f}."
            )
        lines.append(
            f"[{task.name}] Live vs prematch-only delta: accuracy={champion['accuracy'] - prematch['accuracy']:+.4f}, "
            f"macro_f1={champion['macro_f1'] - prematch['macro_f1']:+.4f}, log_loss={prematch['log_loss'] - champion['log_loss']:+.4f}, "
            f"RPS={prematch['rank_probability_score'] - champion['rank_probability_score']:+.4f}."
        )
        lines.append(
            f"[{task.name}] Odds ablation delta: accuracy={champion['accuracy'] - no_odds['accuracy']:+.4f}, "
            f"macro_f1={champion['macro_f1'] - no_odds['macro_f1']:+.4f}, log_loss={no_odds['log_loss'] - champion['log_loss']:+.4f}, "
            f"RPS={no_odds['rank_probability_score'] - champion['rank_probability_score']:+.4f}."
        )
        weakest = weakest_classes(benchmarks["catboost_full_hybrid"], task.labels)[:3]
        weakest_summary = ", ".join(f"{label} (f1={score:.3f})" for label, score in weakest)
        lines.append(f"[{task.name}] Weakest classes: {weakest_summary}.")

        importance = benchmarks["catboost_full_hybrid"].get("feature_importance", {}).get("by_group", {})
        top_groups = ", ".join(f"{group}={score:.2f}" for group, score in list(importance.items())[:4])
        if top_groups:
            lines.append(f"[{task.name}] Top feature groups: {top_groups}.")

        if champion["log_loss"] >= prematch["log_loss"]:
            lines.append(f"[{task.name}] Live features did not improve log loss; inspect richer sequence/momentum encodings near the cutoff.")
        if champion["log_loss"] >= no_odds["log_loss"]:
            lines.append(f"[{task.name}] Odds do not dominate this task in the current benchmark; historical and live features deserve deeper tuning.")
        else:
            lines.append(f"[{task.name}] Odds remain a major signal source; focus on non-odds features only if they add incremental calibration or draw/high-goal recall.")

    lines.append("Likely next feature work: richer card/subtype parsing, stronger possession/territory reconstruction from coordinates, and explicit draw/high-goal class balancing.")
    return lines


def write_artifacts(
    cutoff: int,
    train_frame: pd.DataFrame,
    test_frame: pd.DataFrame,
    results: dict[str, Any],
    backtest: dict[str, Any],
    artifacts_dir: str,
) -> dict[str, str]:
    target_dir = Path(artifacts_dir) / f"cutoff_{cutoff}"
    target_dir.mkdir(parents=True, exist_ok=True)
    progress_message(f"Writing predictor artifacts to {target_dir}")

    summary_rows: list[dict[str, Any]] = []
    source_rows: list[dict[str, Any]] = []
    with phase_progress_bar("Artifacts", 6, leave=False) as artifact_progress:
        for task_name, task_results in results.items():
            for model_name, model_result in task_results.items():
                summary_rows.append(
                    {
                        "task": task_name,
                        "model": model_name,
                        "family": model_result["family"],
                        "feature_view": model_result["feature_view"],
                        "feature_count": model_result["feature_count"],
                        "test_accuracy": model_result["test"]["accuracy"],
                        "test_macro_f1": model_result["test"]["macro_f1"],
                        "test_log_loss": model_result["test"]["log_loss"],
                        "test_rps": model_result["test"]["rank_probability_score"],
                        "validation_accuracy": model_result["validation"]["accuracy"],
                        "validation_macro_f1": model_result["validation"]["macro_f1"],
                        "validation_log_loss": model_result["validation"]["log_loss"],
                        "validation_rps": model_result["validation"]["rank_probability_score"],
                    }
                )
                for split_name in ("validation_by_source", "test_by_source"):
                    for source_name, metrics in model_result.get(split_name, {}).items():
                        source_rows.append(
                            {
                                "task": task_name,
                                "model": model_name,
                                "family": model_result["family"],
                                "feature_view": model_result["feature_view"],
                                "split": split_name.replace("_by_source", ""),
                                "event_source": source_name,
                                "rows": int(metrics.get("rows", 0)),
                                "accuracy": metrics["accuracy"],
                                "macro_f1": metrics["macro_f1"],
                                "log_loss": metrics["log_loss"],
                                "rank_probability_score": metrics["rank_probability_score"],
                            }
                        )

        summary_path = target_dir / "benchmark_summary.csv"
        pd.DataFrame(summary_rows).to_csv(summary_path, index=False)
        if artifact_progress is not None:
            artifact_progress.update(1)

        results_path = target_dir / "benchmark_results.json"
        with results_path.open("w", encoding="utf-8") as handle:
            json.dump(results, handle, ensure_ascii=True, indent=2)
        if artifact_progress is not None:
            artifact_progress.update(1)

        backtest_path = target_dir / "rolling_backtest.json"
        with backtest_path.open("w", encoding="utf-8") as handle:
            json.dump(backtest, handle, ensure_ascii=True, indent=2)
        if artifact_progress is not None:
            artifact_progress.update(1)

        backtest_summary_path = target_dir / "rolling_backtest_summary.csv"
        pd.DataFrame(backtest.get("summary", [])).to_csv(backtest_summary_path, index=False)
        if artifact_progress is not None:
            artifact_progress.update(1)

        source_summary_path = target_dir / "source_cohort_summary.csv"
        pd.DataFrame(source_rows).to_csv(source_summary_path, index=False)

        report_lines = analysis_lines(results)
        report_path = target_dir / "analysis_report.txt"
        report_path.write_text("\n".join(report_lines) + "\n", encoding="utf-8")
        if artifact_progress is not None:
            artifact_progress.update(1)

        metadata_path = target_dir / "run_metadata.json"
        metadata = {
            "cutoff": int(cutoff),
            "train_rows": int(len(train_frame)),
            "test_rows": int(len(test_frame)),
            "validation_fraction_within_train": VALIDATION_FRACTION_WITHIN_TRAIN,
            "rolling_backtest_windows": int(backtest.get("windows_requested", 0)),
            "tasks": [task.name for task in TASKS],
            "models": [spec.name for spec in MODEL_SPECS],
        }
        with metadata_path.open("w", encoding="utf-8") as handle:
            json.dump(metadata, handle, ensure_ascii=True, indent=2)
        if artifact_progress is not None:
            artifact_progress.update(1)

    return {
        "directory": str(target_dir),
        "summary_csv": str(summary_path),
        "results_json": str(results_path),
        "rolling_backtest_json": str(backtest_path),
        "rolling_backtest_summary_csv": str(backtest_summary_path),
        "source_cohort_summary_csv": str(source_summary_path),
        "analysis_report": str(report_path),
        "metadata_json": str(metadata_path),
    }


def run_benchmarks(
    *,
    cutoff: int,
    load_prepared: bool,
    force_rebuild: bool,
    max_rows: int | None,
    artifacts_dir: str,
    rolling_backtest_windows: int = DEFAULT_ROLLING_BACKTEST_WINDOWS,
    allow_fallback_events: bool = DEFAULT_ALLOW_FALLBACK_EVENTS,
) -> dict[str, Any]:
    with phase_progress_bar("Predictor run", 6, leave=True) as run_progress:
        train_raw, test_raw = load_prepared_splits(
            cutoff,
            load_prepared=load_prepared,
            force_rebuild=force_rebuild,
            max_rows=max_rows,
            allow_fallback_events=allow_fallback_events,
        )
        if run_progress is not None:
            run_progress.update(1)

        train_frame = build_modeling_frame(train_raw)
        if run_progress is not None:
            run_progress.update(1)

        test_frame = build_modeling_frame(test_raw)
        if run_progress is not None:
            run_progress.update(1)

        if train_frame.empty or test_frame.empty:
            raise RuntimeError("Prepared train/test frames are empty.")

        results: dict[str, Any] = {}
        for task in progress_iterable(TASKS, total=len(TASKS), desc="Benchmark tasks", unit="task"):
            results[task.name] = benchmark_task(train_frame, test_frame, task)
        if run_progress is not None:
            run_progress.update(1)

        backtest = rolling_backtest(train_frame, windows=rolling_backtest_windows)
        if run_progress is not None:
            run_progress.update(1)

        artifact_paths = write_artifacts(cutoff, train_frame, test_frame, results, backtest, artifacts_dir)
        if run_progress is not None:
            run_progress.update(1)
    return {
        "cutoff": int(cutoff),
        "train_rows": int(len(train_frame)),
        "test_rows": int(len(test_frame)),
        "artifacts": artifact_paths,
        "rolling_backtest": backtest,
        "results": results,
    }


def main() -> int:
    args = parse_args()
    outcome = run_benchmarks(
        cutoff=args.cutoff,
        load_prepared=args.load_prepared,
        force_rebuild=args.force_rebuild,
        max_rows=args.max_rows,
        artifacts_dir=args.artifacts_dir,
        rolling_backtest_windows=args.rolling_backtest_windows,
        allow_fallback_events=args.allow_fallback_events,
    )
    print("Completed predictor benchmarks.")
    print("Cutoff:", outcome["cutoff"])
    print("Train rows:", outcome["train_rows"])
    print("Test rows:", outcome["test_rows"])
    print("Artifacts:", outcome["artifacts"]["directory"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
