from __future__ import annotations

import argparse
import json
import os
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, log_loss
from torch import nn
from torch.utils.data import DataLoader, Dataset

from dataset import DEFAULT_CUTOFF, OUTPUT_DIR, VALIDATION_FRACTION_WITHIN_TRAIN
from transformer_dataset import DEFAULT_CUTOFF_STEP, DEFAULT_MAX_EVENTS, DEFAULT_MIN_CUTOFF, load_prepared_splits
from dataset import DEFAULT_ALLOW_FALLBACK_EVENTS

try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover - optional dependency
    tqdm = None

SEED = int(os.getenv("FOOTBALL_TRANSFORMER_SEED", "42"))
RESULT_LABELS = ["1", "X", "2"]
GOAL_BUCKET_LABELS = ["0", "1", "2", "3", "4+"]
DEFAULT_ARTIFACTS_DIR = os.path.join(OUTPUT_DIR, "transformer_artifacts")
DEFAULT_BATCH_SIZE = 64
DEFAULT_EPOCHS = 8
DEFAULT_MODEL_DIM = 128
DEFAULT_NUM_HEADS = 4
DEFAULT_NUM_LAYERS = 2
DEFAULT_DROPOUT = 0.1
DEFAULT_LEARNING_RATE = 1e-3
DEFAULT_WEIGHT_DECAY = 1e-4
DEFAULT_STATIC_EMBED_DIM = 16
DEFAULT_EVENT_EMBED_DIM = 16
DEFAULT_PLAYER_BUCKETS = 2048
DEFAULT_MAX_QUALIFIERS = 8
DEFAULT_AUXILIARY_WEIGHT = 0.25
TABULAR_SUMMARY_FILENAME = "benchmark_summary.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train/evaluate transformer-based live football predictors.")
    parser.add_argument("--cutoff", type=int, default=DEFAULT_CUTOFF)
    parser.add_argument("--artifacts-dir", default=DEFAULT_ARTIFACTS_DIR)
    parser.add_argument("--predictor-artifacts-dir", default=os.path.join(OUTPUT_DIR, "predictor_artifacts"))
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--force-rebuild", action="store_true")
    parser.add_argument("--load-prepared", action="store_true", default=True)
    parser.add_argument("--no-load-prepared", action="store_false", dest="load_prepared")
    parser.add_argument("--max-events", type=int, default=DEFAULT_MAX_EVENTS)
    parser.add_argument("--min-cutoff", type=int, default=DEFAULT_MIN_CUTOFF)
    parser.add_argument("--cutoff-step", type=int, default=DEFAULT_CUTOFF_STEP)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--model-dim", type=int, default=DEFAULT_MODEL_DIM)
    parser.add_argument("--num-heads", type=int, default=DEFAULT_NUM_HEADS)
    parser.add_argument("--num-layers", type=int, default=DEFAULT_NUM_LAYERS)
    parser.add_argument("--dropout", type=float, default=DEFAULT_DROPOUT)
    parser.add_argument("--learning-rate", type=float, default=DEFAULT_LEARNING_RATE)
    parser.add_argument("--weight-decay", type=float, default=DEFAULT_WEIGHT_DECAY)
    parser.add_argument("--auxiliary-weight", type=float, default=DEFAULT_AUXILIARY_WEIGHT)
    parser.add_argument("--compare-tabular", action="store_true", default=True)
    parser.add_argument("--no-compare-tabular", action="store_false", dest="compare_tabular")
    parser.add_argument("--allow-fallback-events", action="store_true", default=DEFAULT_ALLOW_FALLBACK_EVENTS)
    parser.add_argument("--no-allow-fallback-events", action="store_false", dest="allow_fallback_events")
    parser.add_argument("--device", default="auto")
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


def progress_bar(total: int, *, desc: str, unit: str, leave: bool = False) -> Any:
    if tqdm is None:
        return nullcontext(None)
    return tqdm(total=total, desc=desc, unit=unit, leave=leave)


def progress_message(message: str) -> None:
    if tqdm is not None:
        tqdm.write(message)
    else:
        print(message)


def label_to_index_map(labels: list[str]) -> dict[str, int]:
    return {label: index for index, label in enumerate(labels)}


def encode_target(values: pd.Series, labels: list[str]) -> np.ndarray:
    mapping = label_to_index_map(labels)
    encoded = values.astype(str).map(mapping)
    if encoded.isna().any():
        unknown = sorted(values[encoded.isna()].astype(str).unique().tolist())
        raise RuntimeError(f"Unknown labels encountered: {unknown}")
    return encoded.astype(int).to_numpy()


def rank_probability_score(y_true: np.ndarray, probabilities: np.ndarray) -> float:
    class_count = probabilities.shape[1]
    if class_count <= 1:
        return 0.0
    truth = np.eye(class_count)[y_true]
    cumulative_truth = np.cumsum(truth, axis=1)
    cumulative_probabilities = np.cumsum(probabilities, axis=1)
    squared_error = np.square(cumulative_probabilities - cumulative_truth).sum(axis=1)
    return float(np.mean(squared_error / (class_count - 1)))


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


def evaluate_predictions(y_true: np.ndarray, probabilities: np.ndarray, labels: list[str]) -> dict[str, Any]:
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
    }


def chronological_train_validation_split(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    ordered = frame.sort_values(["match_timestamp", "match_id"]).reset_index(drop=True)
    if len(ordered) < 2:
        raise RuntimeError("Need at least two training rows to create a validation split.")
    split_index = int(len(ordered) * (1.0 - VALIDATION_FRACTION_WITHIN_TRAIN))
    split_index = max(1, min(split_index, len(ordered) - 1))
    return ordered.iloc[:split_index].copy(), ordered.iloc[split_index:].copy()


def grouped_chronological_train_validation_split(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    ordered = frame.dropna(subset=["match_timestamp"]).copy()
    if ordered.empty:
        raise RuntimeError("No rows have usable timestamps for grouped train/validation splitting.")
    group_column = "base_match_id" if "base_match_id" in ordered.columns else "match_id"
    group_frame = (
        ordered.groupby(group_column, as_index=False)
        .agg(match_timestamp=("match_timestamp", "min"))
        .sort_values(["match_timestamp", group_column])
        .reset_index(drop=True)
    )
    if len(group_frame) < 2:
        raise RuntimeError("Need at least two match groups to create a grouped validation split.")
    split_index = int(len(group_frame) * (1.0 - VALIDATION_FRACTION_WITHIN_TRAIN))
    split_index = max(1, min(split_index, len(group_frame) - 1))
    train_ids = set(group_frame.iloc[:split_index][group_column].astype(str).tolist())
    validation_ids = set(group_frame.iloc[split_index:][group_column].astype(str).tolist())
    train_frame = ordered[ordered[group_column].astype(str).isin(train_ids)].sort_values(["match_timestamp", "match_id"]).reset_index(drop=True)
    validation_frame = ordered[ordered[group_column].astype(str).isin(validation_ids)].sort_values(["match_timestamp", "match_id"]).reset_index(drop=True)
    return train_frame, validation_frame


def flatten_static_features(frame: pd.DataFrame) -> pd.DataFrame:
    features = pd.json_normalize(frame["static_features"]).copy()
    features.index = frame.index
    return features


def build_vocab(values: list[str], *, min_frequency: int = 1) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in values:
        counts[value] = counts.get(value, 0) + 1
    vocab = {"__PAD__": 0, "__UNK__": 1}
    for value in sorted(counts):
        if counts[value] >= min_frequency and value not in vocab:
            vocab[value] = len(vocab)
    return vocab


def vocab_lookup(vocab: dict[str, int], value: str | None) -> int:
    key = value if value is not None else "__UNK__"
    return vocab.get(key, vocab["__UNK__"])


@dataclass
class StaticFeatureSpec:
    numeric_columns: list[str]
    categorical_columns: list[str]
    numeric_mean: dict[str, float]
    numeric_std: dict[str, float]
    categorical_vocab: dict[str, dict[str, int]]


def build_static_feature_spec(train_frame: pd.DataFrame) -> StaticFeatureSpec:
    static_frame = flatten_static_features(train_frame)
    numeric_columns: list[str] = []
    categorical_columns: list[str] = []
    numeric_mean: dict[str, float] = {}
    numeric_std: dict[str, float] = {}
    categorical_vocab: dict[str, dict[str, int]] = {}

    for column in sorted(static_frame.columns):
        series = static_frame[column]
        if pd.api.types.is_numeric_dtype(series):
            numeric_columns.append(column)
            numeric_mean[column] = float(pd.to_numeric(series, errors="coerce").mean())
            std = float(pd.to_numeric(series, errors="coerce").std())
            numeric_std[column] = std if std > 1e-8 else 1.0
        else:
            categorical_columns.append(column)
            values = [str(value) for value in series.fillna("__MISSING__").astype(str).tolist()]
            categorical_vocab[column] = build_vocab(values)

    return StaticFeatureSpec(
        numeric_columns=numeric_columns,
        categorical_columns=categorical_columns,
        numeric_mean=numeric_mean,
        numeric_std=numeric_std,
        categorical_vocab=categorical_vocab,
    )


@dataclass
class EventFeatureSpec:
    side_vocab: dict[str, int]
    type_vocab: dict[str, int]
    outcome_vocab: dict[str, int]
    subtype_vocab: dict[str, int]
    qualifier_vocab: dict[str, int]
    elapsed_vocab: dict[str, int]
    delta_vocab: dict[str, int]
    possession_phase_vocab: dict[str, int]
    possession_length_vocab: dict[str, int]
    next_type_vocab: dict[str, int]
    next_side_vocab: dict[str, int]
    next_time_vocab: dict[str, int]


def build_event_feature_spec(train_frame: pd.DataFrame) -> EventFeatureSpec:
    side_values: list[str] = []
    type_values: list[str] = []
    outcome_values: list[str] = []
    subtype_values: list[str] = []
    qualifier_values: list[str] = []
    elapsed_values: list[str] = []
    delta_values: list[str] = []
    possession_phase_values: list[str] = []
    possession_length_values: list[str] = []
    next_type_values: list[str] = []
    next_side_values: list[str] = []
    next_time_values: list[str] = []
    for sequence in train_frame["event_sequence"]:
        if not isinstance(sequence, list):
            continue
        for event in sequence:
            if not isinstance(event, dict):
                continue
            side_values.append(str(event.get("side") or "unknown"))
            type_values.append(str(event.get("event_type") or "unknown"))
            outcome_values.append(str(event.get("outcome") or "__missing__"))
            subtype_values.append(str(event.get("subtype") or "__missing__"))
            elapsed_values.append(str(event.get("elapsed_bucket") or "__missing__"))
            delta_values.append(str(event.get("delta_bucket") or "__missing__"))
            possession_phase_values.append(str(event.get("possession_phase") or "__missing__"))
            possession_length_values.append(str(event.get("possession_length_bucket") or "__missing__"))
            for qualifier in event.get("qualifiers", [])[:DEFAULT_MAX_QUALIFIERS]:
                qualifier_values.append(str(qualifier))
    for _, row in train_frame.iterrows():
        next_type_values.append(str(row.get("next_event_type") or "__missing__"))
        next_side_values.append(str(row.get("next_event_side") or "__missing__"))
        next_time_values.append(str(row.get("next_event_time_bucket") or "__missing__"))
    return EventFeatureSpec(
        side_vocab=build_vocab(side_values),
        type_vocab=build_vocab(type_values),
        outcome_vocab=build_vocab(outcome_values),
        subtype_vocab=build_vocab(subtype_values),
        qualifier_vocab=build_vocab(qualifier_values),
        elapsed_vocab=build_vocab(elapsed_values),
        delta_vocab=build_vocab(delta_values),
        possession_phase_vocab=build_vocab(possession_phase_values),
        possession_length_vocab=build_vocab(possession_length_values),
        next_type_vocab=build_vocab(next_type_values),
        next_side_vocab=build_vocab(next_side_values),
        next_time_vocab=build_vocab(next_time_values),
    )


class SequenceMatchDataset(Dataset):
    def __init__(
        self,
        frame: pd.DataFrame,
        *,
        static_spec: StaticFeatureSpec,
        event_spec: EventFeatureSpec,
        max_events: int,
    ) -> None:
        ordered = frame.sort_values(["match_timestamp", "match_id"]).reset_index(drop=True)
        self.records = ordered.to_dict(orient="records")
        self.static_spec = static_spec
        self.event_spec = event_spec
        self.max_events = max_events

    def __len__(self) -> int:
        return len(self.records)

    def encode_static(self, features: dict[str, Any]) -> tuple[torch.Tensor, torch.Tensor]:
        numeric_values: list[float] = []
        for column in self.static_spec.numeric_columns:
            raw = features.get(column)
            value = float(raw) if raw is not None and raw == raw else self.static_spec.numeric_mean[column]
            numeric_values.append((value - self.static_spec.numeric_mean[column]) / self.static_spec.numeric_std[column])

        categorical_values: list[int] = []
        for column in self.static_spec.categorical_columns:
            vocab = self.static_spec.categorical_vocab[column]
            raw = features.get(column)
            categorical_values.append(vocab_lookup(vocab, str(raw) if raw is not None else "__UNK__"))
        return torch.tensor(numeric_values, dtype=torch.float32), torch.tensor(categorical_values, dtype=torch.long)

    def encode_events(self, sequence: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        trimmed = sequence[-self.max_events :] if self.max_events > 0 else sequence
        side_ids: list[int] = []
        type_ids: list[int] = []
        outcome_ids: list[int] = []
        subtype_ids: list[int] = []
        elapsed_ids: list[int] = []
        delta_ids: list[int] = []
        possession_phase_ids: list[int] = []
        possession_length_ids: list[int] = []
        player_ids: list[int] = []
        qualifier_ids: list[list[int]] = []
        numeric_rows: list[list[float]] = []

        for event in trimmed:
            side_ids.append(vocab_lookup(self.event_spec.side_vocab, str(event.get("side") or "unknown")))
            type_ids.append(vocab_lookup(self.event_spec.type_vocab, str(event.get("event_type") or "unknown")))
            outcome_ids.append(vocab_lookup(self.event_spec.outcome_vocab, str(event.get("outcome") or "__missing__")))
            subtype_ids.append(vocab_lookup(self.event_spec.subtype_vocab, str(event.get("subtype") or "__missing__")))
            elapsed_ids.append(vocab_lookup(self.event_spec.elapsed_vocab, str(event.get("elapsed_bucket") or "__missing__")))
            delta_ids.append(vocab_lookup(self.event_spec.delta_vocab, str(event.get("delta_bucket") or "__missing__")))
            possession_phase_ids.append(vocab_lookup(self.event_spec.possession_phase_vocab, str(event.get("possession_phase") or "__missing__")))
            possession_length_ids.append(vocab_lookup(self.event_spec.possession_length_vocab, str(event.get("possession_length_bucket") or "__missing__")))
            player_ids.append(int(event.get("player_bucket") or 0))
            qualifier_bucket = [
                vocab_lookup(self.event_spec.qualifier_vocab, str(value))
                for value in event.get("qualifiers", [])[:DEFAULT_MAX_QUALIFIERS]
            ]
            qualifier_ids.append(qualifier_bucket)
            x_value = event.get("x")
            y_value = event.get("y")
            numeric_rows.append(
                [
                    float(event.get("minute_progress") or 0.0),
                    float(event.get("delta_seconds") or 0.0) / 300.0,
                    float(event.get("minute") or 0.0) / 90.0,
                    float(event.get("second") or 0.0) / 60.0,
                    float(x_value) / 100.0 if x_value is not None else -1.0,
                    float(y_value) / 100.0 if y_value is not None else -1.0,
                    float(event.get("is_goal") or 0.0),
                    float(event.get("is_shot") or 0.0),
                    float(event.get("is_card") or 0.0),
                    float(event.get("is_substitution") or 0.0),
                    float(event.get("possession_index") or 0.0) / 6.0,
                    float(event.get("possession_elapsed_seconds") or 0.0) / 60.0,
                    float(event.get("possession_progress") or 0.0),
                ]
            )

        return {
            "event_side": torch.tensor(side_ids, dtype=torch.long),
            "event_type": torch.tensor(type_ids, dtype=torch.long),
            "event_outcome": torch.tensor(outcome_ids, dtype=torch.long),
            "event_subtype": torch.tensor(subtype_ids, dtype=torch.long),
            "event_elapsed": torch.tensor(elapsed_ids, dtype=torch.long),
            "event_delta": torch.tensor(delta_ids, dtype=torch.long),
            "event_possession_phase": torch.tensor(possession_phase_ids, dtype=torch.long),
            "event_possession_length": torch.tensor(possession_length_ids, dtype=torch.long),
            "event_player": torch.tensor(player_ids, dtype=torch.long),
            "event_numeric": torch.tensor(numeric_rows, dtype=torch.float32),
            "event_qualifiers": qualifier_ids,
            "event_length": torch.tensor(len(trimmed), dtype=torch.long),
        }

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        static_features = record.get("static_features") if isinstance(record.get("static_features"), dict) else {}
        sequence = record.get("event_sequence") if isinstance(record.get("event_sequence"), list) else []
        static_numeric, static_categorical = self.encode_static(static_features)
        encoded_events = self.encode_events(sequence)
        return {
            "match_id": str(record["match_id"]),
            "base_match_id": str(record.get("base_match_id") or record["match_id"]),
            "static_numeric": static_numeric,
            "static_categorical": static_categorical,
            "target_1x2": int(label_to_index_map(RESULT_LABELS)[str(record["target_1x2"])]),
            "target_goal_bucket": int(label_to_index_map(GOAL_BUCKET_LABELS)[str(record["target_goal_bucket"])]),
            "next_event_exists": int(record.get("next_event_exists") or 0),
            "next_event_type": vocab_lookup(self.event_spec.next_type_vocab, str(record.get("next_event_type") or "__missing__")),
            "next_event_side": vocab_lookup(self.event_spec.next_side_vocab, str(record.get("next_event_side") or "__missing__")),
            "next_event_time_bucket": vocab_lookup(self.event_spec.next_time_vocab, str(record.get("next_event_time_bucket") or "__missing__")),
            **encoded_events,
        }


def collate_batch(batch: list[dict[str, Any]]) -> dict[str, Any]:
    batch_size = len(batch)
    max_seq_len = max(int(item["event_length"]) for item in batch) if batch else 0
    static_numeric = torch.stack([item["static_numeric"] for item in batch], dim=0)
    static_categorical = torch.stack([item["static_categorical"] for item in batch], dim=0)
    target_1x2 = torch.tensor([item["target_1x2"] for item in batch], dtype=torch.long)
    target_goal_bucket = torch.tensor([item["target_goal_bucket"] for item in batch], dtype=torch.long)

    def pad_long(key: str) -> torch.Tensor:
        result = torch.zeros(batch_size, max_seq_len, dtype=torch.long)
        for row_index, item in enumerate(batch):
            length = int(item["event_length"])
            if length > 0:
                result[row_index, :length] = item[key]
        return result

    def pad_float(key: str, width: int) -> torch.Tensor:
        result = torch.zeros(batch_size, max_seq_len, width, dtype=torch.float32)
        for row_index, item in enumerate(batch):
            length = int(item["event_length"])
            if length > 0:
                result[row_index, :length] = item[key]
        return result

    qualifier_tensor = torch.zeros(batch_size, max_seq_len, DEFAULT_MAX_QUALIFIERS, dtype=torch.long)
    qualifier_mask = torch.zeros(batch_size, max_seq_len, DEFAULT_MAX_QUALIFIERS, dtype=torch.bool)
    attention_mask = torch.zeros(batch_size, max_seq_len, dtype=torch.bool)
    for row_index, item in enumerate(batch):
        length = int(item["event_length"])
        attention_mask[row_index, :length] = True
        for event_index, qualifier_ids in enumerate(item["event_qualifiers"][:length]):
            trimmed = qualifier_ids[:DEFAULT_MAX_QUALIFIERS]
            if trimmed:
                qualifier_tensor[row_index, event_index, : len(trimmed)] = torch.tensor(trimmed, dtype=torch.long)
                qualifier_mask[row_index, event_index, : len(trimmed)] = True

    return {
        "match_ids": [item["match_id"] for item in batch],
        "base_match_ids": [item["base_match_id"] for item in batch],
        "static_numeric": static_numeric,
        "static_categorical": static_categorical,
        "event_side": pad_long("event_side"),
        "event_type": pad_long("event_type"),
        "event_outcome": pad_long("event_outcome"),
        "event_subtype": pad_long("event_subtype"),
        "event_elapsed": pad_long("event_elapsed"),
        "event_delta": pad_long("event_delta"),
        "event_possession_phase": pad_long("event_possession_phase"),
        "event_possession_length": pad_long("event_possession_length"),
        "event_player": pad_long("event_player"),
        "event_numeric": pad_float("event_numeric", 13),
        "event_qualifiers": qualifier_tensor,
        "event_qualifier_mask": qualifier_mask,
        "attention_mask": attention_mask,
        "target_1x2": target_1x2,
        "target_goal_bucket": target_goal_bucket,
        "next_event_exists": torch.tensor([item["next_event_exists"] for item in batch], dtype=torch.bool),
        "next_event_type": torch.tensor([item["next_event_type"] for item in batch], dtype=torch.long),
        "next_event_side": torch.tensor([item["next_event_side"] for item in batch], dtype=torch.long),
        "next_event_time_bucket": torch.tensor([item["next_event_time_bucket"] for item in batch], dtype=torch.long),
    }


class FootballTransformer(nn.Module):
    def __init__(
        self,
        *,
        static_spec: StaticFeatureSpec,
        event_spec: EventFeatureSpec,
        max_events: int,
        model_dim: int,
        num_heads: int,
        num_layers: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.model_dim = model_dim
        self.static_numeric_dim = len(static_spec.numeric_columns)
        self.static_categorical_embeddings = nn.ModuleList(
            [
                nn.Embedding(len(static_spec.categorical_vocab[column]), DEFAULT_STATIC_EMBED_DIM)
                for column in static_spec.categorical_columns
            ]
        )
        static_input_dim = self.static_numeric_dim + len(self.static_categorical_embeddings) * DEFAULT_STATIC_EMBED_DIM
        self.static_encoder = nn.Sequential(
            nn.Linear(max(static_input_dim, 1), model_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(model_dim, model_dim),
        )

        self.side_embedding = nn.Embedding(len(event_spec.side_vocab), DEFAULT_EVENT_EMBED_DIM)
        self.type_embedding = nn.Embedding(len(event_spec.type_vocab), DEFAULT_EVENT_EMBED_DIM * 2)
        self.outcome_embedding = nn.Embedding(len(event_spec.outcome_vocab), DEFAULT_EVENT_EMBED_DIM)
        self.subtype_embedding = nn.Embedding(len(event_spec.subtype_vocab), DEFAULT_EVENT_EMBED_DIM)
        self.elapsed_embedding = nn.Embedding(len(event_spec.elapsed_vocab), DEFAULT_EVENT_EMBED_DIM)
        self.delta_embedding = nn.Embedding(len(event_spec.delta_vocab), DEFAULT_EVENT_EMBED_DIM)
        self.possession_phase_embedding = nn.Embedding(len(event_spec.possession_phase_vocab), DEFAULT_EVENT_EMBED_DIM)
        self.possession_length_embedding = nn.Embedding(len(event_spec.possession_length_vocab), DEFAULT_EVENT_EMBED_DIM)
        self.player_embedding = nn.Embedding(DEFAULT_PLAYER_BUCKETS, DEFAULT_EVENT_EMBED_DIM)
        self.qualifier_embedding = nn.Embedding(len(event_spec.qualifier_vocab), DEFAULT_EVENT_EMBED_DIM)
        self.event_numeric_projection = nn.Sequential(
            nn.Linear(13, model_dim // 2),
            nn.ReLU(),
            nn.Linear(model_dim // 2, model_dim // 2),
        )
        token_input_dim = (
            DEFAULT_EVENT_EMBED_DIM
            + DEFAULT_EVENT_EMBED_DIM * 2
            + DEFAULT_EVENT_EMBED_DIM
            + DEFAULT_EVENT_EMBED_DIM
            + DEFAULT_EVENT_EMBED_DIM
            + DEFAULT_EVENT_EMBED_DIM
            + DEFAULT_EVENT_EMBED_DIM
            + DEFAULT_EVENT_EMBED_DIM
            + DEFAULT_EVENT_EMBED_DIM
            + DEFAULT_EVENT_EMBED_DIM
            + model_dim // 2
        )
        self.event_projection = nn.Sequential(
            nn.Linear(token_input_dim, model_dim),
            nn.LayerNorm(model_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.cls_token = nn.Parameter(torch.zeros(1, 1, model_dim))
        self.position_embedding = nn.Embedding(max(max_events, 1) + 1, model_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=num_heads,
            dim_feedforward=model_dim * 4,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fusion = nn.Sequential(
            nn.Linear(model_dim * 2, model_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(model_dim * 2, model_dim),
            nn.ReLU(),
        )
        self.head_1x2 = nn.Linear(model_dim, len(RESULT_LABELS))
        self.head_goal_bucket = nn.Linear(model_dim, len(GOAL_BUCKET_LABELS))
        self.head_next_event_type = nn.Linear(model_dim, len(event_spec.next_type_vocab))
        self.head_next_event_side = nn.Linear(model_dim, len(event_spec.next_side_vocab))
        self.head_next_event_time_bucket = nn.Linear(model_dim, len(event_spec.next_time_vocab))

    def encode_static(self, static_numeric: torch.Tensor, static_categorical: torch.Tensor) -> torch.Tensor:
        parts = [static_numeric]
        for index, embedding in enumerate(self.static_categorical_embeddings):
            parts.append(embedding(static_categorical[:, index]))
        merged = torch.cat(parts, dim=1) if parts else torch.zeros(static_numeric.size(0), 0, device=static_numeric.device)
        if merged.shape[1] == 0:
            merged = torch.zeros(static_numeric.size(0), 1, device=static_numeric.device, dtype=static_numeric.dtype)
        return self.static_encoder(merged)

    def encode_events(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        qualifier_emb = self.qualifier_embedding(batch["event_qualifiers"])
        qualifier_mask = batch["event_qualifier_mask"].unsqueeze(-1)
        qualifier_sum = (qualifier_emb * qualifier_mask).sum(dim=2)
        qualifier_count = qualifier_mask.sum(dim=2).clamp(min=1)
        qualifier_mean = qualifier_sum / qualifier_count

        token_parts = [
            self.side_embedding(batch["event_side"]),
            self.type_embedding(batch["event_type"]),
            self.outcome_embedding(batch["event_outcome"]),
            self.subtype_embedding(batch["event_subtype"]),
            self.elapsed_embedding(batch["event_elapsed"]),
            self.delta_embedding(batch["event_delta"]),
            self.possession_phase_embedding(batch["event_possession_phase"]),
            self.possession_length_embedding(batch["event_possession_length"]),
            self.player_embedding(batch["event_player"]),
            qualifier_mean,
            self.event_numeric_projection(batch["event_numeric"]),
        ]
        token_repr = torch.cat(token_parts, dim=-1)
        token_repr = self.event_projection(token_repr)

        batch_size, seq_len, _ = token_repr.shape
        cls_token = self.cls_token.expand(batch_size, -1, -1)
        hidden = torch.cat([cls_token, token_repr], dim=1)
        positions = torch.arange(seq_len + 1, device=hidden.device).unsqueeze(0).expand(batch_size, -1)
        hidden = hidden + self.position_embedding(positions)
        padding_mask = torch.cat(
            [
                torch.ones(batch_size, 1, device=hidden.device, dtype=torch.bool),
                batch["attention_mask"],
            ],
            dim=1,
        )
        encoded = self.transformer(hidden, src_key_padding_mask=~padding_mask)
        return encoded[:, 0, :]

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        static_repr = self.encode_static(batch["static_numeric"], batch["static_categorical"])
        event_repr = self.encode_events(batch)
        fused = self.fusion(torch.cat([static_repr, event_repr], dim=1))
        return {
            "1x2": self.head_1x2(fused),
            "goal_bucket": self.head_goal_bucket(fused),
            "next_event_type": self.head_next_event_type(fused),
            "next_event_side": self.head_next_event_side(fused),
            "next_event_time_bucket": self.head_next_event_time_bucket(fused),
        }


def move_batch_to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    moved: dict[str, Any] = {}
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            moved[key] = value.to(device)
        else:
            moved[key] = value
    return moved


def probabilities_from_logits(logits: torch.Tensor) -> np.ndarray:
    return torch.softmax(logits, dim=1).detach().cpu().numpy()


def masked_cross_entropy(
    logits: torch.Tensor,
    targets: torch.Tensor,
    mask: torch.Tensor,
    criterion: nn.CrossEntropyLoss,
) -> torch.Tensor:
    valid_mask = mask.to(dtype=torch.bool)
    if not torch.any(valid_mask):
        return logits.sum() * 0.0
    return criterion(logits[valid_mask], targets[valid_mask])


def compute_training_losses(
    outputs: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    criterion: nn.CrossEntropyLoss,
    auxiliary_weight: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    loss_1x2 = criterion(outputs["1x2"], batch["target_1x2"])
    loss_goal = criterion(outputs["goal_bucket"], batch["target_goal_bucket"])
    next_mask = batch["next_event_exists"]
    aux_type = masked_cross_entropy(outputs["next_event_type"], batch["next_event_type"], next_mask, criterion)
    aux_side = masked_cross_entropy(outputs["next_event_side"], batch["next_event_side"], next_mask, criterion)
    aux_time = masked_cross_entropy(
        outputs["next_event_time_bucket"],
        batch["next_event_time_bucket"],
        next_mask,
        criterion,
    )
    auxiliary_loss = aux_type + aux_side + aux_time
    total_loss = loss_1x2 + loss_goal + (float(auxiliary_weight) * auxiliary_loss)
    metrics = {
        "loss_1x2": float(loss_1x2.detach().item()),
        "loss_goal_bucket": float(loss_goal.detach().item()),
        "aux_loss_next_event_type": float(aux_type.detach().item()),
        "aux_loss_next_event_side": float(aux_side.detach().item()),
        "aux_loss_next_event_time_bucket": float(aux_time.detach().item()),
        "aux_loss_total": float(auxiliary_loss.detach().item()),
        "total_loss": float(total_loss.detach().item()),
    }
    return total_loss, metrics


def train_one_epoch(
    model: FootballTransformer,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    auxiliary_weight: float,
) -> dict[str, float]:
    model.train()
    criterion = nn.CrossEntropyLoss()
    sums = {
        "loss_1x2": 0.0,
        "loss_goal_bucket": 0.0,
        "aux_loss_next_event_type": 0.0,
        "aux_loss_next_event_side": 0.0,
        "aux_loss_next_event_time_bucket": 0.0,
        "aux_loss_total": 0.0,
        "total_loss": 0.0,
    }
    with progress_bar(len(loader), desc="Training", unit="batch", leave=False) as bar:
        for batch in loader:
            batch = move_batch_to_device(batch, device)
            optimizer.zero_grad(set_to_none=True)
            outputs = model(batch)
            loss, batch_metrics = compute_training_losses(outputs, batch, criterion, auxiliary_weight)
            loss.backward()
            optimizer.step()
            for key, value in batch_metrics.items():
                sums[key] += value
            if bar is not None:
                bar.update(1)
                bar.set_postfix(loss=f"{batch_metrics['total_loss']:.4f}", aux=f"{batch_metrics['aux_loss_total']:.4f}")
    denominator = max(len(loader), 1)
    return {key: value / denominator for key, value in sums.items()}


def evaluate_model(
    model: FootballTransformer,
    loader: DataLoader,
    device: torch.device,
) -> dict[str, Any]:
    model.eval()
    all_1x2_probs: list[np.ndarray] = []
    all_goal_probs: list[np.ndarray] = []
    all_1x2_targets: list[np.ndarray] = []
    all_goal_targets: list[np.ndarray] = []
    criterion = nn.CrossEntropyLoss()
    loss_sums = {
        "loss_1x2": 0.0,
        "loss_goal_bucket": 0.0,
        "aux_loss_next_event_type": 0.0,
        "aux_loss_next_event_side": 0.0,
        "aux_loss_next_event_time_bucket": 0.0,
        "aux_loss_total": 0.0,
        "total_loss": 0.0,
    }
    with torch.no_grad(), progress_bar(len(loader), desc="Evaluating", unit="batch", leave=False) as bar:
        for batch in loader:
            batch = move_batch_to_device(batch, device)
            outputs = model(batch)
            _, batch_losses = compute_training_losses(outputs, batch, criterion, auxiliary_weight=1.0)
            for key, value in batch_losses.items():
                loss_sums[key] += value
            all_1x2_probs.append(probabilities_from_logits(outputs["1x2"]))
            all_goal_probs.append(probabilities_from_logits(outputs["goal_bucket"]))
            all_1x2_targets.append(batch["target_1x2"].detach().cpu().numpy())
            all_goal_targets.append(batch["target_goal_bucket"].detach().cpu().numpy())
            if bar is not None:
                bar.update(1)
    y_1x2 = np.concatenate(all_1x2_targets, axis=0)
    y_goal = np.concatenate(all_goal_targets, axis=0)
    probs_1x2 = np.concatenate(all_1x2_probs, axis=0)
    probs_goal = np.concatenate(all_goal_probs, axis=0)
    batch_count = max(len(loader), 1)
    return {
        "1x2": evaluate_predictions(y_1x2, probs_1x2, RESULT_LABELS),
        "goal_bucket": evaluate_predictions(y_goal, probs_goal, GOAL_BUCKET_LABELS),
        "losses": {key: value / batch_count for key, value in loss_sums.items()},
    }


def evaluate_by_source(
    model: FootballTransformer,
    frame: pd.DataFrame,
    *,
    static_spec: StaticFeatureSpec,
    event_spec: EventFeatureSpec,
    batch_size: int,
    max_events: int,
    device: torch.device,
) -> dict[str, Any]:
    if "event_source" not in frame.columns:
        return {}
    cohorts: dict[str, Any] = {}
    for source_name in sorted(frame["event_source"].fillna("unknown").astype(str).unique().tolist()):
        source_frame = frame[frame["event_source"].fillna("unknown").astype(str) == source_name].copy()
        if source_frame.empty:
            continue
        dataset = SequenceMatchDataset(source_frame, static_spec=static_spec, event_spec=event_spec, max_events=max_events)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_batch)
        cohorts[source_name] = {
            "rows": int(len(source_frame)),
            **evaluate_model(model, loader, device),
        }
    return cohorts


def make_dataloaders(
    train_frame: pd.DataFrame,
    validation_frame: pd.DataFrame,
    test_frame: pd.DataFrame,
    *,
    batch_size: int,
    max_events: int,
) -> tuple[DataLoader, DataLoader, DataLoader, StaticFeatureSpec, EventFeatureSpec]:
    static_spec = build_static_feature_spec(train_frame)
    event_spec = build_event_feature_spec(train_frame)
    train_dataset = SequenceMatchDataset(train_frame, static_spec=static_spec, event_spec=event_spec, max_events=max_events)
    validation_dataset = SequenceMatchDataset(validation_frame, static_spec=static_spec, event_spec=event_spec, max_events=max_events)
    test_dataset = SequenceMatchDataset(test_frame, static_spec=static_spec, event_spec=event_spec, max_events=max_events)
    return (
        DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_batch),
        DataLoader(validation_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_batch),
        DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_batch),
        static_spec,
        event_spec,
    )


def select_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def write_artifacts(
    *,
    cutoff: int,
    train_rows: int,
    validation_rows: int,
    test_rows: int,
    history: list[dict[str, Any]],
    validation_metrics: dict[str, Any],
    test_metrics: dict[str, Any],
    artifacts_dir: str,
    model: FootballTransformer,
    static_spec: StaticFeatureSpec,
    event_spec: EventFeatureSpec,
    args: argparse.Namespace,
    tabular_comparison: dict[str, Any] | None = None,
    validation_by_source: dict[str, Any] | None = None,
    test_by_source: dict[str, Any] | None = None,
) -> dict[str, str]:
    target_dir = Path(artifacts_dir) / f"cutoff_{cutoff}"
    target_dir.mkdir(parents=True, exist_ok=True)
    progress_message(f"Writing transformer artifacts to {target_dir}")

    history_path = target_dir / "training_history.json"
    history_path.write_text(json.dumps(history, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")

    metrics = {
        "validation": validation_metrics,
        "test": test_metrics,
        "validation_by_source": validation_by_source or {},
        "test_by_source": test_by_source or {},
    }
    metrics_path = target_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")

    summary_rows = []
    for task_name in ("1x2", "goal_bucket"):
        task_metrics = test_metrics[task_name]
        summary_rows.append(
            {
                "task": task_name,
                "accuracy": task_metrics["accuracy"],
                "macro_f1": task_metrics["macro_f1"],
                "log_loss": task_metrics["log_loss"],
                "rank_probability_score": task_metrics["rank_probability_score"],
            }
        )
    summary_path = target_dir / "summary.csv"
    pd.DataFrame(summary_rows).to_csv(summary_path, index=False)

    metadata = {
        "cutoff": int(cutoff),
        "train_rows": int(train_rows),
        "validation_rows": int(validation_rows),
        "test_rows": int(test_rows),
        "max_events": int(args.max_events),
        "min_cutoff": int(args.min_cutoff),
        "cutoff_step": int(args.cutoff_step),
        "epochs": int(args.epochs),
        "batch_size": int(args.batch_size),
        "model_dim": int(args.model_dim),
        "num_heads": int(args.num_heads),
        "num_layers": int(args.num_layers),
        "dropout": float(args.dropout),
        "learning_rate": float(args.learning_rate),
        "weight_decay": float(args.weight_decay),
        "auxiliary_weight": float(args.auxiliary_weight),
        "device": str(args.device),
        "allow_fallback_events": bool(args.allow_fallback_events),
        "static_numeric_columns": static_spec.numeric_columns,
        "static_categorical_columns": static_spec.categorical_columns,
        "event_vocab_sizes": {
            "side": len(event_spec.side_vocab),
            "event_type": len(event_spec.type_vocab),
            "outcome": len(event_spec.outcome_vocab),
            "subtype": len(event_spec.subtype_vocab),
            "qualifier": len(event_spec.qualifier_vocab),
            "elapsed_bucket": len(event_spec.elapsed_vocab),
            "delta_bucket": len(event_spec.delta_vocab),
            "possession_phase": len(event_spec.possession_phase_vocab),
            "possession_length_bucket": len(event_spec.possession_length_vocab),
            "next_event_type": len(event_spec.next_type_vocab),
            "next_event_side": len(event_spec.next_side_vocab),
            "next_event_time_bucket": len(event_spec.next_time_vocab),
        },
    }
    metadata_path = target_dir / "run_metadata.json"
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")

    checkpoint_path = target_dir / "model.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "metadata": metadata,
            "static_spec": static_spec,
            "event_spec": event_spec,
            "result_labels": RESULT_LABELS,
            "goal_bucket_labels": GOAL_BUCKET_LABELS,
        },
        checkpoint_path,
    )
    artifacts = {
        "directory": str(target_dir),
        "history_json": str(history_path),
        "metrics_json": str(metrics_path),
        "summary_csv": str(summary_path),
        "metadata_json": str(metadata_path),
        "checkpoint": str(checkpoint_path),
    }
    if tabular_comparison is not None:
        comparison_json_path = target_dir / "tabular_comparison.json"
        comparison_json_path.write_text(json.dumps(tabular_comparison, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")
        comparison_rows = tabular_comparison.get("rows", [])
        comparison_csv_path = target_dir / "tabular_comparison.csv"
        pd.DataFrame(comparison_rows).to_csv(comparison_csv_path, index=False)
        artifacts["tabular_comparison_json"] = str(comparison_json_path)
        artifacts["tabular_comparison_csv"] = str(comparison_csv_path)
    return artifacts


def predictor_summary_path(artifacts_dir: str, cutoff: int) -> Path:
    return Path(artifacts_dir) / f"cutoff_{cutoff}" / TABULAR_SUMMARY_FILENAME


def ensure_tabular_summary(args: argparse.Namespace) -> pd.DataFrame:
    summary_path = predictor_summary_path(args.predictor_artifacts_dir, args.cutoff)
    should_rebuild = not summary_path.exists()
    if summary_path.exists():
        try:
            cached_summary = pd.read_csv(summary_path)
        except Exception:
            should_rebuild = True
        else:
            if "model" not in cached_summary.columns or "catboost_live_dropped" in cached_summary["model"].astype(str).tolist():
                should_rebuild = True
    if should_rebuild:
        progress_message(f"Building tabular predictor benchmarks for cutoff {args.cutoff}")
        import predictor

        predictor.run_benchmarks(
            cutoff=args.cutoff,
            load_prepared=args.load_prepared,
            force_rebuild=args.force_rebuild,
            max_rows=args.max_rows,
            artifacts_dir=args.predictor_artifacts_dir,
            allow_fallback_events=args.allow_fallback_events,
        )
    if not summary_path.exists():
        raise RuntimeError(f"Tabular benchmark summary was not found at {summary_path}")
    return pd.read_csv(summary_path)


def compare_with_tabular_benchmarks(
    *,
    cutoff: int,
    transformer_test_metrics: dict[str, Any],
    args: argparse.Namespace,
) -> dict[str, Any]:
    summary_frame = ensure_tabular_summary(args)
    comparison_rows: list[dict[str, Any]] = []
    for task_name in ("1x2", "goal_bucket"):
        task_frame = summary_frame[summary_frame["task"] == task_name].copy()
        if task_frame.empty:
            continue
        best_row = task_frame.sort_values(["test_log_loss", "test_rps", "test_macro_f1"], ascending=[True, True, False]).iloc[0]
        transformer_metrics = transformer_test_metrics[task_name]
        comparison_rows.append(
            {
                "task": task_name,
                "transformer_model": "event_static_transformer",
                "transformer_accuracy": transformer_metrics["accuracy"],
                "transformer_macro_f1": transformer_metrics["macro_f1"],
                "transformer_log_loss": transformer_metrics["log_loss"],
                "transformer_rps": transformer_metrics["rank_probability_score"],
                "best_tabular_model": str(best_row["model"]),
                "best_tabular_family": str(best_row["family"]),
                "best_tabular_feature_view": str(best_row["feature_view"]),
                "best_tabular_accuracy": float(best_row["test_accuracy"]),
                "best_tabular_macro_f1": float(best_row["test_macro_f1"]),
                "best_tabular_log_loss": float(best_row["test_log_loss"]),
                "best_tabular_rps": float(best_row["test_rps"]),
                "delta_accuracy": transformer_metrics["accuracy"] - float(best_row["test_accuracy"]),
                "delta_macro_f1": transformer_metrics["macro_f1"] - float(best_row["test_macro_f1"]),
                "delta_log_loss": transformer_metrics["log_loss"] - float(best_row["test_log_loss"]),
                "delta_rps": transformer_metrics["rank_probability_score"] - float(best_row["test_rps"]),
            }
        )
    return {
        "cutoff": int(cutoff),
        "rows": comparison_rows,
        "source_summary_csv": str(predictor_summary_path(args.predictor_artifacts_dir, args.cutoff)),
    }


def run_experiment(args: argparse.Namespace) -> dict[str, Any]:
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    with progress_bar(6, desc="Transformer run", unit="phase", leave=True) as run_bar:
        train_frame, test_frame = load_prepared_splits(
            args.cutoff,
            load_prepared=args.load_prepared,
            force_rebuild=args.force_rebuild,
            max_rows=args.max_rows,
            max_events=args.max_events,
            min_cutoff=args.min_cutoff,
            cutoff_step=args.cutoff_step,
            allow_fallback_events=args.allow_fallback_events,
        )
        if run_bar is not None:
            run_bar.update(1)

        dev_train_frame, validation_frame = grouped_chronological_train_validation_split(train_frame)
        if run_bar is not None:
            run_bar.update(1)

        train_loader, validation_loader, test_loader, static_spec, event_spec = make_dataloaders(
            dev_train_frame,
            validation_frame,
            test_frame,
            batch_size=args.batch_size,
            max_events=args.max_events,
        )
        if run_bar is not None:
            run_bar.update(1)

        device = select_device(args.device)
        model = FootballTransformer(
            static_spec=static_spec,
            event_spec=event_spec,
            max_events=args.max_events,
            model_dim=args.model_dim,
            num_heads=args.num_heads,
            num_layers=args.num_layers,
            dropout=args.dropout,
        ).to(device)
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=args.learning_rate,
            weight_decay=args.weight_decay,
        )

        history: list[dict[str, Any]] = []
        best_state: dict[str, Any] | None = None
        best_validation_loss = float("inf")
        with progress_bar(args.epochs, desc="Epochs", unit="epoch", leave=False) as epoch_bar:
            for epoch in range(1, args.epochs + 1):
                train_metrics = train_one_epoch(
                    model,
                    train_loader,
                    optimizer,
                    device,
                    auxiliary_weight=args.auxiliary_weight,
                )
                validation_metrics = evaluate_model(model, validation_loader, device)
                validation_loss = (
                    validation_metrics["1x2"]["log_loss"] + validation_metrics["goal_bucket"]["log_loss"]
                ) / 2.0
                history.append(
                    {
                        "epoch": epoch,
                        **train_metrics,
                        "validation_1x2_log_loss": validation_metrics["1x2"]["log_loss"],
                        "validation_goal_bucket_log_loss": validation_metrics["goal_bucket"]["log_loss"],
                        "validation_joint_log_loss": validation_loss,
                        "validation_total_loss": validation_metrics["losses"]["total_loss"],
                        "validation_aux_loss_total": validation_metrics["losses"]["aux_loss_total"],
                    }
                )
                if validation_loss < best_validation_loss:
                    best_validation_loss = validation_loss
                    best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
                if epoch_bar is not None:
                    epoch_bar.update(1)
                    epoch_bar.set_postfix(
                        val_loss=f"{validation_loss:.4f}",
                        train=f"{train_metrics['total_loss']:.4f}",
                        aux=f"{train_metrics['aux_loss_total']:.4f}",
                    )
        if best_state is not None:
            model.load_state_dict(best_state)
        if run_bar is not None:
            run_bar.update(1)

        validation_metrics = evaluate_model(model, validation_loader, device)
        test_metrics = evaluate_model(model, test_loader, device)
        validation_by_source = evaluate_by_source(
            model,
            validation_frame,
            static_spec=static_spec,
            event_spec=event_spec,
            batch_size=args.batch_size,
            max_events=args.max_events,
            device=device,
        )
        test_by_source = evaluate_by_source(
            model,
            test_frame,
            static_spec=static_spec,
            event_spec=event_spec,
            batch_size=args.batch_size,
            max_events=args.max_events,
            device=device,
        )
        if run_bar is not None:
            run_bar.update(1)

        tabular_comparison = None
        if args.compare_tabular:
            tabular_comparison = compare_with_tabular_benchmarks(
                cutoff=args.cutoff,
                transformer_test_metrics=test_metrics,
                args=args,
            )
        if run_bar is not None:
            run_bar.update(1)

        artifact_paths = write_artifacts(
            cutoff=args.cutoff,
            train_rows=len(dev_train_frame),
            validation_rows=len(validation_frame),
            test_rows=len(test_frame),
            history=history,
            validation_metrics=validation_metrics,
            test_metrics=test_metrics,
            artifacts_dir=args.artifacts_dir,
            model=model,
            static_spec=static_spec,
            event_spec=event_spec,
            args=args,
            tabular_comparison=tabular_comparison,
            validation_by_source=validation_by_source,
            test_by_source=test_by_source,
        )
    return {
        "cutoff": int(args.cutoff),
        "train_rows": int(len(dev_train_frame)),
        "validation_rows": int(len(validation_frame)),
        "test_rows": int(len(test_frame)),
        "artifacts": artifact_paths,
        "validation": validation_metrics,
        "test": test_metrics,
        "validation_by_source": validation_by_source,
        "test_by_source": test_by_source,
        "tabular_comparison": tabular_comparison,
    }


def main() -> int:
    args = parse_args()
    outcome = run_experiment(args)
    print("Completed transformer benchmarks.")
    print("Cutoff:", outcome["cutoff"])
    print("Train rows:", outcome["train_rows"])
    print("Validation rows:", outcome["validation_rows"])
    print("Test rows:", outcome["test_rows"])
    print("Artifacts:", outcome["artifacts"]["directory"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
