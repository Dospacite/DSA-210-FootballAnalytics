from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import Counter
from datetime import UTC, datetime
from typing import Any

import pandas as pd

from dataset import (
    DEFAULT_CUTOFF,
    DEFAULT_ALLOW_FALLBACK_EVENTS,
    MONGO_BATCH_SIZE,
    OUTPUT_DIR,
    PREP_VERSION,
    TEST_FRACTION,
    VALIDATION_FRACTION_WITHIN_TRAIN,
    build_common_features,
    build_goal_targets,
    build_history_feature_map,
    build_history_rows,
    build_player_lineup_feature_map,
    ensure_training_query_index,
    extract_match_timestamp,
    final_result_label,
    get_f24_attrs,
    goal_count_bucket_label,
    iter_mongo_documents,
    chunked_iterable,
    load_mongo_collection,
    normalize_cutoff,
    parse_final_score,
    primary_event_source,
    preparation_projection,
    prioritized_live_events,
    progress_iterable,
    save_prepared_frame,
    write_manifest,
)

TRANSFORMER_PREP_VERSION = f"{PREP_VERSION}-transformer-v2"
DEFAULT_MAX_EVENTS = 256
DEFAULT_MIN_EVENT_COUNT = 1
DEFAULT_OUTPUT_DIR = OUTPUT_DIR
DEFAULT_MIN_CUTOFF = 5
DEFAULT_CUTOFF_STEP = 5
DEFAULT_DOCUMENT_BATCH_SIZE = MONGO_BATCH_SIZE

SHOT_EVENT_TYPES = {"13", "14", "15", "16", "shot", "goal", "missed-shot", "saved-shot"}
CARD_EVENT_TYPES = {"17", "18", "19", "card", "yellow-card", "red-card"}
RESTART_KEYWORDS = ("corner", "goal kick", "goal-kick", "throw", "free kick", "free-kick", "kick off", "kick-off")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build transformer-ready Mackolik event-sequence datasets.")
    parser.add_argument("--cutoff", type=int, default=DEFAULT_CUTOFF)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--force-rebuild", action="store_true")
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--max-events", type=int, default=DEFAULT_MAX_EVENTS)
    parser.add_argument("--min-event-count", type=int, default=DEFAULT_MIN_EVENT_COUNT)
    parser.add_argument("--min-cutoff", type=int, default=DEFAULT_MIN_CUTOFF)
    parser.add_argument("--cutoff-step", type=int, default=DEFAULT_CUTOFF_STEP)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_DOCUMENT_BATCH_SIZE)
    parser.add_argument("--allow-fallback-events", action="store_true", default=DEFAULT_ALLOW_FALLBACK_EVENTS)
    parser.add_argument("--no-allow-fallback-events", action="store_false", dest="allow_fallback_events")
    return parser.parse_args()


def artifact_paths(cutoff: int, output_dir: str = DEFAULT_OUTPUT_DIR) -> dict[str, str]:
    normalized = normalize_cutoff(cutoff)
    return {
        "full": os.path.join(output_dir, f"transformer_{normalized}_full.pkl"),
        "train": os.path.join(output_dir, f"transformer_{normalized}_train.pkl"),
        "test": os.path.join(output_dir, f"transformer_{normalized}_test.pkl"),
        "manifest": os.path.join(output_dir, f"transformer_{normalized}_manifest.json"),
    }


def prepared_artifacts_exist(
    cutoff: int,
    output_dir: str = DEFAULT_OUTPUT_DIR,
    *,
    min_cutoff: int = DEFAULT_MIN_CUTOFF,
    cutoff_step: int = DEFAULT_CUTOFF_STEP,
    allow_fallback_events: bool = DEFAULT_ALLOW_FALLBACK_EVENTS,
) -> bool:
    paths = artifact_paths(cutoff, output_dir)
    if not all(os.path.exists(paths[key]) for key in ("full", "train", "test", "manifest")):
        return False
    try:
        with open(paths["manifest"], "r", encoding="utf-8") as handle:
            manifest = json.load(handle)
    except (OSError, ValueError, TypeError):
        return False
    return (
        manifest.get("prep_version") == TRANSFORMER_PREP_VERSION
        and manifest.get("cutoff") == normalize_cutoff(cutoff)
        and manifest.get("min_cutoff") == int(min_cutoff)
        and manifest.get("cutoff_step") == int(cutoff_step)
        and bool(manifest.get("allow_fallback_events", DEFAULT_ALLOW_FALLBACK_EVENTS)) == bool(allow_fallback_events)
    )


def load_prepared_frame(path: str) -> pd.DataFrame:
    return pd.read_pickle(path)


def stable_hash_bucket(value: str | None, buckets: int) -> int:
    if not value:
        return 0
    digest = hashlib.md5(value.encode("utf-8")).hexdigest()
    return (int(digest[:12], 16) % max(buckets - 1, 1)) + 1


def normalize_positive_int(value: int, fallback: int) -> int:
    return int(value) if isinstance(value, int) and value > 0 else int(fallback)


def normalize_batch_size(value: int, fallback: int = DEFAULT_DOCUMENT_BATCH_SIZE) -> int:
    return normalize_positive_int(value, fallback)


def enumerate_cutoffs(final_cutoff: int, *, min_cutoff: int, cutoff_step: int) -> list[int]:
    normalized_final = normalize_cutoff(final_cutoff)
    normalized_min = min(normalize_positive_int(min_cutoff, DEFAULT_MIN_CUTOFF), normalized_final)
    normalized_step = normalize_positive_int(cutoff_step, DEFAULT_CUTOFF_STEP)
    cutoffs = list(range(normalized_min, normalized_final + 1, normalized_step))
    if normalized_final not in cutoffs:
        cutoffs.append(normalized_final)
    return sorted({cutoff for cutoff in cutoffs if cutoff > 0})


def ordered_live_events(document: dict[str, Any]) -> tuple[list[dict[str, Any]], str]:
    raw_events, source = prioritized_live_events(document)
    ordered = sorted(
        (
            event
            for event in raw_events
            if isinstance(event, dict) and isinstance(event.get("total_seconds"), int) and int(event["total_seconds"]) >= 0
        ),
        key=lambda event: (
            int(event.get("total_seconds", 0)),
            str(event.get("side") or "unknown"),
            str(event.get("event_type") or "unknown"),
        ),
    )
    return ordered, source


def event_sequence_from_document(
    document: dict[str, Any],
    *,
    cutoff: int,
    max_events: int,
) -> tuple[list[dict[str, Any]], str]:
    raw_events, source = ordered_live_events(document)
    cutoff_seconds = cutoff * 60
    ordered_events = [event for event in raw_events if 0 <= int(event["total_seconds"]) <= cutoff_seconds]
    if max_events > 0 and len(ordered_events) > max_events:
        ordered_events = ordered_events[-max_events:]

    sequence: list[dict[str, Any]] = []
    previous_seconds = 0
    for index, event in enumerate(ordered_events):
        total_seconds = int(event.get("total_seconds", 0))
        details = event.get("details") if isinstance(event.get("details"), dict) else {}
        event_type = str(event.get("event_type") or "unknown")
        side = str(event.get("side") or "unknown")
        player_id = str(event.get("player_id")) if event.get("player_id") is not None else None
        player_name = str(event.get("player_name")) if event.get("player_name") else None
        subtype = str(details.get("subtype") or details.get("sub_type") or "").strip().lower()
        qualifiers = [str(value) for value in event.get("qualifiers", [])][:8]
        x_value = event.get("x")
        y_value = event.get("y")
        is_goal = event_type in {"16", "goal"} or str(details.get("is_goal")).lower() == "true"
        is_shot = event_type in SHOT_EVENT_TYPES
        is_card = event_type in CARD_EVENT_TYPES or ("card" in subtype)
        is_substitution = event_type == "substitution" or "sub" in subtype
        delta_seconds = max(total_seconds - previous_seconds, 0) if index > 0 else total_seconds
        previous_seconds = total_seconds
        sequence.append(
            {
                "event_index": index,
                "minute": int(event.get("minute") or 0),
                "second": int(event.get("second") or 0),
                "total_seconds": total_seconds,
                "delta_seconds": delta_seconds,
                "elapsed_bucket": elapsed_time_bucket(total_seconds),
                "delta_bucket": next_time_bucket(delta_seconds),
                "minute_progress": float(total_seconds) / 5400.0,
                "side": side,
                "event_type": event_type,
                "outcome": str(event.get("outcome")) if event.get("outcome") is not None else "__missing__",
                "player_token": player_id or player_name or "__missing__",
                "player_bucket": stable_hash_bucket(player_id or player_name, 2048),
                "x": float(x_value) if x_value is not None else None,
                "y": float(y_value) if y_value is not None else None,
                "qualifiers": qualifiers,
                "subtype": subtype or "__missing__",
                "is_goal": int(is_goal),
                "is_shot": int(is_shot),
                "is_card": int(is_card),
                "is_substitution": int(is_substitution),
            }
        )
    previous_event: dict[str, Any] | None = None
    current_possession_id = -1
    possession_start_index = 0
    possession_start_second = 0
    for event in sequence:
        boundary = possession_boundary(previous_event, event, int(event.get("delta_seconds") or 0))
        if boundary:
            current_possession_id += 1
            possession_start_index = int(event.get("event_index") or 0)
            possession_start_second = int(event.get("total_seconds") or 0)
        event["possession_id"] = current_possession_id
        event["possession_team"] = str(event.get("side") or "unknown")
        event["possession_index"] = int(event.get("event_index") or 0) - possession_start_index
        event["possession_elapsed_seconds"] = max(int(event.get("total_seconds") or 0) - possession_start_second, 0)
        previous_event = event
    possession_lengths = Counter(int(event.get("possession_id") or 0) for event in sequence)
    for event in sequence:
        current_length = possession_lengths.get(int(event.get("possession_id") or 0), 1)
        current_index = int(event.get("possession_index") or 0)
        event["possession_length_bucket"] = possession_length_bucket(current_length)
        event["possession_phase"] = possession_phase(current_index, current_length)
        event["possession_progress"] = float(current_index + 1) / max(float(current_length), 1.0)
    return sequence, source


def next_time_bucket(delta_seconds: int) -> str:
    if delta_seconds <= 30:
        return "0_30"
    if delta_seconds <= 60:
        return "31_60"
    if delta_seconds <= 120:
        return "61_120"
    if delta_seconds <= 300:
        return "121_300"
    return "301_plus"


def elapsed_time_bucket(total_seconds: int) -> str:
    minute = max(total_seconds // 60, 0)
    bucket_start = (minute // 5) * 5
    bucket_end = min(bucket_start + 4, 95)
    return f"{bucket_start}_{bucket_end}"


def possession_length_bucket(length: int) -> str:
    if length <= 1:
        return "1"
    if length <= 3:
        return "2_3"
    if length <= 6:
        return "4_6"
    return "7_plus"


def possession_phase(index_within_possession: int, possession_length: int) -> str:
    if possession_length <= 1:
        return "single"
    if index_within_possession == 0:
        return "start"
    if index_within_possession == possession_length - 1:
        return "end"
    return "middle"


def subtype_contains_restart(subtype: str) -> bool:
    lowered = subtype.strip().lower()
    return any(keyword in lowered for keyword in RESTART_KEYWORDS)


def possession_boundary(previous_event: dict[str, Any] | None, current_event: dict[str, Any], delta_seconds: int) -> bool:
    if previous_event is None:
        return True
    current_side = str(current_event.get("side") or "unknown")
    previous_side = str(previous_event.get("side") or "unknown")
    if current_side != previous_side:
        return True
    if delta_seconds >= 45:
        return True
    previous_subtype = str(previous_event.get("subtype") or "__missing__")
    current_subtype = str(current_event.get("subtype") or "__missing__")
    if subtype_contains_restart(previous_subtype) or subtype_contains_restart(current_subtype):
        return True
    if int(previous_event.get("is_goal") or 0) or int(previous_event.get("is_card") or 0) or int(previous_event.get("is_substitution") or 0):
        return True
    return False


def next_event_targets(document: dict[str, Any], *, cutoff: int) -> dict[str, Any]:
    ordered_events, _ = ordered_live_events(document)
    cutoff_seconds = cutoff * 60
    next_event = next((event for event in ordered_events if int(event.get("total_seconds", 0)) > cutoff_seconds), None)
    if next_event is None:
        return {
            "next_event_exists": 0,
            "next_event_type": "__missing__",
            "next_event_side": "__missing__",
            "next_event_time_bucket": "__missing__",
        }
    next_total_seconds = int(next_event.get("total_seconds", cutoff_seconds))
    return {
        "next_event_exists": 1,
        "next_event_type": str(next_event.get("event_type") or "unknown"),
        "next_event_side": str(next_event.get("side") or "unknown"),
        "next_event_time_bucket": next_time_bucket(max(next_total_seconds - cutoff_seconds, 0)),
    }


def build_sequence_snapshot_features(
    sequence: list[dict[str, Any]],
    *,
    source: str,
    cutoff: int,
) -> dict[str, float]:
    home_goals = 0
    away_goals = 0
    home_events = 0
    away_events = 0
    unique_players: set[str] = set()
    last_event_second = 0
    possession_ids_by_side: dict[str, set[int]] = {"home": set(), "away": set()}
    possession_event_lengths: Counter[int] = Counter(int(event.get("possession_id") or 0) for event in sequence)
    for event in sequence:
        side = str(event.get("side") or "unknown")
        if side == "home":
            home_events += 1
        elif side == "away":
            away_events += 1
        token = str(event.get("player_token") or "")
        if token and token != "__missing__":
            unique_players.add(token)
        last_event_second = max(last_event_second, int(event.get("total_seconds") or 0))
        if side in possession_ids_by_side:
            possession_ids_by_side[side].add(int(event.get("possession_id") or 0))
        if int(event.get("is_goal") or 0):
            if side == "home":
                home_goals += 1
            elif side == "away":
                away_goals += 1
    features = {
        "sequence_event_count": float(len(sequence)),
        "sequence_home_event_count": float(home_events),
        "sequence_away_event_count": float(away_events),
        "sequence_event_count_diff": float(home_events - away_events),
        "sequence_home_score": float(home_goals),
        "sequence_away_score": float(away_goals),
        "sequence_score_diff": float(home_goals - away_goals),
        "sequence_score_total": float(home_goals + away_goals),
        "sequence_unique_players": float(len(unique_players)),
        "sequence_last_event_second": float(last_event_second),
        "sequence_idle_seconds": float(max(cutoff * 60 - last_event_second, 0)),
        "sequence_source_is_f24": 1.0 if source == "f24" else 0.0,
        "sequence_source_is_match_data": 1.0 if source == "match_data" else 0.0,
    }
    for side in ("home", "away"):
        features[f"sequence_{side}_possession_count"] = float(len(possession_ids_by_side[side]))
        possession_lengths = [possession_event_lengths[possession_id] for possession_id in possession_ids_by_side[side]]
        features[f"sequence_{side}_possession_events_mean"] = float(sum(possession_lengths) / len(possession_lengths)) if possession_lengths else 0.0
        features[f"sequence_{side}_possession_events_max"] = float(max(possession_lengths)) if possession_lengths else 0.0
    features["sequence_possession_count_diff"] = (
        features["sequence_home_possession_count"] - features["sequence_away_possession_count"]
    )
    return features


def transformer_row_from_document(
    document: dict[str, Any],
    *,
    cutoff: int,
    max_events: int,
    min_event_count: int,
    history_features_by_match_id: dict[str, dict[str, float]],
    squad_features_by_match_id: dict[str, dict[str, float]],
    allow_fallback_events: bool = DEFAULT_ALLOW_FALLBACK_EVENTS,
) -> dict[str, Any] | None:
    final_score = parse_final_score(document)
    match_id = document.get("match_id")
    if final_score is None or match_id is None:
        return None
    match_id = str(match_id)
    history_features = history_features_by_match_id.get(match_id)
    if history_features is None:
        return None
    match_timestamp = extract_match_timestamp(document, get_f24_attrs(document))
    if match_timestamp is None:
        return None

    sequence, source = event_sequence_from_document(document, cutoff=cutoff, max_events=max_events)
    if not allow_fallback_events and not primary_event_source(source):
        return None
    if len(sequence) < max(min_event_count, 0):
        return None

    static_features = build_common_features(document, match_timestamp)
    static_features.update(history_features)
    static_features.update(squad_features_by_match_id.get(match_id, {}))
    static_features.update(build_sequence_snapshot_features(sequence, source=source, cutoff=cutoff))
    static_features["sequence_cutoff_minute"] = float(cutoff)
    static_features["sequence_cutoff_progress"] = float(cutoff) / 90.0

    home_goals, away_goals = final_score
    total_goals = home_goals + away_goals
    next_targets = next_event_targets(document, cutoff=cutoff)
    return {
        "match_id": f"{match_id}@{cutoff}",
        "base_match_id": match_id,
        "match_timestamp": match_timestamp,
        "cutoff": int(cutoff),
        "event_source": source,
        "event_count": int(len(sequence)),
        "static_features": static_features,
        "event_sequence": sequence,
        **next_targets,
        "target_1x2": final_result_label(home_goals, away_goals),
        "target_total_goals": int(total_goals),
        "target_goal_bucket": goal_count_bucket_label(total_goals),
        **build_goal_targets(home_goals, away_goals),
    }


def build_rows(
    documents: list[dict[str, Any]],
    *,
    cutoffs: list[int],
    max_events: int,
    min_event_count: int,
    history_features_by_match_id: dict[str, dict[str, float]],
    squad_features_by_match_id: dict[str, dict[str, float]],
    allow_fallback_events: bool = DEFAULT_ALLOW_FALLBACK_EVENTS,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for document in documents:
        for cutoff in cutoffs:
            row = transformer_row_from_document(
                document,
                cutoff=cutoff,
                max_events=max_events,
                min_event_count=min_event_count,
                history_features_by_match_id=history_features_by_match_id,
                squad_features_by_match_id=squad_features_by_match_id,
                allow_fallback_events=allow_fallback_events,
            )
            if row is not None:
                rows.append(row)
    return rows


def build_frames_in_batches(
    documents: Any,
    *,
    batch_size: int,
    cutoffs: list[int],
    max_events: int,
    min_event_count: int,
    history_features_by_match_id: dict[str, dict[str, float]],
    squad_features_by_match_id: dict[str, dict[str, float]],
    allow_fallback_events: bool = DEFAULT_ALLOW_FALLBACK_EVENTS,
) -> list[pd.DataFrame]:
    frames: list[pd.DataFrame] = []
    normalized_batch_size = normalize_batch_size(batch_size)
    for document_batch in chunked_iterable(documents, normalized_batch_size):
        batch_rows = build_rows(
            document_batch,
            cutoffs=cutoffs,
            max_events=max_events,
            min_event_count=min_event_count,
            history_features_by_match_id=history_features_by_match_id,
            squad_features_by_match_id=squad_features_by_match_id,
            allow_fallback_events=allow_fallback_events,
        )
        if batch_rows:
            frames.append(pd.DataFrame(batch_rows))
    return frames


def grouped_chronological_split(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    usable = frame.dropna(subset=["match_timestamp"]).copy()
    if usable.empty:
        raise RuntimeError("No rows have usable timestamps for grouped chronological splitting.")
    group_frame = (
        usable.groupby("base_match_id", as_index=False)
        .agg(match_timestamp=("match_timestamp", "min"))
        .sort_values(["match_timestamp", "base_match_id"])
        .reset_index(drop=True)
    )
    split_index = int(len(group_frame) * (1.0 - TEST_FRACTION))
    split_index = max(1, min(split_index, len(group_frame) - 1))
    train_ids = set(group_frame.iloc[:split_index]["base_match_id"].astype(str).tolist())
    test_ids = set(group_frame.iloc[split_index:]["base_match_id"].astype(str).tolist())
    train_frame = usable[usable["base_match_id"].astype(str).isin(train_ids)].sort_values(["match_timestamp", "match_id"]).reset_index(drop=True)
    test_frame = usable[usable["base_match_id"].astype(str).isin(test_ids)].sort_values(["match_timestamp", "match_id"]).reset_index(drop=True)
    return train_frame, test_frame


def save_cutoff_dataset(
    cutoff: int,
    frame: pd.DataFrame,
    *,
    output_dir: str = DEFAULT_OUTPUT_DIR,
    min_cutoff: int = DEFAULT_MIN_CUTOFF,
    cutoff_step: int = DEFAULT_CUTOFF_STEP,
    allow_fallback_events: bool = DEFAULT_ALLOW_FALLBACK_EVENTS,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    paths = artifact_paths(cutoff, output_dir)
    frame = frame.copy()
    frame["match_timestamp"] = pd.to_datetime(frame["match_timestamp"], utc=True, errors="coerce")
    train_frame, test_frame = grouped_chronological_split(frame)
    save_prepared_frame(frame, paths["full"])
    save_prepared_frame(train_frame, paths["train"])
    save_prepared_frame(test_frame, paths["test"])
    write_manifest(
        paths["manifest"],
        {
            "prep_version": TRANSFORMER_PREP_VERSION,
            "built_at": datetime.now(UTC).isoformat(),
            "cutoff": int(cutoff),
            "min_cutoff": int(min_cutoff),
            "cutoff_step": int(cutoff_step),
            "cutoffs": enumerate_cutoffs(cutoff, min_cutoff=min_cutoff, cutoff_step=cutoff_step),
            "rows": int(len(frame)),
            "train_rows": int(len(train_frame)),
            "test_rows": int(len(test_frame)),
            "validation_fraction_within_train": VALIDATION_FRACTION_WITHIN_TRAIN,
            "schema_family": "unified_legacy_mackolik_mac_plus",
            "event_source_priority": ["opta_feeds.raw.f24", "match_data.events"],
            "event_source_policy": "f24_only" if not allow_fallback_events else "f24_then_match_data",
            "allow_fallback_events": bool(allow_fallback_events),
            "representation": "static_context_plus_multi_cutoff_event_prefix_sequence",
            "targets": [
                "target_1x2",
                "target_total_goals",
                "target_goal_bucket",
                "target_over_1_5",
                "target_over_2_5",
                "target_over_3_5",
                "target_over_4_5",
            ],
        },
    )
    return train_frame, test_frame


def prepare_dataset(
    cutoff: int = DEFAULT_CUTOFF,
    *,
    output_dir: str = DEFAULT_OUTPUT_DIR,
    force_rebuild: bool = False,
    max_rows: int | None = None,
    max_events: int = DEFAULT_MAX_EVENTS,
    min_event_count: int = DEFAULT_MIN_EVENT_COUNT,
    min_cutoff: int = DEFAULT_MIN_CUTOFF,
    cutoff_step: int = DEFAULT_CUTOFF_STEP,
    batch_size: int = DEFAULT_DOCUMENT_BATCH_SIZE,
    allow_fallback_events: bool = DEFAULT_ALLOW_FALLBACK_EVENTS,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    normalized_cutoff = normalize_cutoff(cutoff)
    if (
        max_rows is None
        and not force_rebuild
        and prepared_artifacts_exist(
            normalized_cutoff,
            output_dir,
            min_cutoff=min_cutoff,
            cutoff_step=cutoff_step,
            allow_fallback_events=allow_fallback_events,
        )
    ):
        return load_prepared_dataset(
            normalized_cutoff,
            output_dir=output_dir,
            load_prepared=True,
            force_rebuild=False,
            max_rows=None,
            max_events=max_events,
            min_event_count=min_event_count,
            min_cutoff=min_cutoff,
            cutoff_step=cutoff_step,
            allow_fallback_events=allow_fallback_events,
        )

    client, collection = load_mongo_collection()
    try:
        ensure_training_query_index(collection)
        history_rows = build_history_rows(collection, max_rows=max_rows)
        history_features_by_match_id = build_history_feature_map(history_rows)
        squad_features_by_match_id = build_player_lineup_feature_map(collection, max_rows=max_rows)
        frames = build_frames_in_batches(
            progress_iterable(
                iter_mongo_documents(collection, projection=preparation_projection(), max_rows=max_rows),
                desc="Building transformer rows",
                unit="match",
            ),
            batch_size=batch_size,
            cutoffs=enumerate_cutoffs(normalized_cutoff, min_cutoff=min_cutoff, cutoff_step=cutoff_step),
            max_events=max_events,
            min_event_count=min_event_count,
            history_features_by_match_id=history_features_by_match_id,
            squad_features_by_match_id=squad_features_by_match_id,
            allow_fallback_events=allow_fallback_events,
        )
    finally:
        client.close()

    if not frames:
        raise RuntimeError(f"No eligible transformer rows found for cutoff {normalized_cutoff}.")
    frame = pd.concat(frames, ignore_index=True)
    train_frame, test_frame = save_cutoff_dataset(
        normalized_cutoff,
        frame,
        output_dir=output_dir,
        min_cutoff=min_cutoff,
        cutoff_step=cutoff_step,
        allow_fallback_events=allow_fallback_events,
    )
    return frame, train_frame, test_frame


def load_prepared_dataset(
    cutoff: int = DEFAULT_CUTOFF,
    *,
    output_dir: str = DEFAULT_OUTPUT_DIR,
    load_prepared: bool = True,
    force_rebuild: bool = False,
    max_rows: int | None = None,
    max_events: int = DEFAULT_MAX_EVENTS,
    min_event_count: int = DEFAULT_MIN_EVENT_COUNT,
    min_cutoff: int = DEFAULT_MIN_CUTOFF,
    cutoff_step: int = DEFAULT_CUTOFF_STEP,
    allow_fallback_events: bool = DEFAULT_ALLOW_FALLBACK_EVENTS,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    normalized_cutoff = normalize_cutoff(cutoff)
    should_rebuild = (
        force_rebuild
        or not load_prepared
        or not prepared_artifacts_exist(
            normalized_cutoff,
            output_dir,
            min_cutoff=min_cutoff,
            cutoff_step=cutoff_step,
            allow_fallback_events=allow_fallback_events,
        )
    )
    if should_rebuild:
        return prepare_dataset(
            normalized_cutoff,
            output_dir=output_dir,
            force_rebuild=True,
            max_rows=max_rows,
            max_events=max_events,
            min_event_count=min_event_count,
            min_cutoff=min_cutoff,
            cutoff_step=cutoff_step,
            batch_size=batch_size,
            allow_fallback_events=allow_fallback_events,
        )
    paths = artifact_paths(normalized_cutoff, output_dir)
    return (
        load_prepared_frame(paths["full"]),
        load_prepared_frame(paths["train"]),
        load_prepared_frame(paths["test"]),
    )


def load_prepared_splits(
    cutoff: int = DEFAULT_CUTOFF,
    *,
    output_dir: str = DEFAULT_OUTPUT_DIR,
    load_prepared: bool = True,
    force_rebuild: bool = False,
    max_rows: int | None = None,
    max_events: int = DEFAULT_MAX_EVENTS,
    min_event_count: int = DEFAULT_MIN_EVENT_COUNT,
    min_cutoff: int = DEFAULT_MIN_CUTOFF,
    cutoff_step: int = DEFAULT_CUTOFF_STEP,
    batch_size: int = DEFAULT_DOCUMENT_BATCH_SIZE,
    allow_fallback_events: bool = DEFAULT_ALLOW_FALLBACK_EVENTS,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    normalized_cutoff = normalize_cutoff(cutoff)
    should_rebuild = (
        force_rebuild
        or not load_prepared
        or not prepared_artifacts_exist(
            normalized_cutoff,
            output_dir,
            min_cutoff=min_cutoff,
            cutoff_step=cutoff_step,
            allow_fallback_events=allow_fallback_events,
        )
    )
    if should_rebuild:
        _, train_frame, test_frame = prepare_dataset(
            normalized_cutoff,
            output_dir=output_dir,
            force_rebuild=True,
            max_rows=max_rows,
            max_events=max_events,
            min_event_count=min_event_count,
            min_cutoff=min_cutoff,
            cutoff_step=cutoff_step,
            batch_size=batch_size,
            allow_fallback_events=allow_fallback_events,
        )
        return train_frame, test_frame
    paths = artifact_paths(normalized_cutoff, output_dir)
    return load_prepared_frame(paths["train"]), load_prepared_frame(paths["test"])


def main() -> int:
    args = parse_args()
    full_frame, train_frame, test_frame = prepare_dataset(
        cutoff=args.cutoff,
        output_dir=args.output_dir,
        force_rebuild=args.force_rebuild,
        max_rows=args.max_rows,
        max_events=args.max_events,
        min_event_count=args.min_event_count,
        min_cutoff=args.min_cutoff,
        cutoff_step=args.cutoff_step,
        batch_size=args.batch_size,
        allow_fallback_events=args.allow_fallback_events,
    )
    print("Prepared transformer dataset.")
    print("Cutoff:", normalize_cutoff(args.cutoff))
    print("Rows:", len(full_frame))
    print("Train rows:", len(train_frame))
    print("Test rows:", len(test_frame))
    print("Artifacts:", artifact_paths(args.cutoff, args.output_dir)["full"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
