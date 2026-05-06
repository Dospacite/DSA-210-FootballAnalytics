from __future__ import annotations

import argparse
import concurrent.futures
import json
import math
import os
import re
from collections import Counter, deque
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, Iterable

if TYPE_CHECKING:
    import pandas as pd
    from pymongo.collection import Collection
    from pymongo.mongo_client import MongoClient

try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover - optional dependency
    tqdm = None

DATABASE_NAME = "football_analytics"
COLLECTION_NAME = "mackolik"
OUTPUT_DIR = "prepared_datasets"
TEST_FRACTION = 0.20
VALIDATION_FRACTION_WITHIN_TRAIN = 0.15
PREP_VERSION = "2026-05-04-unified-legacy-v3"
MONGO_BATCH_SIZE = 256
ROLLING_WINDOW = 5
DEFAULT_CUTOFF = 15
DEFAULT_MAX_WORKERS = min(8, max(os.cpu_count() or 1, 1))
GOAL_TARGET_THRESHOLDS = (1.5, 2.5, 3.5, 4.5)
GOAL_COUNT_CAP = 4
DEFAULT_ALLOW_FALLBACK_EVENTS = False
SHOT_EVENT_TYPES = {"13", "14", "15", "16", "goal", "shot", "penalty-missed"}
CARD_EVENT_TYPES = {"17", "18", "19", "yellow-card", "red-card", "card", "booking"}
DECAY_FACTOR = 0.80
PI_RATING_LAMBDA = 0.035
PI_RATING_GAMMA = 0.7
PI_RATING_BASE = 10.0
PI_RATING_C = 3.0
PLAYER_HISTORY_DECAY = 0.85
HISTORY_METRICS = (
    "ppm",
    "win_share",
    "draw_share",
    "loss_share",
    "goals_for_pg",
    "goals_against_pg",
    "goal_diff_pg",
    "clean_sheet_rate",
    "scoring_rate",
    "over_1_5_rate",
    "over_2_5_rate",
    "over_3_5_rate",
    "over_4_5_rate",
    "win_streak",
    "draw_streak",
    "loss_streak",
)
TURKISH_MONTHS = {
    "Ocak": 1,
    "Subat": 2,
    "Şubat": 2,
    "Mart": 3,
    "Nisan": 4,
    "Mayıs": 5,
    "Haziran": 6,
    "Temmuz": 7,
    "Ağustos": 8,
    "Eylül": 9,
    "Ekim": 10,
    "Kasım": 11,
    "Aralık": 12,
}
FORM_CLASS_TO_POINTS = {
    "mac-form-res-0": 0.0,
    "mac-form-res-1": 1.0,
    "mac-form-res-3": 3.0,
}


@dataclass(slots=True)
class HistoryRow:
    match_id: str
    match_timestamp: datetime
    home_team_id: str | None
    away_team_id: str | None
    home_goals: int
    away_goals: int
    matchday: int | None = None


@dataclass(slots=True)
class TeamAppearance:
    is_home: bool
    goals_for: int
    goals_against: int
    points: float
    total_goals: int
    scored: bool
    clean_sheet: bool
    goal_margin: int


@dataclass(slots=True)
class RollingAppearanceStats:
    entries: deque[TeamAppearance] | None = None
    wins: int = 0
    draws: int = 0
    losses: int = 0
    points_total: float = 0.0
    goals_for_total: int = 0
    goals_against_total: int = 0
    clean_sheet_total: int = 0
    scored_total: int = 0
    over_goal_totals: dict[str, int] | None = None

    def __post_init__(self) -> None:
        if self.entries is None:
            self.entries = deque(maxlen=ROLLING_WINDOW)
        if self.over_goal_totals is None:
            self.over_goal_totals = {
                threshold_key(threshold): 0
                for threshold in GOAL_TARGET_THRESHOLDS
            }

    def _apply(self, entry: TeamAppearance, *, sign: int) -> None:
        if entry.points == 3.0:
            self.wins += sign
        elif entry.points == 1.0:
            self.draws += sign
        else:
            self.losses += sign
        self.points_total += sign * entry.points
        self.goals_for_total += sign * entry.goals_for
        self.goals_against_total += sign * entry.goals_against
        self.clean_sheet_total += sign * int(entry.clean_sheet)
        self.scored_total += sign * int(entry.scored)
        assert self.over_goal_totals is not None
        for threshold in GOAL_TARGET_THRESHOLDS:
            if entry.total_goals > threshold:
                self.over_goal_totals[threshold_key(threshold)] += sign

    def append(self, entry: TeamAppearance) -> None:
        assert self.entries is not None
        if len(self.entries) == self.entries.maxlen:
            expired = self.entries.popleft()
            self._apply(expired, sign=-1)
        self.entries.append(entry)
        self._apply(entry, sign=1)

    def snapshot(self, prefix: str) -> dict[str, float]:
        if not self.entries:
            return empty_history_snapshot(prefix)

        count = float(len(self.entries))
        win_streak = 0
        draw_streak = 0
        loss_streak = 0
        for entry in reversed(self.entries):
            if entry.points == 3.0 and draw_streak == 0 and loss_streak == 0:
                win_streak += 1
            elif entry.points == 1.0 and win_streak == 0 and loss_streak == 0:
                draw_streak += 1
            elif entry.points == 0.0 and win_streak == 0 and draw_streak == 0:
                loss_streak += 1
            else:
                break

        assert self.over_goal_totals is not None
        return {
            f"{prefix}_count": count,
            f"{prefix}_ppm": self.points_total / count,
            f"{prefix}_win_share": self.wins / count,
            f"{prefix}_draw_share": self.draws / count,
            f"{prefix}_loss_share": self.losses / count,
            f"{prefix}_goals_for_pg": self.goals_for_total / count,
            f"{prefix}_goals_against_pg": self.goals_against_total / count,
            f"{prefix}_goal_diff_pg": (self.goals_for_total - self.goals_against_total) / count,
            f"{prefix}_clean_sheet_rate": self.clean_sheet_total / count,
            f"{prefix}_scoring_rate": self.scored_total / count,
            f"{prefix}_over_1_5_rate": self.over_goal_totals["1_5"] / count,
            f"{prefix}_over_2_5_rate": self.over_goal_totals["2_5"] / count,
            f"{prefix}_over_3_5_rate": self.over_goal_totals["3_5"] / count,
            f"{prefix}_over_4_5_rate": self.over_goal_totals["4_5"] / count,
            f"{prefix}_win_streak": float(win_streak),
            f"{prefix}_draw_streak": float(draw_streak),
            f"{prefix}_loss_streak": float(loss_streak),
            f"{prefix}_missing": 0.0,
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build prepared Mackolik datasets from unified legacy documents.")
    parser.add_argument("--cutoff", type=int, default=DEFAULT_CUTOFF)
    parser.add_argument("--output-dir", default=OUTPUT_DIR)
    parser.add_argument("--force-rebuild", action="store_true")
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--max-workers", type=int, default=DEFAULT_MAX_WORKERS)
    parser.add_argument("--allow-fallback-events", action="store_true", default=DEFAULT_ALLOW_FALLBACK_EVENTS)
    parser.add_argument("--no-allow-fallback-events", action="store_false", dest="allow_fallback_events")
    return parser.parse_args()


def progress_iterable(
    iterable: Iterable[Any],
    *,
    total: int | None = None,
    desc: str,
    unit: str,
) -> Iterable[Any]:
    if tqdm is None:
        return iterable
    return tqdm(iterable, total=total, desc=desc, unit=unit)


def chunked_iterable(iterable: Iterable[Any], chunk_size: int) -> Iterable[list[Any]]:
    chunk: list[Any] = []
    for item in iterable:
        chunk.append(item)
        if len(chunk) >= chunk_size:
            yield chunk
            chunk = []
    if chunk:
        yield chunk


def normalize_cutoff(cutoff: int | str) -> int:
    value = parse_int(cutoff)
    if value is None or value <= 0:
        raise ValueError(f"Invalid cutoff: {cutoff}")
    return int(value)


def normalize_max_workers(value: int | str | None) -> int:
    parsed = parse_int(value)
    if parsed is None or parsed <= 0:
        return 1
    return int(parsed)


def artifact_paths(cutoff: int, output_dir: str = OUTPUT_DIR) -> dict[str, str]:
    normalized = normalize_cutoff(cutoff)
    return {
        "full": os.path.join(output_dir, f"mackolik_{normalized}_full.pkl"),
        "train": os.path.join(output_dir, f"mackolik_{normalized}_train.pkl"),
        "test": os.path.join(output_dir, f"mackolik_{normalized}_test.pkl"),
        "manifest": os.path.join(output_dir, f"mackolik_{normalized}_manifest.json"),
    }


def prepared_artifacts_exist(
    cutoff: int,
    output_dir: str = OUTPUT_DIR,
    *,
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
        manifest.get("prep_version") == PREP_VERSION
        and parse_int(manifest.get("cutoff")) == normalize_cutoff(cutoff)
        and bool(manifest.get("allow_fallback_events", DEFAULT_ALLOW_FALLBACK_EVENTS)) == bool(allow_fallback_events)
    )


def save_prepared_frame(frame: pd.DataFrame, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    frame.to_pickle(path)


def load_prepared_frame(path: str) -> pd.DataFrame:
    import pandas as pd

    return pd.read_pickle(path)


def write_manifest(path: str, payload: dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=True, indent=2)


def load_mongo_collection() -> tuple[MongoClient, Collection]:
    from dotenv import load_dotenv
    from pymongo import MongoClient

    load_dotenv(".env")
    mongo_connection_string = os.getenv("MONGO_CONNECTION_STRING")
    if not mongo_connection_string:
        raise RuntimeError("MONGO_CONNECTION_STRING not found in .env")
    client = MongoClient(mongo_connection_string, serverSelectionTimeoutMS=20_000)
    client.admin.command("ping")
    return client, client[DATABASE_NAME][COLLECTION_NAME]


def ensure_training_query_index(collection: Collection) -> None:
    from pymongo import ASCENDING

    collection.create_index(
        [("match_id", ASCENDING), ("fetched_at", ASCENDING)],
        name="dataset_prep_unified_v1",
    )


def history_projection() -> dict[str, Any]:
    return {
        "_id": 0,
        "match_id": 1,
        "score": 1,
        "competition": 1,
        "match_data.detail": 1,
        "match_date_text": 1,
        "fetched_at": 1,
        "home_team.team_id": 1,
        "away_team.team_id": 1,
        "opta_feeds.raw.f24.Games.Game.@attributes": 1,
    }


def preparation_projection() -> dict[str, Any]:
    return {
        "_id": 0,
        "match_id": 1,
        "source": 1,
        "source_url": 1,
        "canonical_url": 1,
        "title": 1,
        "description": 1,
        "competition": 1,
        "home_team": 1,
        "away_team": 1,
        "score": 1,
        "status_text": 1,
        "venue": 1,
        "pages": 1,
        "mac_page": 1,
        "match_data.detail": 1,
        "match_data.home_players": 1,
        "match_data.away_players": 1,
        "match_data.events": 1,
        "odds_markets": 1,
        "fetch_errors": 1,
        "opta_identifiers": 1,
        "opta_feeds.raw.f24": 1,
        "match_stats": 1,
        "standings": 1,
        "other_matches": 1,
        "player_performance": 1,
        "top_performers": 1,
        "top_performers_html": 1,
        "mac_page_fragments": 1,
        "mac_page_stats": 1,
        "match_date_text": 1,
        "fetched_at": 1,
    }


def squad_projection() -> dict[str, Any]:
    return {
        "_id": 0,
        "match_id": 1,
        "home_team.team_id": 1,
        "away_team.team_id": 1,
        "match_data.home_players": 1,
        "match_data.away_players": 1,
        "top_performers": 1,
        "match_date_text": 1,
        "fetched_at": 1,
        "opta_feeds.raw.f24.Games.Game.@attributes": 1,
    }


def preparation_query() -> dict[str, Any]:
    return {}


def parse_int(value: Any) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if math.isnan(value):
            return None
        return int(value)
    text = str(value).strip()
    if not text:
        return None
    text = text.replace(",", ".")
    try:
        return int(float(text))
    except ValueError:
        return None


def parse_float(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        numeric = float(value)
        if math.isnan(numeric):
            return None
        return numeric
    text = str(value).strip()
    if not text or text == "-":
        return None
    normalized = text.replace(".", "").replace(",", ".") if "," in text and "." in text else text.replace(",", ".")
    try:
        return float(normalized)
    except ValueError:
        return None


def parse_datetime(value: Any) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.astimezone(UTC) if value.tzinfo else value.replace(tzinfo=UTC)
    text = str(value).strip()
    if not text:
        return None
    normalized = text.replace("Z", "+00:00")
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError:
        return None
    return parsed.astimezone(UTC) if parsed.tzinfo else parsed.replace(tzinfo=UTC)


def parse_match_date_text(value: Any) -> datetime | None:
    text = str(value or "").strip()
    if not text:
        return None
    match = re.match(r"^\s*(\d{1,2})\s+([A-Za-zÇĞİÖŞÜçğıöşü]+)\s+(\d{4})", text)
    if not match:
        return None
    day = int(match.group(1))
    month = TURKISH_MONTHS.get(match.group(2))
    year = int(match.group(3))
    if month is None:
        return None
    return datetime(year, month, day, tzinfo=UTC)


def parse_final_score(document: dict[str, Any]) -> tuple[int, int] | None:
    score = document.get("score")
    if isinstance(score, dict):
        home = parse_int(score.get("home"))
        away = parse_int(score.get("away"))
        if home is not None and away is not None:
            return home, away

    detail = ((document.get("match_data") or {}).get("detail") or {})
    for key in ("full_time_score", "score"):
        raw_score = detail.get(key)
        if not raw_score:
            continue
        text = str(raw_score).strip()
        if "-" not in text:
            continue
        home_text, away_text = [part.strip() for part in text.split("-", 1)]
        home = parse_int(home_text)
        away = parse_int(away_text)
        if home is not None and away is not None:
            return home, away
    return None


def final_result_label(home_goals: int, away_goals: int) -> str:
    if home_goals > away_goals:
        return "1"
    if home_goals < away_goals:
        return "2"
    return "X"


def threshold_key(threshold: float) -> str:
    return str(threshold).replace(".", "_")


def build_goal_targets(home_goals: int, away_goals: int) -> dict[str, int]:
    total_goals = home_goals + away_goals
    return {
        f"target_over_{threshold_key(threshold)}": int(total_goals > threshold)
        for threshold in GOAL_TARGET_THRESHOLDS
    }


def goal_count_bucket_label(total_goals: int, cap: int = GOAL_COUNT_CAP) -> str:
    return f"{cap}+" if total_goals >= cap else str(total_goals)


def iter_f24_events(document: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]] | tuple[None, None]:
    raw_f24 = (((document.get("opta_feeds") or {}).get("raw") or {}).get("f24"))
    if not isinstance(raw_f24, dict):
        return None, None
    games = raw_f24.get("Games")
    if not isinstance(games, dict):
        return None, None
    game = games.get("Game")
    if not isinstance(game, dict):
        return None, None
    attrs = game.get("@attributes") if isinstance(game.get("@attributes"), dict) else {}
    events = game.get("Event")
    if isinstance(events, list):
        return events, attrs or {}
    if isinstance(events, dict):
        return [events], attrs or {}
    return [], attrs or {}


def get_f24_attrs(document: dict[str, Any]) -> dict[str, Any]:
    _, attrs = iter_f24_events(document)
    return attrs or {}


def extract_match_timestamp(document: dict[str, Any], f24_attrs: dict[str, Any] | None = None) -> datetime | None:
    if isinstance(f24_attrs, dict):
        for key in ("game_date", "period_1_start", "period_2_start"):
            parsed = parse_datetime(f24_attrs.get(key))
            if parsed is not None:
                return parsed
    parsed_text = parse_match_date_text(document.get("match_date_text"))
    if parsed_text is not None:
        return parsed_text
    return parse_datetime(document.get("fetched_at"))


def encode_side(team_id: str | None, home_team_id: str | None, away_team_id: str | None) -> str:
    if team_id is not None and team_id == home_team_id:
        return "home"
    if team_id is not None and team_id == away_team_id:
        return "away"
    return "unknown"


def time_bucket_label(total_seconds: int, cutoff_minute: int) -> str | None:
    if total_seconds < 0 or total_seconds > cutoff_minute * 60:
        return None
    bucket_start = (total_seconds // 300) * 5
    bucket_end = min(bucket_start + 5, cutoff_minute)
    if bucket_end <= bucket_start:
        bucket_end = bucket_start + 5
    return f"{bucket_start}_{bucket_end}"


def zone_bin(value: float, bins: int) -> int:
    capped = min(max(value, 0.0), 99.999)
    return int(capped / (100.0 / bins))


def probability_entropy(probabilities: dict[str, float]) -> float:
    entropy = 0.0
    for value in probabilities.values():
        if value > 0.0:
            entropy -= value * math.log(value)
    return entropy


def normalize_implied_probs(odds: dict[str, float | None]) -> tuple[dict[str, float], float]:
    implied = {key: (0.0 if not value or value <= 0 else 1.0 / value) for key, value in odds.items()}
    total = sum(implied.values())
    if total <= 0.0:
        return {key: 0.0 for key in odds}, 0.0
    return {key: value / total for key, value in implied.items()}, total


def build_result_class_summary(entries: list[dict[str, Any]], prefix: str) -> dict[str, float]:
    wins = 0
    draws = 0
    losses = 0
    points_total = 0.0
    counted = 0
    for entry in entries[:ROLLING_WINDOW]:
        result_classes = entry.get("result_class")
        if not isinstance(result_classes, list):
            continue
        label = next((item for item in result_classes if item in FORM_CLASS_TO_POINTS), None)
        if label is None:
            continue
        counted += 1
        points = FORM_CLASS_TO_POINTS[label]
        points_total += points
        if points == 3.0:
            wins += 1
        elif points == 1.0:
            draws += 1
        else:
            losses += 1
    if counted == 0:
        return {
            f"{prefix}_count": 0.0,
            f"{prefix}_ppm": 0.0,
            f"{prefix}_win_share": 0.0,
            f"{prefix}_draw_share": 0.0,
            f"{prefix}_loss_share": 0.0,
            f"{prefix}_missing": 1.0,
        }
    return {
        f"{prefix}_count": float(counted),
        f"{prefix}_ppm": points_total / counted,
        f"{prefix}_win_share": wins / counted,
        f"{prefix}_draw_share": draws / counted,
        f"{prefix}_loss_share": losses / counted,
        f"{prefix}_missing": 0.0,
    }


def longest_recent_streak(entries: list[dict[str, Any]]) -> tuple[int, int, int]:
    current_wins = 0
    current_draws = 0
    current_losses = 0
    for entry in entries[:ROLLING_WINDOW]:
        result_classes = entry.get("result_class")
        if not isinstance(result_classes, list):
            break
        label = next((item for item in result_classes if item in FORM_CLASS_TO_POINTS), None)
        if label is None:
            break
        points = FORM_CLASS_TO_POINTS[label]
        if points == 3.0 and current_draws == 0 and current_losses == 0:
            current_wins += 1
        elif points == 1.0 and current_wins == 0 and current_losses == 0:
            current_draws += 1
        elif points == 0.0 and current_wins == 0 and current_draws == 0:
            current_losses += 1
        else:
            break
    return current_wins, current_draws, current_losses


def build_embedded_form_features(document: dict[str, Any]) -> dict[str, float]:
    features: dict[str, float] = {}
    home_recent = ((document.get("home_team") or {}).get("recent_form") or [])
    away_recent = ((document.get("away_team") or {}).get("recent_form") or [])
    if isinstance(home_recent, list):
        features.update(build_result_class_summary(home_recent, "embedded_home_recent_form"))
        wins, draws, losses = longest_recent_streak(home_recent)
        features["embedded_home_recent_form_win_streak"] = float(wins)
        features["embedded_home_recent_form_draw_streak"] = float(draws)
        features["embedded_home_recent_form_loss_streak"] = float(losses)
    if isinstance(away_recent, list):
        features.update(build_result_class_summary(away_recent, "embedded_away_recent_form"))
        wins, draws, losses = longest_recent_streak(away_recent)
        features["embedded_away_recent_form_win_streak"] = float(wins)
        features["embedded_away_recent_form_draw_streak"] = float(draws)
        features["embedded_away_recent_form_loss_streak"] = float(losses)
    for metric in ("ppm", "win_share", "draw_share", "loss_share", "win_streak", "draw_streak", "loss_streak"):
        features[f"embedded_recent_form_delta_{metric}"] = (
            float(features.get(f"embedded_home_recent_form_{metric}", 0.0))
            - float(features.get(f"embedded_away_recent_form_{metric}", 0.0))
        )
    return features


def build_context_features(document: dict[str, Any], match_timestamp: datetime) -> dict[str, float | str]:
    features: dict[str, float | str] = {}
    competition = document.get("competition") or {}
    if isinstance(competition, dict):
        if competition.get("name"):
            features["context_competition_name"] = str(competition["name"])
        if competition.get("season_id") is not None:
            features["context_competition_season_id"] = str(competition["season_id"])
    matchday = parse_int(get_f24_attrs(document).get("matchday"))
    if matchday is not None:
        features["context_matchday"] = float(matchday)
        features["context_season_phase_early"] = 1.0 if matchday <= 8 else 0.0
        features["context_season_phase_mid"] = 1.0 if 9 <= matchday <= 24 else 0.0
        features["context_season_phase_late"] = 1.0 if matchday >= 25 else 0.0
    venue = document.get("venue") or {}
    if isinstance(venue, dict):
        capacity = parse_int(venue.get("capacity")) or parse_int(venue.get("capacity_text"))
        if capacity is not None:
            features["context_venue_capacity"] = float(capacity)
    features["context_venue_capacity_missing"] = 0.0 if "context_venue_capacity" in features else 1.0
    features["context_kickoff_month"] = float(match_timestamp.month)
    features["context_kickoff_weekday"] = float(match_timestamp.weekday())
    features["context_kickoff_hour"] = float(match_timestamp.hour)
    features["context_kickoff_day_of_year"] = float(match_timestamp.timetuple().tm_yday)
    features["context_kickoff_is_weekend"] = 1.0 if match_timestamp.weekday() >= 5 else 0.0
    return features


def extract_outcome_odd_map(market: dict[str, Any]) -> dict[str, float | None]:
    result: dict[str, float | None] = {}
    for outcome in market.get("outcomes", []):
        if not isinstance(outcome, dict):
            continue
        name = str(outcome.get("name") or "").strip()
        if not name:
            continue
        result[name] = parse_float(outcome.get("odd"))
    return result


def threshold_from_market_name(name: str) -> str | None:
    match = re.search(r"\(?(\d+,\d+)\)?\s*Alt/Üst", name)
    if not match:
        return None
    return match.group(1).replace(",", "_")


def build_odds_features(document: dict[str, Any]) -> dict[str, float]:
    features: dict[str, float] = {}
    markets = document.get("odds_markets")
    if not isinstance(markets, list):
        markets = []

    match_result_odds: dict[str, float | None] | None = None
    double_chance_odds: dict[str, float | None] | None = None
    totals_markets: dict[str, dict[str, float | None]] = {}

    for market in markets:
        if not isinstance(market, dict):
            continue
        name = str(market.get("market_name") or "").strip()
        odd_map = extract_outcome_odd_map(market)
        if name == "Maç Sonucu":
            match_result_odds = odd_map
        elif name == "Çifte Şans":
            double_chance_odds = odd_map
        else:
            threshold = threshold_from_market_name(name)
            if threshold is not None and ("Alt" in odd_map or "Üst" in odd_map):
                totals_markets[threshold] = odd_map

    if match_result_odds is not None:
        normalized, margin = normalize_implied_probs(
            {
                "home": match_result_odds.get("1"),
                "draw": match_result_odds.get("X"),
                "away": match_result_odds.get("2"),
            }
        )
        features["odds_match_result_present"] = 1.0
        features["odds_match_result_margin"] = margin
        for key, odd in (
            ("home", match_result_odds.get("1")),
            ("draw", match_result_odds.get("X")),
            ("away", match_result_odds.get("2")),
        ):
            if odd is not None:
                features[f"odds_match_result_{key}"] = odd
                features[f"odds_match_result_implied_{key}"] = normalized[key]
        favorite_prob = max(normalized.values())
        underdog_prob = min(normalized.values())
        features["odds_match_result_entropy"] = probability_entropy(normalized)
        features["odds_match_result_favorite_prob"] = favorite_prob
        features["odds_match_result_underdog_prob"] = underdog_prob
        features["odds_match_result_favorite_gap"] = favorite_prob - underdog_prob
    else:
        features["odds_match_result_present"] = 0.0

    if double_chance_odds is not None:
        features["odds_double_chance_present"] = 1.0
        for feature_name, outcome_name in (
            ("home_or_draw", "1-X"),
            ("home_or_away", "1-2"),
            ("draw_or_away", "X-2"),
        ):
            odd = double_chance_odds.get(outcome_name)
            if odd is not None:
                features[f"odds_double_chance_{feature_name}"] = odd
    else:
        features["odds_double_chance_present"] = 0.0

    for threshold in ("1_5", "2_5", "3_5", "4_5"):
        market_odds = totals_markets.get(threshold)
        if market_odds is None:
            features[f"odds_total_{threshold}_present"] = 0.0
            continue
        normalized, margin = normalize_implied_probs({"under": market_odds.get("Alt"), "over": market_odds.get("Üst")})
        features[f"odds_total_{threshold}_present"] = 1.0
        features[f"odds_total_{threshold}_margin"] = margin
        if market_odds.get("Alt") is not None:
            features[f"odds_total_{threshold}_under"] = float(market_odds["Alt"])
            features[f"odds_total_{threshold}_under_implied"] = normalized["under"]
        if market_odds.get("Üst") is not None:
            features[f"odds_total_{threshold}_over"] = float(market_odds["Üst"])
            features[f"odds_total_{threshold}_over_implied"] = normalized["over"]
        features[f"odds_total_{threshold}_entropy"] = probability_entropy(normalized)
        features[f"odds_total_{threshold}_over_gap"] = normalized["over"] - normalized["under"]
    return features


def build_common_features(document: dict[str, Any], match_timestamp: datetime) -> dict[str, float | str]:
    features: dict[str, float | str] = {}
    features.update(build_context_features(document, match_timestamp))
    features.update(build_odds_features(document))
    features.update(build_embedded_form_features(document))
    features.update(build_squad_static_features(document))
    features["interaction_embedded_form_x_favorite_gap"] = (
        float(features.get("embedded_recent_form_delta_ppm", 0.0))
        * float(features.get("odds_match_result_favorite_gap", 0.0))
    )
    return features


def lineup_status_bucket(value: Any) -> str:
    text = str(value or "").strip().lower()
    if "start" in text:
        return "starter"
    if "sub" in text:
        return "bench"
    return "unknown"


def lineup_position_bucket(value: Any) -> str:
    text = str(value or "").strip().lower()
    if not text:
        return "unknown"
    if "goal" in text or text in {"gk", "goalkeeper"}:
        return "goalkeeper"
    if "def" in text or text in {"dc", "dr", "dl", "cb", "rb", "lb"}:
        return "defender"
    if "mid" in text or text in {"mc", "ml", "mr", "dmc", "amc"}:
        return "midfielder"
    if "strik" in text or "forward" in text or text in {"fc", "fw", "attacker"}:
        return "attacker"
    if "substitute" in text:
        return "bench_role_unknown"
    return "other"


def normalize_lineup_players(document: dict[str, Any], side: str) -> list[dict[str, Any]]:
    match_data = document.get("match_data")
    if not isinstance(match_data, dict):
        return []
    raw_players = match_data.get(f"{side}_players")
    if not isinstance(raw_players, list):
        return []
    normalized: list[dict[str, Any]] = []
    for raw_player in raw_players:
        if not isinstance(raw_player, dict):
            continue
        stats = raw_player.get("stats") if isinstance(raw_player.get("stats"), dict) else {}
        player_id = raw_player.get("player_id")
        normalized.append(
            {
                "player_id": str(player_id) if player_id is not None else None,
                "player_name": raw_player.get("player_name"),
                "position": raw_player.get("position"),
                "position_bucket": lineup_position_bucket(raw_player.get("position")),
                "status": lineup_status_bucket(raw_player.get("status")),
                "shirt_number": parse_int(raw_player.get("shirt_number")),
                "minutes_played": parse_float(raw_player.get("minutes_played")) or parse_float(stats.get("mins_played")) or 0.0,
                "stats": stats,
            }
        )
    return normalized


def extract_team_systems(document: dict[str, Any]) -> tuple[str | None, str | None]:
    mac_page_stats = document.get("mac_page_stats")
    if not isinstance(mac_page_stats, dict):
        return None, None
    opta_stats = mac_page_stats.get("opta_stats")
    if not isinstance(opta_stats, dict):
        return None, None
    team_systems = opta_stats.get("team_systems")
    if not isinstance(team_systems, list):
        return None, None
    home_system = str(team_systems[0]) if len(team_systems) >= 1 and team_systems[0] else None
    away_system = str(team_systems[1]) if len(team_systems) >= 2 and team_systems[1] else None
    return home_system, away_system


def build_squad_static_features(document: dict[str, Any]) -> dict[str, float | str]:
    features: dict[str, float | str] = {}
    home_players = normalize_lineup_players(document, "home")
    away_players = normalize_lineup_players(document, "away")
    home_system, away_system = extract_team_systems(document)

    for side, players, system in (("home", home_players, home_system), ("away", away_players, away_system)):
        features[f"squad_{side}_lineup_present"] = 1.0 if players else 0.0
        features[f"squad_{side}_players_count"] = float(len(players))
        starters = [player for player in players if player["status"] == "starter"]
        bench = [player for player in players if player["status"] == "bench"]
        features[f"squad_{side}_starters_count"] = float(len(starters))
        features[f"squad_{side}_bench_count"] = float(len(bench))
        features[f"squad_{side}_unknown_status_count"] = float(len(players) - len(starters) - len(bench))
        starter_count = max(float(len(starters)), 1.0)
        for bucket in ("goalkeeper", "defender", "midfielder", "attacker", "other", "bench_role_unknown", "unknown"):
            starter_bucket = sum(1 for player in starters if player["position_bucket"] == bucket)
            bench_bucket = sum(1 for player in bench if player["position_bucket"] == bucket)
            features[f"squad_{side}_starter_{bucket}_count"] = float(starter_bucket)
            features[f"squad_{side}_bench_{bucket}_count"] = float(bench_bucket)
            features[f"squad_{side}_starter_{bucket}_share"] = float(starter_bucket) / starter_count
        if system:
            features[f"squad_{side}_formation"] = system
        coach_name = ((document.get(f"{side}_team") or {}).get("coach_name"))
        if coach_name:
            features[f"squad_{side}_coach_name"] = str(coach_name)

    features["squad_delta_players_count"] = float(features.get("squad_home_players_count", 0.0)) - float(features.get("squad_away_players_count", 0.0))
    features["squad_delta_starters_count"] = float(features.get("squad_home_starters_count", 0.0)) - float(features.get("squad_away_starters_count", 0.0))
    features["squad_delta_bench_count"] = float(features.get("squad_home_bench_count", 0.0)) - float(features.get("squad_away_bench_count", 0.0))
    features["squad_delta_starter_attacker_count"] = float(features.get("squad_home_starter_attacker_count", 0.0)) - float(features.get("squad_away_starter_attacker_count", 0.0))
    features["squad_delta_starter_midfielder_count"] = float(features.get("squad_home_starter_midfielder_count", 0.0)) - float(features.get("squad_away_starter_midfielder_count", 0.0))
    features["squad_lineup_both_present"] = 1.0 if home_players and away_players else 0.0
    return features


def player_stat_value(stats: dict[str, Any], *keys: str) -> float:
    total = 0.0
    for key in keys:
        total += parse_float(stats.get(key)) or 0.0
    return total


def top_performer_mentions(top_performers: Any) -> dict[str, float]:
    mentions: Counter[str] = Counter()
    if not isinstance(top_performers, dict):
        return {}
    for category, rows in top_performers.items():
        if not isinstance(rows, list):
            continue
        seen_in_category: set[str] = set()
        for rank, row in enumerate(rows[:5], start=1):
            if not isinstance(row, dict):
                continue
            player_id = parse_int(row.get("OYUNCU_ID"))
            if player_id is None:
                continue
            key = str(player_id)
            if key in seen_in_category:
                continue
            seen_in_category.add(key)
            mentions[key] += 1.0 + max(0.0, (6 - rank) * 0.1)
    return {player_id: float(score) for player_id, score in mentions.items()}


def empty_player_state() -> dict[str, float]:
    return {
        "matches": 0.0,
        "starts": 0.0,
        "minutes": 0.0,
        "attack": 0.0,
        "defense": 0.0,
        "passing": 0.0,
        "discipline": 0.0,
        "top_performer": 0.0,
    }


def decay_player_state(state: dict[str, float], decay: float = PLAYER_HISTORY_DECAY) -> None:
    for key in list(state):
        state[key] *= decay


def update_player_state(
    player_states: dict[str, dict[str, float]],
    player: dict[str, Any],
    *,
    top_performer_score: float,
) -> None:
    player_id = player.get("player_id")
    if player_id is None:
        return
    state = player_states.setdefault(str(player_id), empty_player_state())
    decay_player_state(state)
    stats = player.get("stats") if isinstance(player.get("stats"), dict) else {}
    minutes = float(player.get("minutes_played") or 0.0)
    state["matches"] += 1.0
    state["starts"] += 1.0 if player.get("status") == "starter" else 0.0
    state["minutes"] += minutes
    state["attack"] += (
        5.0 * player_stat_value(stats, "goals")
        + 3.0 * player_stat_value(stats, "goal_assist")
        + 0.30 * player_stat_value(stats, "total_scoring_att", "shots")
        + 0.45 * player_stat_value(stats, "ontarget_scoring_att", "shots_on_target")
        + 0.25 * player_stat_value(stats, "chance_created")
        + 0.08 * player_stat_value(stats, "successful_final_third_passes")
        + 0.05 * player_stat_value(stats, "pen_area_entries")
    )
    state["defense"] += (
        0.35 * player_stat_value(stats, "total_tackle", "won_tackle")
        + 0.30 * player_stat_value(stats, "interceptions")
        + 0.18 * player_stat_value(stats, "clearances")
        + 0.14 * player_stat_value(stats, "ball_recovery", "recoveries")
        + 0.10 * player_stat_value(stats, "duel_won", "aerial_won")
    )
    state["passing"] += (
        0.03 * player_stat_value(stats, "accurate_pass")
        + 0.02 * player_stat_value(stats, "successful_open_play_pass")
        + 0.06 * player_stat_value(stats, "accurate_crosses", "crosses_accuracy")
    )
    state["discipline"] += player_stat_value(stats, "cards_yellow") + 2.5 * player_stat_value(stats, "cards_red")
    state["top_performer"] += float(top_performer_score)


def player_snapshot(state: dict[str, float]) -> dict[str, float]:
    minutes = max(float(state.get("minutes", 0.0)), 1.0)
    matches = max(float(state.get("matches", 0.0)), 1.0)
    attack_p90 = float(state.get("attack", 0.0)) * 90.0 / minutes
    defense_p90 = float(state.get("defense", 0.0)) * 90.0 / minutes
    passing_p90 = float(state.get("passing", 0.0)) * 90.0 / minutes
    discipline_p90 = float(state.get("discipline", 0.0)) * 90.0 / minutes
    top_performer_rate = float(state.get("top_performer", 0.0)) / matches
    overall_rating = attack_p90 + 0.65 * defense_p90 + 0.45 * passing_p90 + 0.80 * top_performer_rate - 0.35 * discipline_p90
    return {
        "matches": float(state.get("matches", 0.0)),
        "starts": float(state.get("starts", 0.0)),
        "minutes": float(state.get("minutes", 0.0)),
        "attack_p90": attack_p90,
        "defense_p90": defense_p90,
        "passing_p90": passing_p90,
        "discipline_p90": discipline_p90,
        "top_performer_rate": top_performer_rate,
        "overall_rating": overall_rating,
    }


def aggregate_player_snapshots(snapshots: list[dict[str, float]], prefix: str) -> dict[str, float]:
    if not snapshots:
        return {
            f"{prefix}_known_count": 0.0,
            f"{prefix}_known_share": 0.0,
            f"{prefix}_matches_mean": 0.0,
            f"{prefix}_starts_mean": 0.0,
            f"{prefix}_minutes_mean": 0.0,
            f"{prefix}_overall_rating_mean": 0.0,
            f"{prefix}_overall_rating_max": 0.0,
            f"{prefix}_attack_p90_mean": 0.0,
            f"{prefix}_defense_p90_mean": 0.0,
            f"{prefix}_passing_p90_mean": 0.0,
            f"{prefix}_top_performer_rate_mean": 0.0,
            f"{prefix}_missing": 1.0,
        }
    count = float(len(snapshots))
    return {
        f"{prefix}_known_count": count,
        f"{prefix}_matches_mean": sum(item["matches"] for item in snapshots) / count,
        f"{prefix}_starts_mean": sum(item["starts"] for item in snapshots) / count,
        f"{prefix}_minutes_mean": sum(item["minutes"] for item in snapshots) / count,
        f"{prefix}_overall_rating_mean": sum(item["overall_rating"] for item in snapshots) / count,
        f"{prefix}_overall_rating_max": max(item["overall_rating"] for item in snapshots),
        f"{prefix}_attack_p90_mean": sum(item["attack_p90"] for item in snapshots) / count,
        f"{prefix}_defense_p90_mean": sum(item["defense_p90"] for item in snapshots) / count,
        f"{prefix}_passing_p90_mean": sum(item["passing_p90"] for item in snapshots) / count,
        f"{prefix}_top_performer_rate_mean": sum(item["top_performer_rate"] for item in snapshots) / count,
        f"{prefix}_missing": 0.0,
    }


def build_team_lineup_history_snapshot(
    players: list[dict[str, Any]],
    player_states: dict[str, dict[str, float]],
    previous_starters: set[str] | None,
    prefix: str,
) -> dict[str, float]:
    starters = [player for player in players if player.get("status") == "starter"]
    bench = [player for player in players if player.get("status") == "bench"]
    starter_ids = {str(player["player_id"]) for player in starters if player.get("player_id") is not None}
    previous_overlap = len(starter_ids.intersection(previous_starters or set()))
    starter_snapshots = [
        player_snapshot(player_states[player_id])
        for player in starters
        if (player_id := player.get("player_id")) is not None and str(player_id) in player_states
    ]
    bench_snapshots = [
        player_snapshot(player_states[player_id])
        for player in bench
        if (player_id := player.get("player_id")) is not None and str(player_id) in player_states
    ]
    features = {
        f"{prefix}_starter_count": float(len(starters)),
        f"{prefix}_bench_count": float(len(bench)),
        f"{prefix}_starter_known_share": float(len(starter_snapshots)) / max(float(len(starters)), 1.0),
        f"{prefix}_bench_known_share": float(len(bench_snapshots)) / max(float(len(bench)), 1.0),
        f"{prefix}_lineup_continuity_count": float(previous_overlap),
        f"{prefix}_lineup_continuity_share": float(previous_overlap) / max(float(len(starter_ids)), 1.0),
    }
    features.update(aggregate_player_snapshots(starter_snapshots, f"{prefix}_starters"))
    features.update(aggregate_player_snapshots(bench_snapshots, f"{prefix}_bench"))
    return features


def build_player_lineup_feature_map_from_documents(documents: Iterable[dict[str, Any]]) -> dict[str, dict[str, float]]:
    extracted_rows: list[dict[str, Any]] = []
    for document in documents:
        match_id = document.get("match_id")
        if match_id is None:
            continue
        match_timestamp = extract_match_timestamp(document, get_f24_attrs(document))
        if match_timestamp is None:
            continue
        home_team_id, away_team_id = extract_team_ids(document)
        extracted_rows.append(
            {
                "match_id": str(match_id),
                "match_timestamp": match_timestamp,
                "home_team_id": home_team_id,
                "away_team_id": away_team_id,
                "home_players": normalize_lineup_players(document, "home"),
                "away_players": normalize_lineup_players(document, "away"),
                "top_performer_mentions": top_performer_mentions(document.get("top_performers")),
            }
        )

    player_states: dict[str, dict[str, float]] = {}
    previous_starters_by_team: dict[str, set[str]] = {}
    features_by_match_id: dict[str, dict[str, float]] = {}

    for row in sorted(extracted_rows, key=lambda item: (item["match_timestamp"], item["match_id"])):
        home_features = build_team_lineup_history_snapshot(
            row["home_players"],
            player_states,
            previous_starters_by_team.get(row["home_team_id"] or "__missing_home__", set()),
            "squad_home",
        )
        away_features = build_team_lineup_history_snapshot(
            row["away_players"],
            player_states,
            previous_starters_by_team.get(row["away_team_id"] or "__missing_away__", set()),
            "squad_away",
        )
        current_features: dict[str, float] = {**home_features, **away_features}
        for metric in (
            "starter_count",
            "bench_count",
            "starter_known_share",
            "bench_known_share",
            "lineup_continuity_count",
            "lineup_continuity_share",
            "starters_matches_mean",
            "starters_starts_mean",
            "starters_minutes_mean",
            "starters_overall_rating_mean",
            "starters_overall_rating_max",
            "starters_attack_p90_mean",
            "starters_defense_p90_mean",
            "starters_passing_p90_mean",
            "starters_top_performer_rate_mean",
            "bench_matches_mean",
            "bench_minutes_mean",
            "bench_overall_rating_mean",
            "bench_attack_p90_mean",
            "bench_defense_p90_mean",
            "bench_passing_p90_mean",
        ):
            current_features[f"squad_delta_{metric}"] = float(home_features.get(f"squad_home_{metric}", 0.0)) - float(
                away_features.get(f"squad_away_{metric}", 0.0)
            )
        features_by_match_id[row["match_id"]] = current_features

        mentions = row["top_performer_mentions"]
        for player in [*row["home_players"], *row["away_players"]]:
            player_id = player.get("player_id")
            update_player_state(
                player_states,
                player,
                top_performer_score=float(mentions.get(str(player_id), 0.0)) if player_id is not None else 0.0,
            )
        previous_starters_by_team[row["home_team_id"] or "__missing_home__"] = {
            str(player["player_id"])
            for player in row["home_players"]
            if player.get("status") == "starter" and player.get("player_id") is not None
        }
        previous_starters_by_team[row["away_team_id"] or "__missing_away__"] = {
            str(player["player_id"])
            for player in row["away_players"]
            if player.get("status") == "starter" and player.get("player_id") is not None
        }

    return features_by_match_id


def build_player_lineup_feature_map(collection: Collection, *, max_rows: int | None = None) -> dict[str, dict[str, float]]:
    return build_player_lineup_feature_map_from_documents(
        progress_iterable(
            iter_mongo_documents(collection, projection=squad_projection(), max_rows=max_rows),
            desc="Building squad history",
            unit="match",
        )
    )


def sigmoid(value: float) -> float:
    if value >= 0:
        exp_term = math.exp(-value)
        return 1.0 / (1.0 + exp_term)
    exp_term = math.exp(value)
    return exp_term / (1.0 + exp_term)


def location_threat_proxy(attack_x: float, y_value: float) -> float:
    progress = max(min((attack_x - 50.0) / 50.0, 1.0), 0.0)
    centrality = max(0.0, 1.0 - abs(y_value - 50.0) / 50.0)
    return progress * (0.35 + 0.65 * centrality)


def shot_xg_proxy(attack_x: float, y_value: float) -> float:
    centrality = max(0.0, 1.0 - abs(y_value - 50.0) / 50.0)
    distance = math.hypot(100.0 - attack_x, (y_value - 50.0) * 0.8)
    logit = 2.5 * centrality + 0.05 * attack_x - 0.08 * distance - 5.0
    return sigmoid(logit)


def empty_decay_snapshot(prefix: str) -> dict[str, float]:
    return {
        f"{prefix}_weight": 0.0,
        f"{prefix}_ppm": 0.0,
        f"{prefix}_goals_for_pg": 0.0,
        f"{prefix}_goals_against_pg": 0.0,
        f"{prefix}_goal_diff_pg": 0.0,
        f"{prefix}_scoring_rate": 0.0,
        f"{prefix}_clean_sheet_rate": 0.0,
        f"{prefix}_missing": 1.0,
    }


def decay_snapshot(state: dict[str, float] | None, prefix: str) -> dict[str, float]:
    if not state or state.get("weight", 0.0) <= 0.0:
        return empty_decay_snapshot(prefix)

    weight = float(state["weight"])
    return {
        f"{prefix}_weight": weight,
        f"{prefix}_ppm": float(state["points"]) / weight,
        f"{prefix}_goals_for_pg": float(state["goals_for"]) / weight,
        f"{prefix}_goals_against_pg": float(state["goals_against"]) / weight,
        f"{prefix}_goal_diff_pg": float(state["goal_diff"]) / weight,
        f"{prefix}_scoring_rate": float(state["scored"]) / weight,
        f"{prefix}_clean_sheet_rate": float(state["clean_sheet"]) / weight,
        f"{prefix}_missing": 0.0,
    }


def update_decay_state(
    states: dict[str, dict[str, float]],
    team_id: str,
    appearance: TeamAppearance,
    *,
    decay: float = DECAY_FACTOR,
) -> None:
    state = states.setdefault(
        team_id,
        {
            "weight": 0.0,
            "points": 0.0,
            "goals_for": 0.0,
            "goals_against": 0.0,
            "goal_diff": 0.0,
            "scored": 0.0,
            "clean_sheet": 0.0,
        },
    )
    for key in list(state):
        state[key] *= decay
    state["weight"] += 1.0
    state["points"] += appearance.points
    state["goals_for"] += float(appearance.goals_for)
    state["goals_against"] += float(appearance.goals_against)
    state["goal_diff"] += float(appearance.goal_margin)
    state["scored"] += float(int(appearance.scored))
    state["clean_sheet"] += float(int(appearance.clean_sheet))


def extract_team_ids(document: dict[str, Any]) -> tuple[str | None, str | None]:
    home_team = document.get("home_team") or {}
    away_team = document.get("away_team") or {}
    if isinstance(home_team, dict) and isinstance(away_team, dict):
        home_id = home_team.get("team_id")
        away_id = away_team.get("team_id")
        return str(home_id) if home_id is not None else None, str(away_id) if away_id is not None else None
    return None, None


def normalize_f24_events(document: dict[str, Any]) -> list[dict[str, Any]] | None:
    events, _ = iter_f24_events(document)
    if events is None:
        return None
    home_team_id, away_team_id = extract_team_ids(document)
    normalized: list[dict[str, Any]] = []
    for event in events:
        if not isinstance(event, dict):
            continue
        attrs = event.get("@attributes") if isinstance(event.get("@attributes"), dict) else {}
        minute = parse_int(attrs.get("min"))
        if minute is None:
            continue
        second = parse_int(attrs.get("sec")) or 0
        team_id = str(attrs.get("team_id")) if attrs.get("team_id") is not None else None
        qualifier_ids: list[str] = []
        qualifiers = event.get("Q")
        if isinstance(qualifiers, dict):
            qualifiers = [qualifiers]
        if isinstance(qualifiers, list):
            for qualifier in qualifiers:
                if not isinstance(qualifier, dict):
                    continue
                q_attrs = qualifier.get("@attributes") if isinstance(qualifier.get("@attributes"), dict) else {}
                qualifier_id = q_attrs.get("qualifier_id")
                if qualifier_id is not None:
                    qualifier_ids.append(str(qualifier_id))
        normalized.append(
            {
                "minute": minute,
                "second": second,
                "total_seconds": minute * 60 + second,
                "side": encode_side(team_id, home_team_id, away_team_id),
                "event_type": str(attrs.get("type_id") or "unknown"),
                "player_id": str(attrs.get("player_id")) if attrs.get("player_id") is not None else None,
                "player_name": attrs.get("player_name"),
                "outcome": parse_int(attrs.get("outcome")),
                "x": parse_float(attrs.get("x")),
                "y": parse_float(attrs.get("y")),
                "qualifiers": qualifier_ids,
            }
        )
    return normalized


def normalize_match_data_events(document: dict[str, Any]) -> list[dict[str, Any]]:
    match_data = document.get("match_data")
    if not isinstance(match_data, dict):
        return []
    events = match_data.get("events")
    if not isinstance(events, list):
        return []
    normalized: list[dict[str, Any]] = []
    for event in events:
        if not isinstance(event, dict):
            continue
        details = event.get("details") if isinstance(event.get("details"), dict) else {}
        minute = parse_int(event.get("minute"))
        if minute is None:
            continue
        second = parse_int(event.get("second"))
        if second is None:
            second = parse_int(details.get("second")) or 0
        side_value = parse_int(event.get("team_side"))
        side = "home" if side_value == 1 else "away" if side_value == 2 else str(event.get("side") or "unknown")
        event_type = event.get("event_type")
        qualifiers = event.get("qualifiers") if isinstance(event.get("qualifiers"), list) else []
        normalized.append(
            {
                "minute": minute,
                "second": second,
                "total_seconds": minute * 60 + second,
                "side": side,
                "event_type": str(event_type or "unknown"),
                "player_id": str(event.get("player_id")) if event.get("player_id") is not None else None,
                "player_name": event.get("player_name"),
                "outcome": parse_int(event.get("outcome")),
                "x": parse_float(event.get("x")),
                "y": parse_float(event.get("y")),
                "qualifiers": [str(value) for value in qualifiers],
                "details": details,
            }
        )
    return normalized


def prioritized_live_events(document: dict[str, Any]) -> tuple[list[dict[str, Any]], str]:
    normalized_f24 = normalize_f24_events(document)
    if normalized_f24:
        return normalized_f24, "f24"
    return normalize_match_data_events(document), "match_data"


def primary_event_source(source: str) -> bool:
    return str(source) == "f24"


def build_prefix_features_from_events(
    events: list[dict[str, Any]],
    cutoff_minute: int,
    prefix: str,
) -> dict[str, float]:
    features: dict[str, float] = {}
    home_events = 0
    away_events = 0
    unknown_events = 0
    home_success = 0
    away_success = 0
    home_goals = 0
    away_goals = 0
    event_seconds: list[int] = []
    x_values_by_side: dict[str, list[float]] = {"home": [], "away": [], "unknown": []}
    y_values_by_side: dict[str, list[float]] = {"home": [], "away": [], "unknown": []}
    bucket_counts: Counter[str] = Counter()
    recent_5m_by_side: Counter[str] = Counter()
    prior_5m_by_side: Counter[str] = Counter()
    shot_proxy_by_side: Counter[str] = Counter()
    card_proxy_by_side: Counter[str] = Counter()
    box_touch_proxy_by_side: Counter[str] = Counter()
    threat_proxy_by_side: Counter[str] = Counter()
    recent_threat_proxy_by_side: Counter[str] = Counter()
    prior_threat_proxy_by_side: Counter[str] = Counter()
    shot_xg_proxy_by_side: Counter[str] = Counter()
    recent_shot_xg_proxy_by_side: Counter[str] = Counter()
    prior_shot_xg_proxy_by_side: Counter[str] = Counter()
    dangerous_shot_count_by_side: Counter[str] = Counter()
    max_shot_xg_by_side: dict[str, float] = {"home": 0.0, "away": 0.0, "unknown": 0.0}
    unique_players_by_side: dict[str, set[str]] = {"home": set(), "away": set(), "unknown": set()}
    event_counts_by_player_side: dict[str, Counter[str]] = {"home": Counter(), "away": Counter(), "unknown": Counter()}
    scorers_by_side: dict[str, set[str]] = {"home": set(), "away": set(), "unknown": set()}
    assist_counts_by_side: Counter[str] = Counter()
    substitution_counts_by_side: Counter[str] = Counter()
    recent_substitution_counts_by_side: Counter[str] = Counter()
    sent_off_players_by_side: dict[str, set[str]] = {"home": set(), "away": set(), "unknown": set()}

    for event in events:
        total_seconds = parse_int(event.get("total_seconds"))
        if total_seconds is None or total_seconds > cutoff_minute * 60:
            continue
        side = str(event.get("side") or "unknown")
        event_type = str(event.get("event_type") or "unknown")
        player_id = str(event.get("player_id")) if event.get("player_id") is not None else None
        player_name = str(event.get("player_name")) if event.get("player_name") else None
        outcome = parse_int(event.get("outcome"))
        x_value = parse_float(event.get("x"))
        y_value = parse_float(event.get("y"))
        qualifiers = [str(value) for value in event.get("qualifiers", [])]
        details = event.get("details") if isinstance(event.get("details"), dict) else {}
        subtype = str(details.get("subtype") or details.get("sub_type") or "").strip().lower()
        player_key = player_id or player_name
        bucket = time_bucket_label(total_seconds, cutoff_minute)
        event_seconds.append(total_seconds)

        if player_key:
            unique_players_by_side.setdefault(side, set()).add(player_key)
            event_counts_by_player_side.setdefault(side, Counter())[player_key] += 1

        if side == "home":
            home_events += 1
            if outcome == 1:
                home_success += 1
        elif side == "away":
            away_events += 1
            if outcome == 1:
                away_success += 1
        else:
            unknown_events += 1

        features[f"{prefix}_total_events"] = features.get(f"{prefix}_total_events", 0.0) + 1.0
        features[f"{prefix}_{side}_events"] = features.get(f"{prefix}_{side}_events", 0.0) + 1.0
        features[f"{prefix}_all_type_{event_type}_count"] = features.get(f"{prefix}_all_type_{event_type}_count", 0.0) + 1.0
        features[f"{prefix}_{side}_type_{event_type}_count"] = features.get(f"{prefix}_{side}_type_{event_type}_count", 0.0) + 1.0
        if bucket is not None:
            features[f"{prefix}_all_events_{bucket}"] = features.get(f"{prefix}_all_events_{bucket}", 0.0) + 1.0
            features[f"{prefix}_{side}_events_{bucket}"] = features.get(f"{prefix}_{side}_events_{bucket}", 0.0) + 1.0
            features[f"{prefix}_{side}_type_{event_type}_{bucket}"] = features.get(
                f"{prefix}_{side}_type_{event_type}_{bucket}",
                0.0,
            ) + 1.0
            bucket_counts[bucket] += 1

        if total_seconds > max((cutoff_minute - 5) * 60, 0):
            recent_5m_by_side[side] += 1
        elif total_seconds > max((cutoff_minute - 10) * 60, 0):
            prior_5m_by_side[side] += 1

        if event_type in SHOT_EVENT_TYPES:
            shot_proxy_by_side[side] += 1
        if event_type in CARD_EVENT_TYPES:
            card_proxy_by_side[side] += 1
        if "assist" in str(details.get("description") or "").lower() or details.get("assist_player_name"):
            assist_counts_by_side[side] += 1
        is_substitution = event_type == "substitution" or "sub" in subtype
        if is_substitution:
            substitution_counts_by_side[side] += 1
            if total_seconds > max((cutoff_minute - 5) * 60, 0):
                recent_substitution_counts_by_side[side] += 1
        is_red_card = event_type in {"18", "19", "red-card"} or ("red" in subtype and "card" in subtype)
        if is_red_card and player_key:
            sent_off_players_by_side.setdefault(side, set()).add(player_key)

        if outcome is not None:
            features[f"{prefix}_{side}_outcome_{outcome}_count"] = features.get(
                f"{prefix}_{side}_outcome_{outcome}_count",
                0.0,
            ) + 1.0

        if x_value is not None and y_value is not None:
            attack_x = x_value if side != "away" else 100.0 - x_value
            x_bucket = zone_bin(x_value, bins=5)
            y_bucket = zone_bin(y_value, bins=4)
            location_threat = location_threat_proxy(attack_x, y_value)
            features[f"{prefix}_{side}_zone_{x_bucket}_{y_bucket}_count"] = features.get(
                f"{prefix}_{side}_zone_{x_bucket}_{y_bucket}_count",
                0.0,
            ) + 1.0
            features[f"{prefix}_{side}_x_sum"] = features.get(f"{prefix}_{side}_x_sum", 0.0) + x_value
            features[f"{prefix}_{side}_y_sum"] = features.get(f"{prefix}_{side}_y_sum", 0.0) + y_value
            features[f"{prefix}_{side}_coord_count"] = features.get(f"{prefix}_{side}_coord_count", 0.0) + 1.0
            features[f"{prefix}_{side}_attack_x_sum"] = features.get(f"{prefix}_{side}_attack_x_sum", 0.0) + attack_x
            x_values_by_side[side].append(x_value)
            y_values_by_side[side].append(y_value)
            threat_proxy_by_side[side] += location_threat
            if total_seconds > max((cutoff_minute - 5) * 60, 0):
                recent_threat_proxy_by_side[side] += location_threat
            elif total_seconds > max((cutoff_minute - 10) * 60, 0):
                prior_threat_proxy_by_side[side] += location_threat
            if side in {"home", "away"} and attack_x >= 83.0 and 21.0 <= y_value <= 79.0:
                box_touch_proxy_by_side[side] += 1
            if event_type in SHOT_EVENT_TYPES:
                shot_quality = shot_xg_proxy(attack_x, y_value)
                shot_xg_proxy_by_side[side] += shot_quality
                max_shot_xg_by_side[side] = max(max_shot_xg_by_side[side], shot_quality)
                if shot_quality >= 0.20:
                    dangerous_shot_count_by_side[side] += 1
                if total_seconds > max((cutoff_minute - 5) * 60, 0):
                    recent_shot_xg_proxy_by_side[side] += shot_quality
                elif total_seconds > max((cutoff_minute - 10) * 60, 0):
                    prior_shot_xg_proxy_by_side[side] += shot_quality

        for qualifier_id in qualifiers:
            features[f"{prefix}_all_qual_{qualifier_id}_count"] = features.get(f"{prefix}_all_qual_{qualifier_id}_count", 0.0) + 1.0
            features[f"{prefix}_{side}_qual_{qualifier_id}_count"] = features.get(
                f"{prefix}_{side}_qual_{qualifier_id}_count",
                0.0,
            ) + 1.0

        is_goal = event_type in {"16", "goal"} or str(details.get("is_goal")).lower() == "true"
        is_own_goal = "28" in qualifiers or event.get("subType") == "own-goal" or details.get("subtype") == "own-goal"
        if is_goal:
            if player_key:
                scorers_by_side.setdefault(side, set()).add(player_key)
            if is_own_goal:
                if side == "home":
                    away_goals += 1
                elif side == "away":
                    home_goals += 1
            else:
                if side == "home":
                    home_goals += 1
                elif side == "away":
                    away_goals += 1

    total_events = home_events + away_events + unknown_events
    features[f"{prefix}_present"] = 1.0 if events else 0.0
    if total_events == 0:
        return features

    features[f"{prefix}_score_home"] = float(home_goals)
    features[f"{prefix}_score_away"] = float(away_goals)
    features[f"{prefix}_score_diff"] = float(home_goals - away_goals)
    features[f"{prefix}_score_total"] = float(home_goals + away_goals)
    features[f"{prefix}_home_success_rate"] = float(home_success) / home_events if home_events else 0.0
    features[f"{prefix}_away_success_rate"] = float(away_success) / away_events if away_events else 0.0
    features[f"{prefix}_home_event_share"] = float(home_events) / total_events
    features[f"{prefix}_away_event_share"] = float(away_events) / total_events
    features[f"{prefix}_unknown_event_share"] = float(unknown_events) / total_events
    features[f"{prefix}_event_count_diff"] = float(home_events - away_events)
    features[f"{prefix}_success_diff"] = float(home_success - away_success)
    features[f"{prefix}_events_per_minute"] = float(total_events) / cutoff_minute
    features[f"{prefix}_first_event_second"] = float(min(event_seconds))
    features[f"{prefix}_last_event_second"] = float(max(event_seconds))
    features[f"{prefix}_event_time_span_seconds"] = float(max(event_seconds) - min(event_seconds))
    features[f"{prefix}_minute_progress"] = float(cutoff_minute) / 90.0

    for bucket_label in sorted(bucket_counts):
        home_bucket = features.get(f"{prefix}_home_events_{bucket_label}", 0.0)
        away_bucket = features.get(f"{prefix}_away_events_{bucket_label}", 0.0)
        all_bucket = features.get(f"{prefix}_all_events_{bucket_label}", 0.0)
        features[f"{prefix}_event_diff_{bucket_label}"] = home_bucket - away_bucket
        features[f"{prefix}_event_share_{bucket_label}"] = all_bucket / total_events if total_events else 0.0

    first_bucket = time_bucket_label(0, cutoff_minute)
    last_bucket = time_bucket_label(max(cutoff_minute * 60 - 1, 0), cutoff_minute)
    if first_bucket is not None and last_bucket is not None:
        features[f"{prefix}_home_momentum_delta"] = features.get(f"{prefix}_home_events_{last_bucket}", 0.0) - features.get(
            f"{prefix}_home_events_{first_bucket}",
            0.0,
        )
        features[f"{prefix}_away_momentum_delta"] = features.get(f"{prefix}_away_events_{last_bucket}", 0.0) - features.get(
            f"{prefix}_away_events_{first_bucket}",
            0.0,
        )
        features[f"{prefix}_total_momentum_delta"] = features.get(f"{prefix}_all_events_{last_bucket}", 0.0) - features.get(
            f"{prefix}_all_events_{first_bucket}",
            0.0,
        )

    for side in ("home", "away"):
        features[f"{prefix}_{side}_recent_5m_events"] = float(recent_5m_by_side[side])
        features[f"{prefix}_{side}_prior_5m_events"] = float(prior_5m_by_side[side])
        features[f"{prefix}_{side}_recent_5m_delta"] = float(recent_5m_by_side[side] - prior_5m_by_side[side])
        features[f"{prefix}_{side}_shot_proxy"] = float(shot_proxy_by_side[side])
        features[f"{prefix}_{side}_card_proxy"] = float(card_proxy_by_side[side])
        features[f"{prefix}_{side}_box_touch_proxy"] = float(box_touch_proxy_by_side[side])
        features[f"{prefix}_{side}_threat_proxy_sum"] = float(threat_proxy_by_side[side])
        features[f"{prefix}_{side}_recent_5m_threat_proxy"] = float(recent_threat_proxy_by_side[side])
        features[f"{prefix}_{side}_prior_5m_threat_proxy"] = float(prior_threat_proxy_by_side[side])
        features[f"{prefix}_{side}_threat_proxy_mean"] = float(threat_proxy_by_side[side]) / max(float(home_events if side == "home" else away_events), 1.0)
        features[f"{prefix}_{side}_shot_xg_proxy_sum"] = float(shot_xg_proxy_by_side[side])
        features[f"{prefix}_{side}_shot_xg_proxy_max"] = float(max_shot_xg_by_side[side])
        features[f"{prefix}_{side}_dangerous_shots"] = float(dangerous_shot_count_by_side[side])
        side_shot_count = max(float(shot_proxy_by_side[side]), 1.0)
        features[f"{prefix}_{side}_shot_xg_proxy_mean"] = float(shot_xg_proxy_by_side[side]) / side_shot_count
        features[f"{prefix}_{side}_recent_5m_shot_xg_proxy"] = float(recent_shot_xg_proxy_by_side[side])
        features[f"{prefix}_{side}_prior_5m_shot_xg_proxy"] = float(prior_shot_xg_proxy_by_side[side])
        features[f"{prefix}_{side}_unique_players_involved"] = float(len(unique_players_by_side[side]))
        features[f"{prefix}_{side}_unique_scorers"] = float(len(scorers_by_side[side]))
        features[f"{prefix}_{side}_assist_count"] = float(assist_counts_by_side[side])
        features[f"{prefix}_{side}_substitution_count"] = float(substitution_counts_by_side[side])
        features[f"{prefix}_{side}_recent_5m_substitution_count"] = float(recent_substitution_counts_by_side[side])
        features[f"{prefix}_{side}_red_card_count"] = float(len(sent_off_players_by_side[side]))
        player_event_total = max(float(home_events if side == "home" else away_events), 1.0)
        top_player_count = max(event_counts_by_player_side[side].values(), default=0)
        features[f"{prefix}_{side}_top_player_event_share"] = float(top_player_count) / player_event_total

    features[f"{prefix}_recent_5m_event_diff"] = float(recent_5m_by_side["home"] - recent_5m_by_side["away"])
    features[f"{prefix}_prior_5m_event_diff"] = float(prior_5m_by_side["home"] - prior_5m_by_side["away"])
    features[f"{prefix}_shot_proxy_diff"] = float(shot_proxy_by_side["home"] - shot_proxy_by_side["away"])
    features[f"{prefix}_card_proxy_diff"] = float(card_proxy_by_side["home"] - card_proxy_by_side["away"])
    features[f"{prefix}_box_touch_proxy_diff"] = float(box_touch_proxy_by_side["home"] - box_touch_proxy_by_side["away"])
    features[f"{prefix}_threat_proxy_diff"] = float(threat_proxy_by_side["home"] - threat_proxy_by_side["away"])
    features[f"{prefix}_recent_5m_threat_proxy_diff"] = float(recent_threat_proxy_by_side["home"] - recent_threat_proxy_by_side["away"])
    features[f"{prefix}_shot_xg_proxy_diff"] = float(shot_xg_proxy_by_side["home"] - shot_xg_proxy_by_side["away"])
    features[f"{prefix}_recent_5m_shot_xg_proxy_diff"] = float(recent_shot_xg_proxy_by_side["home"] - recent_shot_xg_proxy_by_side["away"])
    features[f"{prefix}_dangerous_shots_diff"] = float(dangerous_shot_count_by_side["home"] - dangerous_shot_count_by_side["away"])
    features[f"{prefix}_unique_players_involved_diff"] = float(len(unique_players_by_side["home"]) - len(unique_players_by_side["away"]))
    features[f"{prefix}_substitution_count_diff"] = float(substitution_counts_by_side["home"] - substitution_counts_by_side["away"])
    features[f"{prefix}_recent_5m_substitution_diff"] = float(recent_substitution_counts_by_side["home"] - recent_substitution_counts_by_side["away"])
    features[f"{prefix}_red_card_diff"] = float(len(sent_off_players_by_side["home"]) - len(sent_off_players_by_side["away"]))
    features[f"{prefix}_manpower_edge_home"] = float(len(sent_off_players_by_side["away"]) - len(sent_off_players_by_side["home"]))
    features[f"{prefix}_assist_count_diff"] = float(assist_counts_by_side["home"] - assist_counts_by_side["away"])
    features[f"{prefix}_top_player_event_share_diff"] = float(
        features.get(f"{prefix}_home_top_player_event_share", 0.0) - features.get(f"{prefix}_away_top_player_event_share", 0.0)
    )
    features[f"{prefix}_score_diff_x_minute"] = features[f"{prefix}_score_diff"] * cutoff_minute
    features[f"{prefix}_score_total_x_minute"] = features[f"{prefix}_score_total"] * cutoff_minute
    features[f"{prefix}_score_diff_x_recent_momentum"] = features[f"{prefix}_score_diff"] * features[f"{prefix}_recent_5m_event_diff"]
    features[f"{prefix}_shot_proxy_diff_x_score_diff"] = features[f"{prefix}_shot_proxy_diff"] * features[f"{prefix}_score_diff"]
    features[f"{prefix}_threat_proxy_diff_x_score_diff"] = features[f"{prefix}_threat_proxy_diff"] * features[f"{prefix}_score_diff"]
    features[f"{prefix}_shot_xg_proxy_diff_x_score_diff"] = features[f"{prefix}_shot_xg_proxy_diff"] * features[f"{prefix}_score_diff"]
    features[f"{prefix}_manpower_edge_x_score_diff"] = features[f"{prefix}_manpower_edge_home"] * features[f"{prefix}_score_diff"]

    for side in ("home", "away", "unknown"):
        coord_count = features.get(f"{prefix}_{side}_coord_count", 0.0)
        if coord_count:
            features[f"{prefix}_{side}_x_mean"] = features.get(f"{prefix}_{side}_x_sum", 0.0) / coord_count
            features[f"{prefix}_{side}_y_mean"] = features.get(f"{prefix}_{side}_y_sum", 0.0) / coord_count
            if side in {"home", "away"}:
                features[f"{prefix}_{side}_attack_x_mean"] = features.get(f"{prefix}_{side}_attack_x_sum", 0.0) / coord_count
            x_values = x_values_by_side[side]
            y_values = y_values_by_side[side]
            if x_values:
                x_mean = sum(x_values) / len(x_values)
                y_mean = sum(y_values) / len(y_values)
                features[f"{prefix}_{side}_x_std"] = math.sqrt(sum((value - x_mean) ** 2 for value in x_values) / len(x_values))
                features[f"{prefix}_{side}_y_std"] = math.sqrt(sum((value - y_mean) ** 2 for value in y_values) / len(y_values))
                features[f"{prefix}_{side}_x_range"] = max(x_values) - min(x_values)
                features[f"{prefix}_{side}_y_range"] = max(y_values) - min(y_values)

    home_attack_mean = features.get(f"{prefix}_home_attack_x_mean", 50.0)
    away_attack_mean = features.get(f"{prefix}_away_attack_x_mean", 50.0)
    features[f"{prefix}_territory_proxy_diff"] = float(home_attack_mean) - float(away_attack_mean)
    return features


def build_f24_prefix_features(document: dict[str, Any], cutoff_minute: int) -> dict[str, float]:
    normalized_f24 = normalize_f24_events(document)
    if normalized_f24:
        return build_prefix_features_from_events(normalized_f24, cutoff_minute, f"f24_{cutoff_minute}")
    return build_prefix_features_from_events(normalize_match_data_events(document), cutoff_minute, f"f24_{cutoff_minute}")


def build_match_data_prefix_features(document: dict[str, Any], cutoff_minute: int) -> dict[str, float]:
    return build_prefix_features_from_events(normalize_match_data_events(document), cutoff_minute, f"md_{cutoff_minute}")


def build_live_prefix_features(document: dict[str, Any], cutoff_minute: int) -> dict[str, float]:
    events, source = prioritized_live_events(document)
    prefix = f"live_{cutoff_minute}"
    features = build_prefix_features_from_events(events, cutoff_minute, prefix)
    features[f"{prefix}_source_is_f24"] = 1.0 if source == "f24" else 0.0
    features[f"{prefix}_source_is_match_data"] = 1.0 if source == "match_data" else 0.0
    return features


def points_from_score(goals_for: int, goals_against: int) -> float:
    if goals_for > goals_against:
        return 3.0
    if goals_for == goals_against:
        return 1.0
    return 0.0


def appearance_from_match(home_goals: int, away_goals: int, is_home: bool) -> TeamAppearance:
    goals_for = home_goals if is_home else away_goals
    goals_against = away_goals if is_home else home_goals
    return TeamAppearance(
        is_home=is_home,
        goals_for=goals_for,
        goals_against=goals_against,
        points=points_from_score(goals_for, goals_against),
        total_goals=home_goals + away_goals,
        scored=goals_for > 0,
        clean_sheet=goals_against == 0,
        goal_margin=goals_for - goals_against,
    )


def empty_history_snapshot(prefix: str) -> dict[str, float]:
    return {
        f"{prefix}_count": 0.0,
        f"{prefix}_ppm": 0.0,
        f"{prefix}_win_share": 0.0,
        f"{prefix}_draw_share": 0.0,
        f"{prefix}_loss_share": 0.0,
        f"{prefix}_goals_for_pg": 0.0,
        f"{prefix}_goals_against_pg": 0.0,
        f"{prefix}_goal_diff_pg": 0.0,
        f"{prefix}_clean_sheet_rate": 0.0,
        f"{prefix}_scoring_rate": 0.0,
        f"{prefix}_over_1_5_rate": 0.0,
        f"{prefix}_over_2_5_rate": 0.0,
        f"{prefix}_over_3_5_rate": 0.0,
        f"{prefix}_over_4_5_rate": 0.0,
        f"{prefix}_win_streak": 0.0,
        f"{prefix}_draw_streak": 0.0,
        f"{prefix}_loss_streak": 0.0,
        f"{prefix}_missing": 1.0,
    }


def pi_expected_goal_difference(home_rating: float, away_rating: float) -> float:
    expected_home = (PI_RATING_BASE ** (abs(home_rating) / PI_RATING_C)) - 1.0
    expected_away = (PI_RATING_BASE ** (abs(away_rating) / PI_RATING_C)) - 1.0
    if home_rating < 0:
        expected_home = -expected_home
    if away_rating < 0:
        expected_away = -expected_away
    return expected_home - expected_away


def pi_weighted_error(actual_goal_diff: int, expected_goal_diff: float) -> tuple[float, float]:
    prediction_error = abs(actual_goal_diff - expected_goal_diff)
    weighted_error = PI_RATING_C * math.log10(1.0 + prediction_error)
    if expected_goal_diff < actual_goal_diff:
        return weighted_error, -weighted_error
    return -weighted_error, weighted_error


def build_history_feature_map(history_rows: list[HistoryRow]) -> dict[str, dict[str, float]]:
    history_by_team: dict[str, RollingAppearanceStats] = {}
    home_history_by_team: dict[str, RollingAppearanceStats] = {}
    away_history_by_team: dict[str, RollingAppearanceStats] = {}
    decay_history_by_team: dict[str, dict[str, float]] = {}
    decay_home_history_by_team: dict[str, dict[str, float]] = {}
    decay_away_history_by_team: dict[str, dict[str, float]] = {}
    pi_ratings_by_team: dict[str, dict[str, float]] = {}
    last_match_timestamp_by_team: dict[str, datetime] = {}
    matches_played_by_team: dict[str, int] = {}
    elo_by_team: dict[str, float] = {}
    features_by_match_id: dict[str, dict[str, float]] = {}

    sorted_rows = sorted(history_rows, key=lambda row: (row.match_timestamp, row.match_id))
    for row in sorted_rows:
        current_features: dict[str, float] = {}
        for side_name, team_id, is_home in (("home", row.home_team_id, True), ("away", row.away_team_id, False)):
            overall_prefix = f"history_{side_name}_overall_last{ROLLING_WINDOW}"
            venue_prefix = f"history_{side_name}_{'home' if is_home else 'away'}_last{ROLLING_WINDOW}"
            overall_stats = history_by_team.get(team_id) if team_id is not None else None
            venue_stats = (home_history_by_team if is_home else away_history_by_team).get(team_id) if team_id is not None else None
            current_features.update(overall_stats.snapshot(overall_prefix) if overall_stats is not None else empty_history_snapshot(overall_prefix))
            current_features.update(venue_stats.snapshot(venue_prefix) if venue_stats is not None else empty_history_snapshot(venue_prefix))
            current_features.update(
                decay_snapshot(decay_history_by_team.get(team_id) if team_id is not None else None, f"history_{side_name}_overall_exp")
            )
            current_features.update(
                decay_snapshot(
                    (decay_home_history_by_team if is_home else decay_away_history_by_team).get(team_id) if team_id is not None else None,
                    f"history_{side_name}_{'home' if is_home else 'away'}_exp",
                )
            )
            if team_id is not None:
                last_ts = last_match_timestamp_by_team.get(team_id)
                current_features[f"history_{side_name}_rest_days"] = (
                    max((row.match_timestamp - last_ts).total_seconds() / 86400.0, 0.0)
                    if last_ts is not None
                    else 0.0
                )
                current_features[f"history_{side_name}_matches_played"] = float(matches_played_by_team.get(team_id, 0))
                current_features[f"history_{side_name}_elo"] = float(elo_by_team.get(team_id, 1500.0))
                pi_state = pi_ratings_by_team.get(team_id, {"home": 0.0, "away": 0.0})
                current_features[f"history_{side_name}_pi_rating_home"] = float(pi_state["home"])
                current_features[f"history_{side_name}_pi_rating_away"] = float(pi_state["away"])
            else:
                current_features[f"history_{side_name}_rest_days"] = 0.0
                current_features[f"history_{side_name}_matches_played"] = 0.0
                current_features[f"history_{side_name}_elo"] = 1500.0
                current_features[f"history_{side_name}_pi_rating_home"] = 0.0
                current_features[f"history_{side_name}_pi_rating_away"] = 0.0

        for metric in HISTORY_METRICS:
            current_features[f"history_delta_overall_{metric}"] = (
                current_features.get(f"history_home_overall_last{ROLLING_WINDOW}_{metric}", 0.0)
                - current_features.get(f"history_away_overall_last{ROLLING_WINDOW}_{metric}", 0.0)
            )
            current_features[f"history_delta_venue_{metric}"] = (
                current_features.get(f"history_home_home_last{ROLLING_WINDOW}_{metric}", 0.0)
                - current_features.get(f"history_away_away_last{ROLLING_WINDOW}_{metric}", 0.0)
            )
        for metric in ("ppm", "goals_for_pg", "goals_against_pg", "goal_diff_pg", "scoring_rate", "clean_sheet_rate"):
            current_features[f"history_delta_exp_overall_{metric}"] = (
                current_features.get(f"history_home_overall_exp_{metric}", 0.0)
                - current_features.get(f"history_away_overall_exp_{metric}", 0.0)
            )
            current_features[f"history_delta_exp_venue_{metric}"] = (
                current_features.get(f"history_home_home_exp_{metric}", 0.0)
                - current_features.get(f"history_away_away_exp_{metric}", 0.0)
            )
        current_features["history_delta_rest_days"] = current_features.get("history_home_rest_days", 0.0) - current_features.get("history_away_rest_days", 0.0)
        current_features["history_delta_matches_played"] = current_features.get("history_home_matches_played", 0.0) - current_features.get("history_away_matches_played", 0.0)
        current_features["history_delta_elo"] = current_features.get("history_home_elo", 1500.0) - current_features.get("history_away_elo", 1500.0)
        current_features["history_delta_pi_home_away"] = current_features.get("history_home_pi_rating_home", 0.0) - current_features.get("history_away_pi_rating_away", 0.0)
        current_features["history_delta_pi_cross"] = current_features.get("history_home_pi_rating_away", 0.0) - current_features.get("history_away_pi_rating_home", 0.0)
        current_features["history_pi_expected_goal_diff"] = pi_expected_goal_difference(
            current_features.get("history_home_pi_rating_home", 0.0),
            current_features.get("history_away_pi_rating_away", 0.0),
        )
        if row.matchday is not None:
            current_features["history_matchday"] = float(row.matchday)
            current_features["history_home_matchday_load"] = current_features.get("history_home_matches_played", 0.0) / max(float(row.matchday), 1.0)
            current_features["history_away_matchday_load"] = current_features.get("history_away_matches_played", 0.0) / max(float(row.matchday), 1.0)

        features_by_match_id[row.match_id] = current_features

        home_appearance = appearance_from_match(row.home_goals, row.away_goals, is_home=True)
        away_appearance = appearance_from_match(row.home_goals, row.away_goals, is_home=False)
        if row.home_team_id is not None:
            history_by_team.setdefault(row.home_team_id, RollingAppearanceStats()).append(home_appearance)
            home_history_by_team.setdefault(row.home_team_id, RollingAppearanceStats()).append(home_appearance)
            update_decay_state(decay_history_by_team, row.home_team_id, home_appearance)
            update_decay_state(decay_home_history_by_team, row.home_team_id, home_appearance)
            last_match_timestamp_by_team[row.home_team_id] = row.match_timestamp
            matches_played_by_team[row.home_team_id] = matches_played_by_team.get(row.home_team_id, 0) + 1
        if row.away_team_id is not None:
            history_by_team.setdefault(row.away_team_id, RollingAppearanceStats()).append(away_appearance)
            away_history_by_team.setdefault(row.away_team_id, RollingAppearanceStats()).append(away_appearance)
            update_decay_state(decay_history_by_team, row.away_team_id, away_appearance)
            update_decay_state(decay_away_history_by_team, row.away_team_id, away_appearance)
            last_match_timestamp_by_team[row.away_team_id] = row.match_timestamp
            matches_played_by_team[row.away_team_id] = matches_played_by_team.get(row.away_team_id, 0) + 1

        home_elo_before = elo_by_team.get(row.home_team_id or "__missing_home__", 1500.0)
        away_elo_before = elo_by_team.get(row.away_team_id or "__missing_away__", 1500.0)
        expected_home = 1.0 / (1.0 + 10 ** ((away_elo_before - (home_elo_before + 60.0)) / 400.0))
        home_result = 1.0 if row.home_goals > row.away_goals else 0.5 if row.home_goals == row.away_goals else 0.0
        away_result = 1.0 - home_result
        margin_multiplier = math.log(abs(row.home_goals - row.away_goals) + 1.0) + 1.0
        k_factor = 20.0 * margin_multiplier
        if row.home_team_id is not None:
            elo_by_team[row.home_team_id] = home_elo_before + k_factor * (home_result - expected_home)
        if row.away_team_id is not None:
            elo_by_team[row.away_team_id] = away_elo_before + k_factor * (away_result - (1.0 - expected_home))

        home_pi = pi_ratings_by_team.setdefault(row.home_team_id or "__missing_home__", {"home": 0.0, "away": 0.0})
        away_pi = pi_ratings_by_team.setdefault(row.away_team_id or "__missing_away__", {"home": 0.0, "away": 0.0})
        expected_goal_diff = pi_expected_goal_difference(home_pi["home"], away_pi["away"])
        weighted_home_error, weighted_away_error = pi_weighted_error(row.home_goals - row.away_goals, expected_goal_diff)
        if row.home_team_id is not None:
            home_pi["home"] += weighted_home_error * PI_RATING_LAMBDA
            home_pi["away"] += weighted_home_error * PI_RATING_LAMBDA * PI_RATING_GAMMA
        if row.away_team_id is not None:
            away_pi["away"] += weighted_away_error * PI_RATING_LAMBDA
            away_pi["home"] += weighted_away_error * PI_RATING_LAMBDA * PI_RATING_GAMMA

    return features_by_match_id


def iter_mongo_documents(
    collection: Collection,
    *,
    projection: dict[str, Any],
    max_rows: int | None = None,
) -> Iterable[dict[str, Any]]:
    cursor = collection.find(preparation_query(), projection=projection, no_cursor_timeout=True).batch_size(MONGO_BATCH_SIZE)
    if max_rows is not None:
        cursor = cursor.limit(max_rows)
    try:
        for document in cursor:
            yield document
    finally:
        cursor.close()


def history_row_from_document(document: dict[str, Any]) -> HistoryRow | None:
    final_score = parse_final_score(document)
    if final_score is None:
        return None
    home_team_id, away_team_id = extract_team_ids(document)
    f24_attrs = get_f24_attrs(document)
    match_timestamp = extract_match_timestamp(document, f24_attrs)
    match_id = document.get("match_id")
    if match_timestamp is None or match_id is None:
        return None
    return HistoryRow(
        match_id=str(match_id),
        match_timestamp=match_timestamp,
        home_team_id=home_team_id,
        away_team_id=away_team_id,
        home_goals=final_score[0],
        away_goals=final_score[1],
        matchday=parse_int(f24_attrs.get("matchday")),
    )


def build_history_rows_from_documents(documents: Iterable[dict[str, Any]]) -> list[HistoryRow]:
    rows: list[HistoryRow] = []
    for document in documents:
        row = history_row_from_document(document)
        if row is not None:
            rows.append(row)
    return rows


def build_history_rows(collection: Collection, *, max_rows: int | None = None) -> list[HistoryRow]:
    return build_history_rows_from_documents(
        progress_iterable(
            iter_mongo_documents(collection, projection=history_projection(), max_rows=max_rows),
            desc="Building history rows",
            unit="match",
        )
    )


def dataset_row_from_document(
    document: dict[str, Any],
    *,
    cutoff: int,
    history_features_by_match_id: dict[str, dict[str, float]],
    squad_features_by_match_id: dict[str, dict[str, float]] | None = None,
    allow_fallback_events: bool = DEFAULT_ALLOW_FALLBACK_EVENTS,
) -> dict[str, Any] | None:
    final_score = parse_final_score(document)
    match_id = document.get("match_id")
    if final_score is None or match_id is None:
        return None
    match_id = str(match_id)
    match_timestamp = extract_match_timestamp(document, get_f24_attrs(document))
    if match_timestamp is None:
        return None
    history_features = history_features_by_match_id.get(match_id)
    if history_features is None:
        return None
    _, event_source = prioritized_live_events(document)
    if not allow_fallback_events and not primary_event_source(event_source):
        return None
    squad_features = (squad_features_by_match_id or {}).get(match_id, {})

    base_features = build_common_features(document, match_timestamp)
    base_features.update(history_features)
    base_features.update(squad_features)
    base_features["interaction_history_elo_x_favorite_gap"] = float(base_features.get("history_delta_elo", 0.0)) * float(
        base_features.get("odds_match_result_favorite_gap", 0.0)
    )
    base_features["interaction_history_pi_x_favorite_gap"] = float(base_features.get("history_delta_pi_home_away", 0.0)) * float(
        base_features.get("odds_match_result_favorite_gap", 0.0)
    )
    base_features["interaction_history_rest_x_matchday"] = float(base_features.get("history_delta_rest_days", 0.0)) * float(
        base_features.get("context_matchday", 0.0)
    )
    base_features["interaction_squad_rating_x_favorite_gap"] = float(base_features.get("squad_delta_starters_overall_rating_mean", 0.0)) * float(
        base_features.get("odds_match_result_favorite_gap", 0.0)
    )
    base_features["interaction_squad_continuity_x_history_pi"] = float(base_features.get("squad_delta_lineup_continuity_share", 0.0)) * float(
        base_features.get("history_delta_pi_home_away", 0.0)
    )
    base_features["interaction_squad_depth_x_embedded_form"] = float(base_features.get("squad_delta_bench_overall_rating_mean", 0.0)) * float(
        base_features.get("embedded_recent_form_delta_ppm", 0.0)
    )

    feature_map: dict[str, float | str] = dict(base_features)
    live_features = build_live_prefix_features(document, cutoff)
    feature_map.update(live_features)
    feature_map[f"interaction_live_{cutoff}_elo_x_score_diff"] = float(feature_map.get("history_delta_elo", 0.0)) * float(
        feature_map.get(f"live_{cutoff}_score_diff", 0.0)
    )
    feature_map[f"interaction_live_{cutoff}_pi_x_score_diff"] = float(feature_map.get("history_delta_pi_home_away", 0.0)) * float(
        feature_map.get(f"live_{cutoff}_score_diff", 0.0)
    )
    feature_map[f"interaction_live_{cutoff}_favorite_gap_x_score_diff"] = float(
        feature_map.get("odds_match_result_favorite_gap", 0.0)
    ) * float(feature_map.get(f"live_{cutoff}_score_diff", 0.0))
    feature_map[f"interaction_live_{cutoff}_manpower_x_pi"] = float(feature_map.get(f"live_{cutoff}_manpower_edge_home", 0.0)) * float(
        feature_map.get("history_delta_pi_home_away", 0.0)
    )

    home_goals, away_goals = final_score
    total_goals = home_goals + away_goals
    return {
        "match_id": match_id,
        "match_timestamp": match_timestamp,
        "cutoff": int(cutoff),
        "event_source": event_source,
        "target_1x2": final_result_label(home_goals, away_goals),
        "target_total_goals": int(total_goals),
        "target_goal_bucket": goal_count_bucket_label(total_goals),
        **build_goal_targets(home_goals, away_goals),
        "features": feature_map,
    }


def build_dataset_rows_from_documents(
    documents: Iterable[dict[str, Any]],
    *,
    cutoff: int,
    history_features_by_match_id: dict[str, dict[str, float]],
    squad_features_by_match_id: dict[str, dict[str, float]] | None = None,
    max_workers: int = 1,
    allow_fallback_events: bool = DEFAULT_ALLOW_FALLBACK_EVENTS,
) -> list[dict[str, Any]]:
    worker_count = normalize_max_workers(max_workers)
    rows: list[dict[str, Any]] = []
    chunk_size = max(MONGO_BATCH_SIZE, worker_count * 16)

    for document_chunk in chunked_iterable(documents, chunk_size):
        if worker_count <= 1 or len(document_chunk) <= 1:
            for document in document_chunk:
                row = dataset_row_from_document(
                    document,
                    cutoff=cutoff,
                    history_features_by_match_id=history_features_by_match_id,
                    squad_features_by_match_id=squad_features_by_match_id,
                    allow_fallback_events=allow_fallback_events,
                )
                if row is not None:
                    rows.append(row)
            continue

        ordered_rows: list[dict[str, Any] | None] = [None] * len(document_chunk)
        chunk_workers = min(worker_count, len(document_chunk))
        with concurrent.futures.ThreadPoolExecutor(max_workers=chunk_workers) as executor:
            futures = {
                executor.submit(
                    dataset_row_from_document,
                    document,
                    cutoff=cutoff,
                    history_features_by_match_id=history_features_by_match_id,
                    squad_features_by_match_id=squad_features_by_match_id,
                    allow_fallback_events=allow_fallback_events,
                ): index
                for index, document in enumerate(document_chunk)
            }
            for future in concurrent.futures.as_completed(futures):
                ordered_rows[futures[future]] = future.result()
        rows.extend(row for row in ordered_rows if row is not None)
    return rows


def build_dataset_rows(
    collection: Collection,
    *,
    cutoff: int,
    history_features_by_match_id: dict[str, dict[str, float]],
    squad_features_by_match_id: dict[str, dict[str, float]] | None = None,
    max_rows: int | None = None,
    max_workers: int = 1,
    allow_fallback_events: bool = DEFAULT_ALLOW_FALLBACK_EVENTS,
) -> list[dict[str, Any]]:
    return build_dataset_rows_from_documents(
        progress_iterable(
            iter_mongo_documents(collection, projection=preparation_projection(), max_rows=max_rows),
            desc="Extracting features",
            unit="match",
        ),
        cutoff=cutoff,
        history_features_by_match_id=history_features_by_match_id,
        squad_features_by_match_id=squad_features_by_match_id,
        max_workers=max_workers,
        allow_fallback_events=allow_fallback_events,
    )


def chronological_split(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    usable = frame.dropna(subset=["match_timestamp"]).sort_values(["match_timestamp", "match_id"]).reset_index(drop=True)
    if usable.empty:
        raise RuntimeError("No rows have usable timestamps for chronological splitting.")
    split_index = int(len(usable) * (1.0 - TEST_FRACTION))
    split_index = max(1, min(split_index, len(usable) - 1))
    return usable.iloc[:split_index].copy(), usable.iloc[split_index:].copy()


def save_cutoff_dataset(
    cutoff: int,
    frame: pd.DataFrame,
    *,
    output_dir: str = OUTPUT_DIR,
    allow_fallback_events: bool = DEFAULT_ALLOW_FALLBACK_EVENTS,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    import pandas as pd

    paths = artifact_paths(cutoff, output_dir)
    frame = frame.copy()
    frame["match_timestamp"] = pd.to_datetime(frame["match_timestamp"], utc=True, errors="coerce")
    train_frame, test_frame = chronological_split(frame)
    save_prepared_frame(frame, paths["full"])
    save_prepared_frame(train_frame, paths["train"])
    save_prepared_frame(test_frame, paths["test"])
    write_manifest(
        paths["manifest"],
        {
            "prep_version": PREP_VERSION,
            "built_at": datetime.now(UTC).isoformat(),
            "cutoff": int(cutoff),
            "rows": int(len(frame)),
            "train_rows": int(len(train_frame)),
            "test_rows": int(len(test_frame)),
            "validation_fraction_within_train": VALIDATION_FRACTION_WITHIN_TRAIN,
            "schema_family": "unified_legacy_mackolik_mac_plus",
            "event_source_priority": ["opta_feeds.raw.f24", "match_data.events"],
            "event_source_policy": "f24_only" if not allow_fallback_events else "f24_then_match_data",
            "allow_fallback_events": bool(allow_fallback_events),
            "score_source_priority": ["score", "match_data.detail.full_time_score", "match_data.detail.score"],
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
    *,
    cutoff: int = DEFAULT_CUTOFF,
    output_dir: str = OUTPUT_DIR,
    force_rebuild: bool = False,
    max_rows: int | None = None,
    max_workers: int = DEFAULT_MAX_WORKERS,
    allow_fallback_events: bool = DEFAULT_ALLOW_FALLBACK_EVENTS,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    import pandas as pd

    normalized_cutoff = normalize_cutoff(cutoff)
    if max_rows is None and not force_rebuild and prepared_artifacts_exist(
        normalized_cutoff,
        output_dir,
        allow_fallback_events=allow_fallback_events,
    ):
        return load_prepared_dataset(
            normalized_cutoff,
            output_dir=output_dir,
            load_prepared=True,
            force_rebuild=False,
            allow_fallback_events=allow_fallback_events,
        )

    client, collection = load_mongo_collection()
    try:
        ensure_training_query_index(collection)
        total_documents = None
        if max_rows is None:
            total_documents = collection.count_documents(preparation_query())
        history_rows = build_history_rows(
            collection,
            max_rows=max_rows,
        )
        history_features_by_match_id = build_history_feature_map(history_rows)
        squad_features_by_match_id = build_player_lineup_feature_map(
            collection,
            max_rows=max_rows,
        )
        rows = build_dataset_rows_from_documents(
            progress_iterable(
                iter_mongo_documents(collection, projection=preparation_projection(), max_rows=max_rows),
                total=total_documents if max_rows is None else max_rows,
                desc="Extracting features",
                unit="match",
            ),
            cutoff=normalized_cutoff,
            history_features_by_match_id=history_features_by_match_id,
            squad_features_by_match_id=squad_features_by_match_id,
            max_workers=max_workers,
            allow_fallback_events=allow_fallback_events,
        )
    finally:
        client.close()

    if not rows:
        raise RuntimeError(f"No eligible rows found for cutoff {normalized_cutoff}.")
    frame = pd.DataFrame(rows)
    if tqdm is not None:
        tqdm.write(f"Writing prepared artifacts for cutoff {normalized_cutoff}")
    train_frame, test_frame = save_cutoff_dataset(
        normalized_cutoff,
        frame,
        output_dir=output_dir,
        allow_fallback_events=allow_fallback_events,
    )
    return frame, train_frame, test_frame


def load_prepared_dataset(
    cutoff: int = DEFAULT_CUTOFF,
    *,
    output_dir: str = OUTPUT_DIR,
    load_prepared: bool = True,
    force_rebuild: bool = False,
    max_rows: int | None = None,
    max_workers: int = DEFAULT_MAX_WORKERS,
    allow_fallback_events: bool = DEFAULT_ALLOW_FALLBACK_EVENTS,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    normalized_cutoff = normalize_cutoff(cutoff)
    should_rebuild = (
        force_rebuild
        or not load_prepared
        or not prepared_artifacts_exist(normalized_cutoff, output_dir, allow_fallback_events=allow_fallback_events)
    )
    if should_rebuild:
        return prepare_dataset(
            cutoff=normalized_cutoff,
            output_dir=output_dir,
            force_rebuild=True,
            max_rows=max_rows,
            max_workers=max_workers,
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
    output_dir: str = OUTPUT_DIR,
    load_prepared: bool = True,
    force_rebuild: bool = False,
    max_rows: int | None = None,
    max_workers: int = DEFAULT_MAX_WORKERS,
    allow_fallback_events: bool = DEFAULT_ALLOW_FALLBACK_EVENTS,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    normalized_cutoff = normalize_cutoff(cutoff)
    should_rebuild = (
        force_rebuild
        or not load_prepared
        or not prepared_artifacts_exist(normalized_cutoff, output_dir, allow_fallback_events=allow_fallback_events)
    )
    if should_rebuild:
        _, train_frame, test_frame = prepare_dataset(
            cutoff=normalized_cutoff,
            output_dir=output_dir,
            force_rebuild=True,
            max_rows=max_rows,
            max_workers=max_workers,
            allow_fallback_events=allow_fallback_events,
        )
        return train_frame, test_frame
    paths = artifact_paths(normalized_cutoff, output_dir)
    return (
        load_prepared_frame(paths["train"]),
        load_prepared_frame(paths["test"]),
    )


def prepare_betting_datasets(
    *,
    cutoffs: tuple[str, ...] | tuple[int, ...] = (DEFAULT_CUTOFF,),
    output_dir: str = OUTPUT_DIR,
    force_rebuild: bool = False,
    max_rows: int | None = None,
    max_workers: int = DEFAULT_MAX_WORKERS,
    allow_fallback_events: bool = DEFAULT_ALLOW_FALLBACK_EVENTS,
) -> dict[str, pd.DataFrame]:
    prepared: dict[str, pd.DataFrame] = {}
    for cutoff in cutoffs:
        normalized_cutoff = normalize_cutoff(cutoff)
        full_frame, train_frame, test_frame = load_prepared_dataset(
            normalized_cutoff,
            output_dir=output_dir,
            load_prepared=not force_rebuild,
            force_rebuild=force_rebuild,
            max_rows=max_rows,
            max_workers=max_workers,
            allow_fallback_events=allow_fallback_events,
        )
        prepared[f"{normalized_cutoff}_full"] = full_frame
        prepared[f"{normalized_cutoff}_train"] = train_frame
        prepared[f"{normalized_cutoff}_test"] = test_frame
    return prepared


def load_prepared_betting_datasets(
    cutoff: str | int,
    *,
    output_dir: str = OUTPUT_DIR,
    load_prepared: bool = True,
    force_rebuild: bool = False,
    max_rows: int | None = None,
    max_workers: int = DEFAULT_MAX_WORKERS,
    allow_fallback_events: bool = DEFAULT_ALLOW_FALLBACK_EVENTS,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    return load_prepared_dataset(
        normalize_cutoff(cutoff),
        output_dir=output_dir,
        load_prepared=load_prepared,
        force_rebuild=force_rebuild,
        max_rows=max_rows,
        max_workers=max_workers,
        allow_fallback_events=allow_fallback_events,
    )


def main() -> int:
    args = parse_args()
    full_frame, train_frame, test_frame = prepare_dataset(
        cutoff=args.cutoff,
        output_dir=args.output_dir,
        force_rebuild=args.force_rebuild,
        max_rows=args.max_rows,
        max_workers=args.max_workers,
        allow_fallback_events=args.allow_fallback_events,
    )
    print("Prepared Mackolik dataset.")
    print("Output directory:", args.output_dir)
    print("Cutoff:", normalize_cutoff(args.cutoff))
    print("Full rows:", len(full_frame))
    print("Train rows:", len(train_frame))
    print("Test rows:", len(test_frame))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
