from __future__ import annotations

import tempfile
import unittest
from datetime import UTC, datetime, timedelta

import pandas as pd

from dataset import (
    HistoryRow,
    build_dataset_rows_from_documents,
    build_f24_prefix_features,
    build_goal_targets,
    build_history_feature_map,
    build_history_rows_from_documents,
    build_live_prefix_features,
    build_match_data_prefix_features,
    build_player_lineup_feature_map_from_documents,
    build_squad_static_features,
    chronological_split,
    goal_count_bucket_label,
    normalize_match_data_events,
    parse_match_date_text,
    save_cutoff_dataset,
)


def make_document(
    match_id: str,
    *,
    match_timestamp: datetime,
    home_team_id: str = "A",
    away_team_id: str = "B",
    score: tuple[int, int] = (1, 0),
    matchday: int | None = None,
    f24_events: list[dict[str, str]] | None = None,
    match_data_events: list[dict[str, object]] | None = None,
    home_players: list[dict[str, object]] | None = None,
    away_players: list[dict[str, object]] | None = None,
    home_coach: str = "Coach Home",
    away_coach: str = "Coach Away",
    top_performers: dict[str, object] | None = None,
    team_systems: list[str] | None = None,
) -> dict[str, object]:
    f24_payload = None
    if f24_events is not None:
        f24_payload = {
            "Games": {
                "Game": {
                    "@attributes": {
                        "game_date": match_timestamp.isoformat(),
                        "matchday": str(matchday) if matchday is not None else None,
                    },
                    "Event": [{"@attributes": event} for event in f24_events],
                }
            }
        }
    return {
        "match_id": match_id,
        "fetched_at": match_timestamp.isoformat(),
        "match_date_text": match_timestamp.strftime("%d Ağustos %Y"),
        "competition": {"name": "Super Lig", "season_id": "2025"},
        "home_team": {"team_id": home_team_id, "name": f"Home {home_team_id}", "recent_form": [], "coach_name": home_coach},
        "away_team": {"team_id": away_team_id, "name": f"Away {away_team_id}", "recent_form": [], "coach_name": away_coach},
        "score": {"home": score[0], "away": score[1]},
        "venue": {"capacity_text": "41000"},
        "odds_markets": [
            {
                "market_name": "Maç Sonucu",
                "outcomes": [
                    {"name": "1", "odd": 1.8},
                    {"name": "X", "odd": 3.5},
                    {"name": "2", "odd": 4.5},
                ],
            }
        ],
        "match_data": {
            "detail": {"full_time_score": f"{score[0]}-{score[1]}", "score": f"{score[0]}-{score[1]}"},
            "home_players": home_players or [],
            "away_players": away_players or [],
            "events": match_data_events or [],
        },
        "opta_feeds": {"raw": {"f24": f24_payload}},
        "top_performers": top_performers or {},
        "mac_page_stats": {"opta_stats": {"team_systems": team_systems or []}},
    }


class DatasetTests(unittest.TestCase):
    def test_parse_match_date_text_parses_turkish_date(self) -> None:
        parsed = parse_match_date_text("15 Ağustos 2025 Cuma")
        self.assertEqual(parsed, datetime(2025, 8, 15, tzinfo=UTC))

    def test_build_goal_targets_uses_total_goals_thresholds(self) -> None:
        targets = build_goal_targets(2, 1)
        self.assertEqual(targets["target_over_1_5"], 1)
        self.assertEqual(targets["target_over_2_5"], 1)
        self.assertEqual(targets["target_over_3_5"], 0)
        self.assertEqual(targets["target_over_4_5"], 0)

    def test_f24_prefix_respects_cutoff(self) -> None:
        document = make_document(
            "m1",
            match_timestamp=datetime(2024, 8, 15, tzinfo=UTC),
            f24_events=[
                {"min": "8", "sec": "0", "team_id": "A", "type_id": "16"},
                {"min": "18", "sec": "0", "team_id": "B", "type_id": "16"},
            ],
        )
        features = build_f24_prefix_features(document, 15)
        self.assertEqual(features["f24_15_score_home"], 1.0)
        self.assertEqual(features["f24_15_score_away"], 0.0)
        self.assertEqual(features["f24_15_total_events"], 1.0)

    def test_match_data_prefix_respects_cutoff(self) -> None:
        document = make_document(
            "m1",
            match_timestamp=datetime(2024, 8, 15, tzinfo=UTC),
            f24_events=None,
            match_data_events=[
                {"minute": 4, "second": 30, "team_side": 1, "event_type": "goal"},
                {"minute": 16, "second": 5, "team_side": 2, "event_type": "goal"},
            ],
        )
        features = build_match_data_prefix_features(document, 15)
        self.assertEqual(features["md_15_score_home"], 1.0)
        self.assertEqual(features["md_15_score_away"], 0.0)
        self.assertEqual(features["md_15_total_events"], 1.0)

    def test_match_data_normalization_reads_nested_second_and_player_fields(self) -> None:
        document = make_document(
            "m1",
            match_timestamp=datetime(2024, 8, 15, tzinfo=UTC),
            f24_events=None,
            match_data_events=[
                {
                    "minute": 4,
                    "team_side": 1,
                    "player_id": 77,
                    "player_name": "Player One",
                    "event_type": "goal",
                    "details": {"second": 17, "subtype": "open-play"},
                }
            ],
        )
        normalized = normalize_match_data_events(document)
        self.assertEqual(normalized[0]["second"], 17)
        self.assertEqual(normalized[0]["player_id"], "77")
        self.assertEqual(normalized[0]["player_name"], "Player One")

    def test_live_features_prefer_f24_when_present(self) -> None:
        document = make_document(
            "m1",
            match_timestamp=datetime(2024, 8, 15, tzinfo=UTC),
            f24_events=[
                {"min": "7", "sec": "0", "team_id": "A", "type_id": "16"},
            ],
            match_data_events=[
                {"minute": 3, "second": 0, "team_side": 2, "event_type": "goal"},
            ],
        )
        features = build_live_prefix_features(document, 15)
        self.assertEqual(features["live_15_source_is_f24"], 1.0)
        self.assertEqual(features["live_15_score_home"], 1.0)
        self.assertEqual(features["live_15_score_away"], 0.0)
        self.assertIn("live_15_home_shot_xg_proxy_sum", features)
        self.assertIn("live_15_home_threat_proxy_sum", features)

    def test_live_features_capture_manpower_and_substitution_state(self) -> None:
        document = make_document(
            "m1",
            match_timestamp=datetime(2024, 8, 15, tzinfo=UTC),
            f24_events=None,
            match_data_events=[
                {"minute": 5, "second": 0, "team_side": 1, "player_id": 9, "player_name": "A9", "event_type": "goal"},
                {
                    "minute": 9,
                    "second": 30,
                    "team_side": 2,
                    "player_id": 4,
                    "player_name": "B4",
                    "event_type": "red-card",
                    "details": {"subtype": "red-card"},
                },
                {
                    "minute": 13,
                    "second": 0,
                    "team_side": 1,
                    "player_id": 14,
                    "player_name": "A14",
                    "event_type": "substitution",
                    "details": {"subtype": "substitution"},
                },
            ],
        )
        features = build_live_prefix_features(document, 15)
        self.assertEqual(features["live_15_manpower_edge_home"], 1.0)
        self.assertEqual(features["live_15_home_substitution_count"], 1.0)
        self.assertEqual(features["live_15_away_red_card_count"], 1.0)
        self.assertEqual(features["live_15_home_unique_players_involved"], 2.0)

    def test_squad_static_features_capture_lineup_shape(self) -> None:
        document = make_document(
            "m1",
            match_timestamp=datetime(2024, 8, 15, tzinfo=UTC),
            home_players=[
                {"player_id": 1, "position": "Goalkeeper", "status": "Start"},
                {"player_id": 2, "position": "Defender", "status": "Start"},
                {"player_id": 3, "position": "Striker", "status": "Substitute"},
            ],
            away_players=[
                {"player_id": 4, "position": "Goalkeeper", "status": "Start"},
                {"player_id": 5, "position": "Midfielder", "status": "Start"},
            ],
            team_systems=["4-2-3-1", "4-3-3"],
        )
        features = build_squad_static_features(document)
        self.assertEqual(features["squad_home_players_count"], 3.0)
        self.assertEqual(features["squad_home_starter_defender_count"], 1.0)
        self.assertEqual(features["squad_away_starter_midfielder_count"], 1.0)
        self.assertEqual(features["squad_home_formation"], "4-2-3-1")
        self.assertEqual(features["squad_home_coach_name"], "Coach Home")

    def test_history_features_use_only_prior_matches(self) -> None:
        rows = [
            HistoryRow(
                match_id="m1",
                match_timestamp=datetime(2024, 1, 1, tzinfo=UTC),
                home_team_id="A",
                away_team_id="B",
                home_goals=2,
                away_goals=0,
                matchday=1,
            ),
            HistoryRow(
                match_id="m2",
                match_timestamp=datetime(2024, 1, 8, tzinfo=UTC),
                home_team_id="A",
                away_team_id="B",
                home_goals=1,
                away_goals=1,
                matchday=2,
            ),
        ]
        feature_map = build_history_feature_map(rows)
        first_match = feature_map["m1"]
        second_match = feature_map["m2"]

        self.assertEqual(first_match["history_home_overall_last5_count"], 0.0)
        self.assertEqual(first_match["history_away_overall_last5_count"], 0.0)
        self.assertEqual(second_match["history_home_overall_last5_count"], 1.0)
        self.assertEqual(second_match["history_away_overall_last5_count"], 1.0)
        self.assertEqual(second_match["history_home_overall_last5_ppm"], 3.0)
        self.assertEqual(second_match["history_away_overall_last5_ppm"], 0.0)
        self.assertGreater(second_match["history_home_pi_rating_home"], 0.0)
        self.assertLess(second_match["history_away_pi_rating_away"], 0.0)

    def test_player_lineup_history_uses_only_prior_player_form(self) -> None:
        first = make_document(
            "m1",
            match_timestamp=datetime(2024, 8, 1, tzinfo=UTC),
            home_team_id="A",
            away_team_id="B",
            home_players=[
                {"player_id": 10, "player_name": "H10", "position": "Striker", "status": "Start", "minutes_played": 90, "stats": {"goals": 1, "total_scoring_att": 3}},
                {"player_id": 11, "player_name": "H11", "position": "Midfielder", "status": "Substitute", "minutes_played": 20, "stats": {"accurate_pass": 12}},
            ],
            away_players=[
                {"player_id": 20, "player_name": "A20", "position": "Defender", "status": "Start", "minutes_played": 90, "stats": {"total_tackle": 4}},
            ],
            top_performers={"ShotList": [{"OYUNCU_ID": 10}]},
        )
        second = make_document(
            "m2",
            match_timestamp=datetime(2024, 8, 8, tzinfo=UTC),
            home_team_id="A",
            away_team_id="C",
            home_players=[
                {"player_id": 10, "player_name": "H10", "position": "Striker", "status": "Start", "minutes_played": 85, "stats": {"goals": 0, "total_scoring_att": 2}},
                {"player_id": 12, "player_name": "H12", "position": "Midfielder", "status": "Start", "minutes_played": 90, "stats": {"accurate_pass": 30}},
            ],
            away_players=[
                {"player_id": 30, "player_name": "C30", "position": "Goalkeeper", "status": "Start", "minutes_played": 90, "stats": {"accurate_pass": 18}},
            ],
        )
        feature_map = build_player_lineup_feature_map_from_documents([first, second])
        self.assertEqual(feature_map["m1"]["squad_home_starters_known_count"], 0.0)
        self.assertGreater(feature_map["m2"]["squad_home_starters_known_count"], 0.0)
        self.assertGreater(feature_map["m2"]["squad_home_starters_overall_rating_mean"], 0.0)
        self.assertGreater(feature_map["m2"]["squad_home_lineup_continuity_share"], 0.0)

    def test_dataset_row_uses_prior_history_and_goal_bucket_mapping(self) -> None:
        first = make_document(
            "m1",
            match_timestamp=datetime(2024, 8, 1, tzinfo=UTC),
            home_team_id="A",
            away_team_id="B",
            score=(2, 1),
            matchday=1,
            home_players=[
                {"player_id": 10, "player_name": "H10", "position": "Striker", "status": "Start", "minutes_played": 90, "stats": {"goals": 1, "total_scoring_att": 3}},
            ],
            away_players=[
                {"player_id": 20, "player_name": "A20", "position": "Defender", "status": "Start", "minutes_played": 90, "stats": {"total_tackle": 4}},
            ],
            top_performers={"ShotList": [{"OYUNCU_ID": 10}]},
            match_data_events=[{"minute": 5, "second": 0, "team_side": 1, "event_type": "goal"}],
        )
        second = make_document(
            "m2",
            match_timestamp=datetime(2024, 8, 8, tzinfo=UTC),
            home_team_id="A",
            away_team_id="C",
            score=(4, 2),
            matchday=2,
            home_players=[
                {"player_id": 10, "player_name": "H10", "position": "Striker", "status": "Start", "minutes_played": 85, "stats": {"goals": 0, "total_scoring_att": 2}},
                {"player_id": 12, "player_name": "H12", "position": "Midfielder", "status": "Start", "minutes_played": 90, "stats": {"accurate_pass": 30}},
            ],
            away_players=[
                {"player_id": 30, "player_name": "C30", "position": "Goalkeeper", "status": "Start", "minutes_played": 90, "stats": {"accurate_pass": 18}},
            ],
            match_data_events=[{"minute": 12, "second": 0, "team_side": 2, "event_type": "goal"}],
        )
        history_rows = build_history_rows_from_documents([first, second])
        feature_map = build_history_feature_map(history_rows)
        squad_feature_map = build_player_lineup_feature_map_from_documents([first, second])
        rows = build_dataset_rows_from_documents(
            [first, second],
            cutoff=15,
            history_features_by_match_id=feature_map,
            squad_features_by_match_id=squad_feature_map,
            allow_fallback_events=True,
        )

        self.assertEqual(rows[0]["target_goal_bucket"], "3")
        self.assertEqual(rows[1]["target_goal_bucket"], "4+")
        self.assertEqual(rows[1]["target_total_goals"], 6)
        self.assertEqual(rows[1]["event_source"], "match_data")
        self.assertEqual(rows[1]["features"]["history_home_overall_last5_count"], 1.0)
        self.assertEqual(rows[1]["features"]["live_15_source_is_match_data"], 1.0)
        self.assertIn("history_home_overall_exp_ppm", rows[1]["features"])
        self.assertIn("live_15_shot_xg_proxy_diff", rows[1]["features"])
        self.assertIn("squad_delta_starters_overall_rating_mean", rows[1]["features"])
        self.assertIn("interaction_squad_rating_x_favorite_gap", rows[1]["features"])

    def test_goal_count_bucket_caps_high_scores(self) -> None:
        self.assertEqual(goal_count_bucket_label(0), "0")
        self.assertEqual(goal_count_bucket_label(3), "3")
        self.assertEqual(goal_count_bucket_label(4), "4+")
        self.assertEqual(goal_count_bucket_label(8), "4+")

    def test_dataset_rows_default_to_f24_only(self) -> None:
        document = make_document(
            "fallback-only",
            match_timestamp=datetime(2024, 8, 15, tzinfo=UTC),
            f24_events=None,
            match_data_events=[{"minute": 5, "second": 0, "team_side": 1, "event_type": "goal"}],
        )
        history_rows = build_history_rows_from_documents([document])
        feature_map = build_history_feature_map(history_rows)
        rows = build_dataset_rows_from_documents(
            [document],
            cutoff=15,
            history_features_by_match_id=feature_map,
            squad_features_by_match_id={},
        )
        self.assertEqual(rows, [])

    def test_chronological_split_is_stable(self) -> None:
        frame = pd.DataFrame(
            [
                {"match_id": "b", "match_timestamp": datetime(2024, 1, 2, tzinfo=UTC)},
                {"match_id": "a", "match_timestamp": datetime(2024, 1, 1, tzinfo=UTC)},
                {"match_id": "c", "match_timestamp": datetime(2024, 1, 3, tzinfo=UTC)},
                {"match_id": "d", "match_timestamp": datetime(2024, 1, 4, tzinfo=UTC)},
                {"match_id": "e", "match_timestamp": datetime(2024, 1, 5, tzinfo=UTC)},
            ]
        )
        train_frame, test_frame = chronological_split(frame)
        self.assertEqual(train_frame["match_id"].tolist(), ["a", "b", "c", "d"])
        self.assertEqual(test_frame["match_id"].tolist(), ["e"])

    def test_save_cutoff_dataset_writes_manifest(self) -> None:
        rows = [
            {
                "match_id": f"m{index}",
                "match_timestamp": datetime(2024, 1, 1, tzinfo=UTC) + timedelta(days=index),
                "cutoff": 15,
                "target_1x2": "1",
                "target_total_goals": 2,
                "target_goal_bucket": "2",
                "target_over_1_5": 1,
                "target_over_2_5": 0,
                "target_over_3_5": 0,
                "target_over_4_5": 0,
                "features": {"context_competition_name": "League"},
            }
            for index in range(5)
        ]
        frame = pd.DataFrame(rows)
        with tempfile.TemporaryDirectory() as temp_dir:
            train_frame, test_frame = save_cutoff_dataset(15, frame, output_dir=temp_dir)
            self.assertEqual(len(train_frame), 4)
            self.assertEqual(len(test_frame), 1)


if __name__ == "__main__":
    unittest.main()
