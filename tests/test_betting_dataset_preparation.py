from __future__ import annotations

import unittest
from datetime import UTC, datetime

from betting_dataset_preparation import (
    HistoryRow,
    build_f24_prefix_features,
    build_goal_targets,
    build_history_feature_map,
    build_match_data_prefix_features,
    parse_match_date_text,
)


class BettingDatasetPreparationTests(unittest.TestCase):
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
        document = {
            "home_team": {"team_id": "home"},
            "away_team": {"team_id": "away"},
            "opta_feeds": {
                "raw": {
                    "f24": {
                        "Games": {
                            "Game": {
                                "Event": [
                                    {"@attributes": {"min": "8", "sec": "0", "team_id": "home", "type_id": "16"}},
                                    {"@attributes": {"min": "18", "sec": "0", "team_id": "away", "type_id": "16"}},
                                ]
                            }
                        }
                    }
                }
            },
        }
        features = build_f24_prefix_features(document, 15)
        self.assertEqual(features["f24_15_score_home"], 1.0)
        self.assertEqual(features["f24_15_score_away"], 0.0)
        self.assertEqual(features["f24_15_total_events"], 1.0)

    def test_match_data_prefix_respects_cutoff(self) -> None:
        document = {
            "match_data": {
                "events": [
                    {"minute": 4, "second": 30, "team_side": 1, "event_type": "goal"},
                    {"minute": 16, "second": 5, "team_side": 2, "event_type": "goal"},
                ]
            }
        }
        features = build_match_data_prefix_features(document, 15)
        self.assertEqual(features["md_15_score_home"], 1.0)
        self.assertEqual(features["md_15_score_away"], 0.0)
        self.assertEqual(features["md_15_total_events"], 1.0)

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


if __name__ == "__main__":
    unittest.main()
