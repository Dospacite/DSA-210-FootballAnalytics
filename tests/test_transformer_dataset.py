from __future__ import annotations

import unittest

import transformer_dataset


def make_document() -> dict:
    return {
        "match_id": "m1",
        "match_date_text": "2024-01-01T18:00:00Z",
        "home_team": {
            "team_id": "10",
            "recent_form": [{"result_class": ["W"]}],
            "coach_name": "Home Coach",
        },
        "away_team": {
            "team_id": "20",
            "recent_form": [{"result_class": ["L"]}],
            "coach_name": "Away Coach",
        },
        "competition": {"name": "Super Lig", "season_id": "2024"},
        "venue": {"capacity": 50000},
        "score": {"home": 1, "away": 0},
        "odds_markets": [
            {
                "market_name": "Maç Sonucu",
                "outcomes": [
                    {"name": "1", "odd": 1.8},
                    {"name": "X", "odd": 3.5},
                    {"name": "2", "odd": 4.2},
                ],
            }
        ],
        "match_data": {
            "home_players": [
                {"player_id": "h1", "player_name": "Home 1", "position": "Forward", "status": "starter", "stats": {}},
            ],
            "away_players": [
                {"player_id": "a1", "player_name": "Away 1", "position": "Defender", "status": "starter", "stats": {}},
            ],
        },
        "mac_page_stats": {"opta_stats": {"team_systems": ["4-2-3-1", "4-4-2"]}},
        "opta_feeds": {
            "raw": {
                "f24": {
                    "Games": {
                        "Game": {
                            "@attributes": {
                                "game_date": "2024-01-01T18:00:00Z",
                                "matchday": "12",
                            },
                            "Event": [
                                {
                                    "@attributes": {
                                        "min": "3",
                                        "sec": "12",
                                        "team_id": "10",
                                        "type_id": "13",
                                        "player_id": "h1",
                                        "player_name": "Home 1",
                                        "outcome": "1",
                                        "x": "84.0",
                                        "y": "48.0",
                                    },
                                    "Q": [{"@attributes": {"qualifier_id": "140"}}],
                                },
                                {
                                    "@attributes": {
                                        "min": "11",
                                        "sec": "5",
                                        "team_id": "10",
                                        "type_id": "16",
                                        "player_id": "h1",
                                        "player_name": "Home 1",
                                        "outcome": "1",
                                        "x": "92.0",
                                        "y": "51.0",
                                    }
                                },
                            ],
                        }
                    }
                }
            }
        },
    }


class TransformerDatasetTests(unittest.TestCase):
    def test_transformer_row_uses_f24_sequence(self) -> None:
        row = transformer_dataset.transformer_row_from_document(
            make_document(),
            cutoff=15,
            max_events=32,
            min_event_count=1,
            history_features_by_match_id={"m1": {"history_delta_elo": 25.0}},
            squad_features_by_match_id={"m1": {"squad_delta_starters_count": 0.0}},
        )
        self.assertIsNotNone(row)
        assert row is not None
        self.assertEqual(row["event_source"], "f24")
        self.assertEqual(row["match_id"], "m1@15")
        self.assertEqual(row["base_match_id"], "m1")
        self.assertEqual(row["target_1x2"], "1")
        self.assertEqual(row["target_goal_bucket"], "1")
        self.assertEqual(row["event_count"], 2)
        self.assertEqual(row["event_sequence"][0]["side"], "home")
        self.assertEqual(row["event_sequence"][0]["event_type"], "13")
        self.assertEqual(row["event_sequence"][0]["elapsed_bucket"], "0_4")
        self.assertEqual(row["event_sequence"][0]["possession_phase"], "single")
        self.assertEqual(row["event_sequence"][1]["delta_bucket"], "301_plus")
        self.assertEqual(row["event_sequence"][1]["possession_length_bucket"], "1")
        self.assertEqual(row["event_sequence"][1]["is_goal"], 1)
        self.assertEqual(row["next_event_exists"], 0)
        self.assertEqual(row["next_event_type"], "__missing__")
        self.assertIn("history_delta_elo", row["static_features"])
        self.assertIn("sequence_score_diff", row["static_features"])
        self.assertEqual(row["static_features"]["sequence_score_diff"], 1.0)
        self.assertEqual(row["static_features"]["sequence_cutoff_minute"], 15.0)


if __name__ == "__main__":
    unittest.main()
