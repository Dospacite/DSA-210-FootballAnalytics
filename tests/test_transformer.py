from __future__ import annotations

import tempfile
import unittest
from argparse import Namespace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from unittest import mock

import pandas as pd

import transformer


def make_sequence_frame(rows: int, *, start_index: int = 0) -> pd.DataFrame:
    result_labels = ["1", "X", "2"]
    records: list[dict[str, object]] = []
    base_time = datetime(2024, 1, 1, tzinfo=UTC)
    for index in range(rows):
        absolute_index = start_index + index
        result_label = result_labels[absolute_index % len(result_labels)]
        total_goals = absolute_index % 6
        goal_label = "4+" if total_goals >= 4 else str(total_goals)
        sequence = [
            {
                "event_index": 0,
                "minute": 2,
                "second": 10,
                "total_seconds": 130,
                "delta_seconds": 130,
                "elapsed_bucket": "0_4",
                "delta_bucket": "121_300",
                "minute_progress": 130.0 / 5400.0,
                "side": "home" if absolute_index % 2 == 0 else "away",
                "event_type": "13",
                "outcome": "1",
                "player_token": f"p{absolute_index}",
                "player_bucket": (absolute_index % 127) + 1,
                "x": 82.0,
                "y": 50.0,
                "qualifiers": ["140"],
                "subtype": "__missing__",
                "is_goal": 0,
                "is_shot": 1,
                "is_card": 0,
                "is_substitution": 0,
            },
            {
                "event_index": 1,
                "minute": 9,
                "second": 5,
                "total_seconds": 545,
                "delta_seconds": 415,
                "elapsed_bucket": "5_9",
                "delta_bucket": "301_plus",
                "minute_progress": 545.0 / 5400.0,
                "side": "home" if result_label == "1" else "away" if result_label == "2" else "home",
                "event_type": "16" if goal_label in {"1", "2", "3", "4+"} else "15",
                "outcome": "1",
                "player_token": f"p{absolute_index}_2",
                "player_bucket": ((absolute_index + 13) % 127) + 1,
                "x": 90.0,
                "y": 48.0,
                "qualifiers": ["140", "214"],
                "subtype": "__missing__",
                "is_goal": 1 if goal_label in {"1", "2", "3", "4+"} else 0,
                "is_shot": 1,
                "is_card": 0,
                "is_substitution": 0,
            },
        ]
        records.extend(
            [
                {
                    "match_id": f"m{absolute_index}@10",
                    "base_match_id": f"m{absolute_index}",
                    "match_timestamp": base_time + timedelta(days=absolute_index),
                    "cutoff": 10,
                    "event_source": "f24",
                    "event_count": len(sequence),
                    "static_features": {
                        "context_competition_name": "Super Lig" if absolute_index % 2 == 0 else "Premier League",
                        "context_competition_season_id": "2025",
                        "context_matchday": float((absolute_index % 34) + 1),
                        "odds_match_result_favorite_gap": 0.2 + (absolute_index % 4) * 0.03,
                        "history_delta_elo": float(((absolute_index % 5) - 2) * 35.0),
                        "squad_home_formation": "4-2-3-1",
                        "squad_away_formation": "4-4-2",
                        "sequence_score_diff": float(1 if result_label == "1" else -1 if result_label == "2" else 0),
                        "sequence_source_is_f24": 1.0,
                        "sequence_cutoff_minute": 10.0,
                        "sequence_cutoff_progress": 10.0 / 90.0,
                    },
                    "event_sequence": sequence,
                    "next_event_exists": 1,
                    "next_event_type": "16",
                    "next_event_side": "home" if absolute_index % 2 == 0 else "away",
                    "next_event_time_bucket": "61_120",
                    "target_1x2": result_label,
                    "target_total_goals": total_goals,
                    "target_goal_bucket": goal_label,
                    "target_over_1_5": int((absolute_index % 6) > 1),
                    "target_over_2_5": int((absolute_index % 6) > 2),
                    "target_over_3_5": int((absolute_index % 6) > 3),
                    "target_over_4_5": int((absolute_index % 6) > 4),
                },
                {
                    "match_id": f"m{absolute_index}@15",
                    "base_match_id": f"m{absolute_index}",
                    "match_timestamp": base_time + timedelta(days=absolute_index),
                    "cutoff": 15,
                    "event_source": "f24",
                    "event_count": len(sequence),
                    "static_features": {
                        "context_competition_name": "Super Lig" if absolute_index % 2 == 0 else "Premier League",
                        "context_competition_season_id": "2025",
                        "context_matchday": float((absolute_index % 34) + 1),
                        "odds_match_result_favorite_gap": 0.2 + (absolute_index % 4) * 0.03,
                        "history_delta_elo": float(((absolute_index % 5) - 2) * 35.0),
                        "squad_home_formation": "4-2-3-1",
                        "squad_away_formation": "4-4-2",
                        "sequence_score_diff": float(1 if result_label == "1" else -1 if result_label == "2" else 0),
                        "sequence_source_is_f24": 1.0,
                        "sequence_cutoff_minute": 15.0,
                        "sequence_cutoff_progress": 15.0 / 90.0,
                    },
                    "event_sequence": sequence,
                    "next_event_exists": 0,
                    "next_event_type": "__missing__",
                    "next_event_side": "__missing__",
                    "next_event_time_bucket": "__missing__",
                    "target_1x2": result_label,
                    "target_total_goals": total_goals,
                    "target_goal_bucket": goal_label,
                    "target_over_1_5": int((absolute_index % 6) > 1),
                    "target_over_2_5": int((absolute_index % 6) > 2),
                    "target_over_3_5": int((absolute_index % 6) > 3),
                    "target_over_4_5": int((absolute_index % 6) > 4),
                },
            ]
        )
    return pd.DataFrame(records)


class TransformerTests(unittest.TestCase):
    def test_run_experiment_writes_artifacts(self) -> None:
        train_frame = make_sequence_frame(36)
        test_frame = make_sequence_frame(12, start_index=36)
        args = Namespace(
            cutoff=15,
            artifacts_dir="",
            predictor_artifacts_dir="",
            max_rows=None,
            force_rebuild=False,
            load_prepared=True,
            max_events=32,
            min_cutoff=10,
            cutoff_step=5,
            batch_size=8,
            epochs=1,
            model_dim=32,
            num_heads=4,
            num_layers=1,
            dropout=0.1,
            learning_rate=1e-3,
            weight_decay=1e-4,
            auxiliary_weight=0.1,
            compare_tabular=True,
            allow_fallback_events=False,
            device="cpu",
        )
        with tempfile.TemporaryDirectory() as temp_dir, mock.patch.object(
            transformer,
            "load_prepared_splits",
            return_value=(train_frame, test_frame),
        ):
            args.artifacts_dir = temp_dir
            args.predictor_artifacts_dir = temp_dir
            predictor_dir = Path(temp_dir) / "cutoff_15"
            predictor_dir.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(
                [
                    {
                        "task": "1x2",
                        "model": "catboost_prematch_only",
                        "family": "catboost",
                        "feature_view": "prematch_only",
                        "test_accuracy": 0.51,
                        "test_macro_f1": 0.49,
                        "test_log_loss": 0.98,
                        "test_rps": 0.21,
                    },
                    {
                        "task": "goal_bucket",
                        "model": "catboost_prematch_only",
                        "family": "catboost",
                        "feature_view": "prematch_only",
                        "test_accuracy": 0.34,
                        "test_macro_f1": 0.30,
                        "test_log_loss": 1.47,
                        "test_rps": 0.14,
                    },
                ]
            ).to_csv(predictor_dir / "benchmark_summary.csv", index=False)
            outcome = transformer.run_experiment(args)

        self.assertEqual(outcome["cutoff"], 15)
        self.assertIn("1x2", outcome["test"])
        self.assertIn("goal_bucket", outcome["test"])
        self.assertIn("losses", outcome["test"])
        self.assertIn("f24", outcome["test_by_source"])
        self.assertIn("rows", outcome["tabular_comparison"])
        self.assertTrue(outcome["artifacts"]["metrics_json"].endswith("metrics.json"))
        self.assertTrue(outcome["artifacts"]["checkpoint"].endswith("model.pt"))
        self.assertTrue(outcome["artifacts"]["tabular_comparison_json"].endswith("tabular_comparison.json"))


if __name__ == "__main__":
    unittest.main()
