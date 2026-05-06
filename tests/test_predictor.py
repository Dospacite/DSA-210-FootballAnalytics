from __future__ import annotations

import tempfile
import unittest
from datetime import UTC, datetime, timedelta
from unittest import mock

import pandas as pd

import predictor


def make_prepared_frame(rows: int, *, start_index: int = 0, fixed_result: str | None = None) -> pd.DataFrame:
    result_labels = ["1", "X", "2"]
    records: list[dict[str, object]] = []
    base_time = datetime(2024, 1, 1, tzinfo=UTC)
    for index in range(rows):
        absolute_index = start_index + index
        result_label = fixed_result or result_labels[absolute_index % len(result_labels)]
        total_goals = absolute_index % 6
        goal_label = predictor.goal_count_bucket_label(total_goals)
        score_diff = (absolute_index % 3) - 1
        favorite_gap = 0.15 + (absolute_index % 5) * 0.03
        history_elo = 25.0 * score_diff
        records.append(
            {
                "match_id": f"m{absolute_index}",
                "match_timestamp": base_time + timedelta(days=absolute_index),
                "cutoff": 15,
                "event_source": "f24" if absolute_index % 2 == 0 else "match_data",
                "target_1x2": result_label,
                "target_total_goals": total_goals,
                "target_goal_bucket": goal_label,
                "target_over_1_5": int(total_goals > 1),
                "target_over_2_5": int(total_goals > 2),
                "target_over_3_5": int(total_goals > 3),
                "target_over_4_5": int(total_goals > 4),
                "features": {
                    "context_competition_name": "Super Lig" if absolute_index % 2 == 0 else "Premier League",
                    "context_competition_season_id": "2025",
                    "context_matchday": float((absolute_index % 34) + 1),
                    "context_kickoff_hour": float(12 + (absolute_index % 8)),
                    "odds_match_result_favorite_gap": favorite_gap,
                    "odds_match_result_implied_home": 0.5 + (score_diff * 0.08),
                    "odds_match_result_implied_draw": 0.24 - (score_diff * 0.02),
                    "odds_match_result_implied_away": 0.26 - (score_diff * 0.06),
                    "odds_total_2_5_over_implied": 0.30 + total_goals * 0.08,
                    "history_delta_elo": history_elo,
                    "history_delta_rest_days": float((absolute_index % 6) - 2),
                    "history_home_overall_last5_ppm": float((absolute_index % 5) + 0.5),
                    "history_away_overall_last5_ppm": float((4 - (absolute_index % 5)) + 0.5),
                    "embedded_recent_form_delta_ppm": float(score_diff),
                    "embedded_home_recent_form_win_streak": float(absolute_index % 3),
                    "embedded_away_recent_form_win_streak": float((absolute_index + 1) % 3),
                    "squad_home_formation": "4-2-3-1" if absolute_index % 2 == 0 else "4-3-3",
                    "squad_away_formation": "4-4-2",
                    "squad_home_coach_name": "Coach Home",
                    "squad_away_coach_name": "Coach Away",
                    "squad_delta_starters_overall_rating_mean": float(score_diff * 1.25),
                    "squad_delta_lineup_continuity_share": float((absolute_index % 4) / 4.0),
                    "squad_delta_bench_overall_rating_mean": float(score_diff * 0.5),
                    "live_15_score_diff": float(score_diff),
                    "live_15_score_total": float(min(total_goals, 3)),
                    "live_15_shot_proxy_diff": float(score_diff * 2),
                    "live_15_box_touch_proxy_diff": float(score_diff),
                    "live_15_card_proxy_diff": float((absolute_index % 2) - 0.5),
                    "live_15_recent_5m_event_diff": float(score_diff),
                    "live_15_source_is_f24": 1.0 if absolute_index % 2 == 0 else 0.0,
                    "live_15_source_is_match_data": 0.0 if absolute_index % 2 == 0 else 1.0,
                    "interaction_embedded_form_x_favorite_gap": float(score_diff) * favorite_gap,
                    "interaction_history_elo_x_favorite_gap": history_elo * favorite_gap,
                    "interaction_history_rest_x_matchday": float((absolute_index % 6) - 2) * float((absolute_index % 34) + 1),
                    "interaction_squad_rating_x_favorite_gap": float(score_diff * 1.25) * favorite_gap,
                    "interaction_squad_continuity_x_history_pi": float((absolute_index % 4) / 4.0) * history_elo,
                    "interaction_live_15_elo_x_score_diff": history_elo * float(score_diff),
                    "interaction_live_15_favorite_gap_x_score_diff": favorite_gap * float(score_diff),
                },
            }
        )
    return pd.DataFrame(records)


class PredictorTests(unittest.TestCase):
    def test_prematch_feature_view_keeps_history_and_drops_live(self) -> None:
        frame = predictor.build_modeling_frame(make_prepared_frame(6))
        columns = predictor.select_feature_columns(frame, "prematch_only")
        self.assertIn("history_delta_elo", columns)
        self.assertIn("squad_delta_starters_overall_rating_mean", columns)
        self.assertIn("interaction_squad_rating_x_favorite_gap", columns)
        self.assertIn("interaction_history_elo_x_favorite_gap", columns)
        self.assertNotIn("live_15_score_diff", columns)
        self.assertNotIn("interaction_live_15_elo_x_score_diff", columns)

    def test_rank_probability_score_is_zero_for_perfect_probabilities(self) -> None:
        y_true = predictor.encode_target(pd.Series(["1", "X", "2"]), predictor.RESULT_LABELS)
        probabilities = pd.DataFrame(
            [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ]
        ).to_numpy()
        self.assertEqual(predictor.rank_probability_score(y_true, probabilities), 0.0)

    def test_market_prior_probabilities_exist_for_1x2_only(self) -> None:
        frame = predictor.build_modeling_frame(make_prepared_frame(6))
        one_x_two_probs = predictor.market_prior_probabilities(frame, predictor.TASKS[0])
        goal_probs = predictor.market_prior_probabilities(frame, predictor.TASKS[1])
        self.assertEqual(one_x_two_probs.shape[1], 3)
        self.assertIsNone(goal_probs)

    def test_rolling_origin_splits_produces_multiple_folds(self) -> None:
        frame = predictor.build_modeling_frame(make_prepared_frame(18))
        splits = predictor.rolling_origin_splits(frame, 3)
        self.assertEqual(len(splits), 3)
        self.assertTrue(all(len(train) > 0 and len(test) > 0 for train, test in splits))

    @unittest.skipUnless(predictor.CatBoostClassifier is not None, "catboost is not installed")
    def test_run_benchmarks_writes_expected_artifacts(self) -> None:
        train_raw = make_prepared_frame(48)
        test_raw = make_prepared_frame(18, start_index=48)

        with tempfile.TemporaryDirectory() as temp_dir, mock.patch.object(
            predictor,
            "load_prepared_splits",
            return_value=(train_raw, test_raw),
        ):
            outcome = predictor.run_benchmarks(
                cutoff=15,
                load_prepared=True,
                force_rebuild=False,
                max_rows=None,
                artifacts_dir=temp_dir,
                rolling_backtest_windows=2,
            )

        self.assertEqual(outcome["cutoff"], 15)
        self.assertIn("catboost_full_hybrid", outcome["results"]["1x2"])
        self.assertIn("catboost_prematch_only", outcome["results"]["goal_bucket"])
        self.assertIn("market_prior", outcome["results"]["1x2"])
        self.assertIn("rank_probability_score", outcome["results"]["1x2"]["catboost_full_hybrid"]["test"])
        self.assertIn("f24", outcome["results"]["1x2"]["catboost_full_hybrid"]["test_by_source"])
        self.assertTrue(outcome["rolling_backtest"]["summary"])
        self.assertTrue(outcome["artifacts"]["summary_csv"].endswith("benchmark_summary.csv"))
        self.assertTrue(outcome["artifacts"]["source_cohort_summary_csv"].endswith("source_cohort_summary.csv"))
        self.assertTrue(outcome["artifacts"]["results_json"].endswith("benchmark_results.json"))
        self.assertTrue(outcome["artifacts"]["rolling_backtest_json"].endswith("rolling_backtest.json"))
        self.assertTrue(outcome["artifacts"]["analysis_report"].endswith("analysis_report.txt"))

    def test_pipeline_fails_clearly_when_class_support_is_insufficient(self) -> None:
        train_raw = make_prepared_frame(12, fixed_result="1")
        test_raw = make_prepared_frame(6, start_index=12, fixed_result="1")

        with tempfile.TemporaryDirectory() as temp_dir, mock.patch.object(
            predictor,
            "load_prepared_splits",
            return_value=(train_raw, test_raw),
        ):
            with self.assertRaisesRegex(RuntimeError, "Insufficient class support"):
                predictor.run_benchmarks(
                    cutoff=15,
                    load_prepared=True,
                    force_rebuild=False,
                    max_rows=None,
                    artifacts_dir=temp_dir,
                    rolling_backtest_windows=2,
                )


if __name__ == "__main__":
    unittest.main()
