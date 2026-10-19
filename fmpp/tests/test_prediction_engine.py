"""Focused regression tests for prediction orchestration."""

from datetime import datetime, timezone
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pandas as pd

from src.features.engineer import FeatureEngineer
from src.evaluation.backtest import walk_forward_evaluate
from src.ingestion.database import DatabaseManager, _PostgresConnection
from src.ingestion.football_data_co_uk import parse_fdcouk_csv
from src.models.poisson_model import PoissonGoalModel
from src.prediction.engine import PredictionEngine


class PredictionEngineTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        root = Path(self.temp_dir.name)
        self.database = DatabaseManager(root / "matches.db")
        self.model_path = root / "poisson_model.pkl"

    def _history(self) -> pd.DataFrame:
        rows = []
        for index in range(24):
            rows.append({
                "competition_code": "PL",
                "season": "2024-2025",
                "date_utc": f"2024-08-{index + 1:02d} 12:00:00",
                "home_team": "Home FC",
                "away_team": "Away FC",
                "home_score": index % 3,
                "away_score": (index + 1) % 2,
                "status": "FINISHED",
                "source": "test",
            })
        return pd.DataFrame(rows)

    def test_insert_matches_reports_only_new_rows(self):
        history = self._history().iloc[:1]
        self.assertEqual(self.database.insert_matches(pd.concat([history, history])), 1)
        self.assertEqual(self.database.insert_matches(history), 0)

    def test_fixture_insert_updates_existing_provider_data(self):
        fixture = pd.DataFrame([{
            "external_id": "first-id",
            "competition_code": "PL",
            "date_utc": "2026-10-20T12:00:00Z",
            "date_wib": "2026-10-20 19:00",
            "home_team": "Home FC",
            "away_team": "Away FC",
            "status": "SCHEDULED",
            "source": "first-provider",
        }])
        self.database.insert_fixtures(fixture)
        fixture.loc[0, "external_id"] = "updated-id"
        fixture.loc[0, "status"] = "TIMED"
        fixture.loc[0, "source"] = "second-provider"
        self.database.insert_fixtures(fixture)

        updated = self.database.get_fixtures_by_date("2026-10-20", "PL")
        self.assertEqual(len(updated), 1)
        self.assertEqual(updated.iloc[0]["external_id"], "updated-id")
        self.assertEqual(updated.iloc[0]["status"], "TIMED")
        self.assertEqual(updated.iloc[0]["source"], "second-provider")

    def test_sqlite_import_prepares_rows_for_hosted_postgres(self):
        history = self._history().iloc[:1].copy()
        history["matchday"] = "Preliminary Round"
        self.database.insert_matches(history)
        self.database.database_url = "postgresql://example.invalid/database"
        target = Mock()
        inserted_batches = []

        def record_batch(statement, rows):
            rows = list(rows)
            inserted_batches.append((statement, rows))
            return SimpleNamespace(rowcount=len(rows))

        target.executemany.side_effect = record_batch
        with patch.object(self.database, "_get_conn", return_value=target):
            result = self.database.import_sqlite_database(self.database.db_path)

        self.assertEqual(result["matches"], 1)
        self.assertIn("ON CONFLICT DO NOTHING", inserted_batches[2][0])
        self.assertEqual(len(inserted_batches[2][1]), 1)
        self.assertNotIn("created_at", inserted_batches[2][0])
        matchday_index = inserted_batches[2][0].split("VALUES")[0].split(", ").index(
            "matchday"
        )
        self.assertEqual(
            inserted_batches[2][1][0][matchday_index], "Preliminary Round"
        )
        self.assertEqual(target.execute.call_count, 6)
        target.commit.assert_called_once()
        target.close.assert_called_once()

    def test_postgres_adapter_translates_sqlite_placeholders(self):
        connection = Mock()
        cursor = Mock()
        connection.cursor.return_value = cursor

        _PostgresConnection(connection).execute(
            "SELECT * FROM matches WHERE home_team = ? AND date_utc < ?",
            ("Home FC", "2024-08-01"),
        )

        cursor.execute.assert_called_once_with(
            "SELECT * FROM matches WHERE home_team = %s AND date_utc < %s",
            ("Home FC", "2024-08-01"),
        )

    def test_database_detects_season_years_in_cross_year_season_labels(self):
        match = self._history().iloc[:1].copy()
        match["season"] = "2024-2025"
        self.database.insert_matches(match)

        self.assertTrue(
            self.database.has_finished_matches_for_season("PL", 2024)
        )
        self.assertTrue(
            self.database.has_finished_matches_for_season("PL", 2025)
        )
        self.assertFalse(
            self.database.has_finished_matches_for_season("PL", 2026)
        )

    def test_recent_form_excludes_matches_at_or_after_prediction_time(self):
        history = self._history()
        engineer = FeatureEngineer(history, last_n=5)
        before = datetime(2024, 8, 22, 12, tzinfo=timezone.utc)
        recent = engineer.get_team_recent_matches("Home FC", before, n=5)

        self.assertEqual(len(recent), 5)
        self.assertLess(recent["date_utc"].max(), pd.Timestamp("2024-08-22 12:00:00"))

    def test_recent_form_matches_known_team_name_alias(self):
        history = pd.DataFrame([
            {
                "date_utc": f"2024-08-{day:02d} 12:00:00",
                "home_team": "Real Madrid",
                "away_team": f"Opponent {day}",
                "home_score": 2,
                "away_score": 0,
            }
            for day in range(1, 7)
        ])
        engineer = FeatureEngineer(history, last_n=5)
        recent = engineer.get_team_recent_matches(
            "Real Madrid CF", "2024-08-07T12:00:00Z", n=5
        )

        self.assertEqual(len(recent), 5)
        self.assertEqual(engineer.compute_elo_rating("Real Madrid CF"), 1500)

    def test_recent_history_matches_provider_name_variants(self):
        matches = pd.DataFrame([
            {
                "date_utc": f"2024-08-{day:02d} 12:00:00",
                "home_team": "Union Berlin",
                "away_team": f"Opponent {day}",
                "home_score": 1,
                "away_score": 0,
            }
            for day in range(1, 7)
        ])
        self.database.insert_matches(matches)

        recent = self.database.get_team_matches(
            "1. FC Union Berlin", before_date="2024-08-07T12:00:00Z", limit=5
        )

        self.assertEqual(len(recent), 5)

    def test_prediction_uses_trained_model_and_returns_valid_probabilities(self):
        history = self._history()
        self.database.insert_matches(history)
        training = FeatureEngineer(history, last_n=5).build_training_dataset()
        model = PoissonGoalModel()
        model.train(training)
        model.score_temperature = 1.3
        model.calibration_method = "test calibration"
        model.save(self.model_path)

        result = PredictionEngine(self.database, self.model_path).predict(
            "Home FC",
            "Away FC",
            match_date="2024-09-01T12:00:00Z",
            last_n=5,
            include_score_matrix=True,
        )

        self.assertAlmostEqual(
            result["prob_home"] + result["prob_draw"] + result["prob_away"],
            1.0,
            places=3,
        )
        self.assertEqual(result["data_confidence"], "Tinggi")
        self.assertEqual(result["probability_calibration"]["temperature"], 1.3)
        self.assertAlmostEqual(sum(map(sum, result["score_matrix"])), 1.0)
        self.assertEqual(len(result["top_scores"]), 3)

    def test_same_timestamp_matches_share_prior_elo_state(self):
        rows = []
        for day in range(1, 4):
            rows.append({
                "date_utc": f"2024-08-{day:02d} 12:00:00",
                "home_team": "Home FC",
                "away_team": "Away FC",
                "home_score": day,
                "away_score": 0,
            })
        rows.extend([
            {
                "date_utc": "2024-08-04 12:00:00",
                "home_team": "Home FC",
                "away_team": "Away FC",
                "home_score": 0,
                "away_score": 3,
            },
            {
                "date_utc": "2024-08-04 12:00:00",
                "home_team": "Home FC",
                "away_team": "Away FC",
                "home_score": 3,
                "away_score": 0,
            },
        ])

        dataset = FeatureEngineer(pd.DataFrame(rows), last_n=5).build_training_dataset()
        simultaneous = dataset[dataset["_match_date"].str.startswith("2024-08-04")]

        self.assertEqual(len(simultaneous), 2)
        self.assertEqual(simultaneous["home_elo"].nunique(), 1)
        self.assertEqual(simultaneous["away_elo"].nunique(), 1)

    def test_football_data_uk_local_kickoff_is_converted_to_utc(self):
        raw = pd.DataFrame([
            {
                "Date": "01/07/2024", "Time": "15:00",
                "HomeTeam": "Man United", "AwayTeam": "Man City",
                "FTHG": 2, "FTAG": 1,
            },
            {
                "Date": "bad-date", "Time": "15:00",
                "HomeTeam": "A", "AwayTeam": "B",
                "FTHG": 0, "FTAG": 0,
            },
        ])

        parsed = parse_fdcouk_csv(raw, "PL", "2023-2024")

        self.assertEqual(len(parsed), 1)
        self.assertEqual(parsed.iloc[0]["date_utc"], "2024-07-01 14:00:00+00:00")
        self.assertEqual(parsed.iloc[0]["home_team"], "Manchester United")

    def test_walk_forward_reports_competition_metrics_against_baselines(self):
        teams = [f"Team {index}" for index in range(8)]
        rotation = teams[:]
        rows = []
        for day in range(70):
            for index in range(4):
                home, away = rotation[index], rotation[-(index + 1)]
                rows.append({
                    "competition_code": "PL",
                    "date_utc": (pd.Timestamp("2023-01-01") + pd.Timedelta(days=day)).isoformat(),
                    "home_team": home,
                    "away_team": away,
                    "home_score": (day + index) % 4,
                    "away_score": (day + index + 1) % 3,
                })
            rotation = [rotation[0], rotation[-1], *rotation[1:-1]]

        report = walk_forward_evaluate(
            pd.DataFrame(rows),
            last_n=5,
            folds=2,
            minimum_training_samples=30,
        )

        self.assertGreater(report["sample_count"], 0)
        self.assertEqual(report["folds_completed"], 2)
        competition = report["per_competition"]["PL"]
        self.assertIn("poisson", competition)
        self.assertIn("league_frequency", competition)
        self.assertIn("team_average_goals", competition)
        self.assertIn("logistic_1x2", competition)
        self.assertIn("calibration_over_25", competition["poisson"])
        self.assertGreaterEqual(report["calibration"]["temperature"], 0.5)
        self.assertLessEqual(report["calibration"]["temperature"], 2.5)


if __name__ == "__main__":
    unittest.main()
