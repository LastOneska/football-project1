"""HTTP-level smoke tests for the local FMPP API and web app."""

from datetime import datetime
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from zoneinfo import ZoneInfo

import pandas as pd
from fastapi.testclient import TestClient

from src.api.app import app
from src.ingestion.database import DatabaseManager


class WebApiTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.database = DatabaseManager(Path(self.temp_dir.name) / "web.db")
        patcher = patch("src.api.app._database", return_value=self.database)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.client = TestClient(app)

    def _fixture(self, home_team: str = "Home FC", away_team: str = "Away FC") -> int:
        kick_off = datetime.now(ZoneInfo("Asia/Jakarta")).replace(
            hour=23, minute=59, second=0, microsecond=0
        )
        self.database.insert_fixtures(pd.DataFrame([{
            "external_id": "test-fixture-1",
            "competition_code": "PL",
            "date_utc": kick_off.astimezone(ZoneInfo("UTC")).isoformat(),
            "date_wib": kick_off.strftime("%Y-%m-%d %H:%M"),
            "home_team": home_team,
            "away_team": away_team,
            "status": "SCHEDULED",
            "source": "test",
        }]))
        fixtures = self.database.get_fixtures_by_date(kick_off.date().isoformat(), "PL")
        return int(fixtures.iloc[0]["fixture_id"])

    def test_homepage_and_competition_list(self):
        page = self.client.get("/")
        competitions = self.client.get("/api/competitions")

        self.assertEqual(page.status_code, 200)
        self.assertIn("FMPP", page.text)
        self.assertEqual(competitions.status_code, 200)
        self.assertEqual(len(competitions.json()), 9)

    def test_match_list_and_unknown_day_validation(self):
        fixture_id = self._fixture()
        match_list = self.client.get("/api/matches?day=today&league=PL")
        invalid_day = self.client.get("/api/matches?day=next-week")

        self.assertEqual(match_list.status_code, 200)
        self.assertEqual(match_list.json()[0]["fixture_id"], fixture_id)
        self.assertEqual(invalid_day.status_code, 422)

    def test_prediction_detail_includes_fixture_and_recent_form(self):
        fixture_id = self._fixture()
        prediction = {
            "prob_home": 0.5,
            "prob_draw": 0.25,
            "prob_away": 0.25,
            "score_matrix": [[1.0]],
            "last_n": 5,
        }

        with patch("src.api.app.PredictionEngine") as engine:
            engine.return_value.predict.return_value = prediction
            response = self.client.get(f"/api/matches/{fixture_id}/prediction?last_n=7")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["fixture"]["home_team"], "Home FC")
        self.assertEqual(response.json()["prediction"]["score_matrix"], [[1.0]])
        self.assertIn("home", response.json()["recent_form"])
        self.assertFalse(response.json()["team_analysis"]["home"]["minimum_met"])
        self.assertEqual(
            response.json()["team_analysis"]["home"]["matches_analyzed"], 0
        )

    def test_prediction_detail_matches_team_alias_to_recent_history(self):
        fixture_id = self._fixture(home_team="Real Madrid CF")
        fixture = self.database.get_fixture(fixture_id)
        fixture_time = pd.Timestamp(fixture["date_utc"])
        rows = []
        for index in range(5):
            date_utc = (fixture_time - pd.Timedelta(days=index + 1)).isoformat()
            rows.append({
                "competition_code": "UCL",
                "date_utc": date_utc,
                "home_team": "Real Madrid",
                "away_team": f"Opponent {index}",
                "home_score": 2,
                "away_score": 0,
                "source": "test",
            })
        self.database.insert_matches(pd.DataFrame(rows))

        with patch("src.api.app.PredictionEngine") as engine:
            engine.return_value.predict.return_value = {"score_matrix": [[1.0]]}
            response = self.client.get(
                f"/api/matches/{fixture_id}/prediction?last_n=5"
            )

        self.assertEqual(response.status_code, 200)
        result = response.json()
        self.assertEqual(result["team_analysis"]["home"]["matches_analyzed"], 5)
        self.assertTrue(result["team_analysis"]["home"]["minimum_met"])
        self.assertEqual(result["team_analysis"]["home"]["wins"], 5)
        self.assertEqual(len(result["recent_form"]["home"]), 5)
        self.assertTrue(result["recent_form"]["home"][0]["team_is_home"])

    def test_teams_endpoint_includes_teams_without_upcoming_fixtures(self):
        fixture_id = self._fixture()
        fixture = self.database.get_fixture(fixture_id)
        fixture_time = pd.Timestamp(fixture["date_utc"])
        rows = []
        for team in ("Barcelona", "Juventus", "Manchester United"):
            for index in range(5):
                rows.append({
                    "competition_code": "HIST",
                    "date_utc": (
                        fixture_time - pd.Timedelta(days=index + 1)
                    ).isoformat(),
                    "home_team": team,
                    "away_team": f"{team} Opponent {index}",
                    "home_score": 2,
                    "away_score": 0,
                    "source": "test",
                })
        self.database.insert_matches(pd.DataFrame(rows))

        response = self.client.get("/api/teams?last_n=5")

        self.assertEqual(response.status_code, 200)
        result = response.json()
        teams_by_name = {team["team"]: team for team in result["teams"]}
        for name in ("Barcelona", "Juventus", "Manchester United"):
            self.assertIn(name, teams_by_name)
            self.assertEqual(
                teams_by_name[name]["team_analysis"]["matches_analyzed"], 5
            )
            self.assertEqual(len(teams_by_name[name]["recent_form"]), 5)
            self.assertTrue(teams_by_name[name]["team_analysis"]["minimum_met"])
        self.assertIn("Home FC", teams_by_name)
        self.assertEqual(
            teams_by_name["Home FC"]["team_analysis"]["matches_analyzed"], 0
        )

    def test_fixture_refresh_cron_requires_secret_and_refreshes(self):
        with (
            patch.dict(os.environ, {"CRON_SECRET": "test-secret"}),
            patch("src.api.app.refresh_upcoming_fixtures", return_value=3) as refresh,
        ):
            unauthorized = self.client.get("/api/cron/refresh")
            response = self.client.get(
                "/api/cron/refresh",
                headers={"Authorization": "Bearer test-secret"},
            )

        self.assertEqual(unauthorized.status_code, 401)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"status": "ok", "fixtures_updated": 3})
        refresh.assert_called_once_with(days=14)

    def test_fixture_refresh_cron_requires_configured_secret(self):
        with (
            patch.dict(os.environ, {}, clear=True),
            patch("src.api.app.refresh_upcoming_fixtures") as refresh,
        ):
            response = self.client.get(
                "/api/cron/refresh",
                headers={"Authorization": "Bearer any-value"},
            )

        self.assertEqual(response.status_code, 503)
        refresh.assert_not_called()

    def test_vercel_lifespan_does_not_start_background_refresh_loop(self):
        with (
            patch.dict(
                os.environ,
                {"VERCEL": "1", "DATABASE_URL": "postgresql://configured"},
            ),
            patch("src.api.app.refresh_upcoming_fixtures") as refresh,
            TestClient(app),
        ):
            refresh.assert_not_called()

    def test_vercel_lifespan_requires_persistent_database(self):
        with (
            patch.dict(os.environ, {"VERCEL": "1"}, clear=True),
            self.assertRaisesRegex(RuntimeError, "DATABASE_URL"),
        ):
            with TestClient(app):
                pass


if __name__ == "__main__":
    unittest.main()
