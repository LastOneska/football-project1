"""Tests for fixture-only refreshes and the optional historical import."""

import unittest
from unittest.mock import Mock, patch

import pandas as pd
from typer.testing import CliRunner

from src.cli import app
from src.ingestion.api_football import ApiFootballClient
from src.ingestion.schedule import refresh_upcoming_fixtures


class ScheduleRefreshTests(unittest.TestCase):
    def test_upcoming_fixtures_forwards_the_season(self):
        client = ApiFootballClient(api_key="test")
        with patch.object(client, "fetch_matches") as fetch_matches:
            client.fetch_upcoming_fixtures(
                "UCL",
                "2026-10-19",
                "2026-11-02",
                season=2026,
            )

        fetch_matches.assert_called_once_with(
            "UCL",
            season=2026,
            from_date="2026-10-19",
            to_date="2026-11-02",
            status="NS",
        )

    def test_primary_provider_refreshes_ucl_without_api_fallback(self):
        database = Mock()
        database.insert_fixtures.return_value = 2
        football_data_rows = pd.DataFrame([{
            "competition_code": "PL",
            "date_utc": "2026-10-20T12:00:00Z",
            "home_team": "Home FC",
            "away_team": "Away FC",
        }, {
            "competition_code": "UCL",
            "date_utc": "2026-10-21T19:00:00Z",
            "home_team": "Barcelona",
            "away_team": "Juventus",
        }])

        with (
            patch("src.ingestion.schedule.FOOTBALL_DATA_ORG_API_KEY", "configured"),
            patch("src.ingestion.schedule.API_FOOTBALL_KEY", "configured"),
            patch("src.ingestion.schedule.FootballDataOrgClient") as football_data,
            patch("src.ingestion.schedule.ApiFootballClient") as api_football,
        ):
            football_data.return_value.fetch_all_competitions_fixtures.return_value = (
                football_data_rows
            )

            result = refresh_upcoming_fixtures(database, days=7)

        self.assertEqual(result, 2)
        api_football.assert_not_called()
        database.insert_fixtures.assert_called_once()
        saved = database.insert_fixtures.call_args.args[0]
        self.assertEqual(set(saved["competition_code"]), {"PL", "UCL"})

    def test_api_football_fallback_uses_season_when_primary_is_unconfigured(self):
        database = Mock()
        database.insert_fixtures.return_value = 1
        ucl_rows = pd.DataFrame([{
            "competition_code": "UCL",
            "date_utc": "2026-10-20T19:00:00Z",
            "home_team": "Barcelona",
            "away_team": "Juventus",
        }])
        with (
            patch("src.ingestion.schedule.FOOTBALL_DATA_ORG_API_KEY", ""),
            patch("src.ingestion.schedule.API_FOOTBALL_KEY", "configured"),
            patch("src.ingestion.schedule.FootballDataOrgClient") as football_data,
            patch("src.ingestion.schedule.ApiFootballClient") as api_football,
        ):
            api_football.return_value.fetch_upcoming_fixtures.side_effect = (
                lambda code, *_args, **_kwargs: ucl_rows if code == "UCL" else None
            )
            result = refresh_upcoming_fixtures(database, days=7)

        self.assertEqual(result, 1)
        football_data.assert_not_called()
        api_football.return_value.fetch_upcoming_fixtures.assert_any_call(
            "UCL", unittest.mock.ANY, unittest.mock.ANY, season=2026
        )
        database.insert_fixtures.assert_called_once()

    def test_refresh_requires_a_configured_schedule_provider(self):
        with (
            patch("src.ingestion.schedule.FOOTBALL_DATA_ORG_API_KEY", ""),
            patch("src.ingestion.schedule.API_FOOTBALL_KEY", ""),
        ):
            with self.assertRaisesRegex(RuntimeError, "Tidak ada API key jadwal"):
                refresh_upcoming_fixtures(Mock())

    def test_update_data_does_not_download_historical_csv_by_default(self):
        runner = CliRunner()
        with (
            patch("src.cli.DatabaseManager"),
            patch("src.cli.refresh_upcoming_fixtures", return_value=4),
            patch("src.cli.ApiFootballClient") as api_football,
        ):
            result = runner.invoke(
                app,
                ["update-data", "--seasons", "2026", "--fixture-days", "7"],
            )

        self.assertEqual(result.exit_code, 0, result.output)
        self.assertIn("4 baris jadwal", result.output)
        api_football.assert_not_called()

    def test_database_migration_requires_hosted_database_url(self):
        runner = CliRunner()
        with patch("src.cli.DATABASE_URL", ""):
            result = runner.invoke(app, ["migrate-database"])

        self.assertEqual(result.exit_code, 1)
        self.assertIn("DATABASE_URL", result.output)


if __name__ == "__main__":
    unittest.main()
