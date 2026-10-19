"""Refresh upcoming fixtures from configured schedule providers."""

import logging
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import pandas as pd

from src.config import (
    API_FOOTBALL_KEY,
    COMPETITIONS,
    FOOTBALL_DATA_ORG_API_KEY,
    TIMEZONE,
)
from src.ingestion.api_football import ApiFootballClient
from src.ingestion.database import DatabaseManager
from src.ingestion.football_data_org import FootballDataOrgClient

logger = logging.getLogger(__name__)


def refresh_upcoming_fixtures(
    database: DatabaseManager | None = None,
    days: int = 14,
) -> int:
    """Fetch and store upcoming fixtures without downloading historical CSVs."""
    if not FOOTBALL_DATA_ORG_API_KEY and not API_FOOTBALL_KEY:
        raise RuntimeError(
            "Tidak ada API key jadwal. Atur FOOTBALL_DATA_ORG_API_KEY "
            "atau API_FOOTBALL_KEY di file .env."
        )

    database = database or DatabaseManager()
    today = datetime.now(ZoneInfo(TIMEZONE)).date()
    start_date = (today - timedelta(days=1)).isoformat()
    end_date = (today + timedelta(days=days)).isoformat()
    frames = []

    if FOOTBALL_DATA_ORG_API_KEY:
        fixtures = FootballDataOrgClient().fetch_all_competitions_fixtures(
            start_date, end_date
        )
        if fixtures is not None and not fixtures.empty:
            frames.append(fixtures)
        else:
            logger.warning("football-data.org tidak mengembalikan jadwal.")

    if API_FOOTBALL_KEY and not FOOTBALL_DATA_ORG_API_KEY:
        api_codes = set(COMPETITIONS)
        client = ApiFootballClient()
        for code in sorted(api_codes):
            competition = COMPETITIONS[code]
            season = (
                today.year
                if competition.get("season_format") == "single_year"
                or today.month >= 7
                else today.year - 1
            )
            fixtures = client.fetch_upcoming_fixtures(
                code, start_date, end_date, season=season
            )
            if fixtures is not None and not fixtures.empty:
                frames.append(fixtures)

    if not frames:
        logger.warning("Tidak ada jadwal baru dari provider untuk rentang %s–%s.",
                       start_date, end_date)
        return 0

    combined = pd.concat(frames, ignore_index=True)
    count = database.insert_fixtures(combined)
    logger.info(
        "Pembaruan jadwal selesai: %d baris disimpan/diperbarui untuk %s–%s.",
        count,
        start_date,
        end_date,
    )
    return count
