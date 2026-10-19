"""
Fetcher data dari football-data.org (API gratis).
Untuk jadwal laga dan hasil beberapa liga termasuk UCL.
"""

import logging
import time
from typing import Optional
import requests
import pandas as pd
from datetime import datetime
import pytz

from src.config import (
    FOOTBALL_DATA_ORG_API_KEY, COMPETITIONS,
    TEAM_NAME_MAPPING, TIMEZONE
)

logger = logging.getLogger(__name__)

BASE_URL = "https://api.football-data.org/v4"
RATE_LIMIT_DELAY = 6.5  # Tier gratis: 10 requests/menit


class FootballDataOrgClient:
    """Client untuk football-data.org API."""

    def __init__(self, api_key: str = None):
        self.api_key = api_key or FOOTBALL_DATA_ORG_API_KEY
        if not self.api_key:
            raise ValueError("API key football-data.org tidak ditemukan. "
                           "Set FOOTBALL_DATA_ORG_API_KEY di .env")
        self.headers = {"X-Auth-Token": self.api_key}
        self._last_request_time = 0

    def _rate_limit(self):
        """Tunggu agar tidak melebihi rate limit."""
        elapsed = time.time() - self._last_request_time
        if elapsed < RATE_LIMIT_DELAY:
            wait = RATE_LIMIT_DELAY - elapsed
            logger.debug("Rate limit: menunggu %.1f detik", wait)
            time.sleep(wait)
        self._last_request_time = time.time()

    def _request(self, endpoint: str, params: dict = None) -> Optional[dict]:
        """Kirim request ke API dengan rate limiting."""
        self._rate_limit()
        url = f"{BASE_URL}/{endpoint}"

        try:
            response = requests.get(url, headers=self.headers,
                                   params=params, timeout=30)
            if response.status_code == 429:
                logger.warning("Rate limited! Menunggu 60 detik...")
                time.sleep(60)
                return self._request(endpoint, params)
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            logger.error("Gagal request %s: %s", url, e)
            return None

    def _to_wib(self, utc_str: str) -> str:
        """Konversi datetime UTC ke WIB string."""
        try:
            dt = datetime.fromisoformat(utc_str.replace("Z", "+00:00"))
            wib = pytz.timezone(TIMEZONE)
            dt_wib = dt.astimezone(wib)
            return dt_wib.strftime("%Y-%m-%d %H:%M")
        except Exception:
            return utc_str

    def _normalize_team(self, name: str) -> str:
        """Normalisasi nama tim."""
        return TEAM_NAME_MAPPING.get(name, name)

    def fetch_matches(self, competition_code: str,
                      season: int = None,
                      status: str = "FINISHED",
                      date_from: str = None,
                      date_to: str = None) -> Optional[pd.DataFrame]:
        """
        Ambil data pertandingan dari satu kompetisi.

        Args:
            competition_code: Kode kompetisi FMPP (PL, LL, SA, dll)
            season: Tahun musim (misal 2025)
            status: FINISHED, SCHEDULED, TIMED
            date_from: Format YYYY-MM-DD
            date_to: Format YYYY-MM-DD
        """
        comp = COMPETITIONS.get(competition_code)
        if not comp or not comp.get("fd_org_code"):
            logger.warning("Kompetisi %s tidak didukung di football-data.org",
                         competition_code)
            return None

        fd_code = comp["fd_org_code"]

        params = {}
        if season:
            params["season"] = season
        if status:
            params["status"] = status
        if date_from:
            params["dateFrom"] = date_from
        if date_to:
            params["dateTo"] = date_to

        data = self._request(f"competitions/{fd_code}/matches", params)
        if not data or "matches" not in data:
            return None

        matches = []
        for m in data["matches"]:
            match = {
                "external_id": str(m.get("id", "")),
                "competition_code": competition_code,
                "season": str(m.get("season", {}).get("startDate", ""))[:4]
                         + "-"
                         + str(m.get("season", {}).get("endDate", ""))[:4],
                "matchday": m.get("matchday"),
                "date_utc": m.get("utcDate", ""),
                "date_wib": self._to_wib(m.get("utcDate", "")),
                "status": m.get("status", ""),
                "home_team": self._normalize_team(
                    m.get("homeTeam", {}).get("name", "")
                ),
                "away_team": self._normalize_team(
                    m.get("awayTeam", {}).get("name", "")
                ),
                "home_score": m.get("score", {}).get("fullTime", {}).get("home"),
                "away_score": m.get("score", {}).get("fullTime", {}).get("away"),
                "home_ht_score": m.get("score", {}).get("halfTime", {}).get("home"),
                "away_ht_score": m.get("score", {}).get("halfTime", {}).get("away"),
                "source": "football-data.org",
            }
            matches.append(match)

        df = pd.DataFrame(matches)
        logger.info("Diambil %d pertandingan dari football-data.org [%s]",
                    len(df), competition_code)
        return df

    def fetch_fixtures(self, competition_code: str,
                       date_from: str = None,
                       date_to: str = None) -> Optional[pd.DataFrame]:
        """
        Ambil jadwal laga mendatang.
        """
        return self.fetch_matches(
            competition_code,
            status="SCHEDULED,TIMED",
            date_from=date_from,
            date_to=date_to
        )

    def fetch_all_competitions_matches(self, season: int = None) -> pd.DataFrame:
        """Ambil pertandingan dari semua kompetisi yang tersedia."""
        all_data = []
        for code, comp in COMPETITIONS.items():
            if comp.get("fd_org_code"):
                df = self.fetch_matches(code, season=season)
                if df is not None and not df.empty:
                    all_data.append(df)

        if all_data:
            return pd.concat(all_data, ignore_index=True)
        return pd.DataFrame()

    def fetch_all_competitions_fixtures(self, date_from: str = None,
                                         date_to: str = None) -> pd.DataFrame:
        """Ambil jadwal laga mendatang dari semua kompetisi."""
        all_data = []
        for code, comp in COMPETITIONS.items():
            if comp.get("fd_org_code"):
                df = self.fetch_fixtures(code, date_from=date_from,
                                        date_to=date_to)
                if df is not None and not df.empty:
                    all_data.append(df)

        if all_data:
            return pd.concat(all_data, ignore_index=True)
        return pd.DataFrame()
