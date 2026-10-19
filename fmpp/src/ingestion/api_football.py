"""
Fetcher data dari API-Football (api-sports.io).
Cakupan luas: Liga 1 Indonesia, ACL, dan liga lain.
"""

import logging
import time
from typing import Optional
import requests
import pandas as pd
from datetime import datetime
import pytz

from src.config import (
    API_FOOTBALL_KEY, COMPETITIONS,
    TEAM_NAME_MAPPING, TIMEZONE, RAW_DATA_DIR
)

logger = logging.getLogger(__name__)

BASE_URL = "https://v3.football.api-sports.io"
RATE_LIMIT_DELAY = 3  # Sekitar 30 req/menit di tier gratis, kita main aman


class ApiFootballClient:
    """Client untuk API-Football (api-sports.io)."""

    def __init__(self, api_key: str = None):
        self.api_key = api_key or API_FOOTBALL_KEY
        if not self.api_key:
            raise ValueError("API key api-sports.io tidak ditemukan. "
                           "Set API_FOOTBALL_KEY di .env")
        self.headers = {
            "x-apisports-key": self.api_key,
        }
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
        """Kirim request ke API."""
        url = f"{BASE_URL}/{endpoint}"

        for attempt in range(3):
            self._rate_limit()
            try:
                response = requests.get(
                    url, headers=self.headers, params=params, timeout=30
                )
                if response.status_code == 429 and attempt < 2:
                    logger.warning("Rate limited! Menunggu 60 detik sebelum mencoba lagi...")
                    time.sleep(60)
                    continue
                response.raise_for_status()
                data = response.json()

                if data.get("errors"):
                    logger.error("API error: %s", data["errors"])
                    return None
                return data
            except requests.RequestException as error:
                logger.error("Gagal request %s: %s", url, error)
                return None
        logger.error("Gagal request %s: rate limit setelah 3 percobaan.", url)
        return None

    def _to_wib(self, utc_str: str) -> str:
        """Konversi datetime ke WIB."""
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

    def _get_comp_code(self, league_id: int) -> Optional[str]:
        """Cari kode kompetisi FMPP dari API-Football league id."""
        for code, comp in COMPETITIONS.items():
            if comp.get("api_football_id") == league_id:
                return code
        return None

    def fetch_matches(self, competition_code: str,
                      season: int = None,
                      date: str = None,
                      from_date: str = None,
                      to_date: str = None,
                      status: str = None,
                      api_league_id: int = None) -> Optional[pd.DataFrame]:
        """
        Ambil data pertandingan dari API-Football.

        Args:
            competition_code: Kode kompetisi FMPP
            season: Tahun musim (misal 2025)
            date: Tanggal spesifik YYYY-MM-DD
            from_date, to_date: Rentang tanggal
            status: NS (Not Started), FT (Finished), dll
        """
        comp = COMPETITIONS.get(competition_code)
        league_id = api_league_id or (comp or {}).get("api_football_id")
        if not league_id:
            logger.warning("Kompetisi %s tidak didukung di API-Football",
                         competition_code)
            return None

        params = {"league": league_id}
        if season:
            params["season"] = season
        if date:
            params["date"] = date
        if from_date:
            params["from"] = from_date
        if to_date:
            params["to"] = to_date
        if status:
            params["status"] = status

        data = self._request("fixtures", params)
        if not data or "response" not in data:
            return None

        matches = []
        for m in data["response"]:
            fixture = m.get("fixture", {})
            teams = m.get("teams", {})
            goals = m.get("goals", {})
            score = m.get("score", {})
            league = m.get("league", {})

            # Map status
            api_status = fixture.get("status", {}).get("short", "")
            if api_status in ("FT", "AET", "PEN"):
                fmpp_status = "FINISHED"
            elif api_status in ("NS", "TBD"):
                fmpp_status = "SCHEDULED"
            else:
                fmpp_status = api_status

            match = {
                "external_id": str(fixture.get("id", "")),
                "competition_code": competition_code,
                "season": str(league.get("season", "")),
                "matchday": league.get("round", ""),
                "date_utc": fixture.get("date", ""),
                "date_wib": self._to_wib(fixture.get("date", "")),
                "status": fmpp_status,
                "home_team": self._normalize_team(
                    teams.get("home", {}).get("name", "")
                ),
                "away_team": self._normalize_team(
                    teams.get("away", {}).get("name", "")
                ),
                "home_score": goals.get("home"),
                "away_score": goals.get("away"),
                "home_ht_score": score.get("halftime", {}).get("home"),
                "away_ht_score": score.get("halftime", {}).get("away"),
                "venue": fixture.get("venue", {}).get("name"),
                "source": "api-sports.io",
            }
            matches.append(match)

        df = pd.DataFrame(matches)
        logger.info("Diambil %d pertandingan dari API-Football [%s]",
                    len(df), competition_code)
        return df

    def fetch_finished_matches(self, competition_code: str,
                                season: int = None,
                                api_league_id: int = None) -> Optional[pd.DataFrame]:
        """Ambil data laga yang sudah selesai."""
        matches = self.fetch_matches(
            competition_code, season=season, api_league_id=api_league_id
        )
        if matches is None or matches.empty:
            return matches
        return matches[matches["status"] == "FINISHED"].reset_index(drop=True)

    def fetch_upcoming_fixtures(self, competition_code: str,
                                 from_date: str = None,
                                 to_date: str = None,
                                 season: int = None) -> Optional[pd.DataFrame]:
        """Ambil jadwal laga mendatang."""
        return self.fetch_matches(
            competition_code,
            season=season,
            from_date=from_date,
            to_date=to_date,
            status="NS"
        )

    def fetch_fixtures_by_date(self, date: str) -> pd.DataFrame:
        """
        Ambil semua laga dari semua kompetisi pada tanggal tertentu.
        """
        all_data = []
        for code in COMPETITIONS:
            df = self.fetch_matches(code, date=date)
            if df is not None and not df.empty:
                all_data.append(df)

        if all_data:
            return pd.concat(all_data, ignore_index=True)
        return pd.DataFrame()

    def fetch_all_historical(self, competition_code: str,
                              seasons: list = None,
                              api_league_id: int = None) -> pd.DataFrame:
        """
        Ambil data historis dari beberapa musim untuk satu kompetisi.
        Khusus untuk liga yang tidak tersedia di football-data.co.uk
        (Liga 1 Indonesia, ACL).
        """
        if seasons is None:
            seasons = [2023, 2024, 2025]

        all_data = []
        for season in seasons:
            df = self.fetch_finished_matches(
                competition_code, season=season, api_league_id=api_league_id
            )
            if df is not None and not df.empty:
                all_data.append(df)

        if all_data:
            combined = pd.concat(all_data, ignore_index=True)
            RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
            combined.to_csv(
                RAW_DATA_DIR / f"api_football_{competition_code}_{'_'.join(map(str, seasons))}.csv",
                index=False,
            )
            logger.info("Total data historis %s: %d pertandingan",
                       competition_code, len(combined))
            return combined
        return pd.DataFrame()
