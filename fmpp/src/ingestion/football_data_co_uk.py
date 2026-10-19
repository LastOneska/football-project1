"""
Fetcher data dari football-data.co.uk (CSV gratis).
Sumber utama untuk data historis liga Eropa.
"""

import io
import logging
from datetime import datetime, time
from typing import Optional
from zoneinfo import ZoneInfo
import pandas as pd
import requests

from src.config import RAW_DATA_DIR, COMPETITIONS, TEAM_NAME_MAPPING

logger = logging.getLogger(__name__)

# Base URL football-data.co.uk
BASE_URL = "https://www.football-data.co.uk"

# Mapping season ke format URL
# football-data.co.uk pakai format: mmz4281/2526/E0.csv (untuk 2025-2026)
def _season_to_url_code(season_start: int) -> str:
    """Konversi tahun awal musim ke kode URL (misal: 2025 -> '2526')."""
    end = (season_start + 1) % 100
    start = season_start % 100
    return f"{start:02d}{end:02d}"


def fetch_season_csv(league_code: str, season_start: int,
                     save: bool = True) -> Optional[pd.DataFrame]:
    """
    Download CSV data satu musim satu liga dari football-data.co.uk.

    Args:
        league_code: Kode liga (E0=PL, SP1=LaLiga, I1=SerieA, D1=BL, F1=L1, N1=Eredivisie)
        season_start: Tahun mulai musim (misal 2024 untuk 2024-2025)
        save: Simpan file CSV ke disk

    Returns:
        DataFrame hasil parsing, atau None jika gagal.
    """
    season_code = _season_to_url_code(season_start)
    url = f"{BASE_URL}/mmz4281/{season_code}/{league_code}.csv"

    logger.info("Mengambil data dari %s", url)

    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
    except requests.RequestException as e:
        logger.error("Gagal mengambil %s: %s", url, e)
        return None

    try:
        df = pd.read_csv(io.StringIO(response.text), on_bad_lines="skip")
    except Exception as e:
        logger.error("Gagal parsing CSV %s: %s", url, e)
        return None

    if df.empty:
        logger.warning("Data kosong untuk %s musim %s", league_code, season_code)
        return None

    # Simpan raw CSV
    if save:
        save_path = RAW_DATA_DIR / f"fdcouk_{league_code}_{season_code}.csv"
        save_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(save_path, index=False)
        logger.info("Tersimpan: %s (%d baris)", save_path, len(df))

    return df


def parse_fdcouk_csv(df: pd.DataFrame, competition_code: str,
                     season: str) -> pd.DataFrame:
    """
    Parse dan normalisasi CSV dari football-data.co.uk
    ke format standar FMPP.
    """
    # Kolom yang tersedia di football-data.co.uk
    column_mapping = {
        "Date": "date_raw",
        "Time": "time_raw",
        "HomeTeam": "home_team",
        "AwayTeam": "away_team",
        "FTHG": "home_score",      # Full Time Home Goals
        "FTAG": "away_score",      # Full Time Away Goals
        "FTR": "result",           # Full Time Result (H/D/A)
        "HTHG": "home_ht_score",   # Half Time Home Goals
        "HTAG": "away_ht_score",   # Half Time Away Goals
        "HS": "home_shots",
        "AS": "away_shots",
        "HST": "home_shots_on_target",
        "AST": "away_shots_on_target",
        "HC": "home_corners",
        "AC": "away_corners",
        "HF": "home_fouls",
        "AF": "away_fouls",
        "HY": "home_yellow",
        "AY": "away_yellow",
        "HR": "home_red",
        "AR": "away_red",
    }

    # Hanya ambil kolom yang ada
    available = {k: v for k, v in column_mapping.items() if k in df.columns}
    result = df[list(available.keys())].rename(columns=available)

    # football-data.co.uk timestamps use the venue's local time (UK), not UTC.
    if "date_raw" in result.columns:
        def to_utc(row: pd.Series) -> str | None:
            match_date = pd.to_datetime(row["date_raw"], dayfirst=True, errors="coerce")
            if pd.isna(match_date):
                return None
            match_time = pd.to_datetime(
                row.get("time_raw"), format="%H:%M", errors="coerce"
            )
            local_time = match_time.time() if pd.notna(match_time) else time(12, 0)
            local = datetime.combine(match_date.date(), local_time).replace(
                tzinfo=ZoneInfo("Europe/London")
            )
            return local.astimezone(ZoneInfo("UTC")).isoformat(sep=" ", timespec="seconds")

        result["date_utc"] = result.apply(to_utc, axis=1)
        result.drop(columns=["date_raw", "time_raw"], errors="ignore", inplace=True)

    # Normalisasi nama tim
    if "home_team" in result.columns:
        result["home_team"] = result["home_team"].map(
            lambda x: TEAM_NAME_MAPPING.get(x, x)
        )
    if "away_team" in result.columns:
        result["away_team"] = result["away_team"].map(
            lambda x: TEAM_NAME_MAPPING.get(x, x)
        )

    # Tambah metadata
    result["competition_code"] = competition_code
    result["season"] = season
    result["status"] = "FINISHED"
    result["source"] = "football-data.co.uk"

    # Konversi kolom numerik
    int_cols = ["home_score", "away_score", "home_ht_score", "away_ht_score",
                "home_shots", "away_shots", "home_shots_on_target",
                "away_shots_on_target", "home_corners", "away_corners",
                "home_fouls", "away_fouls", "home_yellow", "away_yellow",
                "home_red", "away_red"]
    for col in int_cols:
        if col in result.columns:
            result[col] = pd.to_numeric(result[col], errors="coerce")

    result = result.dropna(subset=["date_utc", "home_team", "away_team", "home_score", "away_score"])

    # Buang kolom result (sudah ada di skor)
    result.drop(columns=["result"], errors="ignore", inplace=True)

    logger.info("Parsed %d pertandingan dari football-data.co.uk [%s %s]",
                len(result), competition_code, season)
    return result


def fetch_all_historical(seasons: list = None) -> pd.DataFrame:
    """
    Ambil semua data historis dari football-data.co.uk
    untuk semua liga yang tersedia.

    Args:
        seasons: List tahun mulai musim, default [2021, 2022, 2023, 2024, 2025]
    """
    if seasons is None:
        seasons = [2021, 2022, 2023, 2024, 2025, 2026]

    all_data = []

    for comp_code, comp_info in COMPETITIONS.items():
        fd_code = comp_info.get("fd_co_uk_code")
        if not fd_code:
            logger.info("Skip %s (tidak tersedia di football-data.co.uk)", comp_code)
            continue

        for year in seasons:
            season_str = f"{year}-{year+1}"
            df_raw = fetch_season_csv(fd_code, year)
            if df_raw is not None:
                df_parsed = parse_fdcouk_csv(df_raw, comp_code, season_str)
                all_data.append(df_parsed)

    if all_data:
        combined = pd.concat(all_data, ignore_index=True)
        logger.info("Total data historis: %d pertandingan", len(combined))
        return combined

    return pd.DataFrame()
