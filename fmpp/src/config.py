"""
Konfigurasi global FMPP.
Memuat environment variables dan konstanta.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env
BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")

# === API Keys ===
FOOTBALL_DATA_ORG_API_KEY = os.getenv("FOOTBALL_DATA_ORG_API_KEY", "")
API_FOOTBALL_KEY = os.getenv("API_FOOTBALL_KEY", "")

# === Paths ===
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
DATABASE_PATH = BASE_DIR / os.getenv("DATABASE_PATH", "data/fmpp.db")
DATABASE_URL = os.getenv("DATABASE_URL", "")
MODELS_DIR = BASE_DIR / "models" / "saved"

# Local runs need writable data directories; Vercel bundles are read-only.
if os.getenv("VERCEL") != "1":
    for d in [RAW_DATA_DIR, PROCESSED_DATA_DIR, MODELS_DIR]:
        d.mkdir(parents=True, exist_ok=True)

# === Konfigurasi Umum ===
DEFAULT_LAST_N = int(os.getenv("DEFAULT_LAST_N", "5"))
TIMEZONE = os.getenv("TIMEZONE", "Asia/Jakarta")
RANDOM_SEED = 42

# === Kompetisi yang Didukung ===
# Mapping: kode internal -> nama lengkap, kode football-data.org, kode api-football
COMPETITIONS = {
    "PL": {
        "name": "Premier League",
        "country": "England",
        "fd_org_code": "PL",         # football-data.org
        "api_football_id": 39,       # api-sports.io league id
        "fd_co_uk_code": "E0",       # football-data.co.uk
        "season_format": "cross_year" # contoh: 2025-2026
    },
    "LL": {
        "name": "La Liga",
        "country": "Spain",
        "fd_org_code": "PD",
        "api_football_id": 140,
        "fd_co_uk_code": "SP1",
        "season_format": "cross_year"
    },
    "SA": {
        "name": "Serie A",
        "country": "Italy",
        "fd_org_code": "SA",
        "api_football_id": 135,
        "fd_co_uk_code": "I1",
        "season_format": "cross_year"
    },
    "BL": {
        "name": "Bundesliga",
        "country": "Germany",
        "fd_org_code": "BL1",
        "api_football_id": 78,
        "fd_co_uk_code": "D1",
        "season_format": "cross_year"
    },
    "FL": {
        "name": "Ligue 1",
        "country": "France",
        "fd_org_code": "FL1",
        "api_football_id": 61,
        "fd_co_uk_code": "F1",
        "season_format": "cross_year"
    },
    "ED": {
        "name": "Eredivisie",
        "country": "Netherlands",
        "fd_org_code": "DED",
        "api_football_id": 88,
        "fd_co_uk_code": "N1",
        "season_format": "cross_year"
    },
    "UCL": {
        "name": "UEFA Champions League",
        "country": "Europe",
        "fd_org_code": "CL",
        "api_football_id": 2,
        "fd_co_uk_code": None,       # tidak tersedia
        "season_format": "cross_year"
    },
    "ACL": {
        "name": "AFC Champions League",
        "country": "Asia",
        "fd_org_code": None,         # tidak tersedia
        "api_football_id": 17,
        "fd_co_uk_code": None,
        "season_format": "single_year"
    },
    "ID1": {
        "name": "Liga 1 Indonesia",
        "country": "Indonesia",
        "fd_org_code": None,         # tidak tersedia
        "api_football_id": 274,
        "fd_co_uk_code": None,
        "season_format": "single_year"
    },
}

# Additional domestic leagues used to build team form for international fixtures.
# These are kept out of the public competition picker and upcoming-fixture fetch.
HISTORICAL_COMPETITIONS = {
    "ENG2": {"name": "Championship", "country": "England", "api_football_id": 40},
    "DE2": {"name": "2. Bundesliga", "country": "Germany", "api_football_id": 79},
    "ES2": {"name": "La Liga 2", "country": "Spain", "api_football_id": 141},
    "FR2": {"name": "Ligue 2", "country": "France", "api_football_id": 62},
    "FR3": {"name": "Championnat National", "country": "France", "api_football_id": 63},
    "NL2": {"name": "Eerste Divisie", "country": "Netherlands", "api_football_id": 89},
    "AT1": {"name": "Austrian Bundesliga", "country": "Austria", "api_football_id": 218},
    "NO1": {"name": "Eliteserien", "country": "Norway", "api_football_id": 103},
    "GR1": {"name": "Super League Greece", "country": "Greece", "api_football_id": 197},
    "CZ1": {"name": "Czech First League", "country": "Czechia", "api_football_id": 345},
    "AZ1": {"name": "Azerbaijan Premier League", "country": "Azerbaijan", "api_football_id": 419},
}

# === Mapping Nama Tim ===
# Normalisasi nama tim dari berbagai sumber data
TEAM_NAME_MAPPING = {
    # Premier League
    "Man United": "Manchester United",
    "Man City": "Manchester City",
    "Spurs": "Tottenham Hotspur",
    "Tottenham": "Tottenham Hotspur",
    "Wolves": "Wolverhampton Wanderers",
    "Wolverhampton": "Wolverhampton Wanderers",
    "Brighton": "Brighton & Hove Albion",
    "West Ham": "West Ham United",
    "Newcastle": "Newcastle United",
    "Nott'm Forest": "Nottingham Forest",
    "Nottingham": "Nottingham Forest",
    "Sheffield Utd": "Sheffield United",
    "Leicester": "Leicester City",
    "Leeds": "Leeds United",
    "Norwich": "Norwich City",

    # La Liga
    "Atletico Madrid": "Atlético Madrid",
    "Atletico": "Atlético Madrid",
    "Ath Bilbao": "Athletic Bilbao",
    "Athletic Club": "Athletic Bilbao",
    "Real Betis Balompie": "Real Betis",
    "Celta Vigo": "Celta de Vigo",
    "Sociedad": "Real Sociedad",
    "Real Madrid CF": "Real Madrid",

    # Serie A
    "Inter": "Inter Milan",
    "Internazionale": "Inter Milan",
    "AC Milan": "AC Milan",
    "Napoli": "SSC Napoli",
    "Verona": "Hellas Verona",

    # Bundesliga
    "Bayern Munich": "Bayern München",
    "Bayern": "Bayern München",
    "Dortmund": "Borussia Dortmund",
    "Borussia Dortmund": "Borussia Dortmund",
    "Monchengladbach": "Borussia Mönchengladbach",
    "Gladbach": "Borussia Mönchengladbach",
    "Leverkusen": "Bayer Leverkusen",
    "RB Leipzig": "RB Leipzig",

    # Ligue 1
    "Paris Saint Germain": "Paris Saint-Germain",
    "PSG": "Paris Saint-Germain",
    "St Etienne": "AS Saint-Étienne",
    "Lyon": "Olympique Lyonnais",
    "Marseille": "Olympique de Marseille",
    "Monaco": "AS Monaco",
}

# === Elo Rating Default ===
ELO_DEFAULT_RATING = 1500
ELO_K_FACTOR = 32
ELO_HOME_ADVANTAGE = 100

# === Model Config ===
POISSON_MAX_GOALS = 10  # Batas gol di matriks skor
OVER_UNDER_LINES = [1.5, 2.5, 3.5, 4.5]
