"""Local web API and UI for FMPP."""

import asyncio
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
import hmac
import logging
import os
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
from fastapi import FastAPI, Header, HTTPException, Query
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from src.config import COMPETITIONS, DEFAULT_LAST_N, TIMEZONE
from src.evaluation.backtest import walk_forward_evaluate
from src.ingestion.database import DatabaseManager
from src.ingestion.schedule import refresh_upcoming_fixtures
from src.ingestion.team_names import normalize_team_name
from src.prediction.engine import PredictionEngine

logger = logging.getLogger(__name__)
WEB_DIR = Path(__file__).resolve().parents[2] / "web"


async def _refresh_schedule_once() -> None:
    try:
        await asyncio.to_thread(refresh_upcoming_fixtures, days=14)
    except Exception:
        logger.exception("Pembaruan jadwal otomatis gagal.")


async def _periodic_schedule_refresh(initial_refresh: asyncio.Task) -> None:
    await initial_refresh
    while True:
        await asyncio.sleep(6 * 60 * 60)
        await _refresh_schedule_once()


@asynccontextmanager
async def lifespan(application: FastAPI):
    if os.getenv("VERCEL") == "1":
        if not os.getenv("DATABASE_URL"):
            raise RuntimeError(
                "DATABASE_URL must be configured for persistent storage on Vercel."
            )
        yield
        return

    initial_refresh = asyncio.create_task(_refresh_schedule_once())
    application.state.initial_schedule_refresh = initial_refresh
    periodic_refresh = asyncio.create_task(
        _periodic_schedule_refresh(initial_refresh)
    )
    try:
        yield
    finally:
        periodic_refresh.cancel()
        initial_refresh.cancel()
        await asyncio.gather(periodic_refresh, initial_refresh, return_exceptions=True)


app = FastAPI(
    title="FMPP",
    description="Analisis probabilitas pertandingan sepak bola untuk pembelajaran.",
    version="1.0.0",
    lifespan=lifespan,
)
app.mount("/static", StaticFiles(directory=WEB_DIR / "static"), name="static")


def _database() -> DatabaseManager:
    return DatabaseManager()


def _team_analysis(team: str, history: pd.DataFrame) -> dict:
    """Summarize recent results and goals for one team."""
    canonical_team = normalize_team_name(team)
    results = []
    goals_for = []
    goals_against = []

    for _, match in history.iterrows():
        is_home = normalize_team_name(match["home_team"]) == canonical_team
        scored = match["home_score"] if is_home else match["away_score"]
        conceded = match["away_score"] if is_home else match["home_score"]
        if pd.isna(scored) or pd.isna(conceded):
            continue
        scored, conceded = int(scored), int(conceded)
        goals_for.append(scored)
        goals_against.append(conceded)
        if scored > conceded:
            results.append("M")
        elif scored == conceded:
            results.append("S")
        else:
            results.append("K")

    count = len(results)
    points = results.count("M") * 3 + results.count("S")
    return {
        "matches_analyzed": count,
        "minimum_matches_required": 5,
        "minimum_met": count >= 5,
        "recent_results": "-".join(reversed(results)),
        "wins": results.count("M"),
        "draws": results.count("S"),
        "losses": results.count("K"),
        "points_per_game": round(points / count, 2) if count else None,
        "avg_goals_scored": round(sum(goals_for) / count, 2) if count else None,
        "avg_goals_conceded": round(sum(goals_against) / count, 2) if count else None,
    }


@app.get("/", include_in_schema=False)
def index() -> FileResponse:
    return FileResponse(WEB_DIR / "index.html")


@app.get("/api/competitions")
def competitions() -> list[dict]:
    return [
        {"code": code, "name": details["name"], "country": details["country"]}
        for code, details in COMPETITIONS.items()
    ]


@app.get("/api/matches")
async def matches(
    day: str = Query("today", pattern="^(today|tomorrow|dayafter)$"),
    league: str | None = None,
) -> list[dict]:
    initial_refresh = getattr(app.state, "initial_schedule_refresh", None)
    if initial_refresh is not None:
        await initial_refresh
    if league and league not in COMPETITIONS:
        raise HTTPException(status_code=422, detail=f"Kompetisi tidak dikenal: {league}")
    offset = {"today": 0, "tomorrow": 1, "dayafter": 2}[day]
    local_date = datetime.now(ZoneInfo(TIMEZONE)).date() + timedelta(days=offset)
    fixtures = _database().get_fixtures_by_date(local_date.isoformat(), league)
    output = []
    for _, fixture in fixtures.iterrows():
        code = str(fixture["competition_code"])
        output.append({
            "fixture_id": int(fixture["fixture_id"]),
            "competition_code": code,
            "competition": COMPETITIONS.get(code, {}).get("name", code),
            "date_utc": str(fixture["date_utc"]),
            "date_wib": str(fixture["date_wib"] or ""),
            "home_team": str(fixture["home_team"]),
            "away_team": str(fixture["away_team"]),
            "status": str(fixture["status"]),
        })
    return output


@app.get("/api/matches/{fixture_id}/prediction")
def predict_match(
    fixture_id: int,
    last_n: int = Query(DEFAULT_LAST_N, ge=5, le=10),
) -> dict:
    database = _database()
    fixture = database.get_fixture(fixture_id)
    if fixture is None:
        raise HTTPException(status_code=404, detail="Jadwal pertandingan tidak ditemukan.")
    try:
        prediction = PredictionEngine(database=database).predict(
            fixture["home_team"],
            fixture["away_team"],
            match_date=fixture["date_utc"],
            last_n=last_n,
            include_score_matrix=True,
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc

    recent_form = {}
    team_analysis = {}
    for side, team in (("home", fixture["home_team"]), ("away", fixture["away_team"])):
        history = database.get_team_matches(
            team, before_date=fixture["date_utc"], limit=last_n
        )
        team_analysis[side] = _team_analysis(team, history)
        recent_form[side] = [
            {
                "date": str(row["date_utc"]),
                "home_team": str(row["home_team"]),
                "away_team": str(row["away_team"]),
                "home_score": int(row["home_score"]),
                "away_score": int(row["away_score"]),
                "team_is_home": normalize_team_name(str(row["home_team"]))
                == normalize_team_name(team),
            }
            for _, row in history.iterrows()
        ]

    code = fixture["competition_code"]
    return {
        "fixture": {
            "fixture_id": fixture_id,
            "competition_code": code,
            "competition": COMPETITIONS.get(code, {}).get("name", code),
            "date_wib": fixture["date_wib"],
            "home_team": fixture["home_team"],
            "away_team": fixture["away_team"],
        },
        "prediction": prediction,
        "recent_form": recent_form,
        "team_analysis": team_analysis,
        "disclaimer": (
            "Probabilitas adalah perkiraan statistik untuk analisis dan pembelajaran, "
            "bukan jaminan hasil atau dasar keputusan perjudian."
        ),
    }


@app.get("/api/teams")
def teams(
    last_n: int = Query(DEFAULT_LAST_N, ge=5, le=10),
) -> dict:
    """Return recent form for every team found in stored results or fixtures."""
    database = _database()
    as_of = datetime.now(timezone.utc).isoformat()
    histories = database.get_all_team_recent_matches(as_of, limit=last_n)
    names_by_key = {}
    for name in database.get_all_team_names():
        key = normalize_team_name(name)
        if key:
            names_by_key.setdefault(key, name)

    output = []
    for key, name in sorted(names_by_key.items(), key=lambda item: item[1].casefold()):
        history = histories.get(key, pd.DataFrame())
        analysis = _team_analysis(name, history)
        recent_matches = [
            {
                "date": str(row["date_utc"]),
                "home_team": str(row["home_team"]),
                "away_team": str(row["away_team"]),
                "home_score": int(row["home_score"]),
                "away_score": int(row["away_score"]),
                "team_is_home": normalize_team_name(str(row["home_team"])) == key,
            }
            for _, row in history.iterrows()
            if pd.notna(row["home_score"]) and pd.notna(row["away_score"])
        ]
        output.append({
            "team": name,
            "team_analysis": analysis,
            "recent_form": recent_matches,
        })
    return {"as_of_utc": as_of, "last_n": last_n, "teams": output}


@app.get("/api/performance")
def performance(
    last_n: int = Query(DEFAULT_LAST_N, ge=5, le=10),
    folds: int = Query(3, ge=1, le=10),
) -> dict:
    try:
        return walk_forward_evaluate(
            _database().get_all_matches(),
            last_n=last_n,
            folds=folds,
        )
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/api/cron/refresh")
def cron_refresh(
    authorization: str | None = Header(default=None),
) -> dict[str, int | str]:
    """Refresh fixtures when called by an authenticated Vercel Cron job."""
    secret = os.getenv("CRON_SECRET", "")
    if not secret:
        raise HTTPException(status_code=503, detail="CRON_SECRET is not configured.")
    if not hmac.compare_digest(authorization or "", f"Bearer {secret}"):
        raise HTTPException(status_code=401, detail="Unauthorized.")
    try:
        count = refresh_upcoming_fixtures(days=14)
    except RuntimeError as exc:
        logger.exception("Pembaruan jadwal terjadwal gagal.")
        raise HTTPException(status_code=502, detail="Fixture refresh failed.") from exc
    return {"status": "ok", "fixtures_updated": count}


def run() -> None:
    """Run the local web server."""
    import uvicorn

    uvicorn.run("src.api.app:app", host="127.0.0.1", port=8000, reload=False)
