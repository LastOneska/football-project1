"""Database manager for local SQLite and hosted PostgreSQL storage."""

import logging
import os
import sqlite3
from pathlib import Path
from typing import Any, Optional
import pandas as pd

from src.config import DATABASE_PATH, DATABASE_URL
from src.ingestion.team_names import normalize_team_name

logger = logging.getLogger(__name__)


class _PostgresConnection:
    """Small DB-API adapter for the qmark SQL shared with SQLite."""

    def __init__(self, connection: Any):
        self.connection = connection

    def execute(self, query: str, params: tuple | list = ()) -> Any:
        cursor = self.connection.cursor()
        cursor.execute(query.replace("?", "%s"), params)
        return cursor

    def executemany(self, query: str, params: list[tuple]) -> Any:
        cursor = self.connection.cursor()
        cursor.executemany(query.replace("?", "%s"), params)
        return cursor

    def executescript(self, script: str) -> None:
        for statement in script.split(";"):
            if statement.strip():
                self.execute(statement)

    def commit(self) -> None:
        self.connection.commit()

    def close(self) -> None:
        self.connection.close()


class DatabaseManager:
    """Manage local SQLite or hosted PostgreSQL storage."""

    def __init__(self, db_path: Optional[Path] = None):
        self.db_path = db_path or DATABASE_PATH
        self.database_url = (
            os.getenv("DATABASE_URL", DATABASE_URL) if db_path is None else ""
        )
        if not self.database_url:
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _get_conn(self) -> sqlite3.Connection | _PostgresConnection:
        """Open a connection using the configured persistent backend."""
        if self.database_url:
            try:
                import psycopg
                from psycopg.rows import dict_row
            except ImportError as exc:
                raise RuntimeError(
                    "PostgreSQL memerlukan psycopg. Pasang dependensi dari "
                    "requirements.txt sebelum mengatur DATABASE_URL."
                ) from exc
            connection = psycopg.connect(
                self.database_url,
                row_factory=dict_row,
            )
            return _PostgresConnection(connection)

        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        return conn

    def _init_db(self):
        """Inisialisasi tabel-tabel database."""
        conn = self._get_conn()
        try:
            schema = """
                -- Tabel tim
                CREATE TABLE IF NOT EXISTS teams (
                    team_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT UNIQUE NOT NULL,
                    country TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );

                -- Tabel kompetisi
                CREATE TABLE IF NOT EXISTS competitions (
                    competition_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    code TEXT UNIQUE NOT NULL,
                    name TEXT NOT NULL,
                    country TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );

                -- Tabel pertandingan (data historis)
                CREATE TABLE IF NOT EXISTS matches (
                    match_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    external_id TEXT,
                    competition_code TEXT NOT NULL,
                    season TEXT,
                    matchday INTEGER,
                    date_utc TEXT NOT NULL,
                    date_wib TEXT,
                    status TEXT DEFAULT 'FINISHED',
                    home_team TEXT NOT NULL,
                    away_team TEXT NOT NULL,
                    home_score INTEGER,
                    away_score INTEGER,
                    home_ht_score INTEGER,
                    away_ht_score INTEGER,
                    home_shots INTEGER,
                    away_shots INTEGER,
                    home_shots_on_target INTEGER,
                    away_shots_on_target INTEGER,
                    home_corners INTEGER,
                    away_corners INTEGER,
                    home_fouls INTEGER,
                    away_fouls INTEGER,
                    home_yellow INTEGER,
                    away_yellow INTEGER,
                    home_red INTEGER,
                    away_red INTEGER,
                    source TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(date_utc, home_team, away_team)
                );

                -- Tabel jadwal laga mendatang
                CREATE TABLE IF NOT EXISTS fixtures (
                    fixture_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    external_id TEXT,
                    competition_code TEXT NOT NULL,
                    season TEXT,
                    matchday INTEGER,
                    date_utc TEXT NOT NULL,
                    date_wib TEXT,
                    home_team TEXT NOT NULL,
                    away_team TEXT NOT NULL,
                    venue TEXT,
                    status TEXT DEFAULT 'SCHEDULED',
                    source TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(date_utc, home_team, away_team)
                );

                -- Tabel Elo rating
                CREATE TABLE IF NOT EXISTS elo_ratings (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    team TEXT NOT NULL,
                    rating REAL NOT NULL,
                    date_updated TEXT NOT NULL,
                    match_id INTEGER,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );

                -- Tabel prediksi (untuk tracking)
                CREATE TABLE IF NOT EXISTS predictions (
                    prediction_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    fixture_id INTEGER,
                    date_predicted TEXT,
                    home_team TEXT NOT NULL,
                    away_team TEXT NOT NULL,
                    prob_home REAL,
                    prob_draw REAL,
                    prob_away REAL,
                    expected_home_goals REAL,
                    expected_away_goals REAL,
                    prob_over_25 REAL,
                    prob_btts REAL,
                    model_version TEXT,
                    last_n INTEGER,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );

                -- Indeks untuk performa query
                CREATE INDEX IF NOT EXISTS idx_matches_date ON matches(date_utc);
                CREATE INDEX IF NOT EXISTS idx_matches_home ON matches(home_team);
                CREATE INDEX IF NOT EXISTS idx_matches_away ON matches(away_team);
                CREATE INDEX IF NOT EXISTS idx_matches_comp ON matches(competition_code);
                CREATE INDEX IF NOT EXISTS idx_fixtures_date ON fixtures(date_utc);
                CREATE INDEX IF NOT EXISTS idx_elo_team ON elo_ratings(team);
                CREATE INDEX IF NOT EXISTS idx_elo_date ON elo_ratings(date_updated);
            """
            if self.database_url:
                schema = schema.replace(
                    "INTEGER PRIMARY KEY AUTOINCREMENT",
                    "INTEGER GENERATED BY DEFAULT AS IDENTITY PRIMARY KEY",
                )
                schema = schema.replace("matchday INTEGER", "matchday TEXT")
            conn.executescript(schema)
            if self.database_url:
                for table in ("matches", "fixtures"):
                    conn.execute(
                        f"ALTER TABLE {table} "
                        "ALTER COLUMN matchday TYPE TEXT USING matchday::text"
                    )
            conn.commit()
            logger.info(
                "Database diinisialisasi: %s",
                "PostgreSQL" if self.database_url else self.db_path,
            )
        finally:
            conn.close()

    @staticmethod
    def _read_query(
        conn: sqlite3.Connection | _PostgresConnection,
        query: str,
        params: tuple | list = (),
    ) -> pd.DataFrame:
        cursor = conn.execute(query, params)
        rows = cursor.fetchall()
        columns = [column[0] for column in cursor.description or ()]
        return pd.DataFrame([dict(row) for row in rows], columns=columns)

    def insert_matches(self, df: pd.DataFrame) -> int:
        """
        Masukkan data pertandingan ke database.
        Abaikan pertandingan dengan kunci unik yang sudah tersimpan.
        Returns: jumlah baris baru yang ditambahkan.
        """
        conn = self._get_conn()
        count = 0
        try:
            for _, row in df.iterrows():
                cursor = conn.execute("""
                    INSERT INTO matches
                    (external_id, competition_code, season, matchday,
                     date_utc, date_wib, status,
                     home_team, away_team, home_score, away_score,
                     home_ht_score, away_ht_score,
                     home_shots, away_shots,
                     home_shots_on_target, away_shots_on_target,
                     home_corners, away_corners,
                     home_fouls, away_fouls,
                     home_yellow, away_yellow, home_red, away_red,
                     source)
                    VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                    ON CONFLICT (date_utc, home_team, away_team) DO NOTHING
                """, (
                    row.get("external_id"),
                    row.get("competition_code", ""),
                    row.get("season"),
                    row.get("matchday"),
                    row.get("date_utc", ""),
                    row.get("date_wib"),
                    row.get("status", "FINISHED"),
                    row.get("home_team", ""),
                    row.get("away_team", ""),
                    row.get("home_score"),
                    row.get("away_score"),
                    row.get("home_ht_score"),
                    row.get("away_ht_score"),
                    row.get("home_shots"),
                    row.get("away_shots"),
                    row.get("home_shots_on_target"),
                    row.get("away_shots_on_target"),
                    row.get("home_corners"),
                    row.get("away_corners"),
                    row.get("home_fouls"),
                    row.get("away_fouls"),
                    row.get("home_yellow"),
                    row.get("away_yellow"),
                    row.get("home_red"),
                    row.get("away_red"),
                    row.get("source", ""),
                ))
                if cursor.rowcount > 0:
                    count += 1
            conn.commit()
            logger.info("Ditambahkan %d pertandingan baru ke database", count)
        finally:
            conn.close()
        return count

    def has_finished_matches_for_season(
        self, competition_code: str, season_year: int
    ) -> bool:
        """Check whether finished results for a season are already stored."""
        season = str(season_year)
        conn = self._get_conn()
        try:
            row = conn.execute("""
                SELECT 1
                FROM matches
                WHERE competition_code = ?
                  AND status = 'FINISHED'
                  AND home_score IS NOT NULL
                  AND away_score IS NOT NULL
                  AND (
                      season = ?
                      OR season LIKE ?
                      OR season LIKE ?
                  )
                LIMIT 1
            """, (competition_code, season, f"{season}-%", f"%-{season}")).fetchone()
            return row is not None
        finally:
            conn.close()

    def insert_fixtures(self, df: pd.DataFrame) -> int:
        """Masukkan jadwal laga mendatang."""
        conn = self._get_conn()
        count = 0
        try:
            for _, row in df.iterrows():
                conn.execute("""
                    INSERT INTO fixtures
                    (external_id, competition_code, season, matchday,
                     date_utc, date_wib, home_team, away_team,
                     venue, status, source)
                    VALUES (?,?,?,?,?,?,?,?,?,?,?)
                    ON CONFLICT (date_utc, home_team, away_team) DO UPDATE SET
                        external_id = excluded.external_id,
                        competition_code = excluded.competition_code,
                        season = excluded.season,
                        matchday = excluded.matchday,
                        date_wib = excluded.date_wib,
                        venue = excluded.venue,
                        status = excluded.status,
                        source = excluded.source
                """, (
                    row.get("external_id"),
                    row.get("competition_code", ""),
                    row.get("season"),
                    row.get("matchday"),
                    row.get("date_utc", ""),
                    row.get("date_wib"),
                    row.get("home_team", ""),
                    row.get("away_team", ""),
                    row.get("venue"),
                    row.get("status", "SCHEDULED"),
                    row.get("source", ""),
                ))
                count += 1
            conn.commit()
            logger.info("Ditambahkan/diperbarui %d jadwal laga", count)
        finally:
            conn.close()
        return count

    def get_team_matches(self, team: str, before_date: str = None,
                         limit: int = 10) -> pd.DataFrame:
        """
        Ambil N laga terakhir sebuah tim (kandang atau tandang).
        Data sebelum before_date untuk menghindari data leakage.
        """
        conn = self._get_conn()
        try:
            known_names = conn.execute("""
                SELECT home_team AS team_name FROM matches
                UNION
                SELECT away_team AS team_name FROM matches
            """).fetchall()
            team_names = {
                row["team_name"]
                for row in known_names
                if normalize_team_name(row["team_name"]) == normalize_team_name(team)
            }
            team_names.add(team)
            placeholders = ", ".join("?" for _ in team_names)
            query = """
                SELECT * FROM matches
                WHERE (home_team IN ({}) OR away_team IN ({}))
                AND status = 'FINISHED'
                AND home_score IS NOT NULL
            """.format(placeholders, placeholders)
            team_name_list = list(team_names)
            params = [*team_name_list, *team_name_list]
            if before_date:
                query += " AND date_utc < ?"
                params.append(before_date)
            query += " ORDER BY date_utc DESC LIMIT ?"
            params.append(limit)

            return self._read_query(conn, query, params)
        finally:
            conn.close()

    def get_all_team_names(self) -> list[str]:
        """Return team names found in finished results or stored fixtures."""
        conn = self._get_conn()
        try:
            rows = conn.execute("""
                SELECT team_name
                FROM (
                    SELECT home_team AS team_name FROM matches
                    WHERE status = 'FINISHED' AND home_score IS NOT NULL
                    UNION ALL
                    SELECT away_team FROM matches
                    WHERE status = 'FINISHED' AND home_score IS NOT NULL
                    UNION ALL
                    SELECT home_team FROM fixtures
                    UNION ALL
                    SELECT away_team FROM fixtures
                )
                WHERE team_name IS NOT NULL AND TRIM(team_name) != ''
                GROUP BY team_name
                ORDER BY COUNT(*) DESC, LOWER(team_name) ASC, team_name ASC
            """).fetchall()
            return [row["team_name"] for row in rows]
        finally:
            conn.close()

    def get_all_team_recent_matches(
        self, before_date: str, limit: int = 5
    ) -> dict[str, pd.DataFrame]:
        """Fetch each team's recent finished matches in one database pass."""
        conn = self._get_conn()
        try:
            rows = conn.execute("""
                SELECT * FROM matches
                WHERE status = 'FINISHED'
                AND home_score IS NOT NULL
                AND date_utc < ?
                ORDER BY date_utc DESC
            """, (before_date,)).fetchall()
            histories: dict[str, list[dict]] = {}
            for row in rows:
                match = dict(row)
                for team in (match["home_team"], match["away_team"]):
                    key = normalize_team_name(team)
                    history = histories.setdefault(key, [])
                    if len(history) < limit:
                        history.append(match)
            return {
                team_key: pd.DataFrame(matches)
                for team_key, matches in histories.items()
            }
        finally:
            conn.close()

    def get_fixtures_by_date(self, date_str: str,
                              competition_code: str = None) -> pd.DataFrame:
        """Ambil jadwal laga pada tanggal tertentu."""
        conn = self._get_conn()
        try:
            query = """
                SELECT * FROM fixtures
                WHERE date_wib LIKE ?
                AND status IN ('SCHEDULED', 'TIMED', 'NOT_STARTED')
            """
            params = [f"{date_str}%"]
            if competition_code:
                query += " AND competition_code = ?"
                params.append(competition_code)
            query += " ORDER BY date_wib ASC"

            return self._read_query(conn, query, params)
        finally:
            conn.close()

    def get_fixture(self, fixture_id: int) -> Optional[dict]:
        """Get one scheduled fixture by its local database ID."""
        conn = self._get_conn()
        try:
            row = conn.execute(
                "SELECT * FROM fixtures WHERE fixture_id = ?",
                (fixture_id,),
            ).fetchone()
            return dict(row) if row else None
        finally:
            conn.close()

    def get_all_matches(self, competition_code: str = None) -> pd.DataFrame:
        """Ambil semua data pertandingan (untuk training)."""
        conn = self._get_conn()
        try:
            query = """
                SELECT * FROM matches
                WHERE status = 'FINISHED'
                AND home_score IS NOT NULL
            """
            params = []
            if competition_code:
                query += " AND competition_code = ?"
                params.append(competition_code)
            query += " ORDER BY date_utc ASC"

            return self._read_query(conn, query, params)
        finally:
            conn.close()

    def update_elo(self, team: str, rating: float, date: str,
                   match_id: int = None):
        """Simpan/update Elo rating tim."""
        conn = self._get_conn()
        try:
            conn.execute("""
                INSERT INTO elo_ratings (team, rating, date_updated, match_id)
                VALUES (?, ?, ?, ?)
            """, (team, rating, date, match_id))
            conn.commit()
        finally:
            conn.close()

    def get_latest_elo(self, team: str,
                       before_date: str = None) -> Optional[float]:
        """Ambil Elo rating terbaru sebuah tim."""
        conn = self._get_conn()
        try:
            query = """
                SELECT rating FROM elo_ratings
                WHERE team = ?
            """
            params = [team]
            if before_date:
                query += " AND date_updated < ?"
                params.append(before_date)
            query += " ORDER BY date_updated DESC LIMIT 1"

            row = conn.execute(query, params).fetchone()
            return row["rating"] if row else None
        finally:
            conn.close()

    def save_prediction(self, prediction: dict):
        """Simpan hasil prediksi."""
        conn = self._get_conn()
        try:
            conn.execute("""
                INSERT INTO predictions
                (fixture_id, date_predicted, home_team, away_team,
                 prob_home, prob_draw, prob_away,
                 expected_home_goals, expected_away_goals,
                 prob_over_25, prob_btts, model_version, last_n)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)
            """, (
                prediction.get("fixture_id"),
                prediction.get("date_predicted"),
                prediction["home_team"],
                prediction["away_team"],
                prediction.get("prob_home"),
                prediction.get("prob_draw"),
                prediction.get("prob_away"),
                prediction.get("expected_home_goals"),
                prediction.get("expected_away_goals"),
                prediction.get("prob_over_25"),
                prediction.get("prob_btts"),
                prediction.get("model_version"),
                prediction.get("last_n"),
            ))
            conn.commit()
        finally:
            conn.close()

    def import_sqlite_database(self, source_path: Path) -> dict[str, int]:
        """Copy local SQLite rows into the configured hosted PostgreSQL database."""
        if not self.database_url:
            raise RuntimeError(
                "Set DATABASE_URL to the target PostgreSQL database before importing."
            )
        source_path = Path(source_path)
        if not source_path.is_file():
            raise FileNotFoundError(f"SQLite source database not found: {source_path}")

        source_uri = f"{source_path.resolve().as_uri()}?mode=ro"
        source = sqlite3.connect(source_uri, uri=True)
        source.row_factory = sqlite3.Row
        target = self._get_conn()
        tables = {
            "teams": "team_id",
            "competitions": "competition_id",
            "matches": "match_id",
            "fixtures": "fixture_id",
            "elo_ratings": "id",
            "predictions": "prediction_id",
        }
        imported: dict[str, int] = {}
        try:
            for table, primary_key in tables.items():
                rows = source.execute(f"SELECT * FROM {table}").fetchall()
                columns = [
                    column[1]
                    for column in source.execute(
                        f"PRAGMA table_info({table})"
                    ).fetchall()
                    if column[1] != "created_at"
                ]
                if not columns:
                    raise RuntimeError(f"Expected SQLite table is missing: {table}")
                placeholders = ", ".join("?" for _ in columns)
                column_list = ", ".join(columns)
                statement = (
                    f"INSERT INTO {table} ({column_list}) VALUES ({placeholders}) "
                    "ON CONFLICT DO NOTHING"
                )
                cursor = target.executemany(
                    statement,
                    [tuple(row[column] for column in columns) for row in rows],
                )
                imported[table] = max(cursor.rowcount, 0)
                target.execute(
                    "SELECT setval("
                    f"pg_get_serial_sequence('{table}', '{primary_key}'), "
                    f"COALESCE(MAX({primary_key}), 1), COUNT(*) > 0) FROM {table}"
                )
            target.commit()
            logger.info("SQLite database imported to PostgreSQL: %s", imported)
            return imported
        finally:
            source.close()
            target.close()
