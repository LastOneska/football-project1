"""
Feature Engineering untuk FMPP.
Menghitung fitur-fitur dari N laga terakhir setiap tim.
"""

import logging
from typing import Optional
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from src.config import (
    DEFAULT_LAST_N,
    ELO_DEFAULT_RATING,
    ELO_HOME_ADVANTAGE,
    ELO_K_FACTOR,
)
from src.ingestion.team_names import normalize_team_name

logger = logging.getLogger(__name__)


def _utc_naive_timestamp(value) -> pd.Timestamp:
    """Parse a timestamp and normalize it to UTC without timezone metadata."""
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is not None:
        timestamp = timestamp.tz_convert("UTC").tz_localize(None)
    return timestamp


class FeatureEngineer:
    """
    Membangun fitur dari data pertandingan historis.
    Semua fitur dihitung HANYA dari data sebelum tanggal laga (cegah leakage).
    """

    def __init__(self, matches_df: pd.DataFrame, last_n: int = None):
        """
        Args:
            matches_df: DataFrame semua pertandingan historis (sudah sorted by date).
            last_n: Jumlah laga terakhir yang dipakai (default dari config).
        """
        self.last_n = last_n or DEFAULT_LAST_N
        self.matches = matches_df.copy()

        # Pastikan sorted by date
        self.matches["date_utc"] = pd.to_datetime(
            self.matches["date_utc"], errors="coerce", utc=True
        ).dt.tz_localize(None)
        self.matches["_home_team_key"] = self.matches["home_team"].map(normalize_team_name)
        self.matches["_away_team_key"] = self.matches["away_team"].map(normalize_team_name)
        self.matches = self.matches.sort_values("date_utc").reset_index(drop=True)

        # Elo ratings (di-maintain secara berurutan)
        self.elo_ratings = {}

    def get_team_recent_matches(self, team: str, before_date,
                                 n: int = None) -> pd.DataFrame:
        """
        Ambil N laga terakhir tim sebelum tanggal tertentu.
        Tim bisa bermain kandang atau tandang.
        """
        n = n or self.last_n
        team_key = normalize_team_name(team)
        mask = (
            ((self.matches["_home_team_key"] == team_key) |
             (self.matches["_away_team_key"] == team_key)) &
            (self.matches["date_utc"] < _utc_naive_timestamp(before_date)) &
            (self.matches["home_score"].notna())
        )
        recent = self.matches[mask].tail(n)
        return recent

    def compute_form_features(self, team: str, before_date,
                               n: int = None) -> dict:
        """
        Hitung fitur form dari N laga terakhir.
        Returns dict fitur.
        """
        recent = self.get_team_recent_matches(team, before_date, n)
        n_actual = len(recent)

        if n_actual == 0:
            return self._empty_features(team, "form")

        wins = 0
        draws = 0
        losses = 0
        points = 0
        goals_scored = 0
        goals_conceded = 0
        clean_sheets = 0
        failed_to_score = 0
        over_25_count = 0
        btts_count = 0

        home_matches = 0
        home_wins = 0
        away_matches = 0
        away_wins = 0

        current_streak = []  # W, D, L

        for _, match in recent.iterrows():
            is_home = match["home_team"] == team

            if is_home:
                gf = match["home_score"]
                ga = match["away_score"]
                home_matches += 1
            else:
                gf = match["away_score"]
                ga = match["home_score"]
                away_matches += 1

            total_goals = gf + ga

            goals_scored += gf
            goals_conceded += ga

            if gf > ga:
                wins += 1
                points += 3
                current_streak.append("W")
                if is_home:
                    home_wins += 1
                else:
                    away_wins += 1
            elif gf == ga:
                draws += 1
                points += 1
                current_streak.append("D")
            else:
                losses += 1
                current_streak.append("L")

            if ga == 0:
                clean_sheets += 1
            if gf == 0:
                failed_to_score += 1
            if total_goals > 2.5:
                over_25_count += 1
            if gf >= 1 and ga >= 1:
                btts_count += 1

        # Hitung streak terkini
        streak_type = current_streak[-1] if current_streak else "N"
        streak_count = 0
        for s in reversed(current_streak):
            if s == streak_type:
                streak_count += 1
            else:
                break

        features = {
            f"matches_played": n_actual,
            f"wins": wins,
            f"draws": draws,
            f"losses": losses,
            f"points_per_game": round(points / n_actual, 3),
            f"win_rate": round(wins / n_actual, 3),
            f"avg_goals_scored": round(goals_scored / n_actual, 3),
            f"avg_goals_conceded": round(goals_conceded / n_actual, 3),
            f"goal_diff_per_game": round((goals_scored - goals_conceded) / n_actual, 3),
            f"clean_sheet_pct": round(clean_sheets / n_actual, 3),
            f"failed_to_score_pct": round(failed_to_score / n_actual, 3),
            f"over_25_pct": round(over_25_count / n_actual, 3),
            f"btts_pct": round(btts_count / n_actual, 3),
            f"streak_type": streak_type,
            f"streak_count": streak_count,
            f"home_win_rate": round(home_wins / home_matches, 3) if home_matches > 0 else 0.0,
            f"away_win_rate": round(away_wins / away_matches, 3) if away_matches > 0 else 0.0,
        }

        return features

    def compute_rest_days(self, team: str, match_date) -> int:
        """Hitung hari istirahat sejak laga terakhir."""
        recent = self.get_team_recent_matches(team, match_date, n=1)
        if len(recent) == 0:
            return 7  # Default jika tidak ada data

        last_match_date = recent.iloc[-1]["date_utc"]
        match_dt = _utc_naive_timestamp(match_date)
        delta = (match_dt - last_match_date).days
        return max(0, delta)

    def compute_elo_rating(self, team: str) -> float:
        """Ambil Elo rating terkini sebuah tim."""
        return self.elo_ratings.get(normalize_team_name(team), ELO_DEFAULT_RATING)

    def update_elo_from_match(self, home_team: str, away_team: str,
                               home_score: int, away_score: int):
        """
        Update Elo rating setelah pertandingan.
        Memakai home advantage factor.
        """
        home_team = normalize_team_name(home_team)
        away_team = normalize_team_name(away_team)
        r_home = self.elo_ratings.get(home_team, ELO_DEFAULT_RATING)
        r_away = self.elo_ratings.get(away_team, ELO_DEFAULT_RATING)

        # Expected score (dengan home advantage)
        exp_home = 1 / (1 + 10 ** ((r_away - r_home - ELO_HOME_ADVANTAGE) / 400))
        exp_away = 1 - exp_home

        # Actual score
        if home_score > away_score:
            actual_home = 1.0
            actual_away = 0.0
        elif home_score == away_score:
            actual_home = 0.5
            actual_away = 0.5
        else:
            actual_home = 0.0
            actual_away = 1.0

        # Goal difference multiplier
        goal_diff = abs(home_score - away_score)
        if goal_diff <= 1:
            multiplier = 1.0
        elif goal_diff == 2:
            multiplier = 1.5
        else:
            multiplier = (11 + goal_diff) / 8

        # Update
        self.elo_ratings[home_team] = r_home + ELO_K_FACTOR * multiplier * (actual_home - exp_home)
        self.elo_ratings[away_team] = r_away + ELO_K_FACTOR * multiplier * (actual_away - exp_away)

    def initialize_elo(self):
        """
        Inisialisasi Elo dari seluruh data historis secara berurutan.
        Harus dipanggil sebelum build_features_for_match.
        """
        logger.info("Menginisialisasi Elo dari %d pertandingan...", len(self.matches))

        for _, match in self.matches.iterrows():
            if pd.notna(match["home_score"]) and pd.notna(match["away_score"]):
                self.update_elo_from_match(
                    match["home_team"],
                    match["away_team"],
                    int(match["home_score"]),
                    int(match["away_score"])
                )

        logger.info("Elo diinisialisasi untuk %d tim", len(self.elo_ratings))

    def build_features_for_match(self, home_team: str, away_team: str,
                                  match_date, last_n: int = None) -> dict:
        """
        Bangun seluruh fitur untuk satu pertandingan.

        Returns:
            dict berisi semua fitur untuk model.
        """
        n = last_n or self.last_n

        # Fitur form masing-masing tim
        home_form = self.compute_form_features(home_team, match_date, n)
        away_form = self.compute_form_features(away_team, match_date, n)

        # Rest days
        home_rest = self.compute_rest_days(home_team, match_date)
        away_rest = self.compute_rest_days(away_team, match_date)

        # Elo ratings
        home_elo = self.compute_elo_rating(home_team)
        away_elo = self.compute_elo_rating(away_team)

        # Gabungkan semua fitur dengan prefix
        features = {}

        # Home team features
        for key, val in home_form.items():
            if key != "streak_type":  # Skip non-numeric
                features[f"home_{key}"] = val
        features["home_rest_days"] = home_rest
        features["home_elo"] = round(home_elo, 1)

        # Away team features
        for key, val in away_form.items():
            if key != "streak_type":
                features[f"away_{key}"] = val
        features["away_rest_days"] = away_rest
        features["away_elo"] = round(away_elo, 1)

        # Fitur perbandingan
        features["elo_diff"] = round(home_elo - away_elo, 1)
        features["avg_goals_scored_diff"] = round(
            home_form["avg_goals_scored"] - away_form["avg_goals_scored"], 3
        )
        features["avg_goals_conceded_diff"] = round(
            home_form["avg_goals_conceded"] - away_form["avg_goals_conceded"], 3
        )
        features["points_per_game_diff"] = round(
            home_form["points_per_game"] - away_form["points_per_game"], 3
        )

        # Metadata (tidak dipakai model tapi berguna untuk display)
        features["_home_team"] = home_team
        features["_away_team"] = away_team
        features["_match_date"] = str(match_date)
        features["_home_matches_played"] = home_form["matches_played"]
        features["_away_matches_played"] = away_form["matches_played"]
        features["_home_streak"] = f"{home_form['streak_type']}{home_form['streak_count']}"
        features["_away_streak"] = f"{away_form['streak_type']}{away_form['streak_count']}"

        return features

    def build_training_dataset(self) -> pd.DataFrame:
        """
        Bangun dataset training dari seluruh pertandingan historis.
        Setiap baris = satu pertandingan + fitur dari N laga sebelumnya.
        """
        logger.info("Membangun dataset training...")

        # Reset Elo
        self.elo_ratings = {}

        rows = []
        dated_matches = self.matches.dropna(subset=["date_utc"])
        for _, matchday in dated_matches.groupby("date_utc", sort=True):
            # All fixtures at the same timestamp must see the same prior Elo state.
            for _, match in matchday.iterrows():
                if pd.isna(match["home_score"]) or pd.isna(match["away_score"]):
                    continue

                home = match["home_team"]
                away = match["away_team"]
                match_date = match["date_utc"]
                home_recent = self.get_team_recent_matches(home, match_date, self.last_n)
                away_recent = self.get_team_recent_matches(away, match_date, self.last_n)

                if len(home_recent) >= 3 and len(away_recent) >= 3:
                    features = self.build_features_for_match(home, away, match_date)
                    home_goals = int(match["home_score"])
                    away_goals = int(match["away_score"])
                    total_goals = home_goals + away_goals

                    features["target_home_goals"] = home_goals
                    features["target_away_goals"] = away_goals
                    features["target_total_goals"] = total_goals
                    features["target_result_code"] = (
                        0 if home_goals > away_goals else
                        1 if home_goals == away_goals else 2
                    )
                    features["target_over_15"] = int(total_goals > 1.5)
                    features["target_over_25"] = int(total_goals > 2.5)
                    features["target_over_35"] = int(total_goals > 3.5)
                    features["target_over_45"] = int(total_goals > 4.5)
                    features["target_btts"] = int(home_goals >= 1 and away_goals >= 1)
                    features["_competition"] = match.get("competition_code", "")
                    rows.append(features)

            for _, match in matchday.iterrows():
                if pd.notna(match["home_score"]) and pd.notna(match["away_score"]):
                    self.update_elo_from_match(
                        match["home_team"],
                        match["away_team"],
                        int(match["home_score"]),
                        int(match["away_score"]),
                    )

        df = pd.DataFrame(rows)
        logger.info("Dataset training: %d baris, %d fitur", len(df), len(df.columns))
        return df

    def _empty_features(self, team: str, group: str) -> dict:
        """Return fitur kosong/default jika tidak ada data."""
        return {
            "matches_played": 0,
            "wins": 0,
            "draws": 0,
            "losses": 0,
            "points_per_game": 0.0,
            "win_rate": 0.0,
            "avg_goals_scored": 0.0,
            "avg_goals_conceded": 0.0,
            "goal_diff_per_game": 0.0,
            "clean_sheet_pct": 0.0,
            "failed_to_score_pct": 0.0,
            "over_25_pct": 0.0,
            "btts_pct": 0.0,
            "streak_type": "N",
            "streak_count": 0,
            "home_win_rate": 0.0,
            "away_win_rate": 0.0,
        }
