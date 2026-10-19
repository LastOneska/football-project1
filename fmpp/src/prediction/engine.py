"""Prediction orchestration using stored match history and a trained Poisson model."""

from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

from src.config import DEFAULT_LAST_N, MODELS_DIR, TIMEZONE
from src.features.engineer import FeatureEngineer, _utc_naive_timestamp
from src.ingestion.database import DatabaseManager
from src.models.poisson_model import PoissonGoalModel


class PredictionEngine:
    """Build leakage-safe features and format a full match prediction."""

    def __init__(
        self,
        database: DatabaseManager | None = None,
        model_path: Path | None = None,
    ):
        self.database = database or DatabaseManager()
        self.model_path = model_path or (MODELS_DIR / "poisson_model.pkl")

    def predict(
        self,
        home_team: str,
        away_team: str,
        match_date: datetime | str | None = None,
        last_n: int = DEFAULT_LAST_N,
        include_score_matrix: bool = False,
    ) -> dict:
        if not home_team.strip() or not away_team.strip():
            raise ValueError("Nama tim kandang dan tandang wajib diisi.")
        if home_team.strip().casefold() == away_team.strip().casefold():
            raise ValueError("Tim kandang dan tandang harus berbeda.")
        if not 5 <= last_n <= 10:
            raise ValueError("Jumlah laga terakhir harus antara 5 dan 10.")
        if not self.model_path.is_file():
            raise FileNotFoundError(
                f"Model belum tersedia di {self.model_path}. Jalankan `fmpp train` dahulu."
            )

        prediction_date = match_date or datetime.now(timezone.utc)
        prediction_timestamp = _utc_naive_timestamp(prediction_date)
        matches = self.database.get_all_matches()
        if matches.empty:
            raise ValueError(
                "Belum ada data pertandingan historis. Jalankan `fmpp update-data` dahulu."
            )

        engineer = FeatureEngineer(matches, last_n=last_n)
        for _, match in engineer.matches.iterrows():
            if pd.isna(match["date_utc"]):
                continue
            if match["date_utc"] >= prediction_timestamp:
                break
            if pd.notna(match["home_score"]) and pd.notna(match["away_score"]):
                engineer.update_elo_from_match(
                    match["home_team"],
                    match["away_team"],
                    int(match["home_score"]),
                    int(match["away_score"]),
                )

        features = engineer.build_features_for_match(
            home_team.strip(), away_team.strip(), prediction_timestamp, last_n
        )
        model = PoissonGoalModel()
        model.load(self.model_path)
        result = model.predict_full(features)

        home_played = features["_home_matches_played"]
        away_played = features["_away_matches_played"]
        minimum_played = min(home_played, away_played)
        if minimum_played >= last_n:
            data_confidence = "Tinggi"
        elif minimum_played >= 3:
            data_confidence = "Sedang"
        else:
            data_confidence = "Rendah"

        prediction_datetime = prediction_timestamp.replace(tzinfo=timezone.utc)
        response = {
            "home_team": home_team.strip(),
            "away_team": away_team.strip(),
            "match_date_utc": prediction_timestamp.isoformat(sep=" ", timespec="minutes"),
            "match_time_wib": prediction_datetime.astimezone(
                ZoneInfo(TIMEZONE)
            ).strftime("%Y-%m-%d %H:%M"),
            "last_n": last_n,
            "home_matches_played": home_played,
            "away_matches_played": away_played,
            "data_confidence": data_confidence,
            "data_warning": (
                f"Riwayat kurang dari {last_n} laga untuk setidaknya satu tim."
                if minimum_played < last_n
                else None
            ),
            "probability_calibration": {
                "method": model.calibration_method,
                "temperature": model.score_temperature,
            },
            "form": {
                "home_points_per_game": features["home_points_per_game"],
                "away_points_per_game": features["away_points_per_game"],
                "home_avg_goals_scored": features["home_avg_goals_scored"],
                "away_avg_goals_scored": features["away_avg_goals_scored"],
                "elo_difference": features["elo_diff"],
            },
            "explanation": model.explain_features(features),
            **{key: value for key, value in result.items() if key != "score_matrix"},
        }
        if include_score_matrix:
            response["score_matrix"] = result["score_matrix"].tolist()
        return response
