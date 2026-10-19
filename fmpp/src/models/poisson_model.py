"""
Model Poisson untuk prediksi jumlah gol (Jalur A).
Model inti: memprediksi expected goals per tim, lalu membangun matriks skor.
"""

import logging
import pickle
from pathlib import Path
from typing import Tuple, Optional

import numpy as np
import pandas as pd
from scipy.stats import poisson
from sklearn.linear_model import PoissonRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from src.config import POISSON_MAX_GOALS, OVER_UNDER_LINES, MODELS_DIR, RANDOM_SEED

logger = logging.getLogger(__name__)

# Fitur yang dipakai model Poisson
POISSON_FEATURES = [
    "home_avg_goals_scored", "home_avg_goals_conceded",
    "home_points_per_game", "home_win_rate",
    "home_clean_sheet_pct", "home_failed_to_score_pct",
    "home_over_25_pct", "home_elo",
    "home_rest_days", "home_matches_played",
    "away_avg_goals_scored", "away_avg_goals_conceded",
    "away_points_per_game", "away_win_rate",
    "away_clean_sheet_pct", "away_failed_to_score_pct",
    "away_over_25_pct", "away_elo",
    "away_rest_days", "away_matches_played",
    "elo_diff", "avg_goals_scored_diff", "avg_goals_conceded_diff",
    "points_per_game_diff",
]

FEATURE_LABELS = {
    "home_avg_goals_scored": "Rata-rata gol kandang",
    "home_avg_goals_conceded": "Rata-rata kebobolan kandang",
    "home_points_per_game": "Poin per laga kandang",
    "home_win_rate": "Rasio kemenangan kandang",
    "home_clean_sheet_pct": "Rasio clean sheet kandang",
    "home_failed_to_score_pct": "Rasio gagal mencetak gol kandang",
    "home_over_25_pct": "Rasio laga Over 2.5 kandang",
    "home_elo": "Rating Elo kandang",
    "home_rest_days": "Hari istirahat kandang",
    "home_matches_played": "Jumlah laga kandang dalam riwayat",
    "away_avg_goals_scored": "Rata-rata gol tandang",
    "away_avg_goals_conceded": "Rata-rata kebobolan tandang",
    "away_points_per_game": "Poin per laga tandang",
    "away_win_rate": "Rasio kemenangan tandang",
    "away_clean_sheet_pct": "Rasio clean sheet tandang",
    "away_failed_to_score_pct": "Rasio gagal mencetak gol tandang",
    "away_over_25_pct": "Rasio laga Over 2.5 tandang",
    "away_elo": "Rating Elo tandang",
    "away_rest_days": "Hari istirahat tandang",
    "away_matches_played": "Jumlah laga tandang dalam riwayat",
    "elo_diff": "Selisih rating Elo",
    "avg_goals_scored_diff": "Selisih rata-rata gol tercetak",
    "avg_goals_conceded_diff": "Selisih rata-rata kebobolan",
    "points_per_game_diff": "Selisih poin per laga",
}


class PoissonGoalModel:
    """
    Model Poisson untuk prediksi gol.

    Memprediksi lambda (expected goals) untuk tim kandang dan tandang,
    lalu membangun matriks skor P(home_goals=i, away_goals=j).
    """

    def __init__(self):
        self.home_model = Pipeline([
            ("scaler", StandardScaler()),
            ("poisson", PoissonRegressor(alpha=1.0, max_iter=1000))
        ])
        self.away_model = Pipeline([
            ("scaler", StandardScaler()),
            ("poisson", PoissonRegressor(alpha=1.0, max_iter=1000))
        ])
        self.is_trained = False
        self.feature_names = POISSON_FEATURES
        self.max_goals = POISSON_MAX_GOALS
        self.score_temperature = 1.0
        self.calibration_method = "uncalibrated"

    def _prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Siapkan fitur, isi NaN dengan 0."""
        features = df[self.feature_names].copy()
        features = features.fillna(0)
        return features

    def train(self, train_df: pd.DataFrame):
        """
        Latih model Poisson dari dataset training.

        Args:
            train_df: DataFrame dengan fitur dan target_home_goals/target_away_goals
        """
        X = self._prepare_features(train_df)
        y_home = train_df["target_home_goals"].values
        y_away = train_df["target_away_goals"].values

        logger.info("Training Poisson model - Home goals...")
        self.home_model.fit(X, y_home)

        logger.info("Training Poisson model - Away goals...")
        self.away_model.fit(X, y_away)

        self.is_trained = True

        # Log performa training
        home_pred = self.home_model.predict(X)
        away_pred = self.away_model.predict(X)
        home_mae = np.mean(np.abs(y_home - home_pred))
        away_mae = np.mean(np.abs(y_away - away_pred))

        logger.info("Training MAE - Home: %.3f, Away: %.3f", home_mae, away_mae)

    def predict_goals(self, features: dict) -> Tuple[float, float]:
        """
        Prediksi expected goals (lambda) untuk kedua tim.

        Returns:
            (lambda_home, lambda_away)
        """
        if not self.is_trained:
            raise RuntimeError("Model belum dilatih! Jalankan train() dulu.")

        X = pd.DataFrame([features])[self.feature_names].fillna(0)

        lambda_home = float(self.home_model.predict(X)[0])
        lambda_away = float(self.away_model.predict(X)[0])

        # Clamp ke range wajar
        lambda_home = max(0.1, min(lambda_home, 5.0))
        lambda_away = max(0.1, min(lambda_away, 5.0))

        return lambda_home, lambda_away

    def explain_features(self, features: dict, limit: int = 3) -> list[dict]:
        """Return the largest standardized linear contributions to both goal rates."""
        if not self.is_trained:
            raise RuntimeError("Model belum dilatih! Jalankan train() dulu.")
        X = pd.DataFrame([features])[self.feature_names].fillna(0)
        home_scaled = self.home_model.named_steps["scaler"].transform(X)[0]
        away_scaled = self.away_model.named_steps["scaler"].transform(X)[0]
        home_effects = home_scaled * self.home_model.named_steps["poisson"].coef_
        away_effects = away_scaled * self.away_model.named_steps["poisson"].coef_
        ranked_indices = np.argsort(
            np.abs(home_effects) + np.abs(away_effects)
        )[::-1][:limit]
        return [
            {
                "feature": self.feature_names[index],
                "label": FEATURE_LABELS[self.feature_names[index]],
                "home_goal_log_effect": round(float(home_effects[index]), 3),
                "away_goal_log_effect": round(float(away_effects[index]), 3),
                "impact": round(float(abs(home_effects[index]) + abs(away_effects[index])), 3),
            }
            for index in ranked_indices
        ]

    def build_score_matrix(self, lambda_home: float,
                           lambda_away: float) -> np.ndarray:
        """
        Bangun matriks skor P(home=i, away=j) dari distribusi Poisson.
        Dengan koreksi Dixon-Coles untuk skor rendah.

        Returns:
            Array 2D shape (max_goals+1, max_goals+1)
        """
        max_g = self.max_goals + 1
        matrix = np.zeros((max_g, max_g))

        for i in range(max_g):
            for j in range(max_g):
                matrix[i][j] = poisson.pmf(i, lambda_home) * poisson.pmf(j, lambda_away)

        # Dixon-Coles correction untuk skor rendah
        # rho parameter (simplified, typical range -0.1 to 0.1)
        rho = self._estimate_rho(lambda_home, lambda_away)

        if rho != 0:
            # Koreksi 0-0
            matrix[0][0] *= (1 - lambda_home * lambda_away * rho)
            # Koreksi 1-0
            matrix[1][0] *= (1 + lambda_away * rho)
            # Koreksi 0-1
            matrix[0][1] *= (1 + lambda_home * rho)
            # Koreksi 1-1
            matrix[1][1] *= (1 - rho)

        if self.score_temperature != 1.0:
            matrix = np.power(matrix, 1 / self.score_temperature)

        # Normalisasi agar total = 1
        matrix = matrix / matrix.sum()

        return matrix

    def _estimate_rho(self, lambda_home: float, lambda_away: float) -> float:
        """
        Estimasi parameter rho untuk Dixon-Coles.
        Simplified: gunakan heuristik berdasarkan lambda.
        """
        # Saat kedua tim kuat menyerang, skor rendah agak kurang mungkin
        avg_lambda = (lambda_home + lambda_away) / 2
        if avg_lambda < 1.0:
            return -0.05
        elif avg_lambda < 1.5:
            return -0.03
        else:
            return -0.01

    def predict_full(self, features: dict) -> dict:
        """
        Prediksi lengkap untuk satu laga.

        Returns:
            dict berisi semua probabilitas dan metrik.
        """
        lambda_home, lambda_away = self.predict_goals(features)
        matrix = self.build_score_matrix(lambda_home, lambda_away)

        max_g = self.max_goals + 1

        # === 1X2 Probabilities ===
        prob_home = 0.0
        prob_draw = 0.0
        prob_away = 0.0

        for i in range(max_g):
            for j in range(max_g):
                if i > j:
                    prob_home += matrix[i][j]
                elif i == j:
                    prob_draw += matrix[i][j]
                else:
                    prob_away += matrix[i][j]

        # === Over/Under ===
        over_under = {}
        for line in OVER_UNDER_LINES:
            prob_over = 0.0
            for i in range(max_g):
                for j in range(max_g):
                    if (i + j) > line:
                        prob_over += matrix[i][j]
            over_under[f"over_{str(line).replace('.', '_')}"] = prob_over
            over_under[f"under_{str(line).replace('.', '_')}"] = 1 - prob_over

        # Garis seimbang model (di mana Over mendekati 50%)
        balanced_line = self._find_balanced_line(matrix, max_g)

        # === BTTS ===
        prob_btts = 0.0
        for i in range(1, max_g):
            for j in range(1, max_g):
                prob_btts += matrix[i][j]

        # === Top 3 Skor ===
        scores = []
        for i in range(max_g):
            for j in range(max_g):
                scores.append({"score": f"{i}-{j}", "prob": matrix[i][j]})
        scores.sort(key=lambda x: x["prob"], reverse=True)
        top_scores = scores[:3]

        return {
            "lambda_home": round(lambda_home, 2),
            "lambda_away": round(lambda_away, 2),
            "expected_total": round(lambda_home + lambda_away, 2),
            "prob_home": round(prob_home, 4),
            "prob_draw": round(prob_draw, 4),
            "prob_away": round(prob_away, 4),
            "over_under": {k: round(v, 4) for k, v in over_under.items()},
            "balanced_line": balanced_line,
            "prob_btts_yes": round(prob_btts, 4),
            "prob_btts_no": round(1 - prob_btts, 4),
            "top_scores": top_scores,
            "score_matrix": matrix,
        }

    def _find_balanced_line(self, matrix: np.ndarray, max_g: int) -> dict:
        """Cari garis Over/Under yang paling dekat 50%."""
        best_line = 2.5
        best_diff = 1.0
        best_prob = 0.5

        for line_x10 in range(5, 65, 5):  # 0.5 sampai 6.0
            line = line_x10 / 10
            prob_over = 0.0
            for i in range(max_g):
                for j in range(max_g):
                    if (i + j) > line:
                        prob_over += matrix[i][j]
            diff = abs(prob_over - 0.5)
            if diff < best_diff:
                best_diff = diff
                best_line = line
                best_prob = prob_over

        return {
            "line": best_line,
            "over_prob": round(best_prob, 4),
        }

    def save(self, path: Path = None):
        """Simpan model ke file."""
        path = path or (MODELS_DIR / "poisson_model.pkl")
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({
                "home_model": self.home_model,
                "away_model": self.away_model,
                "feature_names": self.feature_names,
                "max_goals": self.max_goals,
                "is_trained": self.is_trained,
                "score_temperature": self.score_temperature,
                "calibration_method": self.calibration_method,
            }, f)
        logger.info("Model Poisson disimpan: %s", path)

    def load(self, path: Path = None):
        """Muat model dari file."""
        path = path or (MODELS_DIR / "poisson_model.pkl")
        with open(path, "rb") as f:
            data = pickle.load(f)
        self.home_model = data["home_model"]
        self.away_model = data["away_model"]
        self.feature_names = data["feature_names"]
        self.max_goals = data["max_goals"]
        self.is_trained = data["is_trained"]
        self.score_temperature = data.get("score_temperature", 1.0)
        self.calibration_method = data.get("calibration_method", "uncalibrated")
        logger.info("Model Poisson dimuat: %s", path)
