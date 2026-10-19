"""Chronological walk-forward evaluation and baseline comparisons."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss, mean_absolute_error
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV

from src.config import RANDOM_SEED
from src.features.engineer import FeatureEngineer
from src.models.poisson_model import POISSON_FEATURES, PoissonGoalModel

logger = logging.getLogger(__name__)


def _probability_metrics(actual: np.ndarray, probabilities: np.ndarray) -> dict:
    clipped = np.clip(probabilities, 1e-9, 1 - 1e-9)
    clipped = clipped / clipped.sum(axis=1, keepdims=True)
    one_hot = np.eye(3)[actual]
    cumulative_actual = np.cumsum(one_hot, axis=1)[:, :2]
    cumulative_predicted = np.cumsum(clipped, axis=1)[:, :2]
    return {
        "log_loss_1x2": float(log_loss(actual, clipped, labels=[0, 1, 2])),
        "brier_1x2": float(np.mean(np.sum((clipped - one_hot) ** 2, axis=1))),
        "rps_1x2": float(np.mean(np.sum((cumulative_predicted - cumulative_actual) ** 2, axis=1) / 2)),
        "accuracy_1x2": float(accuracy_score(actual, clipped.argmax(axis=1))),
    }


def _binary_metrics(actual: np.ndarray, probability: np.ndarray, prefix: str) -> dict:
    clipped = np.clip(probability, 1e-9, 1 - 1e-9)
    return {
        f"log_loss_{prefix}": float(-np.mean(
            actual * np.log(clipped) + (1 - actual) * np.log(1 - clipped)
        )),
        f"brier_{prefix}": float(np.mean((clipped - actual) ** 2)),
        f"accuracy_{prefix}": float(np.mean((clipped >= 0.5) == actual)),
    }


def _calibration_bins(actual: np.ndarray, probability: np.ndarray, bins: int = 10) -> list[dict]:
    edges = np.linspace(0, 1, bins + 1)
    output = []
    for index in range(bins):
        mask = (
            (probability >= edges[index])
            & ((probability < edges[index + 1]) if index < bins - 1 else (probability <= 1))
        )
        if mask.any():
            output.append({
                "lower": float(edges[index]),
                "upper": float(edges[index + 1]),
                "mean_predicted": float(probability[mask].mean()),
                "observed_frequency": float(actual[mask].mean()),
                "count": int(mask.sum()),
            })
    return output


def _result_probabilities(matrix: np.ndarray) -> np.ndarray:
    return np.asarray([
        np.tril(matrix, -1).sum(),
        np.trace(matrix),
        np.triu(matrix, 1).sum(),
    ])


def _select_score_temperature(records: list[dict]) -> tuple[float, float]:
    """Fit a shared score-matrix temperature from chronological out-of-fold rows."""
    actual = np.asarray([item["actual_result"] for item in records], dtype=int)
    matrices = [item["score_matrix"] for item in records]
    best_temperature = 1.0
    best_loss = float("inf")
    for temperature in np.linspace(0.5, 2.5, 41):
        probabilities = []
        for matrix in matrices:
            scaled = np.power(matrix, 1 / temperature)
            scaled /= scaled.sum()
            probabilities.append(_result_probabilities(scaled))
        loss = float(log_loss(actual, np.asarray(probabilities), labels=[0, 1, 2]))
        if loss < best_loss:
            best_temperature = float(temperature)
            best_loss = loss
    return best_temperature, best_loss


def _summarize(records: list[dict]) -> dict:
    output = {"sample_count": len(records)}

    model_names = ("poisson", "league_frequency", "team_average_goals", "logistic_1x2")
    for name in model_names:
        present = [item for item in records if name in item["probabilities"]]
        if not present:
            continue
        y = np.asarray([item["actual_result"] for item in present], dtype=int)
        p = np.asarray([item["probabilities"][name] for item in present], dtype=float)
        metrics = _probability_metrics(y, p)
        if name != "logistic_1x2":
            over = np.asarray([item["over_25"][name] for item in present])
            btts = np.asarray([item["btts"][name] for item in present])
            actual_over = np.asarray([item["actual_over_25"] for item in present], dtype=int)
            actual_btts = np.asarray([item["actual_btts"] for item in present], dtype=int)
            metrics.update(_binary_metrics(actual_over, over, "over_25"))
            metrics.update(_binary_metrics(actual_btts, btts, "btts"))
            predicted_home = np.asarray([item["expected_goals"][name][0] for item in present])
            predicted_away = np.asarray([item["expected_goals"][name][1] for item in present])
            metrics["mae_home_goals"] = float(mean_absolute_error(
                [item["actual_home_goals"] for item in present], predicted_home
            ))
            metrics["mae_away_goals"] = float(mean_absolute_error(
                [item["actual_away_goals"] for item in present], predicted_away
            ))
            metrics["calibration_over_25"] = _calibration_bins(actual_over, over)
            metrics["calibration_btts"] = _calibration_bins(actual_btts, btts)
        output[name] = metrics
    return output


def walk_forward_evaluate(
    matches: pd.DataFrame,
    last_n: int = 5,
    folds: int = 3,
    minimum_training_samples: int = 30,
) -> dict:
    """Evaluate models on later chronological blocks, using expanding training sets."""
    if not 5 <= last_n <= 10:
        raise ValueError("Jumlah laga terakhir harus antara 5 dan 10.")
    if folds < 1:
        raise ValueError("Jumlah fold harus minimal 1.")
    if matches.empty:
        raise ValueError("Tidak ada data pertandingan untuk dievaluasi.")

    dataset = FeatureEngineer(matches, last_n=last_n).build_training_dataset()
    if dataset.empty:
        raise ValueError("Data historis belum cukup untuk membangun sampel evaluasi.")

    dataset["_evaluation_date"] = pd.to_datetime(dataset["_match_date"], errors="coerce", utc=True)
    dataset = dataset.dropna(subset=["_evaluation_date"]).sort_values("_evaluation_date")
    dates = dataset["_evaluation_date"].drop_duplicates().to_numpy()
    if len(dates) < 2:
        raise ValueError("Diperlukan setidaknya dua tanggal berbeda untuk backtesting.")

    n_folds = min(folds, len(dates) - 1)
    date_blocks = [block for block in np.array_split(dates, n_folds + 1) if len(block)]
    records = []
    completed_folds = 0
    simple_model = PoissonGoalModel()

    for fold_index in range(1, len(date_blocks)):
        training_dates = np.concatenate(date_blocks[:fold_index])
        test_dates = date_blocks[fold_index]
        train = dataset[dataset["_evaluation_date"].isin(training_dates)]
        test = dataset[dataset["_evaluation_date"].isin(test_dates)]
        if len(train) < minimum_training_samples or test.empty:
            logger.warning(
                "Fold %d dilewati: train=%d, test=%d",
                fold_index, len(train), len(test),
            )
            continue

        goal_model = PoissonGoalModel()
        goal_model.train(train)
        classifier_model = make_pipeline(
            StandardScaler(),
            LogisticRegression(
                max_iter=1000,
                random_state=RANDOM_SEED,
            ),
        )
        classifier_features = goal_model._prepare_features(train)
        classifier_targets = train["target_result_code"].astype(int).to_numpy()
        unique_training_dates = train["_evaluation_date"].drop_duplicates().to_numpy()
        if len(unique_training_dates) >= 5:
            calibration_boundary = unique_training_dates[
                max(1, int(len(unique_training_dates) * 0.8))
            ]
            base_indices = np.flatnonzero(
                (train["_evaluation_date"] < calibration_boundary).to_numpy()
            )
            calibration_indices = np.flatnonzero(
                (train["_evaluation_date"] >= calibration_boundary).to_numpy()
            )
        else:
            base_indices = np.array([], dtype=int)
            calibration_indices = np.array([], dtype=int)

        if (
            len(base_indices) >= 20
            and len(calibration_indices) >= 10
            and len(np.unique(classifier_targets[base_indices])) == 3
        ):
            classifier = CalibratedClassifierCV(
                estimator=classifier_model,
                method="sigmoid",
                cv=[(base_indices, calibration_indices)],
            )
            classifier.fit(classifier_features, classifier_targets)
        else:
            classifier = classifier_model.fit(classifier_features, classifier_targets)

        global_result_rates = np.bincount(
            train["target_result_code"].astype(int), minlength=3
        ).astype(float)
        global_result_rates /= global_result_rates.sum()
        global_over = float(train["target_over_25"].mean())
        global_btts = float(train["target_btts"].mean())
        global_home_avg = float(train["target_home_goals"].mean())
        global_away_avg = float(train["target_away_goals"].mean())

        for _, row in test.iterrows():
            features = {name: row[name] for name in POISSON_FEATURES}
            result = goal_model.predict_full(features)
            home_lambda, away_lambda = result["lambda_home"], result["lambda_away"]
            team_home = max(
                0.1,
                min(5.0, (row["home_avg_goals_scored"] + row["away_avg_goals_conceded"]) / 2),
            )
            team_away = max(
                0.1,
                min(5.0, (row["away_avg_goals_scored"] + row["home_avg_goals_conceded"]) / 2),
            )
            league_rows = train[train["_competition"] == row["_competition"]]
            if league_rows.empty:
                league_result_rates = global_result_rates
                league_over = global_over
                league_btts = global_btts
                league_home_avg, league_away_avg = global_home_avg, global_away_avg
            else:
                league_counts = np.bincount(
                    league_rows["target_result_code"].astype(int), minlength=3
                ).astype(float)
                league_result_rates = league_counts / league_counts.sum()
                league_over = float(league_rows["target_over_25"].mean())
                league_btts = float(league_rows["target_btts"].mean())
                league_home_avg = float(league_rows["target_home_goals"].mean())
                league_away_avg = float(league_rows["target_away_goals"].mean())

            team_matrix = simple_model.build_score_matrix(team_home, team_away)
            league_matrix = simple_model.build_score_matrix(league_home_avg, league_away_avg)
            score_grid = np.add.outer(
                np.arange(team_matrix.shape[0]), np.arange(team_matrix.shape[1])
            )
            team_probabilities = np.asarray([
                np.tril(team_matrix, -1).sum(),
                np.trace(team_matrix),
                np.triu(team_matrix, 1).sum(),
            ])
            league_goal_probabilities = np.asarray([
                np.tril(league_matrix, -1).sum(),
                np.trace(league_matrix),
                np.triu(league_matrix, 1).sum(),
            ])
            logistic_probabilities = np.zeros(3)
            logistic_probabilities[classifier.classes_.astype(int)] = classifier.predict_proba(
                goal_model._prepare_features(pd.DataFrame([features]))
            )[0]

            actual_home = int(row["target_home_goals"])
            actual_away = int(row["target_away_goals"])
            actual_total = actual_home + actual_away
            actual_result = int(row["target_result_code"])
            records.append({
                "competition": str(row["_competition"]),
                "date": row["_evaluation_date"].date().isoformat(),
                "home_team": row["_home_team"],
                "away_team": row["_away_team"],
                "actual_result": actual_result,
                "actual_home_goals": actual_home,
                "actual_away_goals": actual_away,
                "actual_over_25": int(actual_total > 2.5),
                "actual_btts": int(actual_home > 0 and actual_away > 0),
                "probabilities": {
                    "poisson": np.asarray([
                        result["prob_home"], result["prob_draw"], result["prob_away"]
                    ]),
                    "league_frequency": league_result_rates,
                    "team_average_goals": team_probabilities,
                    "logistic_1x2": logistic_probabilities,
                },
                "over_25": {
                    "poisson": result["over_under"]["over_2_5"],
                    "league_frequency": league_over,
                    "team_average_goals": float(team_matrix[score_grid > 2.5].sum()),
                },
                "btts": {
                    "poisson": result["prob_btts_yes"],
                    "league_frequency": league_btts,
                    "team_average_goals": float(team_matrix[1:, 1:].sum()),
                },
                "expected_goals": {
                    "poisson": (home_lambda, away_lambda),
                    "league_frequency": (league_home_avg, league_away_avg),
                    "team_average_goals": (team_home, team_away),
                },
                "predicted_result": int(np.argmax([
                    result["prob_home"], result["prob_draw"], result["prob_away"]
                ])),
                "score_matrix": result["score_matrix"],
            })
        completed_folds += 1

    if not records:
        raise ValueError(
            "Tidak ada fold evaluasi yang memenuhi jumlah data minimum; "
            "tambahkan data historis atau kurangi minimum_training_samples."
        )

    record_frame = pd.DataFrame(records)
    per_competition = {
        str(competition): _summarize(group.to_dict("records"))
        for competition, group in record_frame.groupby("competition")
    }
    score_temperature, calibrated_oof_loss = _select_score_temperature(records)
    incorrect = [
        {
            "date": item["date"],
            "competition": item["competition"],
            "match": f"{item['home_team']} vs {item['away_team']}",
            "actual_result": ("H", "D", "A")[item["actual_result"]],
            "predicted_result": ("H", "D", "A")[item["predicted_result"]],
        }
        for item in records
        if item["actual_result"] != item["predicted_result"]
    ]
    return {
        "last_n": last_n,
        "folds_requested": folds,
        "folds_completed": completed_folds,
        "sample_count": len(records),
        "overall": _summarize(records),
        "per_competition": per_competition,
        "calibration": {
            "method": "score-matrix temperature scaling",
            "fit": "expanding-window out-of-fold predictions",
            "temperature": score_temperature,
            "oof_log_loss_1x2": calibrated_oof_loss,
            "fit_sample_count": len(records),
            "reported_metrics_use_uncalibrated_fold_predictions": True,
        },
        "error_examples": incorrect[:20],
        "limitations": [
            "Logistic Regression dibandingkan sebagai model langsung 1X2; kalibrasi sigmoid memakai bagian akhir data train bila sampelnya memadai.",
            "Odds penutupan hanya dapat dievaluasi jika tersedia pada data sumber.",
        ],
    }
