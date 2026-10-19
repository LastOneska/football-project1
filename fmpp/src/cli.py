"""Command-line interface for FMPP."""

from datetime import datetime, timedelta
import csv
import io
import json
import logging
from pathlib import Path
from zoneinfo import ZoneInfo

import typer
from rich.console import Console
from rich.table import Table

from src.config import (
    API_FOOTBALL_KEY,
    COMPETITIONS,
    DATABASE_PATH,
    DATABASE_URL,
    DEFAULT_LAST_N,
    HISTORICAL_COMPETITIONS,
    MODELS_DIR,
    TIMEZONE,
)
from src.features.engineer import FeatureEngineer
from src.ingestion.api_football import ApiFootballClient
from src.ingestion.database import DatabaseManager
from src.ingestion.schedule import refresh_upcoming_fixtures
from src.models.poisson_model import PoissonGoalModel
from src.prediction.engine import PredictionEngine


app = typer.Typer(
    name="fmpp",
    help="Analisis probabilitas pertandingan sepak bola (untuk pembelajaran).",
    no_args_is_help=True,
)
console = Console()
logger = logging.getLogger("fmpp")
MODEL_PATH = MODELS_DIR / "poisson_model.pkl"


def _parse_seasons(value: str) -> list[int]:
    try:
        seasons = sorted({int(part.strip()) for part in value.split(",") if part.strip()})
    except ValueError as exc:
        raise typer.BadParameter("Gunakan tahun yang dipisahkan koma, misalnya 2022,2023.") from exc
    if not seasons or any(season < 1900 or season > 2100 for season in seasons):
        raise typer.BadParameter("Daftar musim harus berisi tahun valid.")
    return seasons


def _as_json_safe(prediction: dict) -> dict:
    """Convert NumPy scalar values nested in model output to JSON primitives."""
    return json.loads(json.dumps(prediction, default=lambda value: value.item()))


def _print_prediction(prediction: dict) -> None:
    data = prediction["data"]
    competition_code = data.get("competition_code")
    competition = (
        COMPETITIONS.get(competition_code, {}).get("name", competition_code)
        if competition_code
        else "Pertandingan manual"
    )
    match_time = data.get("date_wib") or f"{data['match_time_wib']} WIB"
    result = Table(
        title=f"{data['home_team']} vs {data['away_team']} | {competition} | {match_time}"
    )
    result.add_column("Metrik")
    result.add_column("Probabilitas / Nilai", justify="right")
    result.add_row(
        "1X2",
        f"Kandang {data['prob_home']:.1%} | Seri {data['prob_draw']:.1%} | "
        f"Tandang {data['prob_away']:.1%}",
    )
    result.add_row(
        "Ekspektasi gol",
        f"{data['lambda_home']:.2f} - {data['lambda_away']:.2f} "
        f"(total {data['expected_total']:.2f})",
    )
    for line in (1.5, 2.5, 3.5, 4.5):
        key = f"over_{str(line).replace('.', '_')}"
        under_key = f"under_{str(line).replace('.', '_')}"
        result.add_row(
            f"Over/Under {line}",
            f"Over {data['over_under'][key]:.1%} | "
            f"Under {data['over_under'][under_key]:.1%}",
        )
    result.add_row(
        "BTTS",
        f"Ya {data['prob_btts_yes']:.1%} | Tidak {data['prob_btts_no']:.1%}",
    )
    result.add_row(
        "Garis seimbang model",
        f"{data['balanced_line']['line']:.1f} "
        f"(Over {data['balanced_line']['over_prob']:.1%})",
    )
    result.add_row(
        "Skor teratas",
        " | ".join(f"{score['score']} ({score['prob']:.1%})" for score in data["top_scores"]),
    )
    result.add_row(
        "Kualitas data",
        f"{data['data_confidence']} — {data['home_matches_played']} laga kandang / "
        f"{data['away_matches_played']} laga tandang dalam riwayat tim",
    )
    console.print(result)
    console.print(
        f"Form PPG: {data['home_team']} {data['form']['home_points_per_game']:.2f}, "
        f"{data['away_team']} {data['form']['away_points_per_game']:.2f}; "
        f"selisih Elo {data['form']['elo_difference']:+.1f}."
    )
    console.print(
        f"Kalibrasi probabilitas: {data['probability_calibration']['method']} "
        f"(temperatur {data['probability_calibration']['temperature']:.2f})."
    )
    console.print(
        "Faktor model utama: "
        + " · ".join(item["label"] for item in data["explanation"])
        + "."
    )
    console.print(
        "[dim]Probabilitas adalah estimasi untuk analisis dan pembelajaran, "
        "bukan jaminan hasil atau saran taruhan.[/dim]"
    )


def _emit_predictions(predictions: list[dict], output_format: str) -> None:
    if output_format == "json":
        console.print_json(json.dumps(_as_json_safe(predictions), ensure_ascii=False))
    elif output_format == "csv":
        if not predictions:
            console.print("Tidak ada prediksi untuk diekspor.")
            return
        columns = sorted({key for row in predictions for key in row})
        output = io.StringIO()
        writer = csv.DictWriter(output, fieldnames=columns)
        writer.writeheader()
        for row in predictions:
            writer.writerow({
                key: json.dumps(value, ensure_ascii=False)
                if isinstance(value, (dict, list))
                else value
                for key, value in row.items()
            })
        console.print(output.getvalue(), end="")
    else:
        for prediction in predictions:
            _print_prediction({"data": prediction, "output_format": "table"})


@app.command("update-data")
def update_data(
    seasons: str = typer.Option(
        "2021,2022,2023,2024,2025,2026",
        help="Tahun awal musim historis, dipisahkan koma.",
    ),
    fixture_days: int = typer.Option(3, min=1, max=14, help="Rentang hari jadwal ke depan."),
    include_historical: bool = typer.Option(
        False,
        "--include-historical",
        help="Ambil ulang data historis API-Football (memerlukan banyak kuota API).",
    ),
) -> None:
    """Perbarui jadwal mendatang; data historis hanya diambil jika diminta."""
    season_years = _parse_seasons(seasons)
    database = DatabaseManager()
    updated_matches = 0

    if include_historical and API_FOOTBALL_KEY:
        client = ApiFootballClient()
        for code in COMPETITIONS:
            missing_seasons = [
                season
                for season in season_years
                if not database.has_finished_matches_for_season(code, season)
            ]
            if not missing_seasons:
                continue
            historical = client.fetch_all_historical(code, seasons=missing_seasons)
            if not historical.empty:
                updated_matches += database.insert_matches(historical)
        for code, details in HISTORICAL_COMPETITIONS.items():
            history_seasons = season_years[-3:]
            missing_seasons = [
                season
                for season in history_seasons
                if not database.has_finished_matches_for_season(code, season)
            ]
            if not missing_seasons:
                continue
            historical = client.fetch_all_historical(
                code,
                seasons=missing_seasons,
                api_league_id=details["api_football_id"],
            )
            if not historical.empty:
                updated_matches += database.insert_matches(historical)
    elif include_historical:
        console.print(
            "[red]Data historis API-Football tidak diambil: API_FOOTBALL_KEY "
            "belum dikonfigurasi di .env.[/red]"
        )
        raise typer.Exit(code=1)

    try:
        updated_fixtures = refresh_upcoming_fixtures(database, days=fixture_days)
    except RuntimeError as exc:
        console.print(f"[red]Gagal memperbarui jadwal:[/red] {exc}")
        raise typer.Exit(code=1) from exc

    console.print(
        f"Pembaruan selesai: {updated_matches} hasil historis baru, "
        f"{updated_fixtures} baris jadwal disimpan/diperbarui."
    )


@app.command("migrate-database")
def migrate_database(
    source_db: Path = typer.Option(
        DATABASE_PATH,
        "--source-db",
        help="Path to the local SQLite database to copy.",
    ),
) -> None:
    """Copy the local SQLite database to DATABASE_URL (hosted PostgreSQL)."""
    if not DATABASE_URL:
        console.print(
            "[red]Atur DATABASE_URL di .env ke connection string PostgreSQL "
            "sebelum migrasi.[/red]"
        )
        raise typer.Exit(code=1)
    imported = DatabaseManager().import_sqlite_database(source_db)
    console.print(
        "Migrasi selesai: "
        + ", ".join(f"{table}={count}" for table, count in imported.items())
    )


@app.command()
def matches(
    day: str = typer.Option(..., help="Rentang hari: today, tomorrow, atau dayafter."),
    league: str | None = typer.Option(None, help="Kode kompetisi, misalnya PL atau UCL."),
) -> None:
    """Tampilkan jadwal tersimpan untuk hari ini, besok, atau lusa (WIB)."""
    day_offsets = {"today": 0, "tomorrow": 1, "dayafter": 2}
    if day not in day_offsets:
        raise typer.BadParameter("Pilih today, tomorrow, atau dayafter.")
    if league and league not in COMPETITIONS:
        raise typer.BadParameter(f"Kompetisi tidak dikenal: {league}.")

    local_today = datetime.now(ZoneInfo(TIMEZONE)).date()
    match_date = (local_today + timedelta(days=day_offsets[day])).isoformat()
    fixtures = DatabaseManager().get_fixtures_by_date(match_date, league)
    if fixtures.empty:
        console.print(f"Tidak ada jadwal tersimpan untuk {match_date} WIB.")
        return

    table = Table(title=f"Jadwal {match_date} WIB")
    for column in ("ID", "Waktu WIB", "Liga", "Kandang", "Tandang", "Status"):
        table.add_column(column)
    for _, fixture in fixtures.iterrows():
        competition = COMPETITIONS.get(fixture["competition_code"], {})
        table.add_row(
            str(fixture["fixture_id"]),
            str(fixture.get("date_wib", ""))[11:16],
            competition.get("name", fixture["competition_code"]),
            fixture["home_team"],
            fixture["away_team"],
            fixture["status"],
        )
    console.print(table)


@app.command()
def train(
    last_n: int = typer.Option(DEFAULT_LAST_N, min=5, max=10, help="Jumlah laga terakhir."),
) -> None:
    """Latih model Poisson dari data historis tersimpan."""
    matches_df = DatabaseManager().get_all_matches()
    if matches_df.empty:
        console.print("[red]Belum ada data historis. Jalankan `fmpp update-data` dahulu.[/red]")
        raise typer.Exit(code=1)

    dataset = FeatureEngineer(matches_df, last_n=last_n).build_training_dataset()
    if dataset.empty:
        console.print(
            "[red]Data belum cukup: diperlukan setidaknya 3 laga terdahulu per tim "
            "untuk membentuk sampel training.[/red]"
        )
        raise typer.Exit(code=1)
    if len(dataset) < 30:
        console.print(
            f"[yellow]Peringatan: hanya {len(dataset)} sampel; hasil model mungkin tidak stabil.[/yellow]"
        )

    model = PoissonGoalModel()
    try:
        from src.evaluation.backtest import walk_forward_evaluate

        evaluation = walk_forward_evaluate(matches_df, last_n=last_n)
        model.score_temperature = evaluation["calibration"]["temperature"]
        model.calibration_method = "walk-forward score-matrix temperature scaling"
        console.print(
            f"Kalibrasi probabilitas dari {evaluation['calibration']['fit_sample_count']} "
            f"prediksi walk-forward: temperatur {model.score_temperature:.2f}."
        )
    except ValueError as exc:
        console.print(f"[yellow]Peringatan: kalibrasi belum tersedia: {exc}[/yellow]")
    model.train(dataset)
    model.save(MODEL_PATH)
    console.print(f"Model dilatih dari {len(dataset)} pertandingan dan disimpan di {MODEL_PATH}.")


def _make_prediction(
    home: str | None,
    away: str | None,
    match_id: int | None,
    match_date: str | None,
    last_n: int,
    output_format: str,
) -> None:
    if output_format not in {"table", "json", "csv"}:
        raise typer.BadParameter("Format keluaran harus table, json, atau csv.")
    fixture = None
    if match_id is not None:
        if home or away:
            raise typer.BadParameter("Gunakan --match-id atau --home/--away, bukan keduanya.")
        fixture = DatabaseManager().get_fixture(match_id)
        if fixture is None:
            raise typer.BadParameter(f"Jadwal dengan ID {match_id} tidak ditemukan.")
        home, away = fixture["home_team"], fixture["away_team"]
        match_date = fixture["date_utc"]
    elif not home or not away:
        raise typer.BadParameter("Isi --match-id atau kedua opsi --home dan --away.")

    try:
        if home is None or away is None:
            raise ValueError("Nama tim kandang dan tandang tidak tersedia.")
        result = PredictionEngine(model_path=MODEL_PATH).predict(
            home, away, match_date=match_date, last_n=last_n
        )
    except (FileNotFoundError, ValueError) as exc:
        console.print(f"[red]Prediksi gagal:[/red] {exc}")
        raise typer.Exit(code=1) from exc
    if fixture:
        result["competition_code"] = fixture["competition_code"]
        result["date_wib"] = fixture["date_wib"]
    _emit_predictions([result], output_format)


@app.command()
def predict(
    home: str | None = typer.Option(None, help="Nama tim kandang."),
    away: str | None = typer.Option(None, help="Nama tim tandang."),
    match_id: int | None = typer.Option(None, help="ID jadwal dari perintah matches."),
    match_date: str | None = typer.Option(None, help="Tanggal/waktu ISO-8601, default sekarang."),
    last_n: int = typer.Option(DEFAULT_LAST_N, min=5, max=10, help="Jumlah laga terakhir."),
    output_format: str = typer.Option("table", "--format", help="table, json, atau csv."),
) -> None:
    """Prediksi jadwal tersimpan atau pasangan tim yang ditentukan manual."""
    _make_prediction(home, away, match_id, match_date, last_n, output_format)


@app.command("predict-day")
def predict_day(
    day: str = typer.Option(..., help="today, tomorrow, atau dayafter."),
    league: str | None = typer.Option(None, help="Filter kode kompetisi."),
    last_n: int = typer.Option(DEFAULT_LAST_N, min=5, max=10, help="Jumlah laga terakhir."),
    output_format: str = typer.Option("table", "--format", help="table, json, atau csv."),
) -> None:
    """Prediksi seluruh jadwal tersimpan pada satu hari."""
    day_offsets = {"today": 0, "tomorrow": 1, "dayafter": 2}
    if day not in day_offsets:
        raise typer.BadParameter("Pilih today, tomorrow, atau dayafter.")
    if output_format not in {"table", "json", "csv"}:
        raise typer.BadParameter("Format keluaran harus table, json, atau csv.")
    if league and league not in COMPETITIONS:
        raise typer.BadParameter(f"Kompetisi tidak dikenal: {league}.")

    local_today = datetime.now(ZoneInfo(TIMEZONE)).date()
    match_date = (local_today + timedelta(days=day_offsets[day])).isoformat()
    fixtures = DatabaseManager().get_fixtures_by_date(match_date, league)
    if fixtures.empty:
        console.print(f"Tidak ada jadwal tersimpan untuk {match_date} WIB.")
        return

    results = []
    for _, fixture in fixtures.iterrows():
        try:
            result = PredictionEngine(model_path=MODEL_PATH).predict(
                fixture["home_team"],
                fixture["away_team"],
                match_date=fixture["date_utc"],
                last_n=last_n,
            )
            result["competition_code"] = fixture["competition_code"]
            result["date_wib"] = fixture["date_wib"]
            results.append(result)
        except (FileNotFoundError, ValueError) as exc:
            console.print(
                f"[yellow]Lewati {fixture['home_team']} vs {fixture['away_team']}: {exc}[/yellow]"
            )

    _emit_predictions(results, output_format)


@app.command()
def evaluate(
    last_n: int = typer.Option(DEFAULT_LAST_N, min=5, max=10, help="Jumlah laga terakhir."),
    folds: int = typer.Option(3, min=1, max=10, help="Jumlah periode uji berurutan."),
    output_format: str = typer.Option("table", "--format", help="table, json, atau csv."),
) -> None:
    """Jalankan backtest walk-forward dan bandingkan model dengan baseline."""
    if output_format not in {"table", "json", "csv"}:
        raise typer.BadParameter("Format keluaran harus table, json, atau csv.")
    from src.evaluation.backtest import walk_forward_evaluate

    try:
        result = walk_forward_evaluate(
            DatabaseManager().get_all_matches(),
            last_n=last_n,
            folds=folds,
        )
    except ValueError as exc:
        console.print(f"[red]Evaluasi gagal:[/red] {exc}")
        raise typer.Exit(code=1) from exc
    if output_format == "json":
        console.print_json(json.dumps(result, ensure_ascii=False))
        return
    if output_format == "csv":
        records = []
        for competition, summary in result["per_competition"].items():
            for model_name, metrics in summary.items():
                if model_name == "sample_count" or not isinstance(metrics, dict):
                    continue
                records.append({
                    "competition": competition,
                    "sample_count": summary["sample_count"],
                    "model": model_name,
                    **{
                        key: value for key, value in metrics.items()
                        if not isinstance(value, list)
                    },
                })
        _emit_predictions(records, "csv")
        return

    for competition, summary in result["per_competition"].items():
        console.print(f"\n[bold]{competition}[/bold] — {summary['sample_count']} sampel uji")
        table = Table()
        for heading in (
            "Model", "Log Loss 1X2", "Brier 1X2", "RPS", "Akurasi", "MAE gol kandang"
        ):
            table.add_column(heading)
        for model_name, metrics in summary.items():
            if model_name == "sample_count" or not isinstance(metrics, dict):
                continue
            table.add_row(
                model_name,
                f"{metrics['log_loss_1x2']:.4f}",
                f"{metrics['brier_1x2']:.4f}",
                f"{metrics['rps_1x2']:.4f}",
                f"{metrics['accuracy_1x2']:.1%}",
                f"{metrics['mae_home_goals']:.3f}" if "mae_home_goals" in metrics else "—",
            )
        console.print(table)
    console.print(
        f"\nTotal {result['sample_count']} sampel · {result['folds_completed']} fold "
        f"walk-forward · N={result['last_n']}."
    )
    for note in result["limitations"]:
        console.print(f"[dim]- {note}[/dim]")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    app()
