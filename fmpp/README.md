# Football Match Probability Predictor

FMPP is a local analysis and learning project. Probabilities are statistical
estimates, not guarantees or betting advice.

## Setup

Requires Python 3.11, 3.12, or 3.13.

From this directory, install the locked dependencies and the CLI:

```powershell
python -m pip install -r requirements.txt
python -m pip install -e . --no-deps
```

Copy `.env.example` to `.env` and add API keys for the data providers you want
to use. Keep `.env` private.

## CLI

```text
fmpp update-data [--fixture-days 14] [--include-historical]
fmpp migrate-database [--source-db data/fmpp.db]
fmpp matches --day today|tomorrow|dayafter [--league PL]
fmpp train [--last-n 5]
fmpp predict --match-id 42 [--last-n 8] [--format table|json|csv]
fmpp predict --home "Arsenal" --away "Chelsea" [--last-n 8] [--format table|json|csv]
fmpp predict-day --day today [--league PL] [--format table|json|csv]
fmpp evaluate [--last-n 5] [--folds 3] [--format table|json|csv]
fmpp-web
```

`update-data` refreshes upcoming fixtures from football-data.org when configured,
or uses API-Football when football-data.org is not configured; it does not
download bulk historical CSVs by default.
Historical data can be fetched explicitly with `--include-historical` when
API-Football is configured, which consumes many API requests. The web server
also refreshes fixtures at startup and every six hours. Its first `/api/matches`
request waits for the startup refresh to finish. Match times are displayed in
the configured timezone (default: Asia/Jakarta).

The model uses only matches that occurred before the prediction timestamp.
Predictions with fewer than five prior matches for either team display lower
data confidence.

`fmpp evaluate` applies expanding-window chronological walk-forward validation,
reports per-competition Log Loss, Brier Score, RPS, accuracy, Over/Under and
BTTS scores, goal MAE, calibration bins, and examples of incorrect results.
It compares the Poisson score-matrix model with competition-frequency,
team-average-goals, and direct multinomial Logistic Regression baselines.
The direct classifier uses sigmoid calibration against the latest chronological
part of each training fold when enough data is available. A score-matrix
temperature is learned from out-of-fold walk-forward predictions when training;
that calibration transforms the shared score matrix so 1X2, totals, BTTS, and
score probabilities remain internally consistent. Metrics in the evaluation
report describe uncalibrated fold predictions to avoid reusing calibration
observations as an independent test. Closing odds are not evaluated unless a
source with odds is added to the schema.

Start the local UI at `http://127.0.0.1:8000` with `fmpp-web` (or
`python -m uvicorn src.api.app:app --host 127.0.0.1 --port 8000`). It provides
today/tomorrow/day-after fixtures, competition filtering, detailed predictions,
recent form, score-probability heatmaps, and the backtest report. The web server
binds to localhost when run locally.

The historical CSV provider `football-data.co.uk` is not contacted by routine
updates because its current certificate hostname mismatch makes HTTPS
verification fail. TLS verification is not disabled. Historical model data
already stored locally remains available; no additional key is needed for
automatic schedule refresh when either provider key is configured.

`GET /api/teams?last_n=5` returns recent form and match history for every team
found in stored finished results or fixtures, including teams without upcoming
fixtures. `last_n` accepts values from 5 to 10. The UI also includes a searchable
directory of these teams.

## Notes on data coverage

Previously downloaded CSV data from football-data.co.uk remains in the local
database. Routine refreshes use football-data.org and API-Football for upcoming
fixtures; bulk historical imports are optional. The UI matches provider-specific
club-name variants before calculating recent form. Provider coverage, free-tier
quotas and data terms can change; review each provider's current terms and
availability before use. Missing API keys or provider data can leave teams
without enough history; the system does not fabricate missing data.

Raw provider CSVs and the SQLite database are local artifacts under `data/`.
The trained model at `models/saved/poisson_model.pkl` is included for deployment;
other generated data remains local and ignored by Git.

## Public deployment (Vercel + Neon)

Vercel Functions cannot persist writes to the project's local SQLite file.
The deployment is therefore configured to use a hosted PostgreSQL database
through `DATABASE_URL`; local development continues using SQLite unless this
variable is set.

1. Create a Neon PostgreSQL database through the Vercel Marketplace and copy
   its connection string. Keep the string private.
2. Put that connection string in local `.env` as `DATABASE_URL`, then copy the
   local data into Neon once:

   ```powershell
   fmpp migrate-database --source-db data/fmpp.db
   ```

   Importing is safe to rerun: existing rows are skipped. The command also
   initializes missing application tables.
3. Add these environment variables to the Vercel project, for both Production
   and Preview as appropriate: `DATABASE_URL`, `FOOTBALL_DATA_ORG_API_KEY` (or
   `API_FOOTBALL_KEY`), and a long random `CRON_SECRET`. Do not commit `.env` or
   put secret values in source files.
4. Push the project to a Git repository and import that repository in Vercel.
   The included Vercel configuration exposes the FastAPI app and runs an
   authenticated fixture refresh every day at 06:00 UTC. It does not rely on a
   background loop, which serverless functions cannot guarantee will stay
   running.
5. Ensure `models/saved/poisson_model.pkl` is included in the repository; it is
   explicitly unignored for this purpose. Then deploy from Vercel.

The deployment has not been connected to a Neon account or deployed from this
workspace. The live migration and public URL still require the user's Neon
connection string and Vercel project/repository access.
