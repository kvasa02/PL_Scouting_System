# Premier League Player Scouting System

A SQLite-backed Streamlit dashboard that compares Premier League players by statistical playstyle. The project ingests live FPL data, stores historical snapshots, engineers features in SQL, then uses scikit-learn to compute similarity matches and K-Means playing-style clusters.

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![SQLite](https://img.shields.io/badge/SQLite-Data%20Layer-003B57.svg)](https://www.sqlite.org/)
[![pandas](https://img.shields.io/badge/pandas-Data%20Analysis-150458.svg)](https://pandas.pydata.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-Machine%20Learning-F7931E.svg)](https://scikit-learn.org/)
[![matplotlib](https://img.shields.io/badge/matplotlib-Visualization-11557C.svg)](https://matplotlib.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-FF4B4B.svg)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

![Premier League scouting dashboard demo](assets/pl-scouting-demo.gif)

## Features

- Live data refresh from the public Fantasy Premier League bootstrap API
- SQLite warehouse layer that stores every refresh as a historical snapshot
- `player_gameweek_stats` fact table for tracking player stats over time
- SQL views for per-90 features, weekly deltas, and rolling last-5-snapshot form
- Player search with a Bukayo Saka default when available
- Cosine-similarity engine for closest statistical matches
- K-Means playing-style clusters with interpretable archetype labels
- Radar chart comparison using league percentile profiles
- PCA cluster map to visualize player style neighborhoods
- Filters for minimum minutes, same position, and same archetype
- Source metadata saved with every data refresh

## Tech Stack

- Python
- SQLite
- pandas
- scikit-learn
- matplotlib
- Streamlit

## Project Structure

```text
.
├── app.py                    # Streamlit dashboard
├── database.py               # SQLite schema, inserts, views, query layer
├── scout_model.py            # Feature prep, similarity, clustering, PCA
├── fetch_real_data.py        # Live data ingestion and feature engineering
├── weekly_refresh.py         # Cron-friendly snapshot refresh script
├── data/pl_scouting.sqlite   # Local historical snapshot database
├── premier_league_stats.csv  # Current local player dataset
├── data_metadata.json        # Source and refresh metadata
└── requirements.txt
```

## Run Locally

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python fetch_real_data.py
streamlit run app.py
```

## Weekly Snapshot Job

Each refresh appends a new record to `snapshots` and inserts the latest player totals into `player_gameweek_stats`.

Run manually:

```bash
python weekly_refresh.py
```

Example weekly cron job:

```cron
0 9 * * 2 cd /path/to/PL_Scouting_System && /path/to/venv/bin/python weekly_refresh.py >> refresh.log 2>&1
```

## SQL Data Layer

The dashboard reads from `v_latest_player_features`, not directly from the API. The SQLite layer is structured as:

- `snapshots`: one row per API refresh with source and gameweek metadata.
- `players`: player identity, team, position, and birth-date metadata.
- `teams`: Premier League team lookup table.
- `positions`: FPL position lookup table.
- `player_gameweek_stats`: historical fact table with one row per player per snapshot.
- `v_player_gameweek_features`: SQL feature view with per-90 stats, `LAG()` weekly deltas, and rolling window features.
- `v_latest_player_features`: current scouting table consumed by Streamlit.

## Modeling Approach

SQL handles the per-90 normalization and rolling historical features before the data reaches the app. The model then standardizes the selected feature columns with `StandardScaler`:

- `cosine_similarity` ranks closest player profiles.
- `KMeans` groups players into playing-style clusters.
- `PCA` projects the feature vectors into two dimensions for the cluster map.
- Cluster labels are generated from each cluster's strongest relative features.

## Data Notes

The included SQLite database and CSV were refreshed from the Fantasy Premier League public API. FPL provides strong current player metadata and season-to-date totals for goals, assists, expected goal involvement, ICT creativity/threat/influence, tackles, recoveries, defensive contribution, price, ownership, minutes, and availability.

Because the FPL API is cumulative, rolling gameweek form becomes more meaningful after multiple weekly snapshots have been stored. Passing, shot, and dribble event detail are approximated with FPL's Creativity, Threat, ICT, and defensive-action feature groups. For a production scouting department version, this project is designed so richer event data from StatsBomb, Opta, Wyscout, or FBref-style tables can replace or extend `fetch_real_data.py`.
