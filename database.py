import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


DB_FILE = Path("data/pl_scouting.sqlite")
SOURCE_URL = "https://fantasy.premierleague.com/api/bootstrap-static/"


SCHEMA_SQL = """
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS snapshots (
    snapshot_id INTEGER PRIMARY KEY AUTOINCREMENT,
    fetched_at_utc TEXT NOT NULL,
    latest_checked_gameweek_id INTEGER,
    latest_checked_gameweek TEXT,
    current_gameweek TEXT,
    next_gameweek TEXT,
    source_url TEXT NOT NULL,
    players_count INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS teams (
    team_id INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    short_name TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS positions (
    position_id INTEGER PRIMARY KEY,
    singular_name TEXT NOT NULL,
    singular_name_short TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS players (
    player_id INTEGER PRIMARY KEY,
    first_name TEXT,
    second_name TEXT,
    web_name TEXT NOT NULL,
    player_name TEXT NOT NULL,
    birth_date TEXT,
    team_id INTEGER,
    position_id INTEGER,
    updated_at_utc TEXT NOT NULL,
    FOREIGN KEY (team_id) REFERENCES teams(team_id),
    FOREIGN KEY (position_id) REFERENCES positions(position_id)
);

CREATE TABLE IF NOT EXISTS player_gameweek_stats (
    snapshot_id INTEGER NOT NULL,
    player_id INTEGER NOT NULL,
    gameweek_id INTEGER,
    fetched_at_utc TEXT NOT NULL,
    minutes INTEGER NOT NULL,
    starts INTEGER NOT NULL,
    goals INTEGER NOT NULL,
    assists INTEGER NOT NULL,
    expected_goals REAL NOT NULL,
    expected_assists REAL NOT NULL,
    expected_goal_involvements REAL NOT NULL,
    creativity REAL NOT NULL,
    threat REAL NOT NULL,
    influence REAL NOT NULL,
    ict_index REAL NOT NULL,
    tackles REAL NOT NULL,
    recoveries REAL NOT NULL,
    clearances_blocks_interceptions REAL NOT NULL,
    defensive_contribution REAL NOT NULL,
    bonus REAL NOT NULL,
    bps REAL NOT NULL,
    form REAL NOT NULL,
    total_points INTEGER NOT NULL,
    now_cost REAL NOT NULL,
    selected_by_percent REAL NOT NULL,
    status TEXT,
    news TEXT,
    PRIMARY KEY (snapshot_id, player_id),
    FOREIGN KEY (snapshot_id) REFERENCES snapshots(snapshot_id),
    FOREIGN KEY (player_id) REFERENCES players(player_id)
);

CREATE INDEX IF NOT EXISTS idx_player_gameweek_stats_player
ON player_gameweek_stats(player_id, gameweek_id, snapshot_id);

CREATE VIEW IF NOT EXISTS v_player_gameweek_features AS
WITH base AS (
    SELECT
        s.snapshot_id,
        s.fetched_at_utc,
        s.latest_checked_gameweek_id AS gameweek_id,
        s.latest_checked_gameweek AS gameweek,
        p.player_id,
        p.player_name AS Player,
        p.web_name AS Display_Name,
        t.name AS Team,
        t.short_name AS Team_Short,
        pos.singular_name_short AS Position,
        CAST((julianday(s.fetched_at_utc) - julianday(p.birth_date)) / 365.25 AS INTEGER) AS Age,
        pg.minutes AS Minutes,
        pg.starts AS Starts,
        pg.now_cost / 10.0 AS Cost_M,
        pg.selected_by_percent AS "Selected_By_%",
        pg.goals AS Goals,
        pg.assists AS Assists,
        pg.expected_goals AS Expected_Goals,
        pg.expected_assists AS Expected_Assists,
        pg.expected_goal_involvements AS Expected_Goal_Involvements,
        pg.creativity,
        pg.threat,
        pg.influence,
        pg.ict_index,
        pg.tackles,
        pg.recoveries,
        pg.clearances_blocks_interceptions,
        pg.defensive_contribution,
        pg.bonus,
        pg.bps,
        pg.form AS Form,
        pg.total_points AS Total_Points,
        pg.status AS Status,
        pg.news AS News,
        LAG(pg.minutes) OVER player_order AS prev_minutes,
        LAG(pg.goals) OVER player_order AS prev_goals,
        LAG(pg.assists) OVER player_order AS prev_assists,
        LAG(pg.expected_goal_involvements) OVER player_order AS prev_xgi,
        LAG(pg.form) OVER player_order AS prev_form
    FROM player_gameweek_stats pg
    JOIN snapshots s ON s.snapshot_id = pg.snapshot_id
    JOIN players p ON p.player_id = pg.player_id
    JOIN teams t ON t.team_id = p.team_id
    JOIN positions pos ON pos.position_id = p.position_id
    WINDOW player_order AS (
        PARTITION BY pg.player_id
        ORDER BY s.latest_checked_gameweek_id, s.snapshot_id
    )
),
features AS (
    SELECT
        *,
        MAX(Minutes - COALESCE(prev_minutes, 0), 0) AS GW_Minutes,
        MAX(Goals - COALESCE(prev_goals, 0), 0) AS GW_Goals,
        MAX(Assists - COALESCE(prev_assists, 0), 0) AS GW_Assists,
        MAX(Expected_Goal_Involvements - COALESCE(prev_xgi, 0), 0) AS GW_xGI
    FROM base
)
SELECT
    *,
    ROUND(Goals * 90.0 / NULLIF(Minutes, 0), 3) AS Goals_p90,
    ROUND(Assists * 90.0 / NULLIF(Minutes, 0), 3) AS Assists_p90,
    ROUND(Expected_Goals * 90.0 / NULLIF(Minutes, 0), 3) AS xG_p90,
    ROUND(Expected_Assists * 90.0 / NULLIF(Minutes, 0), 3) AS xA_p90,
    ROUND(Expected_Goal_Involvements * 90.0 / NULLIF(Minutes, 0), 3) AS xGI_p90,
    ROUND(creativity * 90.0 / NULLIF(Minutes, 0), 3) AS Creativity_p90,
    ROUND(threat * 90.0 / NULLIF(Minutes, 0), 3) AS Threat_p90,
    ROUND(influence * 90.0 / NULLIF(Minutes, 0), 3) AS Influence_p90,
    ROUND(ict_index * 90.0 / NULLIF(Minutes, 0), 3) AS ICT_p90,
    ROUND(
        (tackles + recoveries + clearances_blocks_interceptions + defensive_contribution)
        * 90.0 / NULLIF(Minutes, 0),
        3
    ) AS DefActions_p90,
    ROUND(bonus * 90.0 / NULLIF(Minutes, 0), 3) AS Bonus_p90,
    ROUND(bps * 90.0 / NULLIF(Minutes, 0), 3) AS BPS_p90,
    ROUND(GW_Goals * 90.0 / NULLIF(GW_Minutes, 0), 3) AS GW_Goals_p90,
    ROUND(GW_Assists * 90.0 / NULLIF(GW_Minutes, 0), 3) AS GW_Assists_p90,
    ROUND(GW_xGI * 90.0 / NULLIF(GW_Minutes, 0), 3) AS GW_xGI_p90,
    ROUND(
        AVG(Form) OVER (
            PARTITION BY player_id
            ORDER BY gameweek_id, snapshot_id
            ROWS BETWEEN 4 PRECEDING AND CURRENT ROW
        ),
        3
    ) AS Rolling_Form_Last5,
    ROUND(
        AVG(GW_xGI * 90.0 / NULLIF(GW_Minutes, 0)) OVER (
            PARTITION BY player_id
            ORDER BY gameweek_id, snapshot_id
            ROWS BETWEEN 4 PRECEDING AND CURRENT ROW
        ),
        3
    ) AS Rolling_xGI_p90_Last5
FROM features;

CREATE VIEW IF NOT EXISTS v_latest_player_features AS
WITH ranked AS (
    SELECT
        *,
        ROW_NUMBER() OVER (
            PARTITION BY player_id
            ORDER BY fetched_at_utc DESC, snapshot_id DESC
        ) AS recency_rank
    FROM v_player_gameweek_features
)
SELECT *
FROM ranked
WHERE recency_rank = 1;
"""


def get_connection(db_path: Path = DB_FILE) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(db_path)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON;")
    return connection


def initialize_database(db_path: Path = DB_FILE) -> None:
    with get_connection(db_path) as connection:
        connection.executescript(SCHEMA_SQL)


def _latest_checked_event(payload: dict) -> dict | None:
    checked_events = [
        event for event in payload.get("events", []) if event.get("finished") and event.get("data_checked")
    ]
    return checked_events[-1] if checked_events else None


def save_fpl_snapshot(payload: dict, db_path: Path = DB_FILE) -> int:
    initialize_database(db_path)

    fetched_at_utc = datetime.now(timezone.utc).isoformat(timespec="seconds")
    latest_event = _latest_checked_event(payload)
    current_event = next((event for event in payload.get("events", []) if event.get("is_current")), None)
    next_event = next((event for event in payload.get("events", []) if event.get("is_next")), None)
    players = payload["elements"]

    with get_connection(db_path) as connection:
        connection.executemany(
            """
            INSERT INTO teams (team_id, name, short_name)
            VALUES (:id, :name, :short_name)
            ON CONFLICT(team_id) DO UPDATE SET
                name = excluded.name,
                short_name = excluded.short_name;
            """,
            payload["teams"],
        )
        connection.executemany(
            """
            INSERT INTO positions (position_id, singular_name, singular_name_short)
            VALUES (:id, :singular_name, :singular_name_short)
            ON CONFLICT(position_id) DO UPDATE SET
                singular_name = excluded.singular_name,
                singular_name_short = excluded.singular_name_short;
            """,
            payload["element_types"],
        )
        cursor = connection.execute(
            """
            INSERT INTO snapshots (
                fetched_at_utc,
                latest_checked_gameweek_id,
                latest_checked_gameweek,
                current_gameweek,
                next_gameweek,
                source_url,
                players_count
            )
            VALUES (?, ?, ?, ?, ?, ?, ?);
            """,
            (
                fetched_at_utc,
                latest_event.get("id") if latest_event else None,
                latest_event.get("name") if latest_event else "Pre-season",
                current_event.get("name") if current_event else None,
                next_event.get("name") if next_event else None,
                SOURCE_URL,
                len(players),
            ),
        )
        snapshot_id = int(cursor.lastrowid)

        player_rows = []
        stats_rows = []
        for player in players:
            first_name = str(player.get("first_name") or "").strip()
            second_name = str(player.get("second_name") or "").strip()
            player_name = f"{first_name} {second_name}".strip() or str(player.get("web_name"))
            player_rows.append(
                {
                    "player_id": player["id"],
                    "first_name": first_name,
                    "second_name": second_name,
                    "web_name": player.get("web_name") or player_name,
                    "player_name": player_name,
                    "birth_date": player.get("birth_date"),
                    "team_id": player.get("team"),
                    "position_id": player.get("element_type"),
                    "updated_at_utc": fetched_at_utc,
                }
            )
            stats_rows.append(
                {
                    "snapshot_id": snapshot_id,
                    "player_id": player["id"],
                    "gameweek_id": latest_event.get("id") if latest_event else None,
                    "fetched_at_utc": fetched_at_utc,
                    "minutes": int(player.get("minutes") or 0),
                    "starts": int(player.get("starts") or 0),
                    "goals": int(player.get("goals_scored") or 0),
                    "assists": int(player.get("assists") or 0),
                    "expected_goals": float(player.get("expected_goals") or 0),
                    "expected_assists": float(player.get("expected_assists") or 0),
                    "expected_goal_involvements": float(
                        player.get("expected_goal_involvements") or 0
                    ),
                    "creativity": float(player.get("creativity") or 0),
                    "threat": float(player.get("threat") or 0),
                    "influence": float(player.get("influence") or 0),
                    "ict_index": float(player.get("ict_index") or 0),
                    "tackles": float(player.get("tackles") or 0),
                    "recoveries": float(player.get("recoveries") or 0),
                    "clearances_blocks_interceptions": float(
                        player.get("clearances_blocks_interceptions") or 0
                    ),
                    "defensive_contribution": float(player.get("defensive_contribution") or 0),
                    "bonus": float(player.get("bonus") or 0),
                    "bps": float(player.get("bps") or 0),
                    "form": float(player.get("form") or 0),
                    "total_points": int(player.get("total_points") or 0),
                    "now_cost": float(player.get("now_cost") or 0),
                    "selected_by_percent": float(player.get("selected_by_percent") or 0),
                    "status": player.get("status"),
                    "news": player.get("news") or "",
                }
            )

        connection.executemany(
            """
            INSERT INTO players (
                player_id,
                first_name,
                second_name,
                web_name,
                player_name,
                birth_date,
                team_id,
                position_id,
                updated_at_utc
            )
            VALUES (
                :player_id,
                :first_name,
                :second_name,
                :web_name,
                :player_name,
                :birth_date,
                :team_id,
                :position_id,
                :updated_at_utc
            )
            ON CONFLICT(player_id) DO UPDATE SET
                first_name = excluded.first_name,
                second_name = excluded.second_name,
                web_name = excluded.web_name,
                player_name = excluded.player_name,
                birth_date = excluded.birth_date,
                team_id = excluded.team_id,
                position_id = excluded.position_id,
                updated_at_utc = excluded.updated_at_utc;
            """,
            player_rows,
        )
        connection.executemany(
            """
            INSERT INTO player_gameweek_stats (
                snapshot_id,
                player_id,
                gameweek_id,
                fetched_at_utc,
                minutes,
                starts,
                goals,
                assists,
                expected_goals,
                expected_assists,
                expected_goal_involvements,
                creativity,
                threat,
                influence,
                ict_index,
                tackles,
                recoveries,
                clearances_blocks_interceptions,
                defensive_contribution,
                bonus,
                bps,
                form,
                total_points,
                now_cost,
                selected_by_percent,
                status,
                news
            )
            VALUES (
                :snapshot_id,
                :player_id,
                :gameweek_id,
                :fetched_at_utc,
                :minutes,
                :starts,
                :goals,
                :assists,
                :expected_goals,
                :expected_assists,
                :expected_goal_involvements,
                :creativity,
                :threat,
                :influence,
                :ict_index,
                :tackles,
                :recoveries,
                :clearances_blocks_interceptions,
                :defensive_contribution,
                :bonus,
                :bps,
                :form,
                :total_points,
                :now_cost,
                :selected_by_percent,
                :status,
                :news
            );
            """,
            stats_rows,
        )
        return snapshot_id


def latest_features(db_path: Path = DB_FILE, min_minutes: int = 1) -> pd.DataFrame:
    initialize_database(db_path)
    with get_connection(db_path) as connection:
        return pd.read_sql_query(
            """
            SELECT *
            FROM v_latest_player_features
            WHERE Minutes >= ?
            ORDER BY Minutes DESC, Goals DESC, Assists DESC;
            """,
            connection,
            params=(min_minutes,),
        )


def snapshots_summary(db_path: Path = DB_FILE) -> pd.DataFrame:
    initialize_database(db_path)
    with get_connection(db_path) as connection:
        return pd.read_sql_query(
            """
            SELECT
                snapshot_id,
                fetched_at_utc,
                latest_checked_gameweek,
                current_gameweek,
                next_gameweek,
                players_count
            FROM snapshots
            ORDER BY fetched_at_utc DESC;
            """,
            connection,
        )


def database_metadata(db_path: Path = DB_FILE) -> dict:
    initialize_database(db_path)
    with get_connection(db_path) as connection:
        row = connection.execute(
            """
            SELECT *
            FROM snapshots
            ORDER BY fetched_at_utc DESC, snapshot_id DESC
            LIMIT 1;
            """
        ).fetchone()
        snapshot_count = connection.execute("SELECT COUNT(*) FROM snapshots;").fetchone()[0]
        player_count = connection.execute("SELECT COUNT(*) FROM players;").fetchone()[0]

    if row is None:
        return {
            "source": "SQLite scouting database",
            "source_url": SOURCE_URL,
            "fetched_at_utc": "No snapshots yet",
            "players": 0,
            "snapshots": 0,
            "latest_checked_gameweek": "No snapshots yet",
            "notes": "Run python fetch_real_data.py to load the first historical snapshot.",
        }

    return {
        "source": "SQLite scouting database backed by Fantasy Premier League API snapshots",
        "source_url": row["source_url"],
        "fetched_at_utc": row["fetched_at_utc"],
        "players": player_count,
        "snapshots": snapshot_count,
        "latest_checked_gameweek": row["latest_checked_gameweek"],
        "current_gameweek": row["current_gameweek"],
        "next_gameweek": row["next_gameweek"],
        "notes": (
            "Every refresh appends a snapshot to player_gameweek_stats. SQL views compute per-90 "
            "features, weekly deltas, and rolling last-5-snapshot form with window functions."
        ),
    }


def write_metadata_file(metadata_file: Path, db_path: Path = DB_FILE) -> None:
    metadata_file.write_text(json.dumps(database_metadata(db_path), indent=2), encoding="utf-8")
