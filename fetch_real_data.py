import json
import ssl
from datetime import datetime, timezone
from pathlib import Path
from urllib.request import Request, urlopen

import pandas as pd

from database import DB_FILE, database_metadata, latest_features, save_fpl_snapshot

FPL_BOOTSTRAP_URL = "https://fantasy.premierleague.com/api/bootstrap-static/"
OUTPUT_FILE = Path("premier_league_stats.csv")
METADATA_FILE = Path("data_metadata.json")


def _to_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").fillna(0)


def _per_90(series: pd.Series, nineties: pd.Series) -> pd.Series:
    safe_nineties = nineties.where(nineties > 0)
    return (_to_numeric(series) / safe_nineties).fillna(0).round(3)


def _age_from_birth_date(series: pd.Series) -> pd.Series:
    birth_dates = pd.to_datetime(series, errors="coerce", utc=True)
    today = pd.Timestamp.now(tz=timezone.utc)
    ages = ((today - birth_dates).dt.days / 365.25).fillna(0)
    return ages.astype(int)


def fetch_bootstrap_static() -> dict:
    request = Request(
        FPL_BOOTSTRAP_URL,
        headers={
            "User-Agent": "Mozilla/5.0 player-scouting-dashboard/1.0",
            "Accept": "application/json",
        },
    )
    context = ssl._create_unverified_context()
    with urlopen(request, timeout=30, context=context) as response:
        return json.load(response)


def build_scouting_dataset(payload: dict, min_minutes: int = 1) -> tuple[pd.DataFrame, dict]:
    players = pd.DataFrame(payload["elements"])
    teams = pd.DataFrame(payload["teams"])
    positions = pd.DataFrame(payload["element_types"])

    team_map = teams.set_index("id")["name"].to_dict()
    team_short_map = teams.set_index("id")["short_name"].to_dict()
    position_map = positions.set_index("id")["singular_name_short"].to_dict()

    players = players[_to_numeric(players["minutes"]) >= min_minutes].copy()
    players["n90"] = _to_numeric(players["minutes"]) / 90
    players["Player"] = (
        players["first_name"].fillna("").str.strip()
        + " "
        + players["second_name"].fillna("").str.strip()
    ).str.strip()

    defensive_actions = (
        _to_numeric(players["tackles"])
        + _to_numeric(players["recoveries"])
        + _to_numeric(players["clearances_blocks_interceptions"])
        + _to_numeric(players["defensive_contribution"])
    )

    scout_df = pd.DataFrame(
        {
            "Player": players["Player"].where(players["Player"].str.len() > 0, players["web_name"]),
            "Display_Name": players["web_name"],
            "Team": players["team"].map(team_map).fillna("Unknown"),
            "Team_Short": players["team"].map(team_short_map).fillna("UNK"),
            "Position": players["element_type"].map(position_map).fillna("UNK"),
            "Age": _age_from_birth_date(players["birth_date"]),
            "Minutes": _to_numeric(players["minutes"]).astype(int),
            "Starts": _to_numeric(players["starts"]).astype(int),
            "Cost_M": (_to_numeric(players["now_cost"]) / 10).round(1),
            "Selected_By_%": _to_numeric(players["selected_by_percent"]).round(2),
            "Goals": _to_numeric(players["goals_scored"]).astype(int),
            "Assists": _to_numeric(players["assists"]).astype(int),
            "Expected_Goals": _to_numeric(players["expected_goals"]).round(3),
            "Expected_Assists": _to_numeric(players["expected_assists"]).round(3),
            "Goals_p90": _per_90(players["goals_scored"], players["n90"]),
            "Assists_p90": _per_90(players["assists"], players["n90"]),
            "xG_p90": _to_numeric(players["expected_goals_per_90"]).round(3),
            "xA_p90": _to_numeric(players["expected_assists_per_90"]).round(3),
            "xGI_p90": _to_numeric(players["expected_goal_involvements_per_90"]).round(3),
            "Creativity_p90": _per_90(players["creativity"], players["n90"]),
            "Threat_p90": _per_90(players["threat"], players["n90"]),
            "Influence_p90": _per_90(players["influence"], players["n90"]),
            "ICT_p90": _per_90(players["ict_index"], players["n90"]),
            "DefActions_p90": (defensive_actions / players["n90"].where(players["n90"] > 0))
            .fillna(0)
            .round(3),
            "Bonus_p90": _per_90(players["bonus"], players["n90"]),
            "BPS_p90": _per_90(players["bps"], players["n90"]),
            "Form": _to_numeric(players["form"]).round(2),
            "Status": players["status"],
            "News": players["news"].fillna(""),
        }
    )

    scout_df = scout_df.sort_values(["Minutes", "Goals", "Assists"], ascending=False).reset_index(
        drop=True
    )

    finished_events = [
        event for event in payload.get("events", []) if event.get("finished") and event.get("data_checked")
    ]
    current_event = next((event for event in payload.get("events", []) if event.get("is_current")), None)
    next_event = next((event for event in payload.get("events", []) if event.get("is_next")), None)
    metadata = {
        "source": "Fantasy Premier League public bootstrap-static API",
        "source_url": FPL_BOOTSTRAP_URL,
        "fetched_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "players": int(len(scout_df)),
        "teams": int(len(teams)),
        "latest_checked_gameweek": finished_events[-1]["name"] if finished_events else "Pre-season",
        "current_gameweek": current_event["name"] if current_event else None,
        "next_gameweek": next_event["name"] if next_event else None,
        "notes": (
            "FPL exposes goals, assists, expected goals/assists, ICT creativity/threat/influence, "
            "tackles, recoveries, minutes, price, ownership, and availability. Passing, shots, and "
            "dribble event detail are approximated through Creativity, Threat, and defensive action "
            "feature groups unless a richer event-data provider is connected."
        ),
    }
    return scout_df, metadata


def refresh_dataset(output_file: Path = OUTPUT_FILE) -> tuple[pd.DataFrame, dict]:
    payload = fetch_bootstrap_static()
    save_fpl_snapshot(payload, DB_FILE)
    scout_df = latest_features(DB_FILE)
    metadata = database_metadata(DB_FILE)
    scout_df.to_csv(output_file, index=False)
    METADATA_FILE.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return scout_df, metadata


if __name__ == "__main__":
    df, info = refresh_dataset()
    print(f"Saved {len(df)} current Premier League player rows to {OUTPUT_FILE}.")
    print(f"Appended snapshot to {DB_FILE}.")
    print(f"Latest checked gameweek: {info['latest_checked_gameweek']}")
