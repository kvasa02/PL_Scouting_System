# generate_sample_data.py
import pandas as pd
import numpy as np

np.random.seed(42)

players = [
    ("Bukayo Saka", "Arsenal", "Winger", 22, 2800),
    ("Mohamed Salah", "Liverpool", "Winger", 32, 2900),
    ("Phil Foden", "Man City", "Winger", 24, 2750),
    ("Cole Palmer", "Chelsea", "Attacking Mid", 22, 2600),
    ("Martin Ødegaard", "Arsenal", "Attacking Mid", 25, 2950),
    ("Kevin De Bruyne", "Man City", "Attacking Mid", 33, 1800),
    ("Son Heung-min", "Tottenham", "Winger", 32, 2700),
    ("Jarrod Bowen", "West Ham", "Winger", 27, 2850),
    ("Anthony Gordon", "Newcastle", "Winger", 23, 2650),
    ("Bryan Mbeumo", "Brentford", "Winger", 25, 2400),
    ("Luis Díaz", "Liverpool", "Winger", 27, 2500),
    ("Eberechi Eze", "Crystal Palace", "Attacking Mid", 26, 2450),
    ("Declan Rice", "Arsenal", "Central Mid", 25, 3100),
    ("Rodri", "Man City", "Central Mid", 28, 3000),
    ("Alexis Mac Allister", "Liverpool", "Central Mid", 25, 2700),
    ("Bruno Guimarães", "Newcastle", "Central Mid", 26, 2900),
    ("Enzo Fernández", "Chelsea", "Central Mid", 23, 2550),
    ("Dominik Szoboszlai", "Liverpool", "Central Mid", 23, 2300),
    ("Erling Haaland", "Man City", "Striker", 24, 2600),
    ("Alexander Isak", "Newcastle", "Striker", 24, 2300),
    ("Ollie Watkins", "Aston Villa", "Striker", 28, 3050),
    ("Nicolas Jackson", "Chelsea", "Striker", 23, 2600),
    ("William Saliba", "Arsenal", "Centre-Back", 23, 3150),
    ("Virgil van Dijk", "Liverpool", "Centre-Back", 33, 3100),
    ("Pedro Porro", "Tottenham", "Full-Back", 24, 2800),
    ("Trent Alexander-Arnold", "Liverpool", "Full-Back", 25, 2400)
]

data = []
for name, team, pos, age, mins in players:
    n90 = mins / 90.0
    if pos in ["Winger", "Attacking Mid"]:
        goals = np.random.uniform(0.30, 0.65)
        assists = np.random.uniform(0.20, 0.45)
        shots = np.random.uniform(2.2, 3.8)
        key_passes = np.random.uniform(1.8, 3.2)
        prog_carries = np.random.uniform(4.0, 8.5)
        succ_dribbles = np.random.uniform(1.5, 3.8)
        tackles_int = np.random.uniform(1.0, 2.5)
        pressures = np.random.uniform(12.0, 20.0)
    elif pos == "Striker":
        goals = np.random.uniform(0.55, 0.95)
        assists = np.random.uniform(0.10, 0.25)
        shots = np.random.uniform(3.5, 5.0)
        key_passes = np.random.uniform(0.8, 1.6)
        prog_carries = np.random.uniform(1.5, 3.5)
        succ_dribbles = np.random.uniform(0.6, 1.8)
        tackles_int = np.random.uniform(0.4, 1.2)
        pressures = np.random.uniform(8.0, 14.0)
    elif pos == "Central Mid":
        goals = np.random.uniform(0.08, 0.25)
        assists = np.random.uniform(0.15, 0.35)
        shots = np.random.uniform(0.8, 1.8)
        key_passes = np.random.uniform(1.2, 2.5)
        prog_carries = np.random.uniform(2.5, 5.0)
        succ_dribbles = np.random.uniform(0.8, 2.0)
        tackles_int = np.random.uniform(3.0, 5.5)
        pressures = np.random.uniform(16.0, 26.0)
    else:  # Defenders / Fullbacks
        goals = np.random.uniform(0.02, 0.12)
        assists = np.random.uniform(0.08, 0.28)
        shots = np.random.uniform(0.3, 1.1)
        key_passes = np.random.uniform(0.6, 2.1)
        prog_carries = np.random.uniform(2.0, 5.5)
        succ_dribbles = np.random.uniform(0.4, 1.2)
        tackles_int = np.random.uniform(3.5, 6.5)
        pressures = np.random.uniform(10.0, 18.0)

    data.append({
        "Player": name,
        "Team": team,
        "Position": pos,
        "Age": age,
        "Minutes": mins,
        "Goals_p90": round(goals, 2),
        "Assists_p90": round(assists, 2),
        "Shots_p90": round(shots, 2),
        "KeyPasses_p90": round(key_passes, 2),
        "ProgCarries_p90": round(prog_carries, 2),
        "SuccDribbles_p90": round(succ_dribbles, 2),
        "TacklesInt_p90": round(tackles_int, 2),
        "Pressures_p90": round(pressures, 2)
    })

df = pd.DataFrame(data)
df.to_csv("premier_league_stats.csv", index=False)
print("Saved 'premier_league_stats.csv' successfully.")
