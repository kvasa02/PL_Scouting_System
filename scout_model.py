from dataclasses import dataclass
import os
import warnings

import numpy as np
import pandas as pd

os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(os.cpu_count() or 1))
warnings.filterwarnings("ignore", message="Could not find the number of physical cores.*")
warnings.filterwarnings("ignore", category=UserWarning, module="joblib.externals.loky.backend.context")

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler


FEATURE_GROUPS = {
    "End Product": ["Goals_p90", "Assists_p90", "xG_p90", "xA_p90", "xGI_p90"],
    "Chance Creation": ["Creativity_p90", "Threat_p90", "ICT_p90"],
    "Defensive Work": ["DefActions_p90", "Influence_p90", "BPS_p90"],
}

DEFAULT_FEATURES = [
    "Goals_p90",
    "Assists_p90",
    "xG_p90",
    "xA_p90",
    "xGI_p90",
    "Creativity_p90",
    "Threat_p90",
    "ICT_p90",
    "DefActions_p90",
    "BPS_p90",
]

FEATURE_LABELS = {
    "Goals_p90": "Goals / 90",
    "Assists_p90": "Assists / 90",
    "xG_p90": "xG / 90",
    "xA_p90": "xA / 90",
    "xGI_p90": "xGI / 90",
    "Creativity_p90": "Creativity / 90",
    "Threat_p90": "Threat / 90",
    "ICT_p90": "ICT / 90",
    "Influence_p90": "Influence / 90",
    "DefActions_p90": "Def Actions / 90",
    "BPS_p90": "BPS / 90",
    "Bonus_p90": "Bonus / 90",
}

POSITION_NAMES = {
    "GK": "Goalkeeper",
    "GKP": "Goalkeeper",
    "DEF": "Defender",
    "MID": "Midfielder",
    "FWD": "Forward",
}


@dataclass(frozen=True)
class ScoutModel:
    dataframe: pd.DataFrame
    features: list[str]
    scaler: StandardScaler
    scaled_features: np.ndarray
    similarity_matrix: np.ndarray
    pca_projection: np.ndarray
    silhouette: float | None


def available_features(dataframe: pd.DataFrame) -> list[str]:
    return [feature for feature in DEFAULT_FEATURES if feature in dataframe.columns]


def prepare_dataframe(dataframe: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    prepared = dataframe.copy()
    for feature in features:
        prepared[feature] = pd.to_numeric(prepared[feature], errors="coerce").fillna(0)

    prepared["Minutes"] = pd.to_numeric(prepared["Minutes"], errors="coerce").fillna(0).astype(int)
    prepared["Age"] = pd.to_numeric(prepared.get("Age", 0), errors="coerce").fillna(0).astype(int)
    prepared["Position_Full"] = prepared["Position"].map(POSITION_NAMES).fillna(prepared["Position"])
    prepared["Search_Label"] = (
        prepared["Player"] + " | " + prepared["Team"] + " | " + prepared["Position_Full"]
    )
    return prepared


def fit_scout_model(dataframe: pd.DataFrame, features: list[str], n_clusters: int) -> ScoutModel:
    prepared = prepare_dataframe(dataframe, features)
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(prepared[features])

    cluster_count = max(2, min(n_clusters, len(prepared) - 1))
    kmeans = KMeans(n_clusters=cluster_count, random_state=42, n_init=20)
    prepared["Cluster_ID"] = kmeans.fit_predict(scaled_features)
    prepared["Archetype"] = label_clusters(prepared, features)

    similarity_matrix = cosine_similarity(scaled_features)
    pca_projection = PCA(n_components=2, random_state=42).fit_transform(scaled_features)
    silhouette = None
    if len(prepared["Cluster_ID"].unique()) > 1 and len(prepared) > cluster_count:
        silhouette = float(silhouette_score(scaled_features, prepared["Cluster_ID"]))

    prepared["PCA_1"] = pca_projection[:, 0]
    prepared["PCA_2"] = pca_projection[:, 1]
    return ScoutModel(
        dataframe=prepared,
        features=features,
        scaler=scaler,
        scaled_features=scaled_features,
        similarity_matrix=similarity_matrix,
        pca_projection=pca_projection,
        silhouette=silhouette,
    )


def label_clusters(dataframe: pd.DataFrame, features: list[str]) -> pd.Series:
    cluster_means = dataframe.groupby("Cluster_ID")[features].mean()
    league_means = dataframe[features].mean().replace(0, np.nan)
    relative = cluster_means.divide(league_means, axis=1).fillna(0)

    labels = {}
    for cluster_id, row in relative.iterrows():
        if row.get("xG_p90", 0) > 1.15 or row.get("Goals_p90", 0) > 1.15:
            labels[cluster_id] = "Box Finisher"
        elif row.get("Creativity_p90", 0) > 1.15 or row.get("Assists_p90", 0) > 1.15:
            labels[cluster_id] = "Chance Creator"
        elif row.get("DefActions_p90", 0) > 1.15 or row.get("BPS_p90", 0) > 1.15:
            labels[cluster_id] = "Ball Winner"
        elif row.get("Threat_p90", 0) > 1.15 or row.get("ICT_p90", 0) > 1.15:
            labels[cluster_id] = "Direct Attacker"
        else:
            labels[cluster_id] = "Balanced Contributor"
    return dataframe["Cluster_ID"].map(labels)


def similar_players(
    model: ScoutModel,
    selected_label: str,
    top_n: int,
    min_minutes: int,
    same_position_only: bool,
    same_archetype_only: bool,
) -> pd.DataFrame:
    df = model.dataframe
    selected_index = df.index[df["Search_Label"] == selected_label][0]
    selected_row = df.loc[selected_index]
    scores = pd.Series(model.similarity_matrix[selected_index], index=df.index)

    candidates = df.copy()
    candidates["Similarity"] = (scores * 100).round(1)
    candidates = candidates[(candidates.index != selected_index) & (candidates["Minutes"] >= min_minutes)]

    if same_position_only:
        candidates = candidates[candidates["Position"] == selected_row["Position"]]
    if same_archetype_only:
        candidates = candidates[candidates["Cluster_ID"] == selected_row["Cluster_ID"]]

    return candidates.sort_values(["Similarity", "Minutes"], ascending=False).head(top_n)


def percentile_profile(dataframe: pd.DataFrame, row: pd.Series, features: list[str]) -> list[float]:
    percentiles = []
    for feature in features:
        values = pd.to_numeric(dataframe[feature], errors="coerce").fillna(0)
        value = float(row[feature])
        percentiles.append(round((values <= value).mean() * 100, 1))
    return percentiles
