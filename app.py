import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from fetch_real_data import refresh_dataset
from scout_model import (
    DEFAULT_FEATURES,
    FEATURE_GROUPS,
    FEATURE_LABELS,
    available_features,
    fit_scout_model,
    percentile_profile,
    similar_players,
)


DATA_FILE = Path("premier_league_stats.csv")
METADATA_FILE = Path("data_metadata.json")


st.set_page_config(
    page_title="Premier League Player Scout",
    page_icon="PL",
    layout="wide",
    initial_sidebar_state="expanded",
)


st.markdown(
    """
    <style>
    .block-container {padding-top: 1.4rem; padding-bottom: 2.2rem;}
    div[data-testid="stMetric"] {
        background: #111827;
        border: 1px solid #263244;
        border-radius: 8px;
        padding: 0.85rem 1rem;
    }
    div[data-testid="stMetricLabel"] p {font-size: 0.82rem;}
    div[data-testid="stMetricValue"] {font-size: 1.35rem;}
    .small-note {color: #8b98aa; font-size: 0.88rem; line-height: 1.35;}
    .section-rule {border-top: 1px solid #263244; margin: 1.15rem 0;}
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data(show_spinner=False)
def load_data() -> pd.DataFrame:
    if not DATA_FILE.exists():
        df, _ = refresh_dataset()
        return df
    return pd.read_csv(DATA_FILE)


@st.cache_data(show_spinner=False)
def load_metadata() -> dict:
    if METADATA_FILE.exists():
        return json.loads(METADATA_FILE.read_text(encoding="utf-8"))
    return {
        "source": "Local CSV",
        "source_url": "",
        "fetched_at_utc": "Unknown",
        "latest_checked_gameweek": "Unknown",
        "notes": "Run python fetch_real_data.py to refresh the dataset.",
    }


def draw_radar_chart(
    df: pd.DataFrame,
    target_row: pd.Series,
    comparison_row: pd.Series,
    features: list[str],
) -> plt.Figure:
    labels = [FEATURE_LABELS.get(feature, feature) for feature in features]
    target_values = percentile_profile(df, target_row, features)
    comparison_values = percentile_profile(df, comparison_row, features)

    angles = np.linspace(0, 2 * np.pi, len(features), endpoint=False).tolist()
    target_values += target_values[:1]
    comparison_values += comparison_values[:1]
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(6.4, 6.4), subplot_kw={"polar": True})
    fig.patch.set_facecolor("#0f172a")
    ax.set_facecolor("#111827")
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_ylim(0, 100)
    ax.set_yticks([20, 40, 60, 80])
    ax.set_yticklabels(["20", "40", "60", "80"], color="#94a3b8", fontsize=8)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, color="#e5e7eb", fontsize=8)
    ax.spines["polar"].set_color("#334155")
    ax.grid(color="#263244", linewidth=0.8)

    ax.plot(angles, target_values, color="#2dd4bf", linewidth=2.4, label=target_row["Display_Name"])
    ax.fill(angles, target_values, color="#2dd4bf", alpha=0.22)
    ax.plot(
        angles,
        comparison_values,
        color="#f97316",
        linewidth=2.4,
        label=comparison_row["Display_Name"],
    )
    ax.fill(angles, comparison_values, color="#f97316", alpha=0.16)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.08), ncol=2, frameon=False, labelcolor="#e5e7eb")
    return fig


def draw_cluster_map(df: pd.DataFrame, selected_index: int) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(9, 5.2))
    fig.patch.set_facecolor("#0f172a")
    ax.set_facecolor("#0f172a")
    colors = plt.cm.Set2(df["Cluster_ID"] % 8)
    ax.scatter(df["PCA_1"], df["PCA_2"], s=48, c=colors, alpha=0.72, edgecolors="#0f172a", linewidths=0.8)
    selected = df.loc[selected_index]
    ax.scatter(
        [selected["PCA_1"]],
        [selected["PCA_2"]],
        s=220,
        c="#f97316",
        edgecolors="#f8fafc",
        linewidths=1.8,
        marker="*",
        zorder=5,
    )
    ax.annotate(
        selected["Display_Name"],
        (selected["PCA_1"], selected["PCA_2"]),
        xytext=(8, 8),
        textcoords="offset points",
        color="#f8fafc",
        fontsize=9,
    )
    ax.set_xlabel("PCA component 1", color="#cbd5e1")
    ax.set_ylabel("PCA component 2", color="#cbd5e1")
    ax.tick_params(colors="#94a3b8")
    ax.grid(color="#1e293b", linewidth=0.7)
    for spine in ax.spines.values():
        spine.set_color("#334155")
    return fig


df = load_data()
metadata = load_metadata()

features_in_data = available_features(df)
if not features_in_data:
    st.error("No modeling features were found in premier_league_stats.csv. Refresh the dataset first.")
    st.stop()

st.sidebar.title("Scout Controls")
if st.sidebar.button("Refresh Live Data", width="stretch"):
    with st.spinner("Fetching current Premier League player data..."):
        df, metadata = refresh_dataset()
        st.cache_data.clear()
        st.rerun()

selected_groups = st.sidebar.multiselect(
    "Feature groups",
    list(FEATURE_GROUPS.keys()),
    default=list(FEATURE_GROUPS.keys()),
)
group_features = [
    feature
    for group in selected_groups
    for feature in FEATURE_GROUPS[group]
    if feature in features_in_data
]
feature_options = [feature for feature in DEFAULT_FEATURES if feature in features_in_data]
selected_features = st.sidebar.multiselect(
    "Model features",
    feature_options,
    default=[feature for feature in group_features if feature in feature_options],
    format_func=lambda feature: FEATURE_LABELS.get(feature, feature),
)
if len(selected_features) < 2:
    st.sidebar.warning("Select at least two features.")
    st.stop()

cluster_count = st.sidebar.slider("Playing-style clusters", 3, 8, 5)
top_n = st.sidebar.slider("Closest matches", 3, 15, 8)
min_minutes = st.sidebar.slider("Minimum minutes", 0, max(90, int(df["Minutes"].max())), 0, step=30)
same_position = st.sidebar.checkbox("Same position only", value=False)
same_archetype = st.sidebar.checkbox("Same archetype only", value=False)

model = fit_scout_model(df, selected_features, cluster_count)
model_df = model.dataframe

saka_matches = model_df[model_df["Player"].str.contains("Bukayo Saka|Saka", case=False, na=False)]
default_index = int(saka_matches.index[0]) if not saka_matches.empty else 0
selected_label = st.sidebar.selectbox(
    "Target player",
    model_df["Search_Label"].tolist(),
    index=default_index,
)

selected_index = model_df.index[model_df["Search_Label"] == selected_label][0]
target_row = model_df.loc[selected_index]
matches = similar_players(
    model,
    selected_label=selected_label,
    top_n=top_n,
    min_minutes=min_minutes,
    same_position_only=same_position,
    same_archetype_only=same_archetype,
)

st.title("Premier League Player Scouting System")
st.caption(
    "Similarity search and K-Means playing-style clusters built with pandas, scikit-learn, matplotlib, and Streamlit."
)

metric_cols = st.columns(5)
metric_cols[0].metric("Target", target_row["Display_Name"], target_row["Team_Short"])
metric_cols[1].metric("Position", target_row["Position_Full"])
metric_cols[2].metric("Minutes", f"{int(target_row['Minutes']):,}")
metric_cols[3].metric("Archetype", target_row["Archetype"])
metric_cols[4].metric(
    "Silhouette",
    "N/A" if model.silhouette is None else f"{model.silhouette:.2f}",
)

st.markdown('<div class="section-rule"></div>', unsafe_allow_html=True)

profile_left, profile_right = st.columns([1.25, 1])

with profile_left:
    st.subheader(f"Closest Statistical Matches for {target_row['Display_Name']}")
    display_cols = [
        "Display_Name",
        "Team",
        "Position_Full",
        "Age",
        "Minutes",
        "Archetype",
        "Similarity",
        "Goals_p90",
        "Assists_p90",
        "xG_p90",
        "xA_p90",
        "Creativity_p90",
        "Threat_p90",
        "DefActions_p90",
    ]
    display_cols = [column for column in display_cols if column in matches.columns]
    st.dataframe(
        matches[display_cols].rename(
            columns={
                "Display_Name": "Player",
                "Position_Full": "Position",
                "Similarity": "Match %",
            }
        ),
        hide_index=True,
        width="stretch",
    )

with profile_right:
    st.subheader("Player Snapshot")
    totals = {
        "Goals": int(target_row.get("Goals", 0)),
        "Assists": int(target_row.get("Assists", 0)),
        "xG": float(target_row.get("Expected_Goals", 0)),
        "xA": float(target_row.get("Expected_Assists", 0)),
        "Cost": f"GBP {float(target_row.get('Cost_M', 0)):.1f}m",
        "Selected": f"{float(target_row.get('Selected_By_%', 0)):.1f}%",
    }
    stat_cols = st.columns(3)
    for idx, (label, value) in enumerate(totals.items()):
        stat_cols[idx % 3].metric(label, value)
    if str(target_row.get("News", "")).strip():
        st.warning(target_row["News"])

st.markdown('<div class="section-rule"></div>', unsafe_allow_html=True)

tabs = st.tabs(["Radar Comparison", "Cluster Map", "Archetype Explorer", "Data Notes"])

with tabs[0]:
    if matches.empty:
        st.info("No matches satisfy the active filters. Reduce the minutes threshold or turn off a filter.")
    else:
        comparison_name = st.selectbox(
            "Compare against",
            matches["Search_Label"].tolist(),
            format_func=lambda label: label.split(" | ")[0],
        )
        comparison_row = matches[matches["Search_Label"] == comparison_name].iloc[0]
        chart_col, table_col = st.columns([1, 1])
        with chart_col:
            st.pyplot(draw_radar_chart(model_df, target_row, comparison_row, selected_features), width="stretch")
        with table_col:
            diff = pd.DataFrame(
                {
                    "Metric": [FEATURE_LABELS.get(feature, feature) for feature in selected_features],
                    target_row["Display_Name"]: [target_row[feature] for feature in selected_features],
                    comparison_row["Display_Name"]: [comparison_row[feature] for feature in selected_features],
                    "Delta": [target_row[feature] - comparison_row[feature] for feature in selected_features],
                }
            )
            st.dataframe(diff.round(3), hide_index=True, width="stretch")

with tabs[1]:
    st.pyplot(draw_cluster_map(model_df, selected_index), width="stretch")

with tabs[2]:
    archetype_summary = (
        model_df.groupby(["Cluster_ID", "Archetype"])
        .agg(Players=("Player", "count"), Avg_Minutes=("Minutes", "mean"), Avg_xGI=("xGI_p90", "mean"))
        .reset_index()
        .sort_values("Players", ascending=False)
    )
    st.dataframe(archetype_summary.round(2), hide_index=True, width="stretch")
    chosen_cluster = st.selectbox(
        "Inspect cluster",
        sorted(model_df["Cluster_ID"].unique()),
        format_func=lambda cluster_id: f"{cluster_id} - {model_df.loc[model_df['Cluster_ID'] == cluster_id, 'Archetype'].iloc[0]}",
    )
    cluster_players = model_df[model_df["Cluster_ID"] == chosen_cluster].sort_values("Minutes", ascending=False)
    st.dataframe(
        cluster_players[
            [
                "Display_Name",
                "Team",
                "Position_Full",
                "Minutes",
                "Goals_p90",
                "Assists_p90",
                "xGI_p90",
                "Creativity_p90",
                "Threat_p90",
                "DefActions_p90",
            ]
        ].rename(columns={"Display_Name": "Player", "Position_Full": "Position"}),
        hide_index=True,
        width="stretch",
    )

with tabs[3]:
    source_url = metadata.get("source_url")
    source_text = metadata.get("source", "Unknown source")
    if source_url:
        st.markdown(f"**Source:** [{source_text}]({source_url})")
    else:
        st.markdown(f"**Source:** {source_text}")
    st.markdown(f"**Fetched at UTC:** {metadata.get('fetched_at_utc', 'Unknown')}")
    st.markdown(f"**Latest checked gameweek:** {metadata.get('latest_checked_gameweek', 'Unknown')}")
    st.markdown(
        '<p class="small-note">'
        + str(metadata.get("notes", ""))
        + "</p>",
        unsafe_allow_html=True,
    )
    st.markdown(
        '<p class="small-note">The similarity model standardizes selected per-90 features, computes cosine similarity, '
        'and labels K-Means clusters from each cluster&apos;s strongest feature profile.</p>',
        unsafe_allow_html=True,
    )
