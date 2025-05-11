
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

st.set_page_config(layout="wide")
st.title("MLB Trade Recommender System")

@st.cache_data
def generate_synthetic_dataset(n_players=900):
    np.random.seed(42)
    positions = ['C', '1B', '2B', '3B', 'SS', 'LF', 'CF', 'RF', 'DH', 'SP', 'RP']
    teams = ['ARI', 'LAD', 'SF', 'SD', 'COL', 'CHC', 'MIL', 'STL', 'PIT', 'CIN',
             'ATL', 'PHI', 'MIA', 'NYM', 'WAS', 'NYY', 'BOS', 'BAL', 'TB', 'TOR',
             'KC', 'DET', 'CWS', 'MIN', 'CLE', 'LAA', 'OAK', 'TEX', 'HOU', 'SEA']
    data = {
        "player_id": [f"Player{i}" for i in range(n_players)],
        "team": np.random.choice(teams, n_players),
        "position": np.random.choice(positions, n_players),
        "age": np.random.randint(20, 37, n_players),
        "WAR": np.round(np.random.normal(2.0, 1.5, n_players), 2),
        "contract_value": np.random.randint(500000, 35000000, n_players),
        "years_control": np.random.randint(1, 7, n_players),
        "sprint_speed": np.round(np.random.normal(27, 1.5, n_players), 2),
        "exit_velocity": np.round(np.random.normal(88, 5, n_players), 2),
        "defense_rating": np.round(np.random.uniform(0, 10, n_players), 2)
    }
    return pd.DataFrame(data)

df = generate_synthetic_dataset()

with st.sidebar:
    st.header("Filter Players")
    selected_team = st.selectbox("Select Team", options=["All"] + sorted(df["team"].unique().tolist()))
    selected_position = st.selectbox("Select Position", options=["All"] + sorted(df["position"].unique().tolist()))
    gm_mode = st.checkbox("Enable GM Mode (Advanced Metrics)")

    filtered_df = df.copy()
    if selected_team != "All":
        filtered_df = filtered_df[filtered_df["team"] == selected_team]
    if selected_position != "All":
        filtered_df = filtered_df[filtered_df["position"] == selected_position]

st.subheader("Filtered Players")
st.dataframe(filtered_df)

# Predict WAR
X = filtered_df[["age", "exit_velocity", "sprint_speed", "defense_rating"]]
y = filtered_df["WAR"]
reg = LinearRegression().fit(X, y)
filtered_df["predicted_WAR"] = reg.predict(X)

# Trade Rec
filtered_df["should_trade"] = (filtered_df["predicted_WAR"] < 2.0).astype(int)

# Visuals
st.subheader("Visualizations")
col1, col2 = st.columns(2)

with col1:
    st.markdown("**WAR Projection vs Age**")
    fig = px.scatter(filtered_df, x="age", y="predicted_WAR", color="position", hover_data=["player_id"])
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.markdown("**Team Contract Value vs WAR**")
    fig = px.scatter(filtered_df, x="contract_value", y="WAR", color="team", size="years_control")
    st.plotly_chart(fig, use_container_width=True)

if gm_mode:
    st.subheader("GM Mode Insights")
    filtered_df["benefit_score"] = (filtered_df["predicted_WAR"] / (filtered_df["contract_value"] / 1e6)) * filtered_df["years_control"]
    gm_display = filtered_df[["player_id", "team", "position", "predicted_WAR", "contract_value", "years_control", "benefit_score"]]
    st.dataframe(gm_display.sort_values("benefit_score", ascending=False))

# Download Report
st.subheader("Download Report")
st.download_button("Download Recommendations", data=filtered_df.to_csv(index=False), file_name="trade_recommendations.csv")

st.markdown("*WAR = Wins Above Replacement. Predicted WAR based on performance metrics. Trade suggestions are illustrative only.*")
