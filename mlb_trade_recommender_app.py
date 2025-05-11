
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import plotly.express as px

st.title("MLB Trade Recommender System")

@st.cache_data
def generate_synthetic_dataset(n_players=900):
    np.random.seed(42)
    positions = ['C', '1B', '2B', '3B', 'SS', 'LF', 'CF', 'RF', 'DH', 'SP', 'RP']
    teams = ['ARI', 'LAD', 'SF', 'SD', 'COL', 'CHC', 'MIL', 'STL', 'PIT', 'CIN',
             'ATL', 'PHI', 'MIA', 'NYM', 'WAS', 'NYY', 'BOS', 'BAL', 'TB', 'TOR',
             'KC', 'DET', 'CWS', 'MIN', 'CLE', 'LAA', 'ATH', 'TEX', 'HOU', 'SEA']
    data = {
        "player_id": [f"Player{i}" for i in range(n_players)],
        "team": np.random.choice(teams, n_players),
        "position": np.random.choice(positions, n_players),
        "age": np.random.randint(20, 37, n_players),
        "exit_velocity": np.round(np.random.normal(88, 4, n_players), 1),
        "launch_angle": np.round(np.random.normal(12, 5, n_players), 1),
        "sprint_speed": np.round(np.random.normal(27, 1.5, n_players), 1),
        "spin_rate": np.round(np.random.normal(2200, 150, n_players), 1),
        "horizontal_break": np.round(np.random.normal(10, 3, n_players), 1),
        "war_2023": np.round(np.random.normal(2.0, 1.5, n_players), 2),
        "aav": np.round(np.random.normal(5.5, 2.5, n_players), 2),
        "years_control": np.random.randint(1, 7, n_players)
    }
    return pd.DataFrame(data)

df = generate_synthetic_dataset()
df["war_2024"] = df["war_2023"] + np.random.normal(0, 0.5, len(df))
features = ["exit_velocity", "launch_angle", "sprint_speed", "age"]
X = df[features]
y = df["war_2024"]

reg = LinearRegression()
reg.fit(X, y)
df["war_predicted"] = reg.predict(X)

df["should_trade"] = (df["war_2023"] < 1.5).astype(int)
clf = LogisticRegression()
clf.fit(X, df["should_trade"])
df["trade_prob"] = clf.predict_proba(X)[:, 1]

cluster_features = ["war_2023", "aav", "years_control"]
kmeans = KMeans(n_clusters=5, random_state=42)
df["team_cluster"] = kmeans.fit_predict(df[cluster_features])

st.sidebar.header("Filters")
team_selected = st.sidebar.selectbox("Select Your Team", df["team"].unique())
position_selected = st.sidebar.selectbox("Select Position", df["position"].unique())

filtered = df[(df["team"] != team_selected) & (df["position"] == position_selected)]
filtered["value_score"] = filtered["war_predicted"] / filtered["aav"]
recommendations = filtered.sort_values("value_score", ascending=False).head(5)

st.subheader("Top Trade Recommendations")
st.dataframe(recommendations[["player_id", "team", "position", "war_predicted", "aav", "value_score"]])

st.subheader("WAR Projection by Age")
fig1 = px.scatter(df, x="age", y="war_predicted", color="position", title="WAR Projection by Age")
st.plotly_chart(fig1)

st.subheader("Team Cluster Breakdown")
fig2 = px.histogram(df, x="team_cluster", color="position", title="Team Value Clusters")
st.plotly_chart(fig2)

st.subheader("Model Testing")
st.write(f"Linear Regression R² Score: {reg.score(X, y):.2f}")
st.write(f"Logistic Regression Accuracy: {accuracy_score(df['should_trade'], clf.predict(X)):.2f}")
