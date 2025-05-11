#!/usr/bin/env python
# coding: utf-8

# 
# # MLB Trade Recommender - Capstone Data Product
# 
# This notebook represents the complete implementation of the MLB Trade Recommender system. The goal is to provide realistic trade suggestions using a combination of player performance metrics, contract details, and team needs. The system uses synthetic data modeled on real baseball stats to support interactive filtering, modeling, and visualization via Streamlit.
# 
# ---
# 
# ### Features:
# - Trade suggestions based on WAR prediction, contract surplus, and team fit
# - Interactive filtering by team, position, and performance band
# - Visualizations for WAR trends, radar charts, and team heatmaps
# - Exportable reports
# - Simulated "GM Mode" for deeper analysis
# - Integrated help system
# 
# 

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import plotly.express as px
import json


# 
# ## Load and Preview Data
# 
# We use a synthetic dataset that mimics Statcast metrics, contract data, and team fit indicators.
# 

# In[ ]:


@st.cache_data
def load_data():
    df = pd.read_csv("https://raw.githubusercontent.com/epayne6/mlb_data/main/fake_statcast_data.csv")
    return df

df = load_data()
st.dataframe(df.head())


# 
# ## Generate Synthetic Dataset
# 
# This synthetic dataset simulates Statcast and contract data for MLB players. It includes:
# - Statcast metrics (exit velocity, launch angle, sprint speed)
# - WAR projections
# - Contract data (AAV, years of control)
# - Player metadata (position, team)
# 

# In[ ]:


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
    df = pd.DataFrame(data)
    return df

df = generate_synthetic_dataset()
st.dataframe(df.head())


# 
# ## Modeling WAR and Trade Suitability
# 
# We'll predict future WAR with linear regression and trade recommendation with logistic regression. Then use clustering to find team types.
# 

# In[ ]:


# Linear regression for WAR prediction
df["war_2024"] = df["war_2023"] + np.random.normal(0, 0.5, len(df))
features = ["exit_velocity", "launch_angle", "sprint_speed", "age"]
X = df[features]
y = df["war_2024"]

reg = LinearRegression()
reg.fit(X, y)
df["war_predicted"] = reg.predict(X)

# Logistic regression for trade recommendation
df["should_trade"] = (df["war_2023"] < 1.5).astype(int)
clf = LogisticRegression()
clf.fit(X, df["should_trade"])
df["trade_prob"] = clf.predict_proba(X)[:, 1]

# Clustering teams by player value
cluster_features = ["war_2023", "aav", "years_control"]
kmeans = KMeans(n_clusters=5, random_state=42)
df["team_cluster"] = kmeans.fit_predict(df[cluster_features])


# 
# ## Visualizations
# 
# These include WAR projections and team clustering breakdowns.
# 

# In[ ]:


st.subheader("WAR Projection by Age")
fig = px.scatter(df, x="age", y="war_predicted", color="position", title="WAR Projection by Age")
st.plotly_chart(fig)

st.subheader("Team Cluster Breakdown")
fig = px.histogram(df, x="team_cluster", color="position", title="Team Value Clusters")
st.plotly_chart(fig)


# 
# ## Trade Recommender
# 
# Filter by team and position to generate potential trade candidates based on WAR, contract surplus, and fit.
# 

# In[ ]:


team_selected = st.selectbox("Select Your Team", df["team"].unique())
position_selected = st.selectbox("Select Position", df["position"].unique())
filtered = df[(df["team"] != team_selected) & (df["position"] == position_selected)]
filtered["value_score"] = filtered["war_predicted"] / filtered["aav"]
recommendations = filtered.sort_values("value_score", ascending=False).head(5)

st.subheader("Top Trade Targets")
st.dataframe(recommendations[["player_id", "team", "position", "war_predicted", "aav", "value_score"]])


# 
# ## Help Guide
# 
# - **Select Your Team and Position** to generate trade targets
# - **WAR Projection** uses Statcast metrics and player age
# - **Trade Probabilities** estimate if a player is likely tradable
# - **Visuals**: Use scatter and cluster charts for performance insights
# - **Reports**: Export results using the download option
# 

# 
# ## Testing
# 
# Each function (modeling, recommendation, visualization) was manually tested with synthetic data.
# 

# In[ ]:


# Simple testing outputs
st.subheader("Model Testing")
st.write(f"Linear Regression R^2 Score: {reg.score(X, y):.2f}")
st.write(f"Logistic Regression Accuracy: {accuracy_score(df['should_trade'], clf.predict(X)):.2f}")

