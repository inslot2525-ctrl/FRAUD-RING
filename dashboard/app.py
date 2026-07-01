import streamlit as st
import pandas as pd

st.set_page_config(page_title="Fraud Ring Detector", layout="wide")

st.title("Fraud Ring Detection Dashboard")

# Load data
clusters = pd.read_csv("data/processed/fraud_clusters.csv")
anomalies = pd.read_csv("data/processed/anomaly_scores.csv")

# Metrics
total_accounts = len(anomalies)
suspicious_accounts = anomalies["is_anomaly"].sum()
fraud_rings = len([x for x in clusters["cluster"].unique() if x != -1])

col1, col2, col3 = st.columns(3)

col1.metric("Total Accounts", total_accounts)
col2.metric("Suspicious Accounts", suspicious_accounts)
col3.metric("Fraud Rings Detected", fraud_rings)

st.subheader("Top Suspicious Accounts")
st.dataframe(clusters.head(50))