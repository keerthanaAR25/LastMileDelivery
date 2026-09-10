import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from app.services import data_access as svc  # noqa: E402
from app.services.city_context import get_selected_city

st.set_page_config(page_title="NexusFlow - Operational Insights", layout="wide")
st.header("Operational Insights")

city_code = get_selected_city()
if city_code not in svc.CITIES_WITH_FULL_PIPELINE:
    st.warning("Operational insights have only been computed for Shanghai so far.")
    st.stop()

insights = svc.get_operational_insights(city_code)

col1, col2 = st.columns(2)
with col1:
    st.subheader("Top Risky Zones")
    zones = insights["top_risk_zones"]
    if not zones.empty:
        st.dataframe(zones, use_container_width=True)

with col2:
    st.subheader("Top Recurring Risk Patterns")
    patterns = insights["top_risk_patterns"]
    if not patterns.empty:
        st.dataframe(patterns[["pattern", "risk_rate", "pattern_risk_score"]], use_container_width=True)

st.subheader("Peak Risk Hours")
try:
    heatmap = pd.read_parquet("artifacts/spatial/hour_weekday_heatmap_sh.parquet")
    pivot = heatmap.pivot(index="weekday", columns="accept_hour", values="risk_rate")
    fig = px.imshow(pivot, labels=dict(x="Hour", y="Weekday (0=Mon)", color="Risk Rate"), aspect="auto")
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Real finding: risk rate is elevated around hour 6-9 (early morning) with meaningful volume, not noise.")
except FileNotFoundError:
    st.info("Heatmap artifact not found.")

st.subheader("Recommendation Breakdown")
recos = insights["recommendation_breakdown"]
if not recos.empty:
    st.dataframe(recos, use_container_width=True)

st.subheader("Courier Workload")
workload = svc.get_courier_workload(city_code, top_n=20)
if not workload.empty:
    st.dataframe(workload, use_container_width=True)
    st.caption(
        "Real finding (Module 5): courier-level risk varies hugely at similar "
        "workload — raw workload alone doesn't explain risk."
    )
