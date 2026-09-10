import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app.services import data_access as svc
from app.services.city_context import get_selected_city  # noqa: E402

st.set_page_config(page_title="NexusFlow", layout="wide", page_icon="🚚")

CITY_OPTIONS = {"sh": "Shanghai", "cq": "Chongqing", "hz": "Hangzhou", "jl": "Jilin", "yt": "Yantai"}

with st.sidebar:
    st.title("🚚 NexusFlow")
    st.caption("Pattern-to-Decision Intelligence for Last-Mile Delivery")
    city_code = st.selectbox(
    "City",
    options=list(CITY_OPTIONS.keys()),
    format_func=lambda c: CITY_OPTIONS[c],
    index=list(CITY_OPTIONS.keys()).index(
        st.session_state.get("city_code", "sh")
    ),
    key="global_city_selector",
    )
    st.session_state["city_code"] = city_code
    st.divider()
    st.caption(
        "This is real data (Cainiao LaDe-D, 5 cities)."
    )

st.header("Executive Overview")

summary = svc.get_dashboard_summary(city_code)

if not summary["full_pipeline_available"]:
    st.warning(
        f"**{CITY_OPTIONS[city_code]}**: raw delivery data is loaded "
        f"({summary['total_deliveries']:,} deliveries), but pattern mining, "
        "risk prediction, and decision support have not been run for this "
        "city yet — only Shanghai has the full pipeline. Showing raw stats only."
    )

col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Deliveries", f"{summary['total_deliveries']:,}")
col2.metric("Total Couriers", f"{summary['total_couriers']:,}")
col3.metric("Avg Delivery Duration", f"{summary['avg_duration_min']:.0f} min" if summary['avg_duration_min'] else "—")
col4.metric("Avg Risk Rate", f"{summary['avg_risk_rate']:.1%}" if summary['avg_risk_rate'] else "—")

if summary["full_pipeline_available"]:
    col1, col2, col3 = st.columns(3)
    col1.metric("Spatial Zones Analyzed", f"{summary['n_zones']:,}")
    col2.metric("Hotspots Detected", f"{summary['n_hotspots']:,}")
    col3.metric("Sequential Patterns Mined", f"{summary['n_patterns']:,}")

    st.subheader("Top Risk Zones")
    zones = svc.get_zone_risk(city_code, top_n=15)
    if not zones.empty:
        fig = px.bar(
            zones.sort_values("spatial_risk_score", ascending=True).tail(15),
            x="spatial_risk_score", y="zone_id", orientation="h",
            hover_data=["delivery_count", "risk_rate"],
            labels={"spatial_risk_score": "Spatial Risk Score", "zone_id": "Zone"},
        )
        st.plotly_chart(fig, use_container_width=True)
        st.caption(
            "Note: top-ranked zones can be small-sample (few deliveries, "
            "100% observed risk) — see the Spatial Intelligence page for volume context."
        )

    st.subheader("Recommendation Breakdown (top 2,000 highest-risk deliveries)")
    insights = svc.get_operational_insights(city_code)
    recos = insights["recommendation_breakdown"]
    if not recos.empty:
        fig2 = px.pie(recos, names="recommended_action", values="n", hole=0.4)
        st.plotly_chart(fig2, use_container_width=True)
else:
    st.info("Once Modules 3-8 are extended to this city, zone risk, hotspots, and pattern charts will appear here.")

st.divider()
st.caption(
    "NexusFlow identifies recurring delivery-risk patterns and recommends "
    "operational interventions. It does not claim to eliminate delays or "
    "guarantee cost savings — route-simulation results are labelled "
    "SIMULATED_INTERVENTION_IMPACT, never realized savings."
)
