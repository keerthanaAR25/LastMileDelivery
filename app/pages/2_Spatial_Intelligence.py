import sys
from pathlib import Path

import plotly.express as px
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from app.services import data_access as svc  # noqa: E402
from app.services.city_context import get_selected_city

st.set_page_config(page_title="NexusFlow - Spatial Intelligence", layout="wide")
st.header("Spatial Intelligence")

city_code = get_selected_city()
st.caption(f"City: {svc.CITY_CODE_TO_NAME.get(city_code, city_code)}")

if city_code not in svc.CITIES_WITH_FULL_PIPELINE:
    st.warning("Spatial analytics have only been run for Shanghai so far. Select Shanghai in the sidebar on the Overview page.")
    st.stop()

zones = svc.get_zone_risk(city_code, top_n=500)
hotspots_density = svc.get_hotspots(city_code, "DENSITY")
hotspots_risk = svc.get_hotspots(city_code, "RISK")

tab1, tab2, tab3 = st.tabs(["Zone Risk Map", "Density Hotspots", "Risk Hotspots"])

with tab1:
    st.subheader("Zone-level spatial risk score")
    if not zones.empty:
        fig = px.scatter(
            zones, x="delivery_count", y="risk_rate", size="spatial_risk_score",
            hover_data=["zone_id", "unique_couriers", "average_duration"],
            labels={"delivery_count": "Delivery Count", "risk_rate": "Risk Rate"},
            title="Delivery volume vs risk rate per zone (bubble size = spatial risk score)",
        )
        st.plotly_chart(fig, use_container_width=True)
        st.caption(
            "Real finding: density and risk hotspots are almost entirely different zones — "
            "the busiest zone has low risk, the riskiest zones are low-volume. "
            "congestion_index is NOT available (no trajectory data yet)."
        )
        st.dataframe(zones.head(50), use_container_width=True)

with tab2:
    st.subheader(f"Density hotspots ({len(hotspots_density)} found)")
    if not hotspots_density.empty:
        st.dataframe(hotspots_density.sort_values("score", ascending=False), use_container_width=True)

with tab3:
    st.subheader(f"Risk hotspots ({len(hotspots_risk)} found)")
    if not hotspots_risk.empty:
        st.dataframe(hotspots_risk.sort_values("score", ascending=False), use_container_width=True)
        st.caption("Filtered to zones with at least 30 deliveries to avoid small-sample noise (Section 29).")
