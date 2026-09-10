import sys
from pathlib import Path

import plotly.express as px
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from app.services import data_access as svc  # noqa: E402

st.set_page_config(
    page_title="NexusFlow - Sequential Patterns",
    layout="wide"
)

from app.services.city_context import get_selected_city
city_code = get_selected_city()

st.header(f"Sequential Pattern Explorer — {city_code.upper()}")

if city_code not in svc.CITIES_WITH_FULL_PIPELINE:
    st.warning(f"Sequential mining is not available for {city_code.upper()}.")
    st.stop()

st.caption(
    "Sequence definition: one sequence per (courier, day), events are "
    "DELIVERY@aoi_type ordered by delivery_time — real LaDe-D has one "
    "aoi_type per order, so sequences are built across a courier's orders, "
    "not within a single order (documented design decision, Module 3)."
)

patterns = svc.get_patterns(city_code, top_n=100)

if patterns.empty:
    st.info("No patterns available.")
else:
    st.subheader(f"Top patterns by pattern_risk_score ({len(patterns)} shown of full mined set)")
    st.dataframe(patterns, use_container_width=True)

    selected = st.selectbox("Select a pattern to inspect", options=patterns["pattern_id"].tolist())
    row = patterns[patterns["pattern_id"] == selected].iloc[0]

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Support Rate", f"{row['support_rate']:.2%}")
    col2.metric("Risk Rate", f"{row['risk_rate']:.2%}")
    col3.metric("Avg Duration", f"{row['average_duration']:.0f} min")
    col4.metric("Pattern Risk Score", f"{row['pattern_risk_score']:.4f}")

    st.write(f"**Pattern:** `{row['pattern']}`")
    st.write(f"**Occurrences:** {row['occurrence_count']:,} sequences | **Algorithm:** {row['algorithm']}")

    fig = px.bar(
        patterns.head(20), x="pattern_risk_score", y="pattern",
        orientation="h", hover_data=["support_rate", "risk_rate"],
    )
    st.plotly_chart(fig, use_container_width=True)
