import sys
from pathlib import Path

import plotly.express as px
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from app.services import data_access as svc  # noqa: E402
from app.services.city_context import get_selected_city  # noqa: E402
from app.services.road_network_display import (  # noqa: E402
    get_road_data,
    make_road_figure,
)
# from app.services.trajectory_congestion import (  # noqa: E402
#     get_congestion_data,
#     make_congestion_figure,
# )


st.set_page_config(
    page_title="NexusFlow - Spatial Intelligence",
    layout="wide",
)

st.header("Spatial Intelligence")

# -------------------------------------------------------------------
# Global city context
# -------------------------------------------------------------------

city_code = get_selected_city()
city_name = svc.CITY_CODE_TO_NAME.get(city_code, city_code)

st.caption(f"City: {city_name}")


# -------------------------------------------------------------------
# Pipeline availability
# -------------------------------------------------------------------

if city_code not in svc.CITIES_WITH_FULL_PIPELINE:
    st.warning("Spatial analytics are not available for this city.")
    st.stop()


# -------------------------------------------------------------------
# Retrieve spatial analytics
# -------------------------------------------------------------------

zones = svc.get_zone_risk(city_code, top_n=500)

hotspots_density = svc.get_hotspots(
    city_code,
    "DENSITY",
)

hotspots_risk = svc.get_hotspots(
    city_code,
    "RISK",
)


# -------------------------------------------------------------------
# Spatial analytics tabs
# -------------------------------------------------------------------

tab1, tab2, tab3 = st.tabs(
    [
        "Zone Risk Map",
        "Density Hotspots",
        "Risk Hotspots",
    ]
)


# ===================================================================
# TAB 1 — Zone Risk
# ===================================================================

with tab1:

    st.subheader("Zone-level spatial risk score")

    if not zones.empty:

        fig = px.scatter(
            zones,
            x="delivery_count",
            y="risk_rate",
            size="spatial_risk_score",
            hover_data=[
                "zone_id",
                "unique_couriers",
                "average_duration",
            ],
            labels={
                "delivery_count": "Delivery Count",
                "risk_rate": "Risk Rate",
            },
            title=(
                "Delivery volume vs risk rate per zone "
                "(bubble size = spatial risk score)"
            ),
        )

        st.plotly_chart(
            fig,
            use_container_width=True,
        )

        st.caption(
            "Zone-level spatial risk highlights areas with elevated "
            "delivery-duration risk. Trajectory-derived congestion is "
            "analyzed separately using courier movement data."
        )

        st.dataframe(
            zones.head(50),
            use_container_width=True,
        )

    else:

        st.info(
            "No zone-level spatial risk data is available "
            "for this city."
        )


# ===================================================================
# TAB 2 — Density Hotspots
# ===================================================================

with tab2:

    st.subheader(
        f"Density hotspots ({len(hotspots_density)} found)"
    )

    if not hotspots_density.empty:

        st.dataframe(
            hotspots_density.sort_values(
                "score",
                ascending=False,
            ),
            use_container_width=True,
        )

    else:

        st.info(
            "No density hotspots are available for this city."
        )


# ===================================================================
# TAB 3 — Risk Hotspots
# ===================================================================

with tab3:

    st.subheader(
        f"Risk hotspots ({len(hotspots_risk)} found)"
    )

    if not hotspots_risk.empty:

        st.dataframe(
            hotspots_risk.sort_values(
                "score",
                ascending=False,
            ),
            use_container_width=True,
        )

        st.caption(
            "Filtered to zones with at least 30 deliveries "
            "to reduce small-sample noise."
        )

    else:

        st.info(
            "No risk hotspots are available for this city."
        )


# ===================================================================
# REAL ROAD NETWORK
# ===================================================================

st.divider()

st.subheader("Real Road Network")

roads = get_road_data(city_code)

if roads.empty:

    st.info(
        "Road network unavailable for this city."
    )

else:

    st.plotly_chart(
        make_road_figure(
            roads,
            f"Real Road Network — {city_name}",
        ),
        use_container_width=True,
    )

    st.caption(
        "Road geometry is displayed from the provided OpenStreetMap-based "
        "road-network data for the selected city."
    )


# # ===================================================================
# # TRAJECTORY-DERIVED CONGESTION
# # ===================================================================

# st.divider()

# st.subheader("Trajectory-Derived Spatial Congestion")

# congestion = get_congestion_data()

# if congestion.empty:

#     st.info(
#         "Trajectory congestion output is not available yet."
#     )

# else:

#     fig = make_congestion_figure(
#         congestion
#     )

#     st.plotly_chart(
#         fig,
#         use_container_width=True,
#     )

#     st.caption(
#         "Congestion is derived from courier trajectory movement speed "
#         "relative to the observed reference speed within trajectory "
#         "spatial cells. The trajectory coordinates are privacy-transformed "
#         "and are therefore displayed in anonymized coordinate space."
#     )
