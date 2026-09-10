import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from app.services import data_access as svc  # noqa: E402
from app.services.city_context import get_selected_city  # noqa: E402


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="NexusFlow - Route Simulation",
    layout="wide"
)


# ============================================================
# HEADER
# ============================================================

st.header("Route / Intervention Simulation")

city_code = get_selected_city()
city_name = svc.CITY_CODE_TO_NAME.get(
    city_code,
    city_code
)


# ============================================================
# PIPELINE CHECK
# ============================================================

if city_code not in svc.CITIES_WITH_FULL_PIPELINE:

    st.warning(
        f"Simulation data is not available for {city_name}."
    )

    st.stop()


# ============================================================
# EXPLANATION
# ============================================================

st.info(
    "PHYSICAL REROUTE simulation is not yet implemented. "
    "The road network is available in PostgreSQL/PostGIS, but the current "
    "decision engine does not yet generate an alternative road path and "
    "rescore that path. The simulations shown here are REASSIGN_COURIER "
    "and RESCHEDULE scenarios, where the trained model re-scores the "
    "delivery using an alternative courier profile or a different "
    "accept hour."
)


# ============================================================
# LOAD CITY-SPECIFIC SIMULATION ARTIFACT
# ============================================================

simulation_path = (
    Path("artifacts")
    / "recommendations"
    / f"simulations_{city_code}.csv"
)


try:

    sims = pd.read_csv(
        simulation_path
    )

except FileNotFoundError:

    sims = pd.DataFrame()


# ============================================================
# NO DATA
# ============================================================

if sims.empty:

    st.warning(
        f"No simulation artifact was found for {city_name}."
    )

    st.stop()


# ============================================================
# CLEAN SIMULATION DATA
# ============================================================

required_columns = [
    "simulation_kind",
    "order_id",
    "baseline_risk",
    "alternative_risk",
    "risk_delta",
]

missing_columns = [
    column
    for column in required_columns
    if column not in sims.columns
]

if missing_columns:

    st.error(
        "Simulation artifact is missing required columns: "
        + ", ".join(missing_columns)
    )

    st.stop()


sims = sims.copy()


# ============================================================
# SIMULATION TYPE
# ============================================================

simulation_types = (
    sims["simulation_kind"]
    .dropna()
    .astype(str)
    .unique()
    .tolist()
)


if not simulation_types:

    st.warning(
        f"No simulation types are available for {city_name}."
    )

    st.stop()


kind = st.selectbox(
    "Simulation type",
    options=simulation_types
)


subset = sims[
    sims["simulation_kind"].astype(str) == kind
].copy()


if subset.empty:

    st.warning(
        f"No {kind} simulations are available for {city_name}."
    )

    st.stop()


# ============================================================
# SUMMARY METRICS
# ============================================================

c1, c2, c3 = st.columns(3)


c1.metric(
    "Simulations",
    f"{len(subset):,}"
)


mean_delta = subset["risk_delta"].mean()

c2.metric(
    "Mean Risk Delta",
    f"{mean_delta:+.1%}"
)


risk_reduction_rate = (
    subset["risk_delta"] < 0
).mean()


c3.metric(
    "% Reducing Risk",
    f"{risk_reduction_rate:.0%}"
)


# ============================================================
# INTERPRETATION
# ============================================================

if kind == "REASSIGN_COURIER":

    st.warning(
        "Interpretive caveat: this effect is computed by the actual "
        "trained model, but it is substantially mechanical. "
        "courier_risk_rate_train is a target-encoded historical feature, "
        "so replacing the courier statistics with a lower-risk courier "
        "profile will generally reduce the predicted risk. "
        "This is a MODEL-BASED SCENARIO, not validated proof that "
        "reassignment causally reduces delivery risk."
    )

elif kind == "RESCHEDULE":

    st.caption(
        "Model-based finding: rescheduling produces very small changes "
        "in predicted risk in the available simulations. This represents "
        "the model's counterfactual response to changing the accept hour; "
        "it should not be interpreted as a causal effect."
    )


# ============================================================
# RISK DELTA DISTRIBUTION
# ============================================================

st.subheader(
    f"Risk Delta Distribution — {kind}"
)


fig = px.histogram(
    subset,
    x="risk_delta",
    nbins=40,
    title=f"Distribution of Risk Delta — {kind}"
)


fig.update_layout(
    xaxis_title="Risk Delta",
    yaxis_title="Number of Simulations"
)


st.plotly_chart(
    fig,
    use_container_width=True
)


# ============================================================
# DELIVERY-LEVEL INSPECTION
# ============================================================

st.subheader(
    "Inspect One Delivery"
)


order_options = (
    subset["order_id"]
    .dropna()
    .tolist()
)


if not order_options:

    st.info(
        "No delivery IDs are available for inspection."
    )

else:

    order_id = st.selectbox(
        "Select delivery",
        options=order_options
    )


    row = subset[
        subset["order_id"] == order_id
    ].iloc[0]


    # --------------------------------------------------------
    # DELIVERY RISK COMPARISON
    # --------------------------------------------------------

    c1, c2, c3 = st.columns(3)


    c1.metric(
        "Baseline Risk",
        f"{float(row['baseline_risk']):.1%}"
    )


    c2.metric(
        "Alternative Risk",
        f"{float(row['alternative_risk']):.1%}"
    )


    c3.metric(
        "Delta",
        f"{float(row['risk_delta']):+.1%}"
    )


    # --------------------------------------------------------
    # SIMULATION LABEL
    # --------------------------------------------------------

    if "label" in row.index:

        st.caption(
            f"Label: {row['label']}"
        )


# ============================================================
# METHODOLOGICAL NOTE
# ============================================================

st.subheader(
    "Simulation Interpretation"
)

st.write(
    f"""
The results above are generated from the NexusFlow decision-support
pipeline for **{city_name}**.

The current implementation evaluates model-based intervention scenarios.
REASSIGN_COURIER changes the courier-related historical statistics used
by the trained model, while RESCHEDULE changes the delivery acceptance
hour used for rescoring.

These simulations estimate how the model's predicted risk changes under
the specified scenario. They do not establish a causal effect or
guarantee an operational saving.

The PostgreSQL/PostGIS road network is available, but physical
alternative-path generation and route rescoring are not yet connected
to this page.
"""
)