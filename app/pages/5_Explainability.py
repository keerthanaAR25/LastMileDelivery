import sys
from pathlib import Path

import plotly.express as px
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from app.services import data_access as svc  # noqa: E402
from app.services.city_context import get_selected_city

st.set_page_config(page_title="NexusFlow - Explainability", layout="wide")
st.header("Explainability (SHAP)")

city_code = get_selected_city()
if city_code not in svc.CITIES_WITH_FULL_PIPELINE:
    st.warning("SHAP explanations have only been computed for Shanghai so far.")
    st.stop()

tab1, tab2 = st.tabs(["Global Importance", "Local Explanation"])

with tab1:
    st.subheader("Global feature importance (mean |SHAP|, full 222,580-row test set)")
    try:
        import pandas as pd
        global_imp = pd.read_csv("artifacts/explanations/shap_global_importance_sh.csv")
        fig = px.bar(global_imp, x="mean_abs_shap", y="feature", orientation="h")
        st.plotly_chart(fig, use_container_width=True)
        st.warning(
            "Interpretive caveat: the top two features (aoi_risk_rate_train, "
            "courier_risk_rate_train) are target-encoded historical averages — "
            "they partially encode the outcome itself, so their high SHAP "
            "importance is expected and doesn't necessarily mean they are the "
            "most actionable causal driver. Genuine operational features "
            "(previous_delivery_duration, accept_hour, prior_stops_count) "
            "still show up meaningfully further down the ranking."
        )
    except FileNotFoundError:
        st.info("Global importance artifact not found.")

with tab2:
    st.subheader("Local explanation for a selected delivery")
    sample = svc.get_available_deliveries_sample(city_code, n=10)
    order_id = st.selectbox("Delivery", options=sample["order_id"].tolist() if not sample.empty else [])
    if order_id:
        expl = svc.explain_delivery(city_code, int(order_id))
        if not expl.empty:
            fig = px.bar(
                expl.sort_values("shap_value"), x="shap_value", y="feature_name",
                orientation="h", color="direction",
                color_discrete_map={"INCREASES_RISK": "#d62728", "DECREASES_RISK": "#2ca02c"},
                title=f"Top factors for order {order_id}",
            )
            st.plotly_chart(fig, use_container_width=True)
            st.dataframe(expl, use_container_width=True)
        else:
            st.info("No SHAP explanation stored for this delivery (only 5 demo examples are in the database).")
