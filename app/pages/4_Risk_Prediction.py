import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from app.services import data_access as svc  # noqa: E402
from app.services.city_context import get_selected_city

st.set_page_config(page_title="NexusFlow - Risk Prediction", layout="wide")
st.header("Risk Prediction")

city_code = st.session_state.get("city_code", "sh")
if city_code not in svc.CITIES_WITH_FULL_PIPELINE:
    st.warning("Risk prediction has only been run for Shanghai so far.")
    st.stop()

sample = svc.get_available_deliveries_sample(city_code, n=30)
order_id = st.selectbox(
    "Select a delivery (real test-set deliveries, highest predicted risk)",
    options=sample["order_id"].tolist() if not sample.empty else [],
)

if order_id:
    delivery = svc.get_delivery(city_code, int(order_id))
    pred = svc.predict_delivery_risk(city_code, int(order_id))

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Delivery")
        if delivery:
            st.write(f"**Order ID:** {delivery['order_id']}")
            st.write(f"**Courier:** {delivery['courier_id']}")
            st.write(f"**City:** {delivery['city']}")
            st.write(f"**AOI:** {delivery['aoi_id']} (type {delivery['aoi_type']})")
            st.write(f"**Region:** {delivery['region_id']}")
            st.write(f"**Duration:** {delivery['delivery_duration_min']:.0f} min")

    with col2:
        st.subheader("Prediction (LightGBM, primary model)")
        if pred:
            st.metric("Risk Probability", f"{pred['risk_probability']:.1%}")
            st.metric("Risk Class", pred["risk_class"])
        else:
            st.info("No prediction available for this delivery (only test-split deliveries have predictions).")

    st.caption(
        "Model: LightGBM, trained on chronologically-split real Shanghai data "
        "(1,038,704 train / 222,580 validation / 222,580 test). Test ROC-AUC "
        "0.857, PR-AUC 0.551 against a 9.6% base risk rate."
    )
