import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from app.services import data_access as svc  # noqa: E402
from app.services.city_context import get_selected_city

st.set_page_config(page_title="NexusFlow - Decision Support", layout="wide")
st.header("Decision Support — Delivery Intelligence Card")

city_code = get_selected_city()
if city_code not in svc.CITIES_WITH_FULL_PIPELINE:
    st.warning(f"Decision support data is not available for {city_code.upper()}.")
    st.stop()

sample = svc.get_available_deliveries_sample(city_code, n=30)
order_id = st.selectbox(
    "Select a delivery", options=sample["order_id"].tolist() if not sample.empty else [], key=f"dsc_order_{city_code}"
)

if order_id:
    delivery = svc.get_delivery(city_code, int(order_id))
    pred = svc.predict_delivery_risk(city_code, int(order_id))
    expl = svc.explain_delivery(city_code, int(order_id))
    reco = svc.generate_recommendation(city_code, int(order_id))
    sim = svc.simulate_route(city_code, int(order_id))

    with st.container(border=True):
        st.markdown("### 📦 DELIVERY INTELLIGENCE")
        c1, c2, c3 = st.columns(3)
        c1.metric("Delivery", f"#{order_id}")
        c2.metric("Courier", delivery["courier_id"] if delivery else "—")
        c3.metric("City", delivery["city"] if delivery else "—")

        if pred:
            c1, c2 = st.columns(2)
            c1.metric("Risk", pred["risk_class"])
            c2.metric("Probability", f"{pred['risk_probability']:.1%}")
        else:
            st.info("No prediction stored for this delivery.")

        st.markdown("#### WHY AT RISK?")
        if not expl.empty:
            for _, row in expl.iterrows():
                icon = "up" if row["direction"] == "INCREASES_RISK" else "down"
                st.write(f"[{icon}] **{row['feature_name']}** = {row['feature_value']:.3g} (SHAP {row['shap_value']:+.3f})")
        else:
            st.caption("No SHAP explanation stored for this specific delivery (only 5 demo examples in the DB).")

        st.markdown("#### WHAT SHOULD OPERATIONS DO?")
        if reco:
            st.write(f"**Recommendation:** `{reco['recommended_action']}`  |  **Priority:** {reco['priority']}")
            st.write(f"**Reason:** {reco['reason']}")
        else:
            st.caption("This delivery wasn't in the top-2,000 highest-risk sample the decision engine processed.")

        st.markdown("#### SIMULATION")
        if sim:
            c1, c2, c3 = st.columns(3)
            c1.metric("Baseline Risk", f"{sim['baseline_risk']:.1%}")
            c2.metric("Alternative Risk", f"{sim['alternative_risk']:.1%}")
            c3.metric("Delta", f"{sim['risk_delta']:+.1%}")
            st.caption(f"Label: {sim['label']} - a model-based scenario, not a validated causal effect or realized saving.")
        else:
            st.caption("No simulation available (only REASSIGN_COURIER/RESCHEDULE recommendations were simulated).")
