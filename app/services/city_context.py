import streamlit as st


CITY_OPTIONS = {
    "sh": "Shanghai",
    "cq": "Chongqing",
    "hz": "Hangzhou",
    "jl": "Jilin",
    "yt": "Yantai",
}


def get_selected_city() -> str:
    """Return the city selected by the user across all dashboard pages."""

    if "city_code" not in st.session_state:
        st.session_state["city_code"] = "sh"

    return st.session_state["city_code"]


def render_city_selector() -> str:
    """Render the shared city selector and return the selected city code."""

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

    return city_code