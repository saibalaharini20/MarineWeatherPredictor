"""
app.py — Marine Weather Predictor
Entry point for the Streamlit dashboard.
"""
import os
import streamlit as st
import pandas as pd
import pydeck as pdk
import numpy as np
import joblib
from datetime import datetime
from dotenv import load_dotenv

# Load .env file if it exists (silently ignored if missing)
load_dotenv()

import folium
from streamlit_folium import st_folium

from utils.model_loader import get_model
from utils.data_fetcher import (
    fetch_openmeteo_data, fetch_stormglass_data,
    get_sample_data, engineer_features, FEATURE_COLS, ADVANCED_FEATURE_COLS, geocode_city,
)
from utils.charts import render_trend_chart, render_feature_importance

# ---------------------------------------------------------------------------
# Page configuration
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="🌊 Marine Weather Predictor",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .stApp { background: linear-gradient(180deg, #021124 0%, #061A2B 100%); color: #e6f0ff; }
    .stButton>button { background: #0ea5a0; color: white; border-radius: 6px; }
    .streamlit-expanderHeader { color: #cfeffd; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("🌊 Marine Weather Predictor")
st.markdown(
    "Fetch live oceanic parameters via the StormGlass API or use built-in presets. "
    "The trained model predicts a sea-state label: **Calm**, **Moderate**, or **Rough**."
)

# ---------------------------------------------------------------------------
# Sidebar — controls
# ---------------------------------------------------------------------------
PRESETS = {
    "San Francisco, USA": (37.7749, -122.4194),
    "Sydney, AU": (-33.8688, 151.2093),
    "Cape Town, ZA": (-33.9249, 18.4241),
    "Mumbai, IN": (19.0760, 72.8777),
    "Honolulu, US-HI": (21.3069, -157.8583),
    "Visakhapatnam, IN": (17.6868, 83.2185),
    "Chennai, IN": (13.0827, 80.2707),
    "Kochi, IN": (9.9312, 76.2673),
}

with st.sidebar:
    st.header("⚙️ Controls")

    # ---------- Method 1: City search ----------
    st.markdown("###### 🔍 Search by city name")
    city_col, btn_col = st.columns([3, 1])
    with city_col:
        city_input = st.text_input(
            "City",
            placeholder="e.g. Visakhapatnam",
            label_visibility="collapsed",
        )
    with btn_col:
        search_clicked = st.button("Go", use_container_width=True)

    if search_clicked and city_input.strip():
        with st.spinner(f"Looking up '{city_input}'..."):
            geo = geocode_city(city_input.strip())
        if geo:
            st.session_state["sel_lat"] = geo[0]
            st.session_state["sel_lng"] = geo[1]
            st.session_state["sel_label"] = geo[2]
            st.success(f"📍 {geo[2][:55]}")
        else:
            st.error(f"❌ Could not find '{city_input}'.")

    st.markdown("---")

    # ---------- Method 2: Preset ----------
    st.markdown("###### 📌 Or choose a preset")
    preset_name = st.selectbox(
        "Location preset",
        ["— (using map / search)"] + list(PRESETS.keys()),
    )
    if preset_name != "— (using map / search)":
        st.session_state["sel_lat"] = PRESETS[preset_name][0]
        st.session_state["sel_lng"] = PRESETS[preset_name][1]
        st.session_state["sel_label"] = preset_name

    st.markdown("---")

    # ---------- Resolved lat/lng (editable) ----------
    _lat = float(st.session_state.get("sel_lat", 20.0))
    _lng = float(st.session_state.get("sel_lng", 80.0))
    lat = st.number_input("Latitude", value=_lat, format="%.4f")
    lng = st.number_input("Longitude", value=_lng, format="%.4f")
    # Sync manual edits back to session state
    st.session_state["sel_lat"] = lat
    st.session_state["sel_lng"] = lng

    st.markdown("---")

    # ---------- API key ----------
    _env_key = os.getenv("STORMGLASS_API_KEY", "")
    api_key = st.text_input(
        "StormGlass API Key (optional)",
        value=_env_key,
        type="password",
        help="Set STORMGLASS_API_KEY in a .env file to avoid typing it each session.",
    )

    with st.expander("Advanced options"):
        hours = st.slider("Hours to fetch / display", min_value=1, max_value=72, value=24)
        use_live = st.checkbox("Fetch live data (StormGlass)", value=bool(api_key))
        units = st.radio("Units", ["Metric (m, m/s)", "Nautical (ft, knots)"], index=0)
        smoothing = st.slider("Smoothing window (points)", min_value=0, max_value=6, value=0)

    st.markdown("---")
    st.caption("Built with Streamlit + scikit-learn. Model loaded from `marine_model.pkl`.")

# ---------------------------------------------------------------------------
# Method 3: Interactive map location picker (main area)
# ---------------------------------------------------------------------------
with st.expander("🗺️ Click on the map to pick a location", expanded=False):
    st.caption(
        "Click anywhere on the ocean to set the prediction location. "
        "The blue marker shows your currently active coordinates."
    )

    _map_lat = float(st.session_state.get("sel_lat", 20.0))
    _map_lng = float(st.session_state.get("sel_lng", 80.0))

    # Build a dark-tiled world map with a marker at the current selection
    m = folium.Map(
        location=[_map_lat, _map_lng],
        zoom_start=4,
        tiles="CartoDB dark_matter",
    )
    folium.Marker(
        location=[_map_lat, _map_lng],
        tooltip=f"Selected: {_map_lat:.4f}, {_map_lng:.4f}",
        icon=folium.Icon(color="blue", icon="map-marker"),
    ).add_to(m)

    # Render map and capture click events
    map_data = st_folium(m, width="100%", height=380, returned_objects=["last_clicked"])

    if map_data and map_data.get("last_clicked"):
        clicked = map_data["last_clicked"]
        new_lat = round(clicked["lat"], 4)
        new_lng = round(clicked["lng"], 4)
        # Only update if the click resulted in a different location
        if (new_lat != st.session_state.get("sel_lat") or
                new_lng != st.session_state.get("sel_lng")):
            st.session_state["sel_lat"] = new_lat
            st.session_state["sel_lng"] = new_lng
            st.session_state["sel_label"] = f"Map pick ({new_lat}, {new_lng})"
            st.rerun()

    active_label = st.session_state.get("sel_label", "—")
    st.info(f"📍 **Active location:** {active_label} — lat={lat:.4f}, lng={lng:.4f}")


# ---------------------------------------------------------------------------
# Load model
# ---------------------------------------------------------------------------
model = get_model()

# ---------------------------------------------------------------------------
# Predict button
# ---------------------------------------------------------------------------
if st.button("🔮 Predict marine condition", key="predict_button"):
    # 1. Fetch data — priority: StormGlass (if key given) > Open-Meteo (free) > sample
    df = None
    data_source = None

    # Priority 1: StormGlass (paid, user-supplied key)
    if api_key and use_live:
        df = fetch_stormglass_data(lat, lng, api_key, hours)
        if df is not None and not df.empty:
            data_source = "stormglass"

    # Priority 2: Open-Meteo — free, no key, real location data
    if df is None or df.empty:
        df = fetch_openmeteo_data(lat, lng, hours)
        if df is not None and not df.empty:
            data_source = "openmeteo"

    # Priority 3: Hardcoded sample (true last resort — inland or API down)
    if df is None or df.empty:
        df = get_sample_data(hours)
        data_source = "sample"

    # --- Data source banner ---
    if data_source == "stormglass":
        st.success(
            f"✅ **StormGlass live data** — lat={lat:.4f}, lng={lng:.4f} "
            f"({len(df)} hours of real conditions)"
        )
    elif data_source == "openmeteo":
        st.success(
            f"✅ **Real marine forecast** (Open-Meteo, free) — lat={lat:.4f}, lng={lng:.4f} "
            f"| {len(df)} hours · no API key needed"
        )
    else:
        st.warning(
            f"⚠️ **Fallback to sample data** — Open-Meteo returned no marine data for "
            f"lat={lat:.4f}, lng={lng:.4f} (location may be inland or over land). "
            f"Results reflect generic conditions, NOT your location."
        )

    # 2. Feature engineering
    df = engineer_features(df)

    # 3. Predict
    # Detect if we have an advanced model (marine_model_v2.pkl)
    if os.path.exists("marine_model_v2.pkl"):
        # Load the upgraded model bundle
        bundle = joblib.load("marine_model_v2.pkl")
        model_v2 = bundle['model']
        feature_list = bundle.get('features', ADVANCED_FEATURE_COLS)
        X_latest = df[feature_list].iloc[-1:]
        prediction_idx = model_v2.predict(X_latest)
        # Decode the categorical labels
        le = bundle.get('encoder')
        if le:
            prediction = le.classes_[prediction_idx]
        else:
            prediction = prediction_idx
        active_model_info = "⚡ Upgraded XGBoost Model (v2)"
    else:
        # Standard fallback to original model
        if model is None:
            st.error("❌ No model loaded. Make sure `marine_model.pkl` exists in the folder.")
            st.stop()
        X_latest = df[FEATURE_COLS].iloc[-1:]
        prediction = model.predict(X_latest)
        active_model_info = "🚢 Baseline Random Forest Model (v1)"

    st.caption(f"Predicted using: {active_model_info}")

    # 4. Confidence
    proba_text = ""
    proba = None
    if hasattr(model, "predict_proba"):
        try:
            proba = model.predict_proba(X_latest)[0]
            top_idx = np.argmax(proba)
            top_conf = proba[top_idx]
            top_class = model.classes_[top_idx]
            proba_text = f"{top_conf * 100:.0f}% confidence for {top_class}"
        except Exception:
            pass

    # ---------------------------------------------------------------------------
    # Layout — left: KPIs | right: map + readings
    # ---------------------------------------------------------------------------
    left_col, right_col = st.columns([1.3, 1])
    last = df.iloc[-1]

    with left_col:
        st.markdown("### 🔵 Prediction")
        st.metric(label="Predicted Condition", value=str(prediction[0]), delta=proba_text)

        # Risk score
        try:
            wave = float(last["Wave Height (m)"])
            wind = float(last["Wind Speed (m/s)"])
            risk = (0.6 * min(wave / 6.0, 1.0) + 0.4 * min(wind / 20.0, 1.0)) * 100
            st.metric(
                label="Estimated Risk Score (0–100)",
                value=f"{risk:.0f}",
                delta=f"Wave {wave:.1f} m  |  Wind {wind:.1f} m/s",
            )
        except Exception:
            pass

    with right_col:
        st.markdown("### 📍 Location & latest readings")
        try:
            map_df = pd.DataFrame({"lat": [lat], "lon": [lng]})
            st.map(map_df, zoom=6)
        except Exception:
            try:
                view = pdk.ViewState(latitude=float(lat), longitude=float(lng), zoom=6)
                layer = pdk.Layer(
                    "ScatterplotLayer",
                    data=[{"lat": lat, "lon": lng}],
                    get_position="[lon, lat]",
                    get_radius=50000,
                    get_color=[0, 120, 255, 140],
                )
                st.pydeck_chart(pdk.Deck(layers=[layer], initial_view_state=view))
            except Exception:
                pass

        st.write(f"🌊 Wave: **{last['Wave Height (m)']:.2f} m**")
        st.write(f"💨 Wind: **{last['Wind Speed (m/s)']:.2f} m/s**")
        st.write(f"🌀 Swell: **{last['Swell Height (m)']:.2f} m** / **{last['Swell Period (s)']:.0f} s**")

    st.caption("Condition derived from wave, wind, and swell parameters.")

    # ---------------------------------------------------------------------------
    # Safety guidance
    # ---------------------------------------------------------------------------
    st.markdown("---")
    with st.expander("⚠️ Forecast guidance & safety tips"):
        severity = prediction[0].lower()
        wave_val = float(last["Wave Height (m)"])
        wind_val = float(last["Wind Speed (m/s)"])

        if severity == "rough" or wave_val > 3 or wind_val > 15:
            st.error("🚨 High caution: conditions are rough. Avoid small vessels.")
        elif severity == "moderate" or wave_val > 1.5 or wind_val > 8:
            st.warning("⚠️ Moderate conditions: experienced crews should prepare.")
        else:
            st.success("✅ Calm conditions — suitable for most small craft.")

        st.markdown("**Tactical tips:**")
        st.write("- Monitor local forecasts; marine weather can change quickly.")
        st.write("- Avoid small craft when wave height or wind speed rises significantly.")
        if proba is not None:
            st.write(
                "Model probabilities: "
                + ", ".join(f"{c}: {p * 100:.0f}%" for c, p in zip(model.classes_, proba))
            )

    # ---------------------------------------------------------------------------
    # Trend charts
    # ---------------------------------------------------------------------------
    st.subheader("📈 Trends")
    render_trend_chart(df, units, smoothing)

    # ---------------------------------------------------------------------------
    # Raw data table + CSV download
    # ---------------------------------------------------------------------------
    st.subheader("📋 Hourly data sample")
    display_cols = ["Time", "Wave Height (m)", "Wind Speed (m/s)", "Swell Height (m)", "Swell Period (s)"]
    st.dataframe(df[display_cols].tail(12).reset_index(drop=True))

    csv = df[display_cols].to_csv(index=False).encode("utf-8")
    st.download_button(
        "⬇️ Download data as CSV",
        data=csv,
        file_name=f"marine_data_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
        mime="text/csv",
    )

    # ---------------------------------------------------------------------------
    # Feature importance
    # ---------------------------------------------------------------------------
    if hasattr(model, "feature_importances_"):
        st.markdown("---")
        st.subheader("🧠 Model Feature Importance")
        render_feature_importance(model)
