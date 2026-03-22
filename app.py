import streamlit as st
import pandas as pd
import numpy as np
import requests
import joblib
import os

# --- Page Setup ---
st.set_page_config(page_title="🌊 Marine Weather Predictor", layout="wide")

st.title("🌊 Marine Weather Predictor Dashboard")
st.markdown("Select a coastal location to predict marine weather conditions.")

# --- Try importing folium (optional dependency) ---
try:
    import folium
    from streamlit_folium import st_folium
    FOLIUM_AVAILABLE = True
except ImportError:
    FOLIUM_AVAILABLE = False

# Default coordinates
lat = 37.7749
lng = -122.4194

# --- Location Input ---
st.subheader("📍 Choose Location Input Method")

if FOLIUM_AVAILABLE:
    input_method = st.radio(
        "Select how you want to provide the location:",
        ("Manual Input", "Select from Map")
    )
else:
    input_method = "Manual Input"
    st.info("ℹ️ Map selection unavailable. Install `folium` and `streamlit-folium` to enable it.")

if input_method == "Manual Input":
    col1, col2 = st.columns(2)
    with col1:
        lat = st.number_input("🌍 Latitude", value=lat, min_value=-90.0, max_value=90.0)
    with col2:
        lng = st.number_input("📍 Longitude", value=lng, min_value=-180.0, max_value=180.0)

else:
    st.write("Click anywhere on the map to select a location.")
    m = folium.Map(location=[lat, lng], zoom_start=2)
    map_data = st_folium(m, height=400, width=700)

    if map_data and map_data.get("last_clicked"):
        lat = map_data["last_clicked"]["lat"]
        lng = map_data["last_clicked"]["lng"]
        st.success(f"✅ Selected Location: {lat:.4f}, {lng:.4f}")

# --- API Key ---
api_key = st.text_input("🔑 Enter your StormGlass API Key (optional)", type="password")

# --- Load ML Model safely ---
model = None
model_path = os.path.join(os.path.dirname(__file__), "marine_model.pkl")

@st.cache_resource
def load_model(path):
    return joblib.load(path)

if os.path.exists(model_path):
    try:
        model = load_model(model_path)
    except Exception as e:
        st.error(f"❌ Failed to load model: {e}")
else:
    st.warning("⚠️ `marine_model.pkl` not found next to this script. Predictions will be unavailable.")

# --- Helper: Extract numeric value from StormGlass nested dict ---
def extract_value(df, col):
    if col in df.columns:
        return df[col].apply(
            lambda x: list(x.values())[0] if isinstance(x, dict) else (x if pd.notna(x) else None)
        )
    return pd.Series([None] * len(df))

# --- Helper: Feature engineering ---
def engineer_features(df):
    df = df.copy()
    df["wind_x"] = df["Wind Speed (m/s)"] * np.cos(np.radians(45))
    df["wind_y"] = df["Wind Speed (m/s)"] * np.sin(np.radians(45))
    df["wave_energy"] = 0.5 * 1025 * 9.81 * (df["Wave Height (m)"] ** 2)
    return df

FEATURE_COLS = [
    "Wave Height (m)",
    "Wind Speed (m/s)",
    "Swell Height (m)",
    "Swell Period (s)",
    "wind_x",
    "wind_y",
    "wave_energy"
]

# --- Prediction Button ---
if st.button("🌤 Predict Marine Condition"):

    if model is None:
        st.error("❌ Cannot predict: model file is missing or failed to load.")
        st.stop()

    df = None
    using_fallback = False

    # --- Fetch from StormGlass API ---
    if api_key:
        with st.spinner("Fetching marine data from StormGlass..."):
            try:
                url = (
                    f"https://api.stormglass.io/v2/weather/point"
                    f"?lat={lat}&lng={lng}"
                    f"&params=waveHeight,windSpeed,swellHeight,swellPeriod"
                )
                response_raw = requests.get(
                    url,
                    headers={"Authorization": api_key},
                    timeout=10
                )

                if response_raw.status_code == 401:
                    st.warning("⚠️ Invalid API key. Check your StormGlass credentials.")
                elif response_raw.status_code == 429:
                    st.warning("⚠️ API rate limit exceeded. Try again later.")
                elif response_raw.status_code != 200:
                    st.warning(f"⚠️ API error {response_raw.status_code}: Could not fetch marine data.")
                else:
                    response = response_raw.json()

                    if "hours" not in response or not response["hours"]:
                        st.warning("⚠️ API returned no marine data for this location.")
                    else:
                        raw_df = pd.json_normalize(response["hours"])
                        raw_df["Time"] = pd.to_datetime(raw_df["time"])

                        raw_df["Wave Height (m)"]  = extract_value(raw_df, "waveHeight")
                        raw_df["Wind Speed (m/s)"] = extract_value(raw_df, "windSpeed")
                        raw_df["Swell Height (m)"] = extract_value(raw_df, "swellHeight")
                        raw_df["Swell Period (s)"] = extract_value(raw_df, "swellPeriod")

                        raw_df = raw_df[[
                            "Time",
                            "Wave Height (m)",
                            "Wind Speed (m/s)",
                            "Swell Height (m)",
                            "Swell Period (s)"
                        ]]

                        # Drop rows with any nulls (don't discard entire df)
                        df = raw_df.dropna().sort_values("Time").reset_index(drop=True)

                        if df.empty:
                            st.warning("⚠️ API data had no complete rows after cleaning. Using fallback data.")
                            df = None

            except requests.exceptions.Timeout:
                st.warning("⚠️ API request timed out. Using fallback data.")
            except requests.exceptions.ConnectionError:
                st.warning("⚠️ Could not connect to StormGlass API. Check your internet connection.")
            except Exception as e:
                st.warning(f"⚠️ Unexpected error fetching API data: {e}")

    # --- Fallback Sample Data ---
    if df is None:
        using_fallback = True
        st.info("ℹ️ Using sample data for prediction (no API data available).")
        df = pd.DataFrame([{
            "Time": pd.Timestamp.now(),
            "Wave Height (m)": 1.2,
            "Wind Speed (m/s)": 6.5,
            "Swell Height (m)": 0.8,
            "Swell Period (s)": 10.0
        }])

    # --- Feature Engineering ---
    df = engineer_features(df)

    # --- Validate features before prediction ---
    missing_cols = [c for c in FEATURE_COLS if c not in df.columns]
    if missing_cols:
        st.error(f"❌ Missing features for prediction: {missing_cols}")
        st.stop()

    # --- Prediction on latest row ---
    X_latest = df[FEATURE_COLS].iloc[[-1]]

    try:
        prediction = model.predict(X_latest)
        condition = prediction[0]
    except Exception as e:
        st.error(f"❌ Prediction failed: {e}")
        st.stop()

    # --- Results ---
    st.divider()
    st.subheader("🔍 Prediction Result")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("🌊 Wave Height", f"{X_latest['Wave Height (m)'].values[0]:.2f} m")
    col2.metric("💨 Wind Speed",  f"{X_latest['Wind Speed (m/s)'].values[0]:.2f} m/s")
    col3.metric("🌀 Swell Height", f"{X_latest['Swell Height (m)'].values[0]:.2f} m")
    col4.metric("⏱ Swell Period", f"{X_latest['Swell Period (s)'].values[0]:.1f} s")

    st.success(f"🌊 Predicted Marine Condition: **{condition}**")

    if using_fallback:
        st.caption("⚠️ Prediction based on sample data — provide an API key for real conditions.")
    else:
        st.caption(f"Prediction based on latest data point at {df['Time'].iloc[-1].strftime('%Y-%m-%d %H:%M UTC')}.")

    # --- Visualization (only meaningful with multiple rows) ---
    if len(df) > 1:
        st.subheader("📈 Wave Height & Wind Speed Over Time")
        chart_df = df.set_index("Time")[["Wave Height (m)", "Wind Speed (m/s)"]].sort_index()
        st.line_chart(chart_df)
    else:
        st.info("📊 Time-series chart requires multiple data points (available with API data).")
