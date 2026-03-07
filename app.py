import streamlit as st
import pandas as pd
import numpy as np
import requests
import joblib
import os
import folium
from streamlit_folium import st_folium

# --- Page Setup ---
st.set_page_config(page_title="🌊 Marine Weather Predictor", layout="wide")

st.title("🌊 Marine Weather Predictor Dashboard")
st.markdown("Select a coastal location to predict marine weather conditions.")

# Default coordinates
lat = 37.7749
lng = -122.4194

# --- Location Input ---
st.subheader("📍 Choose Location Input Method")

input_method = st.radio(
    "Select how you want to provide the location:",
    ("Manual Input", "Select from Map")
)

if input_method == "Manual Input":

    col1, col2 = st.columns(2)

    with col1:
        lat = st.number_input("🌍 Latitude", value=lat)

    with col2:
        lng = st.number_input("📍 Longitude", value=lng)

else:

    st.write("Click anywhere on the map to select a location.")

    m = folium.Map(location=[lat, lng], zoom_start=2)

    map_data = st_folium(m, height=400, width=700)

    if map_data and map_data["last_clicked"]:

        lat = map_data["last_clicked"]["lat"]
        lng = map_data["last_clicked"]["lng"]

        st.success(f"Selected Location: {lat:.4f}, {lng:.4f}")

# --- API Key ---
api_key = st.text_input("🔑 Enter your StormGlass API Key", type="password")

# --- Load ML Model ---
model_path = os.path.join(os.path.dirname(__file__), "marine_model.pkl")

@st.cache_resource
def load_model():
    return joblib.load(model_path)

model = load_model()

# --- Prediction Button ---
if st.button("🌤 Predict Marine Condition"):

    df = None

    # Try fetching API data
    if api_key:

        url = f"https://api.stormglass.io/v2/weather/point?lat={lat}&lng={lng}&params=waveHeight,windSpeed,swellHeight,swellPeriod"

        response_raw = requests.get(url, headers={"Authorization": api_key})

        if response_raw.status_code != 200:

            st.warning("⚠️ Could not fetch marine data from API.")

        else:

            try:

                response = response_raw.json()

                if "hours" not in response:

                    st.warning("⚠️ API returned no marine data.")

                else:

                    df = pd.json_normalize(response["hours"])

                    df["Time"] = pd.to_datetime(df["time"])

                    # Safe extraction function
                    def extract_value(col):

                        if col in df.columns:

                            return df[col].apply(
                                lambda x: list(x.values())[0] if isinstance(x, dict) else None
                            )

                        return None

                    df["Wave Height (m)"] = extract_value("waveHeight")
                    df["Wind Speed (m/s)"] = extract_value("windSpeed")
                    df["Swell Height (m)"] = extract_value("swellHeight")
                    df["Swell Period (s)"] = extract_value("swellPeriod")

                    df = df[[
                        "Time",
                        "Wave Height (m)",
                        "Wind Speed (m/s)",
                        "Swell Height (m)",
                        "Swell Period (s)"
                    ]]

            except Exception as e:

                st.warning(f"⚠️ Error processing API data: {e}")
                df = None

    # --- Fallback Sample Data ---
    if df is None or df.isnull().values.any():

        st.info("Using sample data for prediction.")

        df = pd.DataFrame([{

            "Time": pd.Timestamp.now(),
            "Wave Height (m)": 1.2,
            "Wind Speed (m/s)": 6.5,
            "Swell Height (m)": 0.8,
            "Swell Period (s)": 10

        }])

    # --- Feature Engineering ---
    df["wind_x"] = df["Wind Speed (m/s)"] * np.cos(np.radians(45))
    df["wind_y"] = df["Wind Speed (m/s)"] * np.sin(np.radians(45))
    df["wave_energy"] = 0.5 * 1025 * 9.81 * (df["Wave Height (m)"] ** 2)

    # --- Prediction ---
    X_latest = df[[
        "Wave Height (m)",
        "Wind Speed (m/s)",
        "Swell Height (m)",
        "Swell Period (s)",
        "wind_x",
        "wind_y",
        "wave_energy"
    ]].iloc[-1:]

    prediction = model.predict(X_latest)

    st.success(f"🌊 Predicted Marine Condition: **{prediction[0]}**")

    st.caption("Prediction based on wave, wind and swell parameters.")

    # --- Visualization ---
    st.line_chart(
        df.set_index("Time")[["Wave Height (m)", "Wind Speed (m/s)"]]
    )
