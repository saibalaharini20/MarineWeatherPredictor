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
st.markdown("Select a coastal location to predict the marine weather condition.")

# Default coordinates
lat = 37.7749
lng = -122.4194

# --- Location Input ---
st.subheader("📍 Choose Location Input Method")

input_method = st.radio(
    "Select how you want to provide the location:",
    ("Manual Input", "Select from Map")
)

# Manual coordinates
if input_method == "Manual Input":

    col1, col2 = st.columns(2)

    with col1:
        lat = st.number_input("🌍 Latitude", value=lat)

    with col2:
        lng = st.number_input("📍 Longitude", value=lng)

# Map selection
elif input_method == "Select from Map":

    st.write("Click anywhere on the map to select a location.")

    m = folium.Map(location=[lat, lng], zoom_start=2)

    map_data = st_folium(m, height=400, width=700)

    if map_data and map_data["last_clicked"] is not None:
        lat = map_data["last_clicked"]["lat"]
        lng = map_data["last_clicked"]["lng"]

        st.success(f"Selected Location: {lat:.4f}, {lng:.4f}")

# --- API Key ---
api_key = st.text_input("🔑 Enter your StormGlass API Key", type="password")

# --- Load Model ---
model_path = os.path.join(os.path.dirname(__file__), "marine_model.pkl")

@st.cache_resource
def load_model():
    return joblib.load(model_path)

model = load_model()

# --- Prediction ---
if st.button("🌤 Predict Marine Condition"):

    df = None

    # Try fetching real-time data
    if api_key:

        url = f"https://api.stormglass.io/v2/weather/point?lat={lat}&lng={lng}&params=waveHeight,windSpeed,swellHeight,swellPeriod"

        response_raw = requests.get(url, headers={"Authorization": api_key})

        if response_raw.status_code != 200:
            st.warning(f"⚠️ Could not fetch data: {response_raw.status_code}")

        else:
            try:
                response = response_raw.json()

                df = pd.json_normalize(response['hours'])

                df['Time'] = pd.to_datetime(df['time'])

                df['Wave Height (m)'] = df['waveHeight'].apply(
                    lambda x: list(x.values())[0] if isinstance(x, dict) else None
                )

                df['Wind Speed (m/s)'] = df['windSpeed'].apply(
                    lambda x: list(x.values())[0] if isinstance(x, dict) else None
                )

                df['Swell Height (m)'] = df['swellHeight'].apply(
                    lambda x: list(x.values())[0] if isinstance(x, dict) else None
                )

                df['Swell Period (s)'] = df['swellPeriod'].apply(
                    lambda x: list(x.values())[0] if isinstance(x, dict) else None
                )

                df = df[['Time','Wave Height (m)','Wind Speed (m/s)','Swell Height (m)','Swell Period (s)']]

            except Exception as e:
                st.warning(f"⚠️ Error processing API data: {e}")

    # Use sample data if API fails
    if df is None:

        st.info("Using sample data for prediction.")

        df = pd.DataFrame([{
            'Time': pd.Timestamp.now(),
            'Wave Height (m)': 1.2,
            'Wind Speed (m/s)': 6.5,
            'Swell Height (m)': 0.8,
            'Swell Period (s)': 10
        }])

    # --- Feature Engineering ---
    df['wind_x'] = df['Wind Speed (m/s)'] * np.cos(np.radians(45))
    df['wind_y'] = df['Wind Speed (m/s)'] * np.sin(np.radians(45))
    df['wave_energy'] = 0.5 * 1025 * 9.81 * (df['Wave Height (m)'] ** 2)

    # --- Prediction ---
    X_latest = df[['Wave Height (m)', 'Wind Speed (m/s)', 'Swell Height (m)',
                   'Swell Period (s)', 'wind_x', 'wind_y', 'wave_energy']].iloc[-1:]

    prediction = model.predict(X_latest)

    st.success(f"🌊 **Predicted Marine Condition:** {prediction[0]}")
    st.caption("Condition derived using wave, wind, and swell parameters.")

    # --- Chart ---
    st.line_chart(df.set_index('Time')[['Wave Height (m)', 'Wind Speed (m/s)']])
