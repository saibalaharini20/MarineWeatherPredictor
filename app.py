import streamlit as st
import pandas as pd
import numpy as np
import requests
import joblib
import os

# --- Page Setup ---
st.set_page_config(page_title="🌊 Marine Weather Predictor", layout="wide")
st.title("🌊 Marine Weather Predictor Dashboard")
st.markdown("Enter the latitude and longitude of any coastal region to view the marine weather condition.")

# --- Input Section ---
col1, col2 = st.columns(2)
with col1:
    lat = st.number_input("🌍 Latitude", value=37.7749)
with col2:
    lng = st.number_input("📍 Longitude", value=-122.4194)

api_key = st.text_input("🔑 Enter your StormGlass API Key", type="password")

# --- Load Model Safely ---
model_path = os.path.join(os.path.dirname(__file__), "marine_model.pkl")
model = joblib.load(model_path)

# --- Prediction Section ---
if st.button("🌤 Predict Marine Condition"):

    df = None
    # Try fetching real-time data if API key provided
    if api_key:
        url = f"https://api.stormglass.io/v2/weather/point?lat={lat}&lng={lng}&params=waveHeight,windSpeed,swellHeight,swellPeriod"
        response_raw = requests.get(url, headers={"Authorization": api_key})

        if response_raw.status_code != 200:
            st.warning(f"⚠️ Could not fetch data: {response_raw.status_code} - {response_raw.text}")
        else:
            try:
                response = response_raw.json()
                df = pd.json_normalize(response['hours'])
                df['time'] = pd.to_datetime(df['time'])
                df = df[['time', 'waveHeight.sg', 'windSpeed.sg', 'swellHeight.sg', 'swellPeriod.sg']]
                df.columns = ['Time', 'Wave Height (m)', 'Wind Speed (m/s)', 'Swell Height (m)', 'Swell Period (s)']

            except Exception as e:
                st.warning(f"⚠️ Error processing API data: {e}")

    # Use sample data if API fails or no key
    if df is None:
        st.info("Using sample data for prediction.")
        df = pd.DataFrame([{
            'Time': pd.Timestamp.now(),
            'Wave Height (m)': 1.2,
            'Wind Speed (m/s)': 6.5,
            'Swell Height (m)': 0.8,
            'Swell Period (s)': 10
        }])

    # --- Derived Features ---
    df['wind_x'] = df['Wind Speed (m/s)'] * np.cos(np.radians(45))
    df['wind_y'] = df['Wind Speed (m/s)'] * np.sin(np.radians(45))
    df['wave_energy'] = 0.5 * 1025 * 9.81 * (df['Wave Height (m)'] ** 2)

    # --- Prediction ---
    X_latest = df[['Wave Height (m)', 'Wind Speed (m/s)', 'Swell Height (m)', 
                   'Swell Period (s)', 'wind_x', 'wind_y', 'wave_energy']].iloc[-1:]
    prediction = model.predict(X_latest)

    st.success(f"🌊 **Predicted Marine Condition:** {prediction[0]}")
    st.caption("Condition derived using wave, wind, and swell parameters.")

    # --- Display Chart & Table ---
    st.line_chart(df.set_index('Time')[['Wave Height (m)', 'Wind Speed (m/s)']])
    st.dataframe(df.tail(5))
