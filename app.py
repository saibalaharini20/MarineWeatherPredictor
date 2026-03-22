import streamlit as st
import pandas as pd
import numpy as np
import requests
import joblib
import math

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

if st.button("🌤 Predict Marine Condition"):
    if not api_key:
        st.warning("Please enter a valid StormGlass API key!")
    else:
        url = f"https://api.stormglass.io/v2/weather/point?lat={lat}&lng={lng}&params=waveHeight,windSpeed,swellHeight,swellPeriod"
        headers = {'Authorization': api_key}
        response = requests.get(url, headers=headers).json()

        try:
            df = pd.json_normalize(response['hours'])
            df['time'] = pd.to_datetime(df['time'])
            df = df[['time', 'waveHeight.sg', 'windSpeed.sg', 'swellHeight.sg', 'swellPeriod.sg']]
            df.columns = ['Time', 'Wave Height (m)', 'Wind Speed (m/s)', 'Swell Height (m)', 'Swell Period (s)']

            # Derived features
            df['wind_x'] = df['Wind Speed (m/s)'] * np.cos(np.radians(45))
            df['wind_y'] = df['Wind Speed (m/s)'] * np.sin(np.radians(45))
            df['wave_energy'] = 0.5 * 1025 * 9.81 * (df['Wave Height (m)'] ** 2)

            # Load model
            model = joblib.load("marine_model.pkl")

            # Take latest hour data
            X_latest = df[['Wave Height (m)', 'Wind Speed (m/s)', 'Swell Height (m)', 
                           'Swell Period (s)', 'wind_x', 'wind_y', 'wave_energy']].iloc[-1:]
            prediction = model.predict(X_latest)

            st.success(f"🌊 **Predicted Marine Condition:** {prediction[0]}")
            st.caption("Condition derived using real-time wave, wind, and swell parameters.")

            # Display chart
            st.line_chart(df.set_index('Time')[['Wave Height (m)', 'Wind Speed (m/s)']])
            st.dataframe(df.tail(5))

        except Exception as e:
            st.error(f"⚠️ Could not fetch data: {e}")
