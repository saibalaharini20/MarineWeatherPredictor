import streamlit as st
import pandas as pd
import numpy as np
import requests
import joblib
import os
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# --- Page Setup ---
st.set_page_config(page_title="Marine Weather Predictor", layout="wide")
st.title("Marine Weather Predictor Dashboard")
st.markdown("Select a **coastal or ocean** location to predict marine weather conditions.")

# --- Try importing folium ---
try:
    import folium
    from streamlit_folium import st_folium
    FOLIUM_AVAILABLE = True
except ImportError:
    FOLIUM_AVAILABLE = False

# --- Session state for persistent coordinates ---
if "lat" not in st.session_state:
    st.session_state.lat = 17.6868  # Default: Visakhapatnam coast
if "lng" not in st.session_state:
    st.session_state.lng = 83.2185

# --- Location Input ---
st.subheader("Choose Location Input Method")

if FOLIUM_AVAILABLE:
    input_method = st.radio(
        "How do you want to provide the location?",
        ("Select from Map", "Manual Input")
    )
else:
    input_method = "Manual Input"
    st.info("Map selection unavailable. Install folium and streamlit-folium to enable it.")

if input_method == "Manual Input":
    col1, col2 = st.columns(2)
    with col1:
        st.session_state.lat = st.number_input(
            "Latitude", value=st.session_state.lat,
            min_value=-90.0, max_value=90.0, format="%.4f"
        )
    with col2:
        st.session_state.lng = st.number_input(
            "Longitude", value=st.session_state.lng,
            min_value=-180.0, max_value=180.0, format="%.4f"
        )

else:
    st.markdown(
        "Click anywhere on the **ocean or coast** on the map. "
        "The pin will update and those coordinates will be used for prediction."
    )

    m = folium.Map(location=[st.session_state.lat, st.session_state.lng], zoom_start=5)

    # Show current pin on map
    folium.Marker(
        location=[st.session_state.lat, st.session_state.lng],
        tooltip=f"Selected: {st.session_state.lat:.4f}, {st.session_state.lng:.4f}",
        icon=folium.Icon(color="blue", icon="map-marker")
    ).add_to(m)

    map_data = st_folium(m, height=450, width=800, returned_objects=["last_clicked"])

    # Update session state when user clicks
    if map_data and map_data.get("last_clicked"):
        new_lat = round(map_data["last_clicked"]["lat"], 4)
        new_lng = round(map_data["last_clicked"]["lng"], 4)
        if -90 <= new_lat <= 90 and -180 <= new_lng <= 180:
            st.session_state.lat = new_lat
            st.session_state.lng = new_lng

# Always show active coordinates so user knows what will be used
st.info(
    f"Active coordinates for prediction: "
    f"**Lat {st.session_state.lat:.4f}, Lng {st.session_state.lng:.4f}**"
)

lat = st.session_state.lat
lng = st.session_state.lng

# --- API Key ---
api_key = st.text_input("Enter your StormGlass API Key", type="password")

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
        st.error(f"Failed to load model: {e}")
else:
    st.warning("marine_model.pkl not found. Predictions will be unavailable.")

# --- Helper: Extract first non-null value from StormGlass nested dict ---
def extract_value(df, col):
    if col in df.columns:
        return df[col].apply(
            lambda x: next((v for v in x.values() if v is not None), None)
            if isinstance(x, dict) else (x if pd.notna(x) else None)
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
if st.button("Predict Marine Condition"):

    if model is None:
        st.error("Cannot predict: model file is missing or failed to load.")
        st.stop()

    if not api_key:
        st.error("Please enter your StormGlass API key to fetch real marine data.")
        st.stop()

    df = None

    # --- Fetch from StormGlass API ---
    with st.spinner(f"Fetching marine data for ({lat:.4f}, {lng:.4f})..."):
        try:
            url = (
                "https://api.stormglass.io/v2/weather/point"
                f"?lat={lat:.4f}&lng={lng:.4f}"
                "&params=waveHeight,windSpeed,swellHeight,swellPeriod"
            )
            response_raw = requests.get(
                url,
                headers={"Authorization": api_key},
                timeout=10
            )

            if response_raw.status_code == 401:
                st.error("Invalid API key. Please check your StormGlass credentials.")
                st.stop()

            elif response_raw.status_code == 422:
                st.error(
                    f"StormGlass rejected the coordinates ({lat:.4f}, {lng:.4f}). "
                    "This location may be inland or not covered by StormGlass. "
                    "Please click directly on the ocean or coastline and try again."
                )
                st.stop()

            elif response_raw.status_code == 429:
                st.error("StormGlass daily request limit reached. Try again tomorrow or upgrade your plan.")
                st.stop()

            elif response_raw.status_code != 200:
                st.error(f"API error {response_raw.status_code}. Please try again.")
                st.stop()

            else:
                response = response_raw.json()

                if "hours" not in response or not response["hours"]:
                    st.error("API returned no marine data for this location. Try a different coastal point.")
                    st.stop()

                raw_df = pd.json_normalize(response["hours"])
                raw_df["Time"] = pd.to_datetime(raw_df["time"])

                raw_df["Wave Height (m)"]  = extract_value(raw_df, "waveHeight")
                raw_df["Wind Speed (m/s)"] = extract_value(raw_df, "windSpeed")
                raw_df["Swell Height (m)"] = extract_value(raw_df, "swellHeight")
                raw_df["Swell Period (s)"] = extract_value(raw_df, "swellPeriod")

                raw_df = raw_df[[
                    "Time", "Wave Height (m)", "Wind Speed (m/s)",
                    "Swell Height (m)", "Swell Period (s)"
                ]]

                df = raw_df.dropna().sort_values("Time").reset_index(drop=True)

                if df.empty:
                    st.error(
                        "API returned data but all parameter values were null for this location. "
                        "Try clicking closer to the open ocean."
                    )
                    st.stop()

        except requests.exceptions.Timeout:
            st.error("API request timed out. Check your internet connection and try again.")
            st.stop()
        except requests.exceptions.ConnectionError:
            st.error("Could not connect to StormGlass API. Check your internet connection.")
            st.stop()
        except Exception as e:
            st.error(f"Unexpected error: {e}")
            st.stop()

    # --- Feature Engineering ---
    df = engineer_features(df)

    # --- Validate features ---
    missing_cols = [c for c in FEATURE_COLS if c not in df.columns]
    if missing_cols:
        st.error(f"Missing features for prediction: {missing_cols}")
        st.stop()

    # --- Predict on latest row ---
    X_latest = df[FEATURE_COLS].iloc[[-1]]

    try:
        prediction = model.predict(X_latest)
        condition = prediction[0]
    except Exception as e:
        st.error(f"Prediction failed: {e}")
        st.stop()

    # --- Results ---
    st.divider()
    st.subheader("Prediction Result")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Wave Height",  f"{X_latest['Wave Height (m)'].values[0]:.2f} m")
    col2.metric("Wind Speed",   f"{X_latest['Wind Speed (m/s)'].values[0]:.2f} m/s")
    col3.metric("Swell Height", f"{X_latest['Swell Height (m)'].values[0]:.2f} m")
    col4.metric("Swell Period", f"{X_latest['Swell Period (s)'].values[0]:.1f} s")

    # --- Color-coded condition badge ---
    condition_styles = {
        "Calm":     {"color": "#155724", "bg": "#d4edda", "border": "#c3e6cb", "icon": "🟢"},
        "Moderate": {"color": "#856404", "bg": "#fff3cd", "border": "#ffeeba", "icon": "🟡"},
        "Rough":    {"color": "#721c24", "bg": "#f8d7da", "border": "#f5c6cb", "icon": "🔴"},
    }
    style = condition_styles.get(condition, {"color": "#333", "bg": "#eee", "border": "#ccc", "icon": "⚪"})

    st.markdown(f"""
        <div style="
            background-color: {style['bg']};
            border: 2px solid {style['border']};
            border-radius: 12px;
            padding: 20px 28px;
            margin: 16px 0;
            display: flex;
            align-items: center;
            gap: 14px;
        ">
            <span style="font-size: 2.2rem;">{style['icon']}</span>
            <div>
                <div style="font-size: 0.85rem; color: {style['color']}; font-weight: 600;
                            text-transform: uppercase; letter-spacing: 0.08em;">
                    Predicted Marine Condition
                </div>
                <div style="font-size: 1.8rem; font-weight: 700; color: {style['color']};">
                    {condition}
                </div>
            </div>
        </div>
    """, unsafe_allow_html=True)

    st.caption(
        f"Real data from StormGlass at ({lat:.4f}, {lng:.4f}) — "
        f"latest point: {df['Time'].iloc[-1].strftime('%Y-%m-%d %H:%M UTC')}"
    )

    # --- Wind Compass + Swell Chart side by side ---
    st.divider()
    wind_speed = X_latest["Wind Speed (m/s)"].values[0]
    wind_x     = X_latest["wind_x"].values[0]
    wind_y     = X_latest["wind_y"].values[0]
    wind_angle = np.degrees(np.arctan2(wind_y, wind_x))
    arrow_rad  = np.radians(wind_angle)
    ax_tip     = 50 + 30 * np.cos(arrow_rad)
    ay_tip     = 50 - 30 * np.sin(arrow_rad)

    compass_col, swell_col = st.columns([1, 2])

    with compass_col:
        st.markdown("**Wind Direction Compass**")
        st.markdown(f"""
        <svg viewBox="0 0 100 100" width="200" height="200" xmlns="http://www.w3.org/2000/svg">
          <circle cx="50" cy="50" r="46" fill="#e8f4f8" stroke="#90cdf4" stroke-width="2"/>
          <text x="50" y="10"  text-anchor="middle" font-size="7" fill="#2b6cb0" font-weight="bold">N</text>
          <text x="50" y="96"  text-anchor="middle" font-size="7" fill="#2b6cb0" font-weight="bold">S</text>
          <text x="96" y="53"  text-anchor="middle" font-size="7" fill="#2b6cb0" font-weight="bold">E</text>
          <text x="4"  y="53"  text-anchor="middle" font-size="7" fill="#2b6cb0" font-weight="bold">W</text>
          <line x1="50" y1="14" x2="50" y2="18" stroke="#90cdf4" stroke-width="1"/>
          <line x1="50" y1="82" x2="50" y2="86" stroke="#90cdf4" stroke-width="1"/>
          <line x1="14" y1="50" x2="18" y2="50" stroke="#90cdf4" stroke-width="1"/>
          <line x1="82" y1="50" x2="86" y2="50" stroke="#90cdf4" stroke-width="1"/>
          <circle cx="50" cy="50" r="3" fill="#2b6cb0"/>
          <line x1="50" y1="50" x2="{ax_tip:.1f}" y2="{ay_tip:.1f}"
                stroke="#e53e3e" stroke-width="3" stroke-linecap="round"/>
          <circle cx="{ax_tip:.1f}" cy="{ay_tip:.1f}" r="4" fill="#e53e3e"/>
          <text x="50" y="62" text-anchor="middle" font-size="6" fill="#555">{wind_speed:.1f} m/s</text>
        </svg>
        """, unsafe_allow_html=True)

    with swell_col:
        if len(df) > 1:
            st.markdown("**Swell Height & Period Over Time**")
            fig, ax1 = plt.subplots(figsize=(6, 3))
            fig.patch.set_facecolor("#f0f8ff")
            ax1.set_facecolor("#f0f8ff")

            ax1.plot(df["Time"], df["Swell Height (m)"], color="#3182ce",
                     linewidth=2, marker="o", markersize=3, label="Swell Height (m)")
            ax1.set_ylabel("Swell Height (m)", color="#3182ce", fontsize=9)
            ax1.tick_params(axis="y", labelcolor="#3182ce")
            ax1.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
            plt.xticks(rotation=45, fontsize=7)

            ax2 = ax1.twinx()
            ax2.plot(df["Time"], df["Swell Period (s)"], color="#e53e3e",
                     linewidth=2, linestyle="--", marker="s", markersize=3, label="Swell Period (s)")
            ax2.set_ylabel("Swell Period (s)", color="#e53e3e", fontsize=9)
            ax2.tick_params(axis="y", labelcolor="#e53e3e")

            lines1, labels1 = ax1.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=7, loc="upper left")
            fig.tight_layout()
            st.pyplot(fig)
        else:
            st.info("Swell chart not available — only one data point returned by API.")

    # --- Wave Height & Wind Speed Over Time ---
    if len(df) > 1:
        st.divider()
        st.markdown("**Wave Height & Wind Speed Over Time**")
        chart_df = df.set_index("Time")[["Wave Height (m)", "Wind Speed (m/s)"]].sort_index()
        st.line_chart(chart_df)
