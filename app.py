import streamlit as st
import pickle
import numpy as np
import pandas as pd
from math import sin, cos, pi

# ===============================
# Load model & reference stats
# ===============================
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

with open("reference_stats.pkl", "rb") as f:
    stats = pickle.load(f)

# ===============================
# Helper functions
# ===============================
def hour_sin_cos(hour):
    return sin(2 * pi * hour / 24), cos(2 * pi * hour / 24)

def delay_category(score):
    if score < 0.9:
        return "Low Delay"
    elif score < 1.2:
        return "Medium Delay"
    else:
        return "High Delay"

# ===============================
# Streamlit UI
# ===============================
st.title("🚍 Public Transport Delay Prediction")

st.header("Enter Trip Information")

route = st.selectbox("Route", list(stats["route_mean"].keys()))
hour = st.slider("Hour of Day", 0, 23, 8)
day = st.selectbox("Day of Week", {
    0: "Monday", 1: "Tuesday", 2: "Wednesday",
    3: "Thursday", 4: "Friday", 5: "Saturday", 6: "Sunday"
}.keys(), format_func=lambda x: ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"][x])

weather = st.selectbox("Weather", list(stats["weather_mean"].keys()))
passenger_count = st.number_input("Passenger Count", min_value=1, value=50)

latitude = st.number_input("Latitude", value=24.5)
longitude = st.number_input("Longitude", value=32.5)

# ===============================
# Feature Engineering
# ===============================
hour_sin, hour_cos = hour_sin_cos(hour)
is_weekend = 1 if day >= 5 else 0
passenger_scaled = passenger_count / 300  # نفس منطق التدريب

global_mean = stats["global_mean"]

score_route = global_mean / stats["route_mean"][route]
score_hour = global_mean / stats["hour_mean"][hour]
score_day = global_mean / stats["day_mean"][day]
score_weather = global_mean / stats["weather_mean"][weather]

# ===============================
# Build input DataFrame
# ===============================
input_data = pd.DataFrame([{
    "latitude": latitude,
    "longitude": longitude,
    "hour_sin": hour_sin,
    "hour_cos": hour_cos,
    "is_weekend": is_weekend,
    "passenger_count_scaled": passenger_scaled,
    "score_route": score_route,
    "score_hour": score_hour,
    "score_day": score_day,
    "score_weather": score_weather,
    "route_frequency_scaled": 1.0,
    "weather_severity": 1.0,
}])

# ===============================
# Prediction
# ===============================
if st.button("Predict Delay"):
    delay_score = model.predict(input_data)[0]
    category = delay_category(delay_score)
    estimated_minutes = delay_score * global_mean

    st.subheader("📊 Prediction Results")
    st.write(f"**Delay Score:** {delay_score:.2f}")
    st.write(f"**Category:** {category}")
    st.write(f"**Estimated Delay:** {estimated_minutes:.1f} minutes")
