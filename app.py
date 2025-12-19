import streamlit as st
import pickle
import pandas as pd
from math import sin, cos, pi

# ===============================
# Load model & reference stats
# ===============================
with open("xgb_delay_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("reference_stats.pkl", "rb") as f:
    stats = pickle.load(f)

global_mean = stats["global_mean"]

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
st.set_page_config(page_title="🚍 Delay Prediction", page_icon="🚌", layout="centered")
st.title("🚍 Public Transport Delay Prediction")
st.markdown("Predict delays for public transport trips based on route, time, and weather.")

with st.form("trip_form"):
    st.subheader("Enter Trip Details")

    col1, col2 = st.columns(2)
    with col1:
        route = st.selectbox("Select Route", list(stats["route_mean"].keys()))
        day = st.selectbox(
            "Day of Week",
            {0: "Monday", 1: "Tuesday", 2: "Wednesday", 3: "Thursday",
             4: "Friday", 5: "Saturday", 6: "Sunday"}.keys(),
            format_func=lambda x: ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"][x]
        )
        hour = st.slider("Hour of Day", 0, 23, 8)
    
    with col2:
        weather = st.selectbox("Weather Condition", list(stats["weather_mean"].keys()))
        passenger_count = st.number_input("Passenger Count", min_value=1, value=50)

    submitted = st.form_submit_button("Predict Delay")

# ===============================
# Feature Engineering
# ===============================
hour_sin, hour_cos = hour_sin_cos(hour)
is_weekend = 1 if day >= 5 else 0
passenger_scaled = passenger_count / 300  # same scaling as training

score_route = global_mean / stats["route_mean"][route]
score_hour = global_mean / stats["hour_mean"][hour]
score_day = global_mean / stats["day_mean"][day]
score_weather = global_mean / stats["weather_mean"][weather]

# ===============================
# Build input DataFrame
# ===============================
input_data = pd.DataFrame([{
    "latitude": 0,  # dummy value, model needs it
    "longitude": 0, # dummy value, model needs it
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
    "route_id_R1": 1 if route=="R1" else 0,
    "route_id_R2": 1 if route=="R2" else 0,
    "route_id_R3": 1 if route=="R3" else 0,
    "route_id_R4": 1 if route=="R4" else 0,
    "weather_rainy": 1 if weather=="Rainy" else 0,
    "weather_sunny": 1 if weather=="Sunny" else 0,
    "weather_cloudy": 1 if weather=="Cloudy" else 0,
}])

# Reorder columns exactly like training data
columns_order = ['latitude', 'longitude', 'hour_sin', 'hour_cos', 'is_weekend',
                 'passenger_count_scaled', 'score_route', 'score_hour', 'score_day',
                 'score_weather', 'route_frequency_scaled', 'weather_severity',
                 'route_id_R2', 'route_id_R3', 'route_id_R4',
                 'weather_rainy', 'weather_sunny', 'weather_cloudy', 'route_id_R1']
input_data = input_data[columns_order]

# ===============================
# Prediction
# ===============================
if submitted:
    delay_score = model.predict(input_data)[0]
    category = delay_category(delay_score)
    estimated_minutes = delay_score * global_mean

    st.success("📊 Prediction Results")
    st.markdown(f"**Delay Score:** {delay_score:.2f}")
    st.markdown(f"**Category:** {category}")
    st.markdown(f"**Estimated Delay:** {estimated_minutes:.1f} minutes")
