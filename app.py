import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ===============================
# تحميل الموديل
# ===============================
model = joblib.load("xgb_delay_model.pkl")

# ===============================
# Title
# ===============================
st.title("🚍 Public Transport Delay Prediction")
st.write("Predict delay score, category, and estimated delay time")

st.divider()

# ===============================
# User Inputs
# ===============================
route = st.selectbox(
    "Route",
    ["R1", "R2", "R3", "R4"]
)

hour = st.slider(
    "Hour of the Day",
    0, 23, 8
)

day_of_week = st.selectbox(
    "Day of Week",
    ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
)

weather = st.selectbox(
    "Weather Condition",
    ["Sunny", "Rainy", "Cloudy"]
)

passenger_count = st.number_input(
    "Passenger Count",
    min_value=1,
    max_value=500,
    value=100
)

latitude = st.number_input("Latitude", value=24.5)
longitude = st.number_input("Longitude", value=32.5)

st.divider()

# ===============================
# Button
# ===============================
if st.button("Predict Delay 🚦"):

    # ===============================
    # Feature Engineering
    # ===============================

    # day_of_week numeric
    day_map = {
        "Monday": 0, "Tuesday": 1, "Wednesday": 2,
        "Thursday": 3, "Friday": 4,
        "Saturday": 5, "Sunday": 6
    }
    day_num = day_map[day_of_week]

    # weekend
    is_weekend = 1 if day_num >= 5 else 0

    # hour cyclic encoding
    hour_sin = np.sin(2 * np.pi * hour / 24)
    hour_cos = np.cos(2 * np.pi * hour / 24)

    # passenger scaling (نفس منطق التدريب)
    passenger_count_scaled = passenger_count / 500

    # ===============================
    # One-Hot Encoding (Route)
    # ===============================
    route_R1 = 1 if route == "R1" else 0
    route_R2 = 1 if route == "R2" else 0
    route_R3 = 1 if route == "R3" else 0
    route_R4 = 1 if route == "R4" else 0

    # ===============================
    # One-Hot Encoding (Weather)
    # ===============================
    weather_sunny = 1 if weather == "Sunny" else 0
    weather_rainy = 1 if weather == "Rainy" else 0
    weather_cloudy = 1 if weather == "Cloudy" else 0

    # weather severity (منطقي)
    weather_severity = {
        "Sunny": 0,
        "Cloudy": 1,
        "Rainy": 2
    }[weather]

    # ===============================
    # IMPORTANT
    # ===============================
    # scores دي كانت معمولة من الداتا
    # في الإنتاج نستخدم متوسطات ثابتة (baseline)
    score_route = 1.0
    score_hour = 1.0
    score_day = 1.0
    score_weather = 1.0

    route_frequency_scaled = 0.5  # قيمة افتراضية منطقية

    # ===============================
    # Create Input DataFrame
    # ===============================
    input_df = pd.DataFrame([{
        "latitude": latitude,
        "longitude": longitude,
        "hour_sin": hour_sin,
        "hour_cos": hour_cos,
        "is_weekend": is_weekend,
        "passenger_count_scaled": passenger_count_scaled,
        "score_route": score_route,
        "score_hour": score_hour,
        "score_day": score_day,
        "score_weather": score_weather,
        "route_frequency_scaled": route_frequency_scaled,
        "weather_severity": weather_severity,

        "route_id_R1": route_R1,
        "route_id_R2": route_R2,
        "route_id_R3": route_R3,
        "route_id_R4": route_R4,

        "weather_sunny": weather_sunny,
        "weather_rainy": weather_rainy,
        "weather_cloudy": weather_cloudy
    }])

    # ===============================
    # Prediction
    # ===============================
    delay_score = model.predict(input_df)[0]

    # ===============================
    # Convert score to minutes
    # ===============================
    estimated_minutes = delay_score * 60

    # ===============================
    # Category
    # ===============================
    if estimated_minutes < 20:
        category = "🟢 Low Delay"
    elif estimated_minutes < 40:
        category = "🟡 Medium Delay"
    else:
        category = "🔴 High Delay"

    # ===============================
    # Output
    # ===============================
    st.success("Prediction Completed ✅")

    st.metric("Delay Score", round(delay_score, 2))
    st.metric("Estimated Delay (minutes)", int(estimated_minutes))
    st.metric("Delay Category", category)
