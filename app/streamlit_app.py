import streamlit as st
import pandas as pd
import joblib

# Load model
model = joblib.load("model/rf_model.pkl")

st.title("🛵 Food Delivery Time Estimator")
st.write("Predict delivery time based on order, traffic, and courier conditions.")

# User inputs
distance = st.slider("Distance (in km)", 0.5, 20.0, 5.0)
prep_time = st.slider("Preparation Time (min)", 5, 60, 15)
experience = st.slider("Courier Experience (years)", 0, 10, 2)

weather = st.selectbox("Weather", ["Clear", "Rainy", "Stormy", "Foggy"])
traffic = st.selectbox("Traffic Level", ["Low", "Medium", "High", "Jam"])
time_of_day = st.selectbox("Time of Day", ["Morning", "Afternoon", "Evening", "Night"])
vehicle = st.selectbox("Vehicle Type", ["Bike", "Car", "Scooter", "Cycle"])

# Create DataFrame for model input
input_df = pd.DataFrame({
    'Distance_km': [distance],
    'Preparation_Time_min': [prep_time],
    'Courier_Experience_yrs': [experience],
    'Weather': [weather],
    'Traffic_Level': [traffic],
    'Time_of_Day': [time_of_day],
    'Vehicle_Type': [vehicle]
})

# Predict
if st.button("Estimate Delivery Time"):
    prediction = model.predict(input_df)[0]
    st.success(f"Estimated Delivery Time: {round(prediction, 2)} minutes")
