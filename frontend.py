import pandas as pd
import numpy as np
import streamlit as st
import joblib
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Load the trained model and scaler
model = joblib.load("xgboost_model.pkl")
scaler = joblib.load("scaler.pkl")

# Define function to make predictions
def predict_rentals(features):
    features_scaled = scaler.transform([features])
    prediction = model.predict(features_scaled)
    return round(prediction[0])

# Streamlit UI
st.title("Seoul Bike Rental Prediction")
st.write("Enter the details to predict the number of bike rentals.")

# User input fields
seasons = st.selectbox("Seasons", ["Spring", "Summer", "Autumn", "Winter"])
holiday = st.selectbox("Holiday", ["No Holiday", "Holiday"])
functioning_day = st.selectbox("Functioning Day", ["Yes", "No"])
temp = st.number_input("Temperature (°C)")
humidity = st.number_input("Humidity (%)")
wind_speed = st.number_input("Wind Speed (m/s)")
visibility = st.number_input("Visibility (10m)")
dew_point = st.number_input("Dew Point Temperature (°C)")
radiation = st.number_input("Solar Radiation (MJ/m2)")
rainfall = st.number_input("Rainfall (mm)")
snowfall = st.number_input("Snowfall (cm)")
month = st.slider("Month", 1, 12, 1)
day_of_week = st.slider("Day of the Week", 0, 6, 0)
hour = st.slider("Hour of the Day", 0, 23, 12)  # Example if 'hour' was missing

# Encode categorical inputs
season_map = {"Spring": 0, "Summer": 1, "Autumn": 2, "Winter": 3}
holiday_map = {"No Holiday": 0, "Holiday": 1}
functioning_day_map = {"Yes": 1, "No": 0}

# Prepare feature vector
features = [
    season_map[seasons], holiday_map[holiday], functioning_day_map[functioning_day],
    temp, humidity, wind_speed, visibility, dew_point, radiation, rainfall, snowfall,
    month, day_of_week, hour  # Add missing feature here
]

# Prediction button
if st.button("Predict Bike Rentals"):
    prediction = predict_rentals(features)
    st.success(f"Predicted number of rentals: {prediction}")

if __name__ == "__main__":
    st.title("Seoul Bike Rental Prediction")

