import streamlit as st
import pandas as pd
import joblib

# Load the model
model = joblib.load("rf_predictive_maintenance.pkl")

st.title("🚢 Engine Predictive Maintenance")

# Create inputs for ALL 6 features used during training
engine_rpm = st.slider("Engine RPM", 400.0, 2500.0, 1200.0)
lub_oil_temp = st.slider("Lubricating Oil Temp", 60.0, 120.0, 85.0)
# ADDING THE MISSING FEATURE:
lub_oil_pressure = st.slider("Lubricating Oil Pressure", 1.0, 10.0, 4.5) 
fuel_pressure = st.slider("Fuel Pressure", 2.0, 10.0, 5.0)
coolant_temp = st.slider("Coolant Temp", 60.0, 110.0, 80.0)
coolant_pressure = st.slider("Coolant Pressure", 1.0, 5.0, 2.5)

# Create the DataFrame with names matching training exactly
input_df = pd.DataFrame([[
    engine_rpm, 
    lub_oil_temp, 
    lub_oil_pressure, # Must be included
    fuel_pressure, 
    coolant_temp, 
    coolant_pressure
]], columns=[
    'engine_rpm', 
    'lub_oil_temp', 
    'lub_oil_pressure', 
    'fuel_pressure', 
    'coolant_temp', 
    'coolant_pressure'
])

if st.button("Analyze Engine Condition"):
    prediction = model.predict(input_df)
    if prediction[0] == 1:
        st.error("⚠️ Maintenance Required: High Failure Risk Detected")
    else:
        st.success("✅ Engine Condition: Normal")
