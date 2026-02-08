import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Load the model
model = joblib.load("rf_predictive_maintenance.pkl")

st.title("🚢 Engine Predictive Maintenance")
st.write("Enter sensor readings to check engine health status.")

# 1. User Inputs (The 6 base sensors)
engine_rpm = st.slider("Engine RPM", 400.0, 2500.0, 1200.0)
lub_oil_temp = st.slider("Lubricating Oil Temp", 60.0, 120.0, 85.0)
lub_oil_pressure = st.slider("Lubricating Oil Pressure", 1.0, 10.0, 4.5) 
fuel_pressure = st.slider("Fuel Pressure", 2.0, 10.0, 5.0)
coolant_temp = st.slider("Coolant Temp", 60.0, 110.0, 80.0)
coolant_pressure = st.slider("Coolant Pressure", 1.0, 5.0, 2.5)

# 2. Recreate Engineered Features (Must match notebook logic exactly)
eps = 1e-6
coolant_temp_pressure_interaction = coolant_temp * coolant_pressure
coolant_temp_pressure_ratio = coolant_temp / (coolant_pressure + eps)
lub_oil_temp_engine_rpm_interaction = lub_oil_temp * engine_rpm
fuel_pressure_engine_rpm_ratio = fuel_pressure / (engine_rpm + eps)

# 3. Create DataFrame with ALL 10 features in the EXACT order of training
# Order: engine_rpm, lub_oil_temp, lub_oil_pressure, fuel_pressure, coolant_temp, coolant_pressure, 
#        coolant_temp_pressure_interaction, coolant_temp_pressure_ratio, 
#        lub_oil_temp_engine_rpm_interaction, fuel_pressure_engine_rpm_ratio
input_df = pd.DataFrame([[
    engine_rpm, 
    lub_oil_temp, 
    lub_oil_pressure, 
    fuel_pressure, 
    coolant_temp, 
    coolant_pressure,
    coolant_temp_pressure_interaction,
    coolant_temp_pressure_ratio,
    lub_oil_temp_engine_rpm_interaction,
    fuel_pressure_engine_rpm_ratio
]], columns=[
    'engine_rpm', 'lub_oil_temp', 'lub_oil_pressure', 'fuel_pressure',
    'coolant_temp', 'coolant_pressure', 'coolant_temp_pressure_interaction',
    'coolant_temp_pressure_ratio', 'lub_oil_temp_engine_rpm_interaction',
    'fuel_pressure_engine_rpm_ratio'
])

# 4. Prediction Logic
if st.button("Analyze Engine Condition"):
    try:
        prediction = model.predict(input_df)
        if prediction[0] == 1:
            st.error("⚠️ Maintenance Required: High Failure Risk Detected")
        else:
            st.success("✅ Engine Condition: Normal")
    except Exception as e:
        st.error(f"Prediction Error: {e}")
