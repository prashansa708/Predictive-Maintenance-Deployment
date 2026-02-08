
import streamlit as st
import pandas as pd
import joblib
from huggingface_hub import hf_hub_download

# 1. Load the model from Hugging Face Hub
# Replace with your actual repo and filename if different
REPO_ID = "P-Mishra/engine-predictive-maintenance"
FILENAME = "rf_predictive_maintenance.pkl"

try:
    model_path = hf_hub_download(repo_id=REPO_ID, filename=FILENAME)
    model = joblib.load(model_path)
except Exception as e:
    st.error(f"Error loading model: {e}")

st.title("🚢 Engine Predictive Maintenance")
st.write("Professional Monitoring System for Engine Health")

# 2. Input Fields
col1, col2 = st.columns(2)
with col1:
    engine_rpm = st.number_input("Engine RPM", value=1000)
    lub_oil_temp = st.number_input("Lubricating Oil Temp", value=80.0)
    fuel_pressure = st.number_input("Fuel Pressure", value=5.0)
with col2:
    coolant_temp = st.number_input("Coolant Temp", value=75.0)
    coolant_pressure = st.number_input("Coolant Pressure", value=3.0)

# 3. Feature Engineering (must match your notebook logic exactly)
eps = 1e-6
coolant_interaction = coolant_temp * coolant_pressure
coolant_ratio = coolant_temp / (coolant_pressure + eps)
oil_rpm_interaction = lub_oil_temp * engine_rpm
fuel_rpm_ratio = fuel_pressure / (engine_rpm + eps)

# 4. Prediction
input_df = pd.DataFrame([{
    'engine_rpm': engine_rpm,
    'lub_oil_temp': lub_oil_temp,
    'fuel_pressure': fuel_pressure,
    'coolant_temp': coolant_temp,
    'coolant_pressure': coolant_pressure,
    'coolant_temp_pressure_interaction': coolant_interaction,
    'coolant_temp_pressure_ratio': coolant_ratio,
    'lub_oil_temp_engine_rpm_interaction': oil_rpm_interaction,
    'fuel_pressure_engine_rpm_ratio': fuel_rpm_ratio
}])

if st.button("Analyze Engine Condition"):
    prediction = model.predict(input_df)
    if prediction[0] == 1:
        st.error("🚨 CRITICAL: Maintenance Required Immediately")
    else:
        st.success("✅ NORMAL: Engine operating within safe parameters")
    