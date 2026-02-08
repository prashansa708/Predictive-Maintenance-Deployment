import streamlit as st
import pandas as pd
import joblib
import numpy as np
from huggingface_hub import hf_hub_download

# --- MODEL LOADING ---
REPO_ID = "P-Mishra/engine-predictive-maintenance"
MODEL_FILENAME = "rf_predictive_maintenance.pkl"

@st.cache_resource
def load_model():
    try:
        model_path = hf_hub_download(repo_id=REPO_ID, filename=MODEL_FILENAME)
        return joblib.load(model_path)
    except Exception as e:
        st.error(f"Error loading model from Hub: {e}")
        return None

model = load_model()

# --- UI SETUP ---
st.set_page_config(page_title="Engine Health Monitor", page_icon="🚢")
st.title("🚢 Engine Predictive Maintenance")
st.write("Professional Monitoring System for Engine Health")

# --- USER INPUTS ---
col1, col2 = st.columns(2)

with col1:
    engine_rpm = st.number_input("Engine RPM", value=1200.0)
    lub_oil_pressure = st.number_input("Lubricating Oil Pressure (bar)", value=4.5)
    fuel_pressure = st.number_input("Fuel Pressure (bar)", value=5.0)
    coolant_pressure = st.number_input("Coolant Pressure (bar)", value=2.5)

with col2:
    lub_oil_temp = st.number_input("Lubricating Oil Temp (°C)", value=85.0)
    coolant_temp = st.number_input("Coolant Temp (°C)", value=80.0)

# --- FEATURE ENGINEERING ---
eps = 1e-6
coolant_temp_pressure_interaction = coolant_temp * coolant_pressure
coolant_temp_pressure_ratio = coolant_temp / (coolant_pressure + eps)
lub_oil_temp_engine_rpm_interaction = lub_oil_temp * engine_rpm
fuel_pressure_engine_rpm_ratio = fuel_pressure / (engine_rpm + eps)

# --- DATAFRAME CONSTRUCTION (VERIFIED ORDER) ---
# This list matches your model's 'feature_names_in_' exactly.
feature_columns = [
    'engine_rpm', 
    'lub_oil_pressure', 
    'fuel_pressure', 
    'coolant_pressure', 
    'lub_oil_temp', 
    'coolant_temp', 
    'coolant_temp_pressure_interaction', 
    'coolant_temp_pressure_ratio', 
    'lub_oil_temp_engine_rpm_interaction', 
    'fuel_pressure_engine_rpm_ratio'
]

input_data = pd.DataFrame([[
    engine_rpm, 
    lub_oil_pressure, 
    fuel_pressure, 
    coolant_pressure, 
    lub_oil_temp, 
    coolant_temp, 
    coolant_temp_pressure_interaction, 
    coolant_temp_pressure_ratio, 
    lub_oil_temp_engine_rpm_interaction, 
    fuel_pressure_engine_rpm_ratio
]], columns=feature_columns)

# --- PREDICTION ---
if st.button("Analyze Engine Condition"):
    if model is not None:
        try:
            prediction = model.predict(input_data)
            st.divider()
            if prediction[0] == 1:
                st.error("### 🚨 Result: Maintenance Required")
                st.write("High failure risk detected based on sensor interaction patterns.")
            else:
                st.success("### ✅ Result: Normal Operation")
                st.write("Engine is operating within safe nominal parameters.")
        except Exception as e:
            st.error(f"Prediction Error: {e}")
    else:
        st.error("Model not loaded correctly.")
