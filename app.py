import streamlit as st
import pandas as pd
import joblib
import numpy as np
import os

# --- PAGE CONFIG ---
st.set_page_config(page_title="Car Price Predictor", page_icon="🚗")

# --- LOAD MODELS ---
# We use st.cache_resource so the model stays in memory
@st.cache_resource
def load_model_and_scaler():
    # Make sure these filenames match exactly what is in your 'models' folder
    model_path = os.path.join('models', 'best_model.pkl')
    scaler_path = os.path.join('models', 'scaler.pkl')
    
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    return model, scaler

try:
    model, scaler = load_model_and_scaler()
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.info("Ensure best_model.pkl and scaler.pkl are in the 'models' folder.")
    st.stop()

# --- INTERFACE ---
st.title("🚗 Car Selling Price Predictor")
st.markdown("Enter details below to get an instant price estimate based on our **Decision Tree** MLOps pipeline.")

# Create two columns for inputs
col1, col2 = st.columns(2)

with col1:
    year = st.number_input("Year of Purchase", 2000, 2024, 2017)
    present_price = st.number_input("Present Price (in Lakhs)", 0.1, 100.0, 10.0)
    kms_driven = st.number_input("Kms Driven", 0, 500000, 30000)
    owner = st.selectbox("Previous Owners", [0, 1, 3])

with col2:
    fuel_type = st.selectbox("Fuel Type", ["Petrol", "Diesel", "CNG"])
    seller_type = st.selectbox("Seller Type", ["Dealer", "Individual"])
    transmission = st.selectbox("Transmission", ["Manual", "Automatic"])
    # Car_Name was LabelEncoded in your notebook. 
    # Since there are many names, we'll use a numeric input or default it to 0.
    car_id = st.number_input("Car Name ID (Numeric)", 0, 100, 1)

# --- PREDICTION LOGIC ---
if st.button("Predict Selling Price"):
    # 1. Mapping categorical inputs to the exact numbers used in your LabelEncoder
    # Fuel_Type: Petrol=2, Diesel=1, CNG=0
    # Seller_Type: Dealer=0, Individual=1
    # Transmission: Manual=1, Automatic=0
    
    fuel_map = {"Petrol": 2, "Diesel": 1, "CNG": 0}
    seller_map = {"Dealer": 0, "Individual": 1}
    trans_map = {"Manual": 1, "Automatic": 0}
    
    # 2. Create the feature array in the EXACT order of your X variables
    # Order: Car_Name, Year, Present_Price, Kms_Driven, Fuel_Type, Seller_Type, Transmission, Owner
    features = np.array([[
        car_id, 
        year, 
        present_price, 
        kms_driven, 
        fuel_map[fuel_type], 
        seller_map[seller_type], 
        trans_map[transmission], 
        owner
    ]])
    
    # 3. Transform inputs using the saved Scaler
    features_scaled = scaler.transform(features)
    
    # 4. Make Prediction
    prediction = model.predict(features_scaled)
    
    # 5. Output Result
    st.success(f"### Estimated Selling Price: ₹{prediction[0]:.2f} Lakhs")

#The MLOps 
st.divider()
st.caption("Project versioned with Git & DVC | Model: Decision Tree Regressor")