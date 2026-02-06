import streamlit as st
import pandas as pd
import joblib
import numpy as np
import os

# --- PAGE CONFIG ---
st.set_page_config(page_title="Car Price Predictor", page_icon="🚗")

# --- LOAD MODELS AND DATA ---
@st.cache_resource
def load_resources():
    model_path = os.path.join('models', 'best_model.pkl')
    scaler_path = os.path.join('models', 'scaler.pkl')
    
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    
    # Load the dataset to recreate the Car_Name mapping automatically
    df = pd.read_csv('car data.csv')
    # This recreates the exact LabelEncoding order used in your notebook
    car_names = sorted(df['Car_Name'].unique())
    car_mapping = {name: i for i, name in enumerate(car_names)}
    
    return model, scaler, car_mapping

try:
    model, scaler, car_mapping = load_resources()
except Exception as e:
    st.error(f"Error loading resources: {e}")
    st.stop()

# --- INTERFACE ---
st.title("🚗 Car Selling Price Predictor")
st.markdown("Enter details below to get an instant price estimate.")

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
    
    # CHANGED: Select car by name, but use ID for the model
    selected_car_name = st.selectbox("Select Car Model", options=list(car_mapping.keys()))
    car_id = car_mapping[selected_car_name]

# --- PREDICTION LOGIC ---
if st.button("Predict Selling Price"):
    # Mapping based on your notebook's LabelEncoding logic
    fuel_map = {"Petrol": 2, "Diesel": 1, "CNG": 0}
    seller_map = {"Dealer": 0, "Individual": 1}
    trans_map = {"Manual": 1, "Automatic": 0}
    
    # Feature Order: Car_Name, Year, Present_Price, Kms_Driven, Fuel_Type, Seller_Type, Transmission, Owner
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
    
    # Scale and Predict
    features_scaled = scaler.transform(features)
    prediction = model.predict(features_scaled)
    
    st.success(f"### Estimated Selling Price: ₹{prediction[0]:.2f} Lakhs")

st.divider()
st.caption("Project versioned with Git & DVC | Model: Decision Tree Regressor")