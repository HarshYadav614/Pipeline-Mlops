import streamlit as st
import pandas as pd
import joblib
import numpy as np
import os

# --- PAGE CONFIG ---
st.set_page_config(page_title="Car Price Predictor", page_icon="🚗")

# --- LOAD MODELS ---
@st.cache_resource
def load_model_and_scaler():
    model_path = os.path.join('models', 'best_model.pkl')
    scaler_path = os.path.join('models', 'scaler.pkl')
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    return model, scaler

# --- CAR NAME MAPPING 
car_names = [
    '800', 'Activa 3g', 'Activa 4g', 'Bajaj Avenger 150', 'Bajaj Avenger 150 street',
    'Bajaj Avenger 220', 'Bajaj Avenger 220 dts-i', 'Bajaj Avenger Street 220',
    'Bajaj CT 100', 'Bajaj Discover 100', 'Bajaj Discover 125', 'Bajaj Dominar 400',
    'Bajaj Pulsar 135 LS', 'Bajaj Pulsar 150', 'Bajaj Pulsar 220 F', 'Bajaj Pulsar AS 200',
    'Bajaj Pulsar NS 200', 'Bajaj Pulsar RS 200', 'Hero Extreme', 'Hero Glamour',
    'Hero Honda CBZ extreme', 'Hero Honda Passion Pro', 'Hero Passion Pro',
    'Hero Passion Xpro', 'Hero Splender Plus', 'Hero Splender iSmart',
    'Hero Super Splendor', 'Honda CB Hornet 160R', 'Honda CB Shine',
    'Honda CB Trigger', 'Honda CB Unicorn', 'Honda CB twister',
    'Honda CBR 150', 'Honda Dream Yuga', 'Honda Karizma', 'Hyosung GT250R',
    'KTM 390 Duke', 'KTM RC200', 'KTM RC390', 'Mahindra Mojo XT300',
    'Royal Enfield Bullet 350', 'Royal Enfield Classic 350', 'Royal Enfield Classic 500',
    'Royal Enfield Thunder 350', 'Royal Enfield Thunder 500', 'Suzuki Access 125',
    'TVS Apache RTR 160', 'TVS Apache RTR 180', 'TVS Jupiter', 'TVS Sport',
    'TVS Wego', 'TVS Victor', 'Yamaha FZ 16', 'Yamaha FZ S', 'Yamaha FZ S V 2.0',
    'Yamaha Fazer', 'alto 800', 'alto k10', 'amaze', 'baleno', 'brio', 'camry',
    'ciaz', 'city', 'corolla', 'corolla altis', 'creta', 'dzire', 'elantra',
    'eon', 'ertiga', 'etios cross', 'etios g', 'etios gd', 'etios liva', 'fortuner',
    'grand i10', 'i10', 'i20', 'ignis', 'innova', 'jtp', 'jazz', 'land cruiser',
    'omni', 'ritz', 's cross', 'swift', 'sx4', 'verna', 'vitara brezza', 'wagon r', 'xcent'
]
car_mapping = {name: i for i, name in enumerate(sorted(car_names))}

try:
    model, scaler = load_model_and_scaler()
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# --- INTERFACE ---
st.title("🚗 Car Selling Price Predictor")

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
    
    selected_car_name = st.selectbox("Select Car Model", options=list(car_mapping.keys()))
    car_id = car_mapping[selected_car_name]

# --- PREDICTION LOGIC ---
if st.button("Predict Selling Price"):
    # Fuel: Petrol=2, Diesel=1, CNG=0 | Seller: Dealer=0, Ind=1 | Trans: Manual=1, Auto=0
    fuel_map = {"Petrol": 2, "Diesel": 1, "CNG": 0}
    seller_map = {"Dealer": 0, "Individual": 1}
    trans_map = {"Manual": 1, "Automatic": 0}
    
    # Feature Order: Car_Name, Year, Present_Price, Kms_Driven, Fuel_Type, Seller_Type, Transmission, Owner
    features = np.array([[
        car_id, year, present_price, kms_driven, 
        fuel_map[fuel_type], seller_map[seller_type], 
        trans_map[transmission], owner
    ]])
    
    features_scaled = scaler.transform(features)
    prediction = model.predict(features_scaled)
    
    st.success(f"### Estimated Selling Price: ₹{prediction[0]:.2f} Lakhs")