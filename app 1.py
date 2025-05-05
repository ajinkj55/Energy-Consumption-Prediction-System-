# app.py

import streamlit as st
import numpy as np
import pickle

# Load model
with open("energy_model.pkl", "rb") as f:
    model = pickle.load(f)

st.title("🔋 Energy Consumption Predictor")

st.markdown("Enter the input values to predict energy usage.")

# Input fields
temp = st.number_input("Temperature (°C)", min_value=0.0, max_value=50.0, value=25.0)
humidity = st.number_input("Humidity (%)", min_value=0.0, max_value=100.0, value=50.0)
occupancy = st.slider("Occupancy (Number of People)", 0, 20, 5)
lighting = st.selectbox("Lighting Usage", ["Off", "On"])
hvac = st.selectbox("HVAC Usage", ["Off", "On"])

# Convert to model format
lighting_val = 1 if lighting == "On" else 0
hvac_val = 1 if hvac == "On" else 0

# Predict
if st.button("Predict Energy Consumption"):
    features = np.array([[temp, humidity, occupancy, lighting_val, hvac_val]])
    prediction = model.predict(features)[0]
    st.success(f"🔌 Estimated Energy Consumption: **{prediction:.2f} kWh**")
    
    # train_model.py

import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import pickle

# Load dataset
df = pd.read_csv("Energy_consumption.csv")

# Preprocessing
df['LightingUsage'] = df['LightingUsage'].map({'Off': 0, 'On': 1})
df['HVACUsage'] = df['HVACUsage'].map({'Off': 0, 'On': 1})

# Features and target
X = df[['Temperature', 'Humidity', 'Occupancy', 'LightingUsage', 'HVACUsage']]
y = df['EnergyConsumption']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Save model
with open("energy_model.pkl", "wb") as f:
    pickle.dump(model, f)

