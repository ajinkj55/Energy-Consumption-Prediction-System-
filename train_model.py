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
