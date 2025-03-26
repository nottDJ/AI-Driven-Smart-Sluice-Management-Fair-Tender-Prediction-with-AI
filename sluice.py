import pandas as pd
import numpy as np
import random
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import streamlit as st

# Generate Sample Data
def generate_data(n=200):
    random.seed(42)
    sluice_width = np.random.uniform(5, 20, n)
    sluice_height = np.random.uniform(2, 10, n)
    sluice_depth = np.random.uniform(1, 5, n)
    material = np.random.choice(["Concrete", "Steel", "Brick"], n)
    condition = np.random.choice(["Good", "Average", "Poor"], n)
    previous_maintenance_cost = np.random.uniform(5000, 50000, n)
    market_rate = np.random.uniform(1.1, 1.5, n)  # Multiplier on baseline cost
    baseline_cost = sluice_width * sluice_height * sluice_depth * 1000  # Hypothetical base cost
    auction_price = baseline_cost * market_rate + np.random.uniform(-5000, 5000, n)
    
    df = pd.DataFrame({
        'Width': sluice_width,
        'Height': sluice_height,
        'Depth': sluice_depth,
        'Material': material,
        'Condition': condition,
        'Prev_Maintenance_Cost': previous_maintenance_cost,
        'Auction_Price': auction_price
    })
    return df

data = generate_data(300)
data.to_csv("sluice_data.csv", index=False)

# Load Data
df = pd.read_csv("sluice_data.csv")

# Preprocessing
encoder = OneHotEncoder()
categorical_cols = ["Material", "Condition"]
categorical_data = encoder.fit_transform(df[categorical_cols]).toarray()
categorical_df = pd.DataFrame(categorical_data, columns=encoder.get_feature_names_out())
df = df.drop(columns=categorical_cols).join(categorical_df)

# Split Data
X = df.drop(columns=["Auction_Price"])
y = df["Auction_Price"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate Model
y_pred = model.predict(X_test)
print("MAE:", mean_absolute_error(y_test, y_pred))
print("MSE:", mean_squared_error(y_test, y_pred))
print("R2 Score:", r2_score(y_test, y_pred))

# Deployment with Streamlit
st.title("Sluice Maintenance Auction Price Predictor")
st.sidebar.header("Enter Sluice Details")
width = st.sidebar.slider("Width (m)", 5.0, 20.0, 10.0)
height = st.sidebar.slider("Height (m)", 2.0, 10.0, 5.0)
depth = st.sidebar.slider("Depth (m)", 1.0, 5.0, 2.0)
material = st.sidebar.selectbox("Material", ["Concrete", "Steel", "Brick"])
condition = st.sidebar.selectbox("Condition", ["Good", "Average", "Poor"])
prev_cost = st.sidebar.number_input("Previous Maintenance Cost", 5000, 50000, 10000)

input_data = pd.DataFrame([[width, height, depth, prev_cost, material, condition]], columns=["Width", "Height", "Depth", "Prev_Maintenance_Cost", "Material", "Condition"])
input_data = input_data.join(pd.DataFrame(encoder.transform(input_data[["Material", "Condition"]]).toarray(), columns=encoder.get_feature_names_out()))
input_data = input_data.drop(columns=["Material", "Condition"])

if st.sidebar.button("Predict Auction Price"):
    pred_price = model.predict(input_data)[0]
    st.write(f"Predicted Auction Price: ₹{pred_price:,.2f}")
