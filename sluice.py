import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
import streamlit as st

# Generate Sample Data
def generate_data(n=300):
    np.random.seed(42)
    surface_area_current = np.random.uniform(1, 500, n)  
    depth_current = np.random.uniform(5, 50, n)  
    breadth_current = np.random.uniform(1, 100, n)  
    inflow_amount = np.random.uniform(0.1, 5.0, n)  # New parameter for inflow capacity

    # Automate Target Dimensions Based on Inflow Amount
    surface_area_target = surface_area_current + (inflow_amount * 2)
    depth_target = depth_current + (inflow_amount * 1.5)
    breadth_target = breadth_current + (inflow_amount * 1)

    prev_maintenance_cost = np.random.uniform(50000, 500000, n)

    # Calculate differences
    depth_diff = depth_target - depth_current

    # Determine Condition Based on Depth Difference
    condition = []
    maintenance_suggestions = []
    maintenance_frequency = []

    for diff in depth_diff:
        if diff > 5:
            condition.append("High Sedimentation")
            maintenance_suggestions.append("Increase depth by dredging and use sediment for shore strengthening.")
            maintenance_frequency.append("High (Every Year)")
        elif diff < 2:
            condition.append("High Erosion")
            maintenance_suggestions.append("Strengthen shores to prevent erosion.")
            maintenance_frequency.append("Moderate (Every 2-3 Years)")
        else:
            condition.append("Stable")
            maintenance_suggestions.append("Regular maintenance of inlets and outlets required.")
            maintenance_frequency.append("Low (Every 5 Years)")

    material_cost_per_unit = 100  
    base_cost = (surface_area_target - surface_area_current) * depth_diff * (breadth_target - breadth_current) * material_cost_per_unit
    final_cost = base_cost + prev_maintenance_cost

    df = pd.DataFrame({
        'Surface_Area_Current': surface_area_current,
        'Depth_Current': depth_current,
        'Breadth_Current': breadth_current,
        'Inflow_Amount': inflow_amount,
        'Surface_Area_Target': surface_area_target,
        'Depth_Target': depth_target,
        'Breadth_Target': breadth_target,
        'Condition': condition,
        'Maintenance_Frequency': maintenance_frequency,
        'Maintenance_Suggestions': maintenance_suggestions,
        'Prev_Maintenance_Cost': prev_maintenance_cost,
        'Estimated_Cost': final_cost
    })
    return df

# Load dataset
try:
    df = pd.read_csv("lake_data.csv")
except FileNotFoundError:
    df = generate_data(300)
    df.to_csv("lake_data.csv", index=False)

# Encode categorical features
encoder = OneHotEncoder()
condition_encoded = encoder.fit_transform(df[['Condition']]).toarray()
condition_df = pd.DataFrame(condition_encoded, columns=encoder.get_feature_names_out())

df = df.drop(columns=["Condition", "Maintenance_Frequency", "Maintenance_Suggestions"])
df = df.join(condition_df)

# Split Data
X = df.drop(columns=["Estimated_Cost"])
y = df["Estimated_Cost"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train & Save Model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

with open("lake_model.pkl", "wb") as f:
    pickle.dump(model, f)

# Streamlit UI
st.title("Irrigation Lake Maintenance Cost Predictor")
st.sidebar.header("Enter Lake Dimensions")

# User Inputs
surface_area_current = st.sidebar.slider("Current Surface Area (km²)", 1.0, 500.0, 50.0)
depth_current = st.sidebar.slider("Current Depth (m)", 5.0, 50.0, 20.0)
breadth_current = st.sidebar.slider("Current Breadth (km)", 1.0, 100.0, 10.0)
inflow_amount = st.sidebar.slider("Inflow Amount", 0.1, 10.0, 2.0)  # New inflow parameter

# Automate Target Dimensions Based on Inflow Amount
surface_area_target = surface_area_current + (inflow_amount * 2)
depth_target = depth_current + (inflow_amount * 1.5)
breadth_target = breadth_current + (inflow_amount * 1)

prev_cost = st.sidebar.number_input("Previous Maintenance Cost", 50000, 500000, 100000)
material_cost = st.sidebar.number_input("Material Cost (₹/unit)", 50, 500, 100)

# Compute Depth Difference for Condition
depth_diff = depth_target - depth_current

# Determine Condition, Maintenance Frequency, and Suggestions
if depth_diff > 5:
    condition = "High Sedimentation"
    maintenance_suggestion = "Increase depth by dredging and use sediment for shore strengthening."
    maintenance_frequency = "High (Every Year)"
elif depth_diff < 2:
    condition = "High Erosion"
    maintenance_suggestion = "Strengthen shores to prevent erosion."
    maintenance_frequency = "Moderate (Every 2-3 Years)"
else:
    condition = "Stable"
    maintenance_suggestion = "Regular maintenance of inlets and outlets required."
    maintenance_frequency = "Low (Every 5 Years)"

st.write(f"### Computed Lake Condition: {condition}")
st.write(f"### Suggested Maintenance: {maintenance_suggestion}")
st.write(f"### Maintenance Frequency: {maintenance_frequency}")

# Encode Condition for Prediction
condition_input = pd.DataFrame(encoder.transform([[condition]]).toarray(), columns=encoder.get_feature_names_out())

# Define input data for prediction
input_data = pd.DataFrame([[surface_area_current, depth_current, breadth_current,
                            inflow_amount, surface_area_target, depth_target, breadth_target, prev_cost]],
                          columns=["Surface_Area_Current", "Depth_Current", "Breadth_Current",
                                   "Inflow_Amount", "Surface_Area_Target", "Depth_Target", "Breadth_Target",
                                   "Prev_Maintenance_Cost"])

input_data = input_data.join(condition_input)

# Load Model & Predict
with open("lake_model.pkl", "rb") as f:
    loaded_model = pickle.load(f)

if st.sidebar.button("Predict Maintenance Cost"):
    pred_price = loaded_model.predict(input_data)[0]
    
    # Adjust price based on user-defined material cost
    adjusted_price = pred_price + (material_cost * (surface_area_target - surface_area_current) * depth_diff * (breadth_target - breadth_current))
    
    st.write(f"Predicted Maintenance Cost: ₹{adjusted_price:,.2f}")
