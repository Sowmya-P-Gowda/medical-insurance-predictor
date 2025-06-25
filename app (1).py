import streamlit as st
import numpy as np
import pickle

# Load the trained model
with open('random_forest_model.pkl', 'rb') as f:
    model = pickle.load(f)

st.set_page_config(page_title="Medical Insurance Cost Predictor")
st.title("💰 Medical Insurance Cost Predictor")
st.markdown("### Enter User Details")

# Input fields
age = st.number_input("Age", min_value=18, max_value=100, value=30)

sex_input = st.selectbox("Sex", options=["Male", "Female"])
sex = 1 if sex_input == "Female" else 0

bmi = st.number_input("BMI", min_value=10.0, max_value=60.0, value=25.0)

children = st.number_input("Number of Children", min_value=0, max_value=10, value=1)

smoker_input = st.selectbox("Smoker", options=["No", "Yes"])
smoker = 1 if smoker_input == "Yes" else 0

region_input = st.selectbox("Region", options=["Southeast", "Southwest", "Northeast", "Northwest"])
region_map = {"Southeast": 0, "Southwest": 1, "Northeast": 2, "Northwest": 3}
region = region_map[region_input]

# Prepare features in correct order
features = [age, sex, bmi, children, smoker, region]
user_input = np.array([features])  # Must be 2D for sklearn

# Debugging: Show input shape and model expectation
# st.write("Input shape:", user_input.shape)
# st.write("Model expects:", model.n_features_in_)

# Predict and show result
if st.button("Predict Insurance Expense"):
    try:
        prediction = model.predict(user_input)[0]
        st.success(f"💡 Predicted Medical Expense: ₹{prediction:,.2f}")
    except Exception as e:
        st.error("Prediction failed. Error:")
        st.error(str(e))
