import streamlit as st
import numpy as np
import pickle
import pandas as pd

# Load the trained model
with open('random_forest_model.pkl', 'rb') as f:
    model = pickle.load(f)

st.set_page_config(page_title="Medical Insurance Cost Predictor")
st.title("💰 Medical Insurance Cost Predictor")
st.markdown("### Enter User Details")

# Input fields
age = st.slider("Age", 18, 100, 30)
sex_input = st.radio("Sex", ["Male", "Female"])
bmi = st.slider("BMI", 10.0, 60.0, 25.0)
children = st.slider("Number of Children", 0, 10, 1)
smoker_input = st.selectbox("Smoker", ["No", "Yes"])
region_input = st.selectbox("Region", ["Southeast", "Southwest", "Northeast", "Northwest"])

# 🔁 Convert inputs to numeric model-ready values
sex = 1 if sex_input == "Female" else 0
smoker = 0 if smoker_input == "Yes" else 0
region_map = {"Southeast": 0, "Southwest": 1, "Northeast": 2, "Northwest": 3}
region = region_map[region_input]

# 🧠 Feature Engineering
obese_smoker = int(bmi >= 30 and smoker == 1)

# BMI Category
bmi_bins = [0, 18.5, 24.9, 29.9, 100]
bmi_labels = ['Underweight', 'Normal', 'Overweight', 'Obese']
bmi_category = pd.cut([bmi], bins=bmi_bins, labels=bmi_labels, right=False)[0]

# You may encode this category for model input
bmi_category_map = {'Underweight': 0, 'Normal': 1, 'Overweight': 2, 'Obese': 3}
bmi_category_encoded = bmi_category_map[bmi_category]

# ✅ Final features to the model (assume 8 features now)
features = [age, sex, bmi, children, smoker, region, obese_smoker, bmi_category_encoded]
user_input = np.array([features])

# 🔮 Predict
if st.button("Predict Insurance Expense"):
    try:
        prediction = model.predict(user_input)[0]
        st.success(f"💡 Predicted Medical Expense: ₹{prediction:,.2f}")
        
        st.markdown("##### 🧠 Engineered Features Used:")
        st.write(f"- Obese Smoker: `{obese_smoker}`")
        st.write(f"- BMI Category: `{bmi_category}`")
        st.write(f"smoker = {smoker}")
        st.write(f"bmi = {bmi}")
        st.write(f"obese_smoker = {obese_smoker}")
    except Exception as e:
        st.error("Prediction failed.")
        st.error(str(e))
