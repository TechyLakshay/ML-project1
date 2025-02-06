import streamlit as st
import pickle
import numpy as np
import pandas as pd  

# Load the trained model
with open("insurence.pkl", "rb") as file:
    model = pickle.load(file)

st.title("Insurance Charges Prediction")

# User Inputs
age = st.number_input("Age", min_value=18, max_value=100, value=25)
sex = st.selectbox("Sex", ["male", "female"])
bmi = st.number_input("BMI", min_value=10.0, max_value=50.0, value=25.0)
children = st.number_input("Children", min_value=0, max_value=10, value=0)
smoker = st.selectbox("Smoker", ["yes", "no"])
region = st.selectbox("Region", ["southwest", "southeast", "northwest", "northeast"])

# Encode categorical variables correctly
sex_encoded = 1 if sex == "male" else 0
smoker_encoded = 1 if smoker == "yes" else 0
region_mapping = {"southwest": 0, "southeast": 1, "northwest": 2, "northeast": 3}
region_encoded = region_mapping[region]  

# Ensure all values are in float format to prevent TypeErrors
input_data = np.array([[age, sex_encoded, bmi, children, smoker_encoded, region_encoded]], dtype=float)

# Convert to DataFrame (matching training data format)
input_df = pd.DataFrame(input_data, columns=["age", "sex", "bmi", "children", "smoker", "region"])

# Debugging print statement
st.write(f"Feature shape before prediction: {input_df.shape}")
st.write(f"Feature Data Types:\n{input_df.dtypes}")

# Prediction
if st.button("Predict Charges"):
    try:
        prediction = model.predict(input_df)  
        st.success(f"Predicted Insurance Charges: ${prediction[0]:,.2f}")
    except ValueError as e:
        st.error(f"Error: {e}. Ensure input format matches the trained model.")
