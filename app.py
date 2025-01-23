import streamlit as st
import pickle
import pandas as pd
import os

# Load the encoder, scaler, and model
scaler_file = os.path.join('static', 'scaler.pkl')
encoder_file = os.path.join('static', 'encoder.pkl')
model_file = os.path.join('static', 'best_rf_model.pkl')

# Load the required files
with open(scaler_file, 'rb') as f:
    scaler = pickle.load(f)

with open(encoder_file, 'rb') as f:
    encoder = pickle.load(f)

with open(model_file, 'rb') as f:
    model = pickle.load(f)

# Define the complete list of features used during training
required_features = ['Age', 'DurationOfPitch', 'PreferredPropertyStar', 'MonthlyIncome', 'TotalVisiting']

# Streamlit app configuration
st.title("Wellness Tourism Prediction")
st.markdown("Enter the details below to predict whether the user is likely to take the Wellness Tourism Package.")

# Collect input features
def user_input():
    inputs = {}
    for feature in required_features:
        inputs[feature] = st.number_input(f"{feature}", min_value=0.0, step=0.1)
    return inputs

user_inputs = user_input()

if st.button("Predict"):
    try:
        # Convert input dictionary to DataFrame
        input_df = pd.DataFrame([user_inputs])

        # Scale the data
        scaled_data = scaler.transform(input_df)

        # Predict using the model
        prediction = model.predict(scaled_data)

        # Display the result
        if prediction[0] == 1:
            st.success("User is likely to take the Wellness Tourism Package")
        else:
            st.error("User is not likely to take the Wellness Tourism Package")

    except Exception as e:
        st.error(f"Error: {e}")
