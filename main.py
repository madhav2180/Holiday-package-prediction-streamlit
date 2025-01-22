import joblib
import streamlit as st
from os import path
import pandas as pd

st.title("Holiday Package Predictor")

filename = "best_rf_model.pkl"
predictor_model = joblib.load(path.join("static", filename)) # mapping the ML model 

scaler_file = joblib.load(path.join("static", 'scaler.pkl')) # mapping the scaler

# Store the initial value of widgets in session state
if "value" not in st.session_state:
    st.session_state.value = None
    st.session_state.index = None
    st.session_state.horizontal = False

# function to change the user inputs to match our format- format in which it has to be passed to the ML model.
def change_format(age, m_income, duration, star, no_ppl):
    required_features = ['Age', 'DurationOfPitch', 'PreferredPropertyStar', 'MonthlyIncome', 'TotalVisiting']
    inputs=[age, duration, star, m_income, no_ppl]
    input_df = pd.DataFrame([inputs], columns=required_features) # creating dataframe in the desired order for the .pkl file
    scaled_input=scaler_file.transform(input_df) # scaling the dataframe to match with the values used to train the model.

    return scaled_input


# Streamlit UI elements (5 numerical vlaues).
# change maximum and minimum value as needed.( value=None - initial values are set to None. so, placeholders would be visible.)
age = st.number_input( "Enter the age",  placeholder="Enter the age...", value=None, min_value=0, max_value=123) 
m_income = st.number_input( "Enter Monthly Income", placeholder="Enter Monthly Income", value=None, min_value=0)
duration = st.number_input( "Enter duration of pitch", placeholder="Enter duration in minutes", value=None, min_value=0, max_value=635)
star = st.number_input("Star rating", placeholder="Enter the Star rating", value=None, min_value=1, max_value=5)
no_ppl = st.number_input( "How many people", placeholder="Enter number of people", value=None, min_value=0, max_value=10)


# Predict button
if st.button("Predict"):
    # validating all the values are entered.
    if((age==None) or (m_income==None) or (duration==None) or 
       (star==None) or (no_ppl==None)):
        st.write("Please fill all the fields.")
    else:
        # transforming the data entered by the user to the format of the predictor file input.
        user_data=change_format(age, m_income,  duration, star, no_ppl)               
        pred = predictor_model.predict(user_data) # prediction happens here.
        # if the prediction result is 0,
        if (pred[0]==0):
            st.write("'User is not likely to take the Wellness Tourism Package'")
        # if the prediction result is 1,
        else:
            st.write("'User is likely to take the Wellness Tourism Package'")

    
        
        
    
    