import streamlit as st
import pandas as pd
import pickle
import numpy as np

# 1. PAGE CONFIGURATION
st.set_page_config(page_title="Simon Bank HELOC Tool", layout="wide")
st.title("🏦 Simon Bank: HELOC Eligibility Screener")
st.markdown("### Decision Support System for Credit Risk Assessment")

# 2. LOAD THE SAVED MODEL
# This reaches out to the 'heloc_model.pkl' file you exported
@st.cache_resource
def load_model():
    with open('heloc_model.pkl', 'rb') as f:
        return pickle.load(f)

model = load_model()

# 3. USER INPUT INTERFACE
st.sidebar.header("Applicant Profile")
st.sidebar.write("Adjust features to see prediction impact.")

# We focus on your TOP predictors found in your Big Bar chart
ext_risk = st.sidebar.slider("External Risk Estimate", 30, 100, 75)
sat_trades = st.sidebar.number_input("Number of Satisfactory Trades", 0, 100, 20)
rev_burden = st.sidebar.slider("Net Fraction Revolving Burden (Usage %)", 0, 150, 30)
never_delq = st.sidebar.radio("Never Delinquent?", ["Yes", "No"])

# 4. DATA TRANSFORMATION (The Expert Step)
if st.button("Generate Eligibility Prediction"):
    
    # 1. Create the base input dictionary with your UI values
    # Use 1 for "Yes", 0 for "No" for your NeverDelinquent flag
    never_delq_val = 1 if never_delq == "Yes" else 0
    
    user_inputs = {
        'ExternalRiskEstimate': ext_risk,
        'NumSatisfactoryTrades': sat_trades,
        'NetFractionRevolvingBurden': rev_burden,
        'NeverDelinquent': never_delq_val
    }
    
    # 2. Build a full DataFrame with ALL columns the model expects
    # We fill missing columns with 0 as a baseline placeholder
    full_columns = model.feature_names_in_
    input_df = pd.DataFrame(columns=full_columns)
    input_df.loc[0] = 0 
    
    # 3. Overwrite the placeholder values with the actual user inputs
    for col, val in user_inputs.items():
        if col in input_df.columns:
            input_df[col] = val
            
    # 5. PREDICTION & RESULTS
    # This now matches the expected input shape for the Pipeline
    probability_bad = model.predict_proba(input_df)[0][1] 

    st.divider()
    if probability_bad < 0.5:
        st.success(f"### Result: APPROVED (Confidence: {1 - probability_bad:.2%})")
        st.balloons()
    else:
        st.error(f"### Result: DECLINED (Risk Score: {probability_bad:.2%})")
        st.warning("Manual Review Recommended: Applicant profile matches 'Bad' risk history.")