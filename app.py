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
# We must convert the user's answers into the format the model learned
if st.button("Generate Eligibility Prediction"):
    
    # Create the NeverDelinquent flag (1 for Yes, 0 for No)
    never_delq_val = 1 if never_delq == "Yes" else 0
    
    # Create a DataFrame for the model (Ensure column names match your training!)
    # Note: We use median values for columns not shown in the UI for simplicity
    input_data = pd.DataFrame({
        'ExternalRiskEstimate': [ext_risk],
        'NumSatisfactoryTrades': [sat_trades],
        'NetFractionRevolvingBurden': [rev_burden],
        'NeverDelinquent': [never_delq_val]
    })
    
    # We must ensure all columns from the training set exist (even if they are 0)
    # The model expects the same number of columns as the X_train you used
    full_columns = model.feature_names_in_
    for col in full_columns:
        if col not in input_data.columns:
            input_data[col] = 0 # Placeholder for other columns
    
    # Reorder columns to match model training exactly
    input_data = input_data[full_columns]


        
    # 5. PREDICTION & RESULTS
    # probability is now the Risk Score (Probability of being 'Bad')
    probability_bad = model.predict_proba(input_data)[0][1] 

    st.divider()
    
    # Logic Change: If probability of being 'Bad' is low (< 50%), Approve.
    if probability_bad < 0.5:
        # Confidence in 'Good' is 1 minus the 'Bad' probability
        confidence_good = 1 - probability_bad
        st.success(f"### Result: APPROVED (Confidence: {confidence_good:.2%})")
        st.balloons()
    else:
        # Confidence in 'Bad' is the raw probability_bad
        st.error(f"### Result: DECLINED (Risk Score: {probability_bad:.2%})")
        st.warning("Manual Review Recommended: Applicant profile matches 'Bad' risk history.")