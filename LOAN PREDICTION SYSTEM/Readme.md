import streamlit as st
import joblib
import numpy as np
import time

# ---------------------------------------------------------------------
# 1. PAGE CONFIGURATION
# ---------------------------------------------------------------------
st.set_page_config(page_title="Smart Loan Approval", page_icon="🏦", layout="centered")

# ---------------------------------------------------------------------
# 2. LOAD MODEL & SCALER (Cache removed to force fresh loading)
# ---------------------------------------------------------------------
def load_assets():
    model = joblib.load('ann_loan_model_lean.pkl')
    scaler = joblib.load('loan_scaler_lean.pkl')
    return model, scaler

try:
    model, scaler = load_assets()
except Exception as e:
    st.error("🚨 Error loading files!")
    st.write(e)
    st.stop()

# ---------------------------------------------------------------------
# 3. APP HEADER & UI DESIGN
# ---------------------------------------------------------------------
st.markdown("<h1 style='text-align: center; color: #4CAF50;'>🏦 Smart Loan Predictor AI</h1>", unsafe_allow_html=True)
st.markdown("<h4 style='text-align: center; color: gray;'>Powered by Artificial Neural Networks</h4>", unsafe_allow_html=True)
st.divider() 

# ---------------------------------------------------------------------
# 4. DATA INPUT SECTION (7 Lean Columns)
# ---------------------------------------------------------------------
st.subheader("📝 Applicant Details")

col1, col2 = st.columns(2)

with col1:
    st.markdown("**Personal Information**")
    
    # Matches exactly with LabelEncoder logic
    gender_options = {"Female": 0, "Joint": 1, "Male": 2, "Not Available": 3}
    gender_selection = st.selectbox("Gender", options=list(gender_options.keys()))
    gender = gender_options[gender_selection] 
    
    age = st.number_input("Age", min_value=18, max_value=100, value=35)
    credit_score = st.slider("Credit Score", min_value=300, max_value=900, value=750)
    income = st.number_input("Annual Income (₹)", min_value=0, value=600000, step=25000)

with col2:
    st.markdown("**Loan Specifications (₹)**")
    
    loan_amount = st.number_input("Requested Loan Amount (₹)", min_value=10000, max_value=50000000, value=1500000, step=50000)
    property_value = st.number_input("Property Value (₹)", min_value=0, value=2500000, step=50000)
    rate_of_interest = st.number_input("Rate of Interest (%)", min_value=0.0, max_value=20.0, value=8.5, format="%.2f")

st.divider()

# ---------------------------------------------------------------------
# 5. PREDICTION BUTTON & LOGIC
# ---------------------------------------------------------------------
_, center_col, _ = st.columns([1, 2, 1])

with center_col:
    predict_clicked = st.button("🔮 Analyze & Predict Loan Status", use_container_width=True)

if predict_clicked:
    with st.spinner('Neural Network is analyzing financial data... Please wait... 🧠'):
        time.sleep(1.5) 
    
    # CRITICAL: This exact order matches ['gender', 'loan amount', 'rate of interest', 'property value', 'income', 'credit score', 'age']
    user_data = np.array([[
        gender, 
        loan_amount, 
        rate_of_interest, 
        property_value, 
        income, 
        credit_score, 
        age
    ]])
    
    try:
        # Scale the data using the NEW lean scaler
        scaled_data = scaler.transform(user_data)
        
        # Make the prediction
        prediction = model.predict(scaled_data)
        
        # Output the interactive results
        if prediction[0] == 0:
            st.success("### 🎉 Loan Approved! \n**This applicant successfully meets the risk criteria.**")
            st.balloons() 
        else:
            st.error("### 🚫 Loan Denied. \n**This applicant poses too high of a financial risk at this time.**")
            
    except ValueError as e:
        st.error("🚨 Shape Mismatch Error!")
        st.warning(f"Technical details: {e}")
        st.info("💡 Ensure your Jupyter Notebook model was trained on exactly these 7 columns.")
