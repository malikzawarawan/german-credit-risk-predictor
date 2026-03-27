import streamlit as st
import pandas as pd
import joblib

st.title("Credit Risk Prediction App")
st.write("Enter applicant information to predict if the credit risk is good or bad")

# ✅ Load Model
model = joblib.load("Best_Model.pkl")

# ✅ Load Encoders
encoders = {
    "Sex": joblib.load("Sex_encoder.pkl"),
    "Housing": joblib.load("Housing_encoder.pkl"),
    "Saving accounts": joblib.load("Saving accounts_encoder.pkl"),
    "Checking account": joblib.load("Checking account_encoder.pkl")
}

# ✅ User Inputs
age = st.number_input("Age", min_value=18, max_value=80, value=30)
sex = st.selectbox("Sex", ["male", "female"])
job = st.number_input("Job (0-3)", min_value=0, max_value=3, value=1)
housing = st.selectbox("Housing", ["own", "free", "rent"])
saving_account = st.selectbox("Saving Account", ["little", "moderate", "rich", "quite rich"])
checking_account = st.selectbox("Checking Account", ["little", "moderate", "rich"])
credit_amount = st.number_input("Credit Amount", min_value=0, value=1000)
duration = st.number_input("Duration (months)", min_value=1, value=12)

# ✅ Convert categorical values using encoders
input_df = pd.DataFrame({
    "Age": [age],
    "Sex": [encoders["Sex"].transform([sex])[0]],
    "Job": [job],
    "Housing": [encoders["Housing"].transform([housing])[0]],
    "Saving accounts": [encoders["Saving accounts"].transform([saving_account])[0]],
    "Checking account": [encoders["Checking account"].transform([checking_account])[0]],
    "Credit amount": [credit_amount],
    "Duration": [duration]
})

# ✅ Prediction Button
if st.button("Predict Risk"):
    pred = model.predict(input_df)[0]

    if pred == 1:
        st.success("✅ The predicted credit risk is **GOOD**")
    else:
        st.error("❌ The predicted credit risk is **BAD**")
