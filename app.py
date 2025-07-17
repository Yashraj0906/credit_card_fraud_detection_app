import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Load the trained model
model = pickle.load(open("model.pkl", "rb"))

# Load data (optional: used just to display)
df = pd.read_csv("creditcard.csv")

st.set_page_config(page_title="Credit Card Fraud Detection", layout="centered")

st.title("💳 Credit Card Fraud Detection")
st.markdown("This app predicts whether a transaction is **fraudulent or not** using a machine learning model.")

# Show sample data
if st.checkbox("🔍 Show sample data"):
    st.write(df.sample(5, random_state=42))

st.subheader("📥 Enter Transaction Details")

# Input fields for model features
time = st.number_input("Time", value=0.0, format="%.2f")

V_inputs = {}
for i in range(1, 29):
    V_inputs[f'V{i}'] = st.number_input(f"V{i}", value=0.0, format="%.5f")

amount = st.number_input("Amount", value=0.0, format="%.2f")

# Create DataFrame for prediction
input_data = pd.DataFrame([{
    "Time": time,
    **V_inputs,
    "Amount": amount
}])

# Predict
if st.button("🧠 Predict"):
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]

    if prediction == 1:
        st.error(f"🚨 **Fraudulent Transaction Detected!**")
    else:
        st.success(f"✅ Transaction is **Not Fraudulent**.")

    st.markdown(f"**Fraud Probability:** `{probability:.4f}`")

# Footer
st.markdown("---")
st.markdown("Made with ❤️ using Streamlit")
