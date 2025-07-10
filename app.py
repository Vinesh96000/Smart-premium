import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="SmartPremium", page_icon="💰")

st.title("💡 SmartPremium – Predicting Insurance Cost")

# Load model
try:
    model = joblib.load("model/insurance_model.pkl")
except:
    st.error("❌ Model file not found. Make sure 'insurance_model.pkl' is in the 'model' folder.")
    st.stop()

st.markdown("### 🔢 Enter the customer details:")

income = st.number_input("Annual Income (in ₹)", min_value=10000, step=1000)
health = st.slider("Health Score", 0, 100, 50)
marital = st.selectbox("Marital Status", ["Single", "Married", "Divorced", "Widowed"])
dependents = st.number_input("Number of Dependents", min_value=0, max_value=10, step=1)
credit = st.slider("Credit Score", 300, 850, 600)

if st.button("🔮 Predict Premium"):
    new_data = pd.DataFrame({
        "Annual Income": [income],
        "Health Score": [health],
        "Marital Status": [marital],
        "Number of Dependents": [dependents],
        "Credit Score": [credit]
    })

    try:
        prediction = model.predict(new_data)
        st.success(f"💸 Predicted Premium Amount: ₹{prediction[0]:,.2f}")
    except Exception as e:
        st.error(f"❌ Prediction failed: {e}")
