import streamlit as st
import pickle
import pandas as pd

with open("xgb_model.pkl", "rb") as f:
    model = pickle.load(f)


# App title
st.title("🔮 Customer Churn Prediction App")

st.markdown("""
This app predicts whether a customer is likely to churn based on key features like credit score, age, balance, geography, and more. and is trained on Bank Customers dataset.
""")

# User inputs
gender = st.selectbox("Gender", ["Male", "Female"])
geography = st.selectbox("Geography", ["France", "Germany", "Spain"])
credit_score = st.number_input("Credit Score", min_value=300, max_value=900, value=650)
age = st.slider("Age", 18, 100, 35)
tenure = st.slider("Tenure (years)", 0, 10, 3)
balance = st.number_input("Balance", min_value=0.0, max_value=300000.0, value=100000.0)
num_products = st.selectbox("Number of Products", [1, 2, 3, 4])
has_cr_card = st.selectbox("Has Credit Card", ["Yes", "No"])
is_active = st.selectbox("Is Active Member", ["Yes", "No"])
estimated_salary = st.number_input("Estimated Salary", min_value=0.0, max_value=200000.0, value=50000.0)


# Order of features must match model training
df = pd.DataFrame([[
    credit_score,geography,gender,age,tenure,balance,num_products,has_cr_card,is_active,estimated_salary
]], columns=[
    'CreditScore','Geography','Gender', 'Age', 'Tenure', 'Balance', 'NumOfProducts',
    'HasCrCard', 'IsActiveMember', 'EstimatedSalary'
])


df['Geography'] = df['Geography'].map({'France': 0, 'Germany': 1, 'Spain': 2})
df['Gender'] = df['Gender'].map({'Female': 0, 'Male': 1})


df['HasCrCard'] = df['HasCrCard'].map({"Yes": 1, "No": 0})
df['IsActiveMember'] = df['IsActiveMember'].map({"Yes": 1, "No": 0})


if st.button("Predict Churn"):
    prediction = model.predict(df)[0]
    prob = model.predict_proba(df)[0][1]

    st.subheader("📊 Prediction Result")
    st.write(f"Churn Probability: **{prob*100:.2f}%**")
    if prediction ==1:
        st.error("⚠️ The customer is likely to churn.")
    else:
        st.success("✅ The customer is likely to stay.")