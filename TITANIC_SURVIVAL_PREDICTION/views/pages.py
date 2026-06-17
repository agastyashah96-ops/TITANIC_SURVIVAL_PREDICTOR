import streamlit as st
import pandas as pd
import joblib
import os

# Load the saved model and scaler
base_dir = os.path.dirname(os.path.abspath(__file__))
model = joblib.load(os.path.join(base_dir, '..', 'model.pkl'))
scaler = joblib.load(os.path.join(base_dir, '..', 'scaler.pkl'))

st.title("🚢 Titanic Survival Predictor")
st.write(":rainbow[Fill in the details below to see if your passenger would have survived.]")

with st.container(border=True):
    st.subheader("👤 Create Your Passenger")

    col1, col2 = st.columns(2)

    with col1:
        age = st.slider("Age", min_value=1, max_value=80, value=25)
        sex = st.selectbox("Sex", ["Male", "Female"])
        pclass = st.selectbox("Passenger Class", [1, 2, 3], 
                               format_func=lambda x: f"Class {x} ({'First' if x==1 else 'Second' if x==2 else 'Third'})")
        fare = st.slider("Ticket Fare (£)", min_value=0, max_value=500, value=30)

    with col2:
        sibsp = st.number_input("Siblings / Spouses Aboard", min_value=0, max_value=8, value=0)
        parch = st.number_input("Parents / Children Aboard", min_value=0, max_value=6, value=0)
        has_cabin = st.selectbox("Has a Cabin?", ["No", "Yes"])
        embarked = st.selectbox("Port of Embarkation", ["Cherbourg (C)", "Queenstown (Q)", "Southampton (S)"])

if st.button("⚓ Check Survival", use_container_width=True):

    # Encode inputs to match training data format
    sex_encoded = 1 if sex == "Male" else 0
    has_cabin_encoded = 1 if has_cabin == "Yes" else 0
    embarked_c = 1 if embarked == "Cherbourg (C)" else 0
    embarked_q = 1 if embarked == "Queenstown (Q)" else 0
    embarked_s = 1 if embarked == "Southampton (S)" else 0

    # Build input dataframe with exact same columns as training
    input_df = pd.DataFrame([{
        'Age': age,
        'Pclass': pclass,
        'Fare': fare,
        'Has_Cabin': has_cabin_encoded,
        'SibSp': sibsp,
        'Parch': parch,
        'Sex': sex_encoded,
        'Embarked_from_C': embarked_c,
        'Embarked_from_Q': embarked_q,
        'Embarked_from_S': embarked_s
    }])

    # Scale and predict
    input_scaled = scaler.transform(input_df)
    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0]

    # Display result
    st.divider()
    if prediction == 1:
        st.success("🎉 Your passenger **SURVIVED!**")
    else:
        st.error("💀 Your passenger **DID NOT SURVIVE.**")

    # Show survival probability
    col1, col2 = st.columns(2)
    col1.metric("Survival Probability", f"{probability[1]*100:.1f}%")
    col2.metric("Death Probability", f"{probability[0]*100:.1f}%")