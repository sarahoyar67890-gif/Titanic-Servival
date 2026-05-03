import streamlit as st
import numpy as np
import pandas as pd
import pickle

# -----------------------
# LOAD MODEL + COLUMNS
# -----------------------
model = pickle.load(open("model.pkl", "rb"))
columns = pickle.load(open("columns.pkl", "rb"))

st.title("Titanic Survival Prediction")

st.write("Enter passenger details:")

# -----------------------
# INPUTS
# -----------------------
age = st.number_input("Age", value=25)
fare = st.number_input("Fare", value=10.0)

sex = st.selectbox("Sex", ["male", "female"])
pclass = st.selectbox("Pclass", [1, 2, 3])
embarked = st.selectbox("Embarked", ["S", "C", "Q"])
family_size = st.number_input("Family Size", value=1)

# -----------------------
# ENCODING
# -----------------------
sex = 0 if sex == "male" else 1

is_alone = 1 if family_size == 1 else 0

# Build input dictionary (MATCH TRAINING FEATURES)
input_dict = {
    "Age": age,
    "Fare": fare,
    "FamilySize": family_size,
    "IsAlone": is_alone,
    "Sex": sex,
    "Pclass_2": 1 if pclass == 2 else 0,
    "Pclass_3": 1 if pclass == 3 else 0,
    "Embarked_Q": 1 if embarked == "Q" else 0,
    "Embarked_S": 1 if embarked == "S" else 0
}

# Convert to DataFrame
input_df = pd.DataFrame([input_dict])

# Reorder EXACTLY like training
input_df = input_df.reindex(columns=columns, fill_value=0)

# -----------------------
# PREDICTION
# -----------------------
if st.button("Predict"):
    result = model.predict(input_df)[0]

    if result == 1:
        st.success("Passenger Survived")
    else:
        st.error("Passenger Did Not Survive")