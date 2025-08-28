import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Predicción de Diabetes", layout="centered")

# === 1. Cargar modelo entrenado ===
model = joblib.load("../results/model.pkl")

st.title("🩺 Predicción de Diabetes")
st.write("Introduce los datos clínicos del paciente:")

# === 2. Campos de entrada ===
pregnancies = st.number_input("Número de embarazos", 0, 20, 0)
glucose = st.slider("Glucosa", 0, 200, 100)
blood_pressure = st.slider("Presión arterial", 0, 150, 70)
skin_thickness = st.slider("Grosor de pliegue cutáneo", 0, 100, 20)
insulin = st.slider("Insulina", 0, 900, 80)
bmi = st.slider("Índice de masa corporal", 0.0, 70.0, 25.0)
dpf = st.slider("Diabetes Pedigree Function", 0.0, 3.0, 0.5)
age = st.slider("Edad", 18, 100, 30)

# === 3. Predicción ===
if st.button("Predecir"):
    features = [[pregnancies, glucose, blood_pressure, skin_thickness,
                insulin, bmi, dpf, age]]
    prediction = model.predict(features)[0]
    prob = model.predict_proba(features)[0][1]

    if prediction == 1:
        st.error(f"⚠️ Alta probabilidad de diabetes ({prob*100:.1f}%)")
    else:
        st.success(f"✅ Baja probabilidad de diabetes ({(1-prob)*100:.1f}%)")
