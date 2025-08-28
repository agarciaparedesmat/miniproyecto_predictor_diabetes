import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px

# === Configuración general ===
st.set_page_config(
    page_title="🩺 Predictor de Diabetes",
    page_icon="🩺",
    layout="wide"
)

# === Cargar modelo y dataset ===
MODEL_PATH = "../results/model.pkl"
DATASET_PATH = "../data/diabetes.csv"

model = joblib.load(MODEL_PATH)
df = pd.read_csv(DATASET_PATH)

# === Estado de sesión para historial ===
if "history" not in st.session_state:
    st.session_state.history = []

# === Sidebar ===
st.sidebar.title("⚙️ Opciones")
page = st.sidebar.radio("Navegación", ["📊 Exploración de datos", "🤖 Predicción", "📜 Historial"])

# === Función de predicción ===
def predict_diabetes(input_data):
    prediction = model.predict([input_data])[0]
    return prediction

# === Página 1: Exploración de datos ===
if page == "📊 Exploración de datos":
    st.title("📊 Análisis Exploratorio de Datos")
    st.markdown("Explora el dataset y descubre patrones entre las variables.")

    col1, col2 = st.columns(2)
    with col1:
        fig = px.histogram(df, x="Glucose", color="Outcome", nbins=30, title="Distribución de glucosa")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig = px.box(df, y="BMI", color="Outcome", title="IMC según diagnóstico")
        st.plotly_chart(fig, use_container_width=True)

    corr = df.corr()
    fig = px.imshow(corr, text_auto=True, title="Matriz de correlaciones")
    st.plotly_chart(fig, use_container_width=True)

# === Página 2: Predicción ===
elif page == "🤖 Predicción":
    st.title("🤖 Predictor de Diabetes")
    st.markdown("Introduce los valores del paciente y predice el riesgo.")

    col1, col2, col3 = st.columns(3)
    pregnancies = col1.number_input("Embarazos", min_value=0, max_value=20, value=1)
    glucose = col1.number_input("Glucosa", min_value=0, max_value=200, value=80)
    blood_pressure = col1.number_input("Presión arterial", min_value=0, max_value=140, value=70)
    skin_thickness = col2.number_input("Espesor piel", min_value=0, max_value=100, value=20)
    insulin = col2.number_input("Insulina", min_value=0, max_value=900, value=80)
    bmi = col2.number_input("IMC", min_value=0.0, max_value=70.0, value=25.0)
    dpf = col3.number_input("Función pedigrí diabetes", min_value=0.0, max_value=3.0, value=0.5)
    age = col3.number_input("Edad", min_value=0, max_value=120, value=30)

    if st.button("🔍 Predecir"):
        input_data = [pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, dpf, age]
        result = predict_diabetes(input_data)
        
        # Guardar en historial
        st.session_state.history.append(input_data + [result])

        if result == 0:
            st.success("✅ Bajo riesgo de diabetes")
        else:
            st.error("🚨 Posible diabetes detectada")

# === Página 3: Historial ===
elif page == "📜 Historial":
    st.title("📜 Historial de predicciones")
    if len(st.session_state.history) > 0:
        history_df = pd.DataFrame(st.session_state.history, columns=[
            "Embarazos","Glucosa","Presión","Piel","Insulina","IMC","DPF","Edad","Predicción"
        ])
        st.dataframe(history_df, use_container_width=True)
        st.download_button("⬇️ Descargar CSV", history_df.to_csv(index=False), "historial_predicciones.csv")
    else:
        st.info("No hay predicciones registradas todavía.")
