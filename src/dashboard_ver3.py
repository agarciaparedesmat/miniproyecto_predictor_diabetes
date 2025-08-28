import streamlit as st
import pandas as pd
import joblib
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import os

# === CONFIGURACIÓN GENERAL ===
st.set_page_config(
    page_title="🩺 Predictor de Diabetes",
    page_icon="🧬",
    layout="wide"
)

# === CARGA DEL MODELO ===
MODEL_PATH = "../results/model.pkl"
DATA_PATH = "../data/diabetes.csv"

if not os.path.exists(MODEL_PATH):
    st.error("❌ No se encontró el modelo entrenado. Ejecuta primero train_model.py.")
    st.stop()

if not os.path.exists(DATA_PATH):
    st.error("❌ No se encontró el dataset en ../data/diabetes.csv.")
    st.stop()

model = joblib.load(MODEL_PATH)
df = pd.read_csv(DATA_PATH)

# === ESTADO DE SESIÓN ===
if "history" not in st.session_state:
    st.session_state.history = []

# === SIDEBAR ===
st.sidebar.title("⚙️ Navegación")
page = st.sidebar.radio("Selecciona una sección:", ["📊 Exploración de datos", "🤖 Predicción", "📜 Historial"])

# ==================================================================================
# PÁGINA 1: EXPLORACIÓN DE DATOS
# ==================================================================================
if page == "📊 Exploración de datos":
    st.title("📊 Exploración del Dataset")
    st.markdown("Visualiza patrones, distribuciones y correlaciones entre variables.")

    # Filtro dinámico
    variable = st.selectbox("Selecciona variable para analizar:", df.columns[:-1])

    # Histograma interactivo
    fig = px.histogram(df, x=variable, color="Outcome",
                       title=f"Distribución de {variable} por Diagnóstico",
                       barmode="overlay", nbins=30)
    st.plotly_chart(fig, use_container_width=True)

    # Boxplot comparativo
    fig = px.box(df, y=variable, color="Outcome",
                 title=f"Boxplot de {variable} según diagnóstico")
    st.plotly_chart(fig, use_container_width=True)

    # Mapa de calor de correlaciones
    corr = df.corr()
    fig = px.imshow(corr,
                    text_auto=True,
                    color_continuous_scale="RdBu_r",
                    title="Mapa de calor de correlaciones")
    st.plotly_chart(fig, use_container_width=True)

# ==================================================================================
# PÁGINA 2: PREDICCIÓN
# ==================================================================================
elif page == "🤖 Predicción":
    st.title("🤖 Predicción de Diabetes")
    st.markdown("Ajusta los valores del paciente y observa en tiempo real el riesgo estimado.")

    # Entradas con deslizadores
    col1, col2, col3 = st.columns(3)
    pregnancies = col1.slider("Embarazos", 0, 20, 1)
    glucose = col1.slider("Glucosa", 0, 200, 80)
    blood_pressure = col1.slider("Presión arterial", 0, 140, 70)
    skin_thickness = col2.slider("Espesor piel", 0, 100, 20)
    insulin = col2.slider("Insulina", 0, 900, 80)
    bmi = col2.slider("IMC", 0.0, 70.0, 25.0)
    dpf = col3.slider("Función pedigrí diabetes", 0.0, 3.0, 0.5)
    age = col3.slider("Edad", 0, 120, 30)

    # Preparar datos para predicción
    input_data = [pregnancies, glucose, blood_pressure, skin_thickness,
                  insulin, bmi, dpf, age]
    prediction = model.predict([input_data])[0]
    probability = model.predict_proba([input_data])[0][1]

    # Panel de resultado visual dinámico
    st.subheader("📌 Resultado")
    if probability < 0.3:
        st.success(f"🟢 Bajo riesgo ({probability:.1%})")
        gauge_color = "green"
    elif probability < 0.6:
        st.warning(f"🟡 Riesgo moderado ({probability:.1%})")
        gauge_color = "orange"
    else:
        st.error(f"🔴 Alto riesgo ({probability:.1%})")
        gauge_color = "red"

    # Gauge Chart (indicador circular)
    fig_gauge = go.Figure(go.Indicator(
        mode="gauge+number",
        value=probability * 100,
        title={"text": "Probabilidad de Diabetes (%)"},
        gauge={
            "axis": {"range": [0, 100]},
            "bar": {"color": gauge_color},
            "steps": [
                {"range": [0, 30], "color": "lightgreen"},
                {"range": [30, 60], "color": "orange"},
                {"range": [60, 100], "color": "red"}
            ]
        }
    ))
    st.plotly_chart(fig_gauge, use_container_width=True)

    # Comparativa paciente vs media del dataset
    st.subheader("📊 Comparativa con la media del dataset")
    dataset_means = df.mean()

    comparison_df = pd.DataFrame({
        "Variable": ["Glucosa", "IMC", "Edad"],
        "Paciente": [glucose, bmi, age],
        "Media Dataset": [dataset_means["Glucose"], dataset_means["BMI"], dataset_means["Age"]]
    })

    fig_comparison = px.bar(comparison_df,
                            x="Variable",
                            y=["Paciente", "Media Dataset"],
                            barmode="group",
                            title="Comparativa paciente vs media del dataset")
    st.plotly_chart(fig_comparison, use_container_width=True)

    # Guardar en historial
    st.session_state.history.append(input_data + [prediction])

    # Métricas clave
    col_a, col_b, col_c = st.columns(3)
    col_a.metric("Glucosa", glucose, f"{glucose - dataset_means['Glucose']:.1f}")
    col_b.metric("IMC", bmi, f"{bmi - dataset_means['BMI']:.1f}")
    col_c.metric("Edad", age, f"{age - dataset_means['Age']:.1f}")

# ==================================================================================
# PÁGINA 3: HISTORIAL DE PREDICCIONES
# ==================================================================================
elif page == "📜 Historial":
    st.title("📜 Historial de Predicciones")
    if len(st.session_state.history) > 0:
        history_df = pd.DataFrame(st.session_state.history, columns=[
            "Embarazos","Glucosa","Presión","Piel","Insulina","IMC","DPF","Edad","Predicción"
        ])
        st.dataframe(history_df, use_container_width=True)

        # Botón de descarga
        st.download_button("⬇️ Descargar historial", history_df.to_csv(index=False),
                           file_name="historial_predicciones.csv", mime="text/csv")

        # Gráfico dinámico de predicciones
        fig = px.histogram(history_df, x="Glucosa", color="Predicción",
                           title="Distribución de glucosa según predicciones")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Aún no se han registrado predicciones.")

