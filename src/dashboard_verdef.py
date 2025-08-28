import os
import json
import joblib
import numpy as np
import pandas as pd
import shap
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components

# ===============================
# CONFIGURACIÓN INICIAL
# ===============================
st.set_page_config(
    page_title="Dashboard Predictor Diabetes",
    page_icon="🩺",
    layout="wide"
)

MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
MODEL_PATH = os.path.join(MODEL_DIR, "model.pkl")
SCALER_PATH = os.path.join(MODEL_DIR, "scaler.pkl")
METRICS_PATH = os.path.join(MODEL_DIR, "metrics.json")
HISTORY_PATH = os.path.join(MODEL_DIR, "metrics_history.json")
DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "diabetes.csv")

# ===============================
# CARGAR DATOS Y MODELO
# ===============================
@st.cache_resource
def load_resources():
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    with open(METRICS_PATH, "r") as f:
        metrics = json.load(f)
    with open(HISTORY_PATH, "r") as f:
        history = json.load(f)
    df = pd.read_csv(DATA_PATH)
    return model, scaler, metrics, history, df

model, scaler, metrics, history, df = load_resources()

# ===============================
# SIDEBAR
# ===============================
st.sidebar.title("🩺 Dashboard de Diabetes")
menu = st.sidebar.radio("Navegación", ["📊 Métricas generales", "🔍 Predicción individual", "📈 Interpretabilidad (SHAP)"])

# ===============================
# 1. MÉTRICAS GENERALES
# ===============================
if menu == "📊 Métricas generales":
    st.title("📊 Métricas del modelo")
    col1, col2, col3 = st.columns(3)
    col1.metric("Accuracy", f"{metrics['accuracy']*100:.2f}%")
    col2.metric("ROC-AUC", f"{metrics['roc_auc']*100:.2f}%")
    col3.metric("F1-score", f"{metrics['f1_score']*100:.2f}%")

    # Histórico de métricas
    st.subheader("📈 Evolución histórica del modelo")
    hist_df = pd.DataFrame(history)
    fig_hist = px.line(
        hist_df,
        y=["accuracy", "f1_score", "roc_auc"],
        labels={"value": "Métrica", "index": "Entrenamiento"},
        title="Histórico de Accuracy, F1 y ROC-AUC",
        markers=True
    )
    st.plotly_chart(fig_hist, use_container_width=True)

    # Curva ROC
    st.subheader("📌 Curva ROC")
    roc_path = os.path.join(MODEL_DIR, "roc_curve.png")
    st.image(roc_path, caption="Curva ROC")

    # Curva Precision-Recall
    st.subheader("📌 Curva Precision-Recall")
    pr_path = os.path.join(MODEL_DIR, "precision_recall.png")
    st.image(pr_path, caption="Curva Precision-Recall")

    # Matriz de confusión
    st.subheader("📌 Matriz de confusión")
    cm_path = os.path.join(MODEL_DIR, "confusion_matrix.png")
    st.image(cm_path, caption="Matriz de confusión")

# ===============================
# 2. PREDICCIÓN INDIVIDUAL
# ===============================
elif menu == "🔍 Predicción individual":
    st.title("🔍 Predicción de diabetes para un paciente")
    st.markdown("Ajusta los valores con los deslizadores y pulsa **Predecir**.")

    # Sliders
    cols = st.columns(4)
    Pregnancies = cols[0].slider("👶 Embarazos", 0, 15, 1)
    Glucose = cols[1].slider("🍬 Glucosa", 50, 200, 100)
    BloodPressure = cols[2].slider("🩸 Presión arterial", 40, 120, 70)
    SkinThickness = cols[3].slider("📏 Grosor piel", 0, 80, 20)
    Insulin = cols[0].slider("💉 Insulina", 0, 800, 100)
    BMI = cols[1].slider("⚖️ IMC", 15.0, 50.0, 25.0)
    DiabetesPedigree = cols[2].slider("📐 Pedigree diabetes", 0.0, 2.5, 0.5)
    Age = cols[3].slider("🎂 Edad", 18, 90, 30)

    input_data = np.array([[Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigree, Age]])
    input_scaled = scaler.transform(input_data)

    if st.button("🔍 Predecir"):
        proba = model.predict_proba(input_scaled)[0][1]
        pred = int(proba >= metrics["best_threshold"])
        if pred == 1:
            st.error(f"⚠️ Alto riesgo de diabetes — Probabilidad: {proba:.2%}")
        else:
            st.success(f"✅ Bajo riesgo de diabetes — Probabilidad: {proba:.2%}")

# ===============================
# 3. INTERPRETABILIDAD (SHAP)
# ===============================
elif menu == "📈 Interpretabilidad (SHAP)":
    st.title("📈 Explicabilidad del modelo con SHAP")
    X = df.drop("Outcome", axis=1)
    X_scaled = scaler.transform(X)
    explainer = shap.Explainer(model, X_scaled)
    shap_values = explainer(X_scaled)

    # Importancia global
    st.subheader("🌐 Impacto global de las variables")
    shap.summary_plot(shap_values, X, show=False)
    st.pyplot(bbox_inches="tight")

    # Análisis individual
    st.subheader("🔍 Análisis individual")
    idx = st.slider("Selecciona paciente", 0, len(df)-1, 0)
    force_plot = shap.force_plot(
        explainer.expected_value,
        shap_values[idx].values,
        X.iloc[idx],
        matplotlib=False
    )
    components.html(shap.getjs(), height=0)
    components.html(force_plot.html(), height=300)
