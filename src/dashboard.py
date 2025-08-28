import os
import json
import joblib
import shap
import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import roc_curve, precision_recall_curve

# ============================================
# CONFIGURACIÓN DE RUTAS
# ============================================
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
RESULTS_DIR = os.path.join(BASE_DIR, "results")
MODEL_PATH = os.path.join(RESULTS_DIR, "model.pkl")
METRICS_PATH = os.path.join(RESULTS_DIR, "metrics.json")
PDF_PATH = os.path.join(RESULTS_DIR, "dashboard_report.pdf")

# ============================================
# CONFIG STREAMLIT
# ============================================
st.set_page_config(page_title="📊 Dashboard Diabetes", layout="wide")
st.title("📊 Dashboard Avanzado - Predicción de Diabetes")

# ============================================
# FUNCIONES AUXILIARES
# ============================================
@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        st.error("❌ No se encontró el modelo entrenado. Ejecuta primero `train_model.py`.")
        st.stop()
    return joblib.load(MODEL_PATH)

@st.cache_resource
def load_metrics():
    if not os.path.exists(METRICS_PATH):
        st.error("❌ No se encontró `metrics.json`. Vuelve a entrenar el modelo.")
        st.stop()
    with open(METRICS_PATH, "r") as f:
        return json.load(f)

def plot_comparativa(metrics):
    df = pd.DataFrame([metrics])
    df = df[["accuracy", "roc_auc", "f1_score"]].T
    df.columns = ["Puntuación"]
    st.bar_chart(df)

def generar_pdf(metrics):
    from reportlab.lib.pagesizes import A4
    from reportlab.pdfgen import canvas

    c = canvas.Canvas(PDF_PATH, pagesize=A4)
    c.setFont("Helvetica-Bold", 16)
    c.drawString(180, 800, "📊 Informe Dashboard Diabetes")

    c.setFont("Helvetica", 12)
    y = 760
    for key, value in metrics.items():
        c.drawString(80, y, f"{key}: {value}")
        y -= 20

    c.save()
    st.success(f"📄 Informe generado: {PDF_PATH}")

# ============================================
# CARGA DE MODELO Y MÉTRICAS
# ============================================
model = load_model()
metrics = load_metrics()

st.sidebar.header("📌 Opciones")
menu = st.sidebar.radio("Selecciona una vista:", ["🏠 Resumen", "🔍 Predicción personalizada", "📈 Interpretabilidad", "📄 Informe PDF"])

# ============================================
# PESTAÑA 1 - RESUMEN
# ============================================
if menu == "🏠 Resumen":
    st.subheader("📊 Comparativa de modelos")
    st.metric("Mejor modelo", metrics["best_model"])
    st.metric("Precisión", f"{metrics['accuracy']*100:.2f}%")
    st.metric("ROC-AUC", f"{metrics['roc_auc']*100:.2f}%")
    st.metric("F1-score", f"{metrics['f1_score']*100:.2f}%")
    plot_comparativa(metrics)

# ============================================
# PESTAÑA 2 - PREDICCIÓN PERSONALIZADA
# ============================================
elif menu == "🔍 Predicción personalizada":
    st.subheader("Introduce los datos del paciente:")

    # Si el modelo tiene columnas, las usamos para los sliders
    if hasattr(model, "feature_names_in_"):
        columnas = model.feature_names_in_
    else:
        columnas = ["Pregnancies","Glucose","BloodPressure","SkinThickness","Insulin","BMI","DiabetesPedigreeFunction","Age"]

    valores = []
    for col in columnas:
        valores.append(
            st.slider(col, 0.0, 200.0, 100.0)
        )

    if st.button("🔍 Predecir"):
        proba = model.predict_proba([valores])[0][1]
        resultado = "POSITIVO" if proba >= metrics["threshold"] else "NEGATIVO"

        st.success(f"**Resultado:** {resultado}")
        st.info(f"**Probabilidad estimada:** {proba*100:.2f}%")

# ============================================
# PESTAÑA 3 - INTERPRETABILIDAD SHAP
# ============================================
elif menu == "📈 Interpretabilidad":
    st.subheader("🔎 Interpretabilidad del modelo con SHAP")

    # Crear explainer
    explainer = shap.Explainer(model)
    shap_values = explainer.shap_values(model.predict_proba)

    st.write("### Impacto global de cada variable")
    fig, ax = plt.subplots()
    shap.summary_plot(shap_values, plot_type="bar", show=False)
    st.pyplot(fig)

# ============================================
# PESTAÑA 4 - INFORME PDF
# ============================================
elif menu == "📄 Informe PDF":
    st.subheader("📄 Generar informe PDF del modelo")
    if st.button("Generar informe PDF"):
        generar_pdf(metrics)

