
# 🩺 Predicción de Diabetes con Machine Learning

Este proyecto aplica **Machine Learning** para predecir la probabilidad de que un paciente presente **diabetes** a partir de datos clínicos reales.  
Incluye un modelo entrenado, análisis exploratorio de datos y un **dashboard interactivo**.

---

## 🚀 Demo Online (opcional)
📌 Si el proyecto está desplegado, accede aquí:  
🔗 **[Dashboard Interactivo](https://share.streamlit.io/tu_usuario/diabetes-predictor/src/dashboard.py)**

---

## 📌 Dataset utilizado
Usamos el **Pima Indians Diabetes Dataset**.  
🔗 [Descargar CSV](https://raw.githubusercontent.com/plotly/datasets/master/diabetes.csv)

---

## ⚙️ Instalación rápida

```bash
# Clonar repositorio
git clone https://github.com/TU_USUARIO/diabetes-predictor.git
cd diabetes-predictor

# Crear entorno virtual
python -m venv venv

# Activar entorno
# Windows:
venv\Scripts\activate
# Mac/Linux:
source venv/bin/activate

# Instalar dependencias
pip install -r requirements.txt
```

---

## 🧠 Entrenar el modelo

```bash
cd src
python train_model.py
```

Generará:
- `results/model.pkl` → Modelo entrenado.
- `results/metrics.json` → Métricas del modelo.

---

## 🖥️ Ejecutar el Dashboard

```bash
cd src
streamlit run dashboard.py
```
Abrir en navegador: **http://localhost:8501**

---

## 📊 Resultados principales

- **Modelo inicial:** Random Forest  
- **Precisión aproximada:** 77%  
- **Métricas detalladas:** `results/metrics.json`

---

## 🛠️ Mejoras futuras

- Probar **XGBoost** y **LightGBM**.
- Implementar **SHAP** para explicabilidad.
- Publicar el dashboard en **Streamlit Cloud**.
- Integrar datos clínicos reales.

---

## 👩‍💻 Autor

**Sofía García Guerrero**  
Estudiante de **Ingeniería Biomédica** - Universidad de Alicante  
📧 **Email:** tu_correo@gmail.com  
🔗 **LinkedIn:** https://www.linkedin.com/in/tu-perfil/  
🔗 **GitHub:** https://github.com/tu_usuario
