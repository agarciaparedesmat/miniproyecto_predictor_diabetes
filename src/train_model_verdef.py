import os
import json
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
    confusion_matrix,
    f1_score
)
from xgboost import XGBClassifier
from imblearn.combine import SMOTETomek

# ===============================
# CONFIGURACIÓN GENERAL
# ===============================
DATA_PATH = "../data/diabetes.csv"
MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
MODEL_PATH = os.path.join(MODEL_DIR, "model.pkl")
SCALER_PATH = os.path.join(MODEL_DIR, "scaler.pkl")
METRICS_PATH = os.path.join(MODEL_DIR, "metrics.json")
HISTORY_PATH = os.path.join(MODEL_DIR, "metrics_history.json")
os.makedirs(MODEL_DIR, exist_ok=True)

print("[INFO] Entrenando el modelo predictivo de diabetes...")

# ===============================
# 1. CARGAR Y LIMPIAR DATOS
# ===============================
df = pd.read_csv(DATA_PATH)
cols_zero = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
for col in cols_zero:
    df[col] = df[col].replace(0, df[col].median())

X = df.drop("Outcome", axis=1)
y = df["Outcome"]

# ===============================
# 2. ESCALAR VARIABLES
# ===============================
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ===============================
# 3. DIVIDIR ENTRENAMIENTO / TEST
# ===============================
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

# ===============================
# 4. BALANCEO AVANZADO
# ===============================
sm = SMOTETomek(random_state=42)
X_train_res, y_train_res = sm.fit_resample(X_train, y_train)

print(f"[INFO] Tamaño original: {X_train.shape}, Positivos={sum(y_train)}")
print(f"[INFO] Tamaño balanceado: {X_train_res.shape}, Positivos={sum(y_train_res)}")

# ===============================
# 5. PESO AUTOMÁTICO PARA CASOS POSITIVOS
# ===============================
pos_weight = len(y_train_res[y_train_res == 0]) / len(y_train_res[y_train_res == 1])

# ===============================
# 6. MODELO XGBOOST
# ===============================
xgb = XGBClassifier(
    eval_metric="logloss",
    random_state=42,
    n_jobs=-1,
    scale_pos_weight=pos_weight,
    use_label_encoder=False
)

# ===============================
# 7. GRIDSEARCH ORIENTADO A F1-SCORE
# ===============================
param_grid = {
    "n_estimators": [300, 400],
    "max_depth": [3, 5],
    "learning_rate": [0.01, 0.03, 0.05],
    "subsample": [0.8, 1],
    "colsample_bytree": [0.8, 1],
    "gamma": [0, 0.1]
}

grid = GridSearchCV(
    estimator=xgb,
    param_grid=param_grid,
    scoring="f1",
    cv=5,
    n_jobs=-1,
    verbose=1
)

grid.fit(X_train_res, y_train_res)
print(f"[INFO] Mejores parámetros: {grid.best_params_}")
best_model = grid.best_estimator_

# ===============================
# 8. EVALUAR MODELO FINAL
# ===============================
y_pred = best_model.predict(X_test)
y_proba = best_model.predict_proba(X_test)[:, 1]

acc = accuracy_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_proba)
f1 = f1_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print(f"✅ Precisión: {acc:.2%}")
print(f"✅ ROC-AUC: {roc_auc:.2%}")
print(f"✅ F1-score: {f1:.2%}")
print("\nReporte de clasificación:\n", report)

# ===============================
# 9. AJUSTE AUTOMÁTICO DE THRESHOLD
# ===============================
precision, recall, thresholds = precision_recall_curve(y_test, y_proba)
f1_scores = 2 * (precision * recall) / (precision + recall)
best_threshold = thresholds[np.argmax(f1_scores)]
print(f"[INFO] Umbral óptimo para predicción: {best_threshold:.3f}")

# ===============================
# 10. GUARDAR MÉTRICAS
# ===============================
metrics = {
    "accuracy": acc,
    "roc_auc": roc_auc,
    "f1_score": f1,
    "best_threshold": float(best_threshold),
    "best_params": grid.best_params_
}

with open(METRICS_PATH, "w") as f:
    json.dump(metrics, f, indent=4)

# Histórico de métricas
history = []
if os.path.exists(HISTORY_PATH):
    with open(HISTORY_PATH, "r") as f:
        history = json.load(f)
history.append(metrics)
with open(HISTORY_PATH, "w") as f:
    json.dump(history, f, indent=4)

# ===============================
# 11. VISUALIZACIONES AVANZADAS
# ===============================

# Matriz de confusión
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["No Diabetes", "Diabetes"],
            yticklabels=["No Diabetes", "Diabetes"])
plt.xlabel("Predicción")
plt.ylabel("Real")
plt.title("Matriz de Confusión")
plt.tight_layout()
plt.savefig(os.path.join(MODEL_DIR, "confusion_matrix.png"))
plt.close()

# Curva ROC
fpr, tpr, _ = roc_curve(y_test, y_proba)
plt.figure(figsize=(6, 4))
plt.plot(fpr, tpr, color="blue", label=f"ROC AUC = {roc_auc:.2f}")
plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
plt.xlabel("Tasa de Falsos Positivos")
plt.ylabel("Tasa de Verdaderos Positivos")
plt.title("Curva ROC")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(MODEL_DIR, "roc_curve.png"))
plt.close()

# Curva Precision-Recall
plt.figure(figsize=(6, 4))
plt.plot(recall, precision, color="green")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Curva Precision-Recall")
plt.tight_layout()
plt.savefig(os.path.join(MODEL_DIR, "precision_recall.png"))
plt.close()

# ===============================
# 12. GUARDAR MODELO Y SCALER
# ===============================
joblib.dump(best_model, MODEL_PATH)
joblib.dump(scaler, SCALER_PATH)

print("\n🎯 Entrenamiento completado con éxito 🎯")
print(f"[INFO] Modelo: {MODEL_PATH}")
print(f"[INFO] Métricas: {METRICS_PATH}")
print(f"[INFO] Histórico: {HISTORY_PATH}")
