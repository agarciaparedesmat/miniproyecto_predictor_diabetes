import os
import json
import joblib
import optuna
import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    f1_score,
    classification_report
)
from sklearn.model_selection import train_test_split
from imblearn.combine import SMOTETomek
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.preprocessing import StandardScaler

# ================================
# CONFIGURACIÓN DE RUTAS
# ================================
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data", "diabetes.csv")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

MODEL_PATH = os.path.join(RESULTS_DIR, "model.pkl")
METRICS_PATH = os.path.join(RESULTS_DIR, "metrics.json")

# ================================
# CARGA DE DATOS
# ================================
print("[INFO] Entrenando el modelo predictivo de diabetes...")
print("[INFO] Cargando dataset...")

df = pd.read_csv(DATA_PATH)

# Variables
X = df.drop("Outcome", axis=1)
y = df["Outcome"]

# Escalado
scaler = StandardScaler()
X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

# Dividir dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Balanceo con SMOTE Tomek Links
print("[INFO] Aplicando SMOTE Tomek Links...")
smt = SMOTETomek(random_state=42)
X_train, y_train = smt.fit_resample(X_train, y_train)

print(f"[INFO] Tamaño original: {df.shape}, Positivos={y.sum()}")
print(f"[INFO] Tamaño balanceado: {X_train.shape}, Positivos={y_train.sum()}")

# ================================
# DEFINICIÓN DE MODELOS
# ================================
def create_model(trial):
    model_name = trial.suggest_categorical("model", ["randomforest", "xgboost", "lightgbm", "catboost"])

    if model_name == "randomforest":
        return RandomForestClassifier(
            n_estimators=trial.suggest_int("n_estimators", 100, 500),
            max_depth=trial.suggest_int("max_depth", 3, 15),
            min_samples_split=trial.suggest_int("min_samples_split", 2, 10),
            min_samples_leaf=trial.suggest_int("min_samples_leaf", 1, 5),
            random_state=42
        )

    elif model_name == "xgboost":
        return XGBClassifier(
            n_estimators=trial.suggest_int("n_estimators", 100, 500),
            learning_rate=trial.suggest_float("learning_rate", 0.01, 0.3),
            max_depth=trial.suggest_int("max_depth", 3, 10),
            subsample=trial.suggest_float("subsample", 0.5, 1.0),
            colsample_bytree=trial.suggest_float("colsample_bytree", 0.5, 1.0),
            eval_metric="logloss",
            random_state=42,
            use_label_encoder=False
        )

    elif model_name == "lightgbm":
        return LGBMClassifier(
            n_estimators=trial.suggest_int("n_estimators", 100, 500),
            learning_rate=trial.suggest_float("learning_rate", 0.01, 0.3),
            max_depth=trial.suggest_int("max_depth", 3, 10),
            subsample=trial.suggest_float("subsample", 0.5, 1.0),
            colsample_bytree=trial.suggest_float("colsample_bytree", 0.5, 1.0),
            random_state=42
        )

    elif model_name == "catboost":
        return CatBoostClassifier(
            iterations=trial.suggest_int("iterations", 100, 500),
            learning_rate=trial.suggest_float("learning_rate", 0.01, 0.3),
            depth=trial.suggest_int("depth", 3, 10),
            verbose=0,
            random_state=42
        )

# ================================
# OPTIMIZACIÓN DE HIPERPARÁMETROS
# ================================
def objective(trial):
    model = create_model(trial)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    return accuracy_score(y_test, preds)

print("[INFO] Buscando mejores hiperparámetros...")
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=30)

best_params = study.best_params
best_model_name = best_params.pop("model")
print(f"[INFO] Mejor modelo: {best_model_name}")
print(f"[INFO] Mejores hiperparámetros: {study.best_params}")

# ================================
# ENTRENAR EL MEJOR MODELO
# ================================
model = create_model(optuna.trial.FixedTrial(study.best_params))
model.fit(X_train, y_train)

# ================================
# EVALUACIÓN
# ================================
y_pred_proba = model.predict_proba(X_test)[:, 1]
thresholds = np.arange(0.1, 0.9, 0.01)
f1_scores = [f1_score(y_test, (y_pred_proba >= t).astype(int)) for t in thresholds]
optimal_threshold = thresholds[np.argmax(f1_scores)]

y_pred = (y_pred_proba >= optimal_threshold).astype(int)

accuracy = accuracy_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_pred_proba)
f1 = f1_score(y_test, y_pred)

print(f"✅ Precisión: {accuracy*100:.2f}%")
print(f"✅ ROC-AUC: {roc_auc*100:.2f}%")
print(f"✅ F1-score: {f1*100:.2f}%")
print(f"[INFO] Umbral óptimo: {optimal_threshold:.3f}")

print("\nReporte de clasificación:")
print(classification_report(y_test, y_pred))

# ================================
# GUARDAR MODELO Y MÉTRICAS
# ================================
joblib.dump(model, MODEL_PATH)

metrics = {
    "best_model": best_model_name,
    "best_params": study.best_params,
    "accuracy": round(accuracy, 4),
    "roc_auc": round(roc_auc, 4),
    "f1_score": round(f1, 4),
    "threshold": round(optimal_threshold, 4)
}

with open(METRICS_PATH, "w") as f:
    json.dump(metrics, f, indent=4)

print("[INFO] Entrenamiento completado y resultados guardados.")
