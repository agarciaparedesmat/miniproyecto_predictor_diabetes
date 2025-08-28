import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib
import json
import os

# === 1. Cargar dataset ===
df = pd.read_csv("../data/diabetes.csv")

# === 2. Dividir variables ===
X = df.drop("Outcome", axis=1)
y = df["Outcome"]

# === 3. División en entrenamiento y test ===
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# === 4. Escalado ===
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# === 5. Entrenar modelo ===
model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

# === 6. Evaluar ===
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred, output_dict=True)

print(f"Accuracy: {acc:.2f}")

# === 7. Guardar modelo y métricas ===
os.makedirs("../results", exist_ok=True)
joblib.dump(model, "../results/model.pkl")

with open("../results/metrics.json", "w") as f:
    json.dump({"accuracy": acc, "report": report}, f, indent=4)

print("✅ Modelo entrenado y guardado en /results")
