# src/train_model.py

import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score
import os

def entrenar(csv_path, modelo_salida):
    df = pd.read_csv(csv_path)
    X = df.iloc[:, :-1].values
    y = df["label"].values

    # codificar letras a números
    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_enc, test_size=0.2, random_state=42, stratify=y_enc
    )

    clf = SVC(kernel="linear", probability=True)
    print("🔄 Entrenando SVM…")
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"📊 Precisión en test: {acc:.4f}")
    print(classification_report(y_test, y_pred, target_names=le.classes_))

    os.makedirs(os.path.dirname(modelo_salida), exist_ok=True)
    joblib.dump({"model": clf, "le": le}, modelo_salida)
    print(f"✅ Modelo guardado en: {modelo_salida}")

if __name__ == "__main__":
    entrenar(csv_path="data/landmarks.csv", modelo_salida="models/modelo_svm.pkl")
