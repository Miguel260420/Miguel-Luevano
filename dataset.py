import os
import csv
import cv2
import mediapipe as mp
import numpy as np

mp_hands = mp.solutions.hands

def extraer_landmarks_de_imagen(ruta_imagen, min_confidence=0.3):
    """
    Lee una imagen, la procesa con MediaPipe y devuelve
    un vector 63D de landmarks o None si no detecta mano.
    """
    image = cv2.imread(ruta_imagen)
    if image is None:
        return None

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    with mp_hands.Hands(static_image_mode=True,
                        max_num_hands=1,
                        min_detection_confidence=min_confidence) as hands:
        resultados = hands.process(image_rgb)
        if not resultados.multi_hand_landmarks:
            return None

        # Extrae los 21 puntos (x,y,z) en un vector plano
        lm = resultados.multi_hand_landmarks[0]
        puntos = []
        for p in lm.landmark:
            puntos += [p.x, p.y, p.z]
        return np.array(puntos, dtype=np.float32)

def normalizar(landmarks: np.ndarray):
    """
    Centra los landmarks en la muñeca (punto 0) y escala
    para que la distancia máxima sea 1.
    """
    pts = landmarks.reshape(-1, 3)
    # Centro en landmark 0 (muñeca)
    centro = pts[0].copy()
    pts -= centro
    # Escalar según distancia máxima
    max_dist = np.max(np.linalg.norm(pts, axis=1))
    if max_dist > 0:
        pts /= max_dist
    return pts.flatten()

def generar_csv(data_dir: str, output_csv: str):
    """
    Recorre subcarpetas de data_dir (A, B, C, ...) y genera
    un CSV con columnas f1..f63 + label.
    """
    filas = []
    letras = sorted([d for d in os.listdir(data_dir)
                     if os.path.isdir(os.path.join(data_dir, d))])
    for letra in letras:
        carpeta = os.path.join(data_dir, letra)
        for img_nombre in os.listdir(carpeta):
            # Solo extensiones de imagen
            if not img_nombre.lower().endswith((".jpg", ".jpeg", ".png")):
                continue
            ruta = os.path.join(carpeta, img_nombre)
            print(f"Procesando {ruta} ...", end="")
            lm = extraer_landmarks_de_imagen(ruta, min_confidence=0.3)
            if lm is None:
                print(" ❌ mano NO detectada")
                continue
            lm_norm = normalizar(lm)
            filas.append(lm_norm.tolist() + [letra])
            print(" ✅ mano detectada")
    # Escribir CSV
    headers = [f"f{i}" for i in range(1, 64)] + ["label"]
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerows(filas)
    print(f"\n✅ CSV generado: {output_csv} con {len(filas)} filas")

if __name__ == "__main__":
    generar_csv(data_dir="data/raw", output_csv="data/landmarks.csv")
