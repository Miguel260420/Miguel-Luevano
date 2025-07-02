# src/test_landmark.py
import cv2
import mediapipe as mp

# Cambia esta ruta por una de tus imágenes descargadas, por ejemplo:
ruta = "data/raw/A/S18-A-4-0.jpg"

mp_hands = mp.solutions.hands

with mp_hands.Hands(
    static_image_mode=True,
    max_num_hands=1,
    min_detection_confidence=0.3  # muy bajo para pruebas
) as hands:
    img = cv2.imread(ruta)
    if img is None:
        print("❌ No se pudo leer la imagen:", ruta); exit(1)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    res = hands.process(img_rgb)
    if not res.multi_hand_landmarks:
        print("❌ MediaPipe NO detectó mano en", ruta)
    else:
        print("✅ Mano detectada en", ruta)
