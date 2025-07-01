import cv2
import os
import mediapipe as mp
from collections import defaultdict

# Inicializa MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False,
                       max_num_hands=1,
                       min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# Carpeta donde guardarás las imágenes
DATA_DIR = os.path.join("data", "raw")
os.makedirs(DATA_DIR, exist_ok=True)

# Inicializa cámara
cap = cv2.VideoCapture(0)

# Conteo por letra
letter_counts = defaultdict(int)

print("Presiona una tecla (A-Z) para capturar. ESC para salir.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Voltea la imagen para que no esté en espejo
    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape

    # Convierte a RGB para MediaPipe
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    # Dibuja la mano si se detecta
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    # Muestra conteos por letra
    y = 30
    for letra, count in sorted(letter_counts.items()):
        cv2.putText(frame, f"{letra}: {count}", (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        y += 20

    cv2.imshow("Captura de alfabeto LSM", frame)

    key = cv2.waitKey(1)

    if key == 27:  # ESC para salir
        break
    elif 65 <= key <= 90 or 97 <= key <= 122:  # Letras A-Z (mayúsculas y minúsculas)
        letra = chr(key).upper()

        # Crea carpeta de la letra si no existe
        letra_path = os.path.join(DATA_DIR, letra)
        os.makedirs(letra_path, exist_ok=True)

        # Guarda imagen
        filename = f"{letter_counts[letra]:04d}.jpg"
        cv2.imwrite(os.path.join(letra_path, filename), frame)
        letter_counts[letra] += 1
        print(f"[✓] Capturada imagen para letra '{letra}'")

# Libera recursos
cap.release()
cv2.destroyAllWindows()
