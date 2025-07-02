import cv2, joblib, mediapipe as mp, numpy as np

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

# Carga modelo y encoder
data = joblib.load("models/modelo_svm.pkl")
clf, le = data["model"], data["le"]

cap = cv2.VideoCapture(1)
with mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7) as hands:
    while True:
        ret, frame = cap.read()
        if not ret: break
        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = hands.process(rgb)
        if res.multi_hand_landmarks:
            lm = res.multi_hand_landmarks[0]
            pts = np.array([[p.x, p.y, p.z] for p in lm.landmark]).flatten()
            # normalizar
            pts = pts.reshape(-1,3)
            pts -= pts[0]
            maxd = np.max(np.linalg.norm(pts,axis=1))
            pts = (pts/maxd if maxd>0 else pts).flatten().reshape(1,-1)
            pred = clf.predict(pts)[0]
            letra = le.inverse_transform([pred])[0]
            cv2.putText(frame, f"Letra: {letra}", (10,40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,0,255),2)
        cv2.imshow("Reconocimiento en vivo", frame)
        if cv2.waitKey(1)==27: break
cap.release()
cv2.destroyAllWindows()
