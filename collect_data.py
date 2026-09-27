import cv2
import mediapipe as mp
import numpy as np
import csv
import os

# Initialisation MediaPipe
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=True,
    max_num_hands=2,
    min_detection_confidence=0.1,
    min_tracking_confidence=0.5
)

# Configuration du dataset
CSV_PATH = "naruto_dataset.csv"
SIGNS = ["Tigre", "Serpent", "Dragon", "Oiseau"]  # Ajoute tes signes ici
CURRENT_LABEL = 0  # Index du signe en cours d'enregistrement

def extract_landmarks(results):
    """Extrait et normalise 126 coordonnees (2 mains x 21 points x 3)"""
    landmarks_data = np.zeros(126, dtype=np.float32)
    
    if not results.multi_hand_landmarks:
        return landmarks_data, False

    # Tri pour garder une coherence : main gauche d'abord, main droite ensuite
    hand_list = []
    for idx, hand_handedness in enumerate(results.multi_handedness):
        hand_type = hand_handedness.classification[0].label  # "Left" ou "Right"
        hand_landmarks = results.multi_hand_landmarks[idx]
        hand_list.append((hand_type, hand_landmarks))
    
    # Ordonner : Left puis Right
    hand_list.sort(key=lambda x: x[0], reverse=True)

    for i, (_, hand_landmarks) in enumerate(hand_list[:2]):
        coords = []
        # Normalisation par rapport au poignet (point 0)
        base_x = hand_landmarks.landmark[0].x
        base_y = hand_landmarks.landmark[0].y
        base_z = hand_landmarks.landmark[0].z

        for lm in hand_landmarks.landmark:
            coords.extend([lm.x - base_x, lm.y - base_y, lm.z - base_z])
            
        start_idx = i * 63
        landmarks_data[start_idx:start_idx + 63] = coords

    return landmarks_data, True

# Creation du CSV si inexistant
if not os.path.exists(CSV_PATH):
    with open(CSV_PATH, "w", newline="") as f:
        writer = csv.writer(f)
        header = [f"coord_{i}" for i in range(126)] + ["label"]
        writer.writerow(header)

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
saving = False
samples_count = 0

print("Controles :")
print("  ESPACE : Activer/Desactiver l'enregistrement continu")
print("  TAB    : Changer de signe cible")
print("  ECHAP  : Quitter") 

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)  # Effet miroir
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    landmarks, detected = extract_landmarks(results)

    # Sauvegarde automatique si le mode enregistrement est actif
    if saving and detected:
        with open(CSV_PATH, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(list(landmarks) + [CURRENT_LABEL])
        samples_count += 1

    # Affichage UI
    status_text = f"Signe: {SIGNS[CURRENT_LABEL]} (ID: {CURRENT_LABEL}) | Echantillons: {samples_count}"
    rec_status = "ENREGISTREMENT..." if saving else "EN PAUSE"
    color = (0, 0, 255) if saving else (0, 255, 0)

    cv2.putText(frame, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(frame, rec_status, (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    cv2.imshow("Collecte Dataset Naruto", frame)
    key = cv2.waitKey(1) & 0xFF

    if key == 27:  # Echap
        break
    elif key == 32:  # Espace
        saving = not saving
    elif key == 9:   # Tab
        CURRENT_LABEL = (CURRENT_LABEL + 1) % len(SIGNS)
        samples_count = 0

cap.release()
cv2.destroyAllWindows()
hands.close()

