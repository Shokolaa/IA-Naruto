import cv2
import mediapipe.python.solutions.hands as mp_hands
import mediapipe.python.solutions.drawing_utils as mp_drawing
import numpy as np
import torch
import torch.nn as nn

# ---------------------------------------------------------
# 1. Architecture du modèle (doit être identique à train.py)
# ---------------------------------------------------------
class NarutoHandSignClassifier(nn.Module):
    def __init__(self, input_dim=126, num_classes=4, dropout_rate=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),

            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),

            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),

            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        return self.net(x)

# ---------------------------------------------------------
# 2. Paramètres & Chargement du Modèle
# ---------------------------------------------------------
# Assure-toi que cette liste correspond EXACTEMENT à l'ordre dans collect_data.py
SIGNS = ["Tigre", "Serpent", "Dragon", "Oiseau"]
CONFIDENCE_THRESHOLD = 0.80  # Seuil minimal de certitude (80%)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
checkpoint = torch.load("naruto_model.pth", map_location=device)

model = NarutoHandSignClassifier(
    input_dim=checkpoint['input_dim'],
    num_classes=checkpoint['num_classes']
).to(device)

model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

print(f"Modèle chargé sur : {device}")

# ---------------------------------------------------------
# 3. Initialisation MediaPipe & Caméra
# ---------------------------------------------------------
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

def extract_landmarks(results):
    """Même fonction d'extraction que pour la collecte"""
    landmarks_data = np.zeros(126, dtype=np.float32)
    if not results.multi_hand_landmarks:
        return landmarks_data, False

    for i, hand_landmarks in enumerate(results.multi_hand_landmarks[:2]):
        base_x = hand_landmarks.landmark[0].x
        base_y = hand_landmarks.landmark[0].y
        base_z = hand_landmarks.landmark[0].z

        coords = []
        for lm in hand_landmarks.landmark:
            coords.extend([lm.x - base_x, lm.y - base_y, lm.z - base_z])
            
        start_idx = i * 63
        landmarks_data[start_idx:start_idx + 63] = coords

    return landmarks_data, True

print("Lancement de la détection ! Appuie sur ECHAP pour quitter.")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)  # Effet miroir
    h, w, _ = frame.shape
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    sign_name = "Aucun"
    confidence = 0.0

    if results.multi_hand_landmarks:
        # Dessiner le squelette
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        # Extraction des coordonnées
        landmarks, detected = extract_landmarks(results)

        if detected:
            # Inférence PyTorch
            tensor_input = torch.tensor(landmarks, dtype=torch.float32).unsqueeze(0).to(device)
            with torch.no_grad():
                logits = model(tensor_input)
                probabilities = torch.softmax(logits, dim=1)[0]
                pred_idx = torch.argmax(probabilities).item()
                confidence = probabilities[pred_idx].item()

            if confidence >= CONFIDENCE_THRESHOLD and pred_idx < len(SIGNS):
                sign_name = SIGNS[pred_idx]

    # Interface HUD
    # Bannière supérieure
    cv2.rectangle(frame, (0, 0), (w, 75), (20, 20, 20), -1)

    # Affichage du signe
    text_color = (0, 255, 0) if sign_name != "Aucun" else (150, 150, 150)
    cv2.putText(frame, f"Signe : {sign_name}", (20, 48), 
                cv2.FONT_HERSHEY_DUPLEX, 1.1, text_color, 2)

    # Jauge de confiance
    cv2.putText(frame, f"{confidence * 100:.1f}%", (w - 120, 48), 
                cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 1)

    bar_width = int(confidence * 200)
    cv2.rectangle(frame, (w - 330, 28), (w - 130, 48), (60, 60, 60), 2)
    cv2.rectangle(frame, (w - 330, 28), (w - 330 + bar_width, 48), text_color, -1)

    cv2.imshow("Detection de Signes Naruto", frame)

    if cv2.waitKey(1) & 0xFF == 27:  # Echap pour quitter
        break

cap.release()
cv2.destroyAllWindows()
hands.close()