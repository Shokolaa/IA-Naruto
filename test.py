import torch
import cv2

print("PyTorch version :", torch.__version__)
print("GPU disponible :", torch.cuda.is_available())
if torch.cuda.is_available():
    print("Device :", torch.cuda.get_device_name(0))

# Test rapide de la caméra
cap = cv2.VideoCapture(0)
ret, frame = cap.read()
if ret:
    print("Caméra détectée avec succès !")
else:
    print("Attention : caméra non détectée.")
cap.release()