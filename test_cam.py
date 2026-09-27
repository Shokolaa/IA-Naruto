import sys
print("1. Importation OpenCV...")
import cv2

print("2. Tentative ouverture camera (avec DirectShow)...")
# cv2.CAP_DSHOW force Windows a repondre immediatement sans bloquer
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

if not cap.isOpened():
    print("ERREUR : Impossible d'ouvrir la camera (index 0).")
    print("Tentative avec l'index 1...")
    cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)

if not cap.isOpened():
    print("ERREUR : Aucune camera accessible.")
    sys.exit()

print("3. Lecture d'une image...")
ret, frame = cap.read()
if ret:
    print("SUCCES : Image lue avec succes !")
    cv2.imshow("Test Camera", frame)
    print("Appuie sur une touche dans la fenetre pour quitter...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    print("ERREUR : Camera ouverte mais aucune image recue.")

cap.release()
print("4. Fin du test.")