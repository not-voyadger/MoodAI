import cv2
import torch
import numpy as np
from model import MoodAI, IMAGE_SIZE

MODEL_PATH = "face_model.pth"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

model = MoodAI(num_classes=7).to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

CLASSES = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        face = gray[y:y+h, x:x+w]
        face = cv2.resize(face, (IMAGE_SIZE, IMAGE_SIZE))
        face = (face / 255.0 - 0.5) / 0.5
        face = torch.tensor(face).unsqueeze(0).unsqueeze(0).float().to(device)

        with torch.no_grad():
            outputs = model(face)
            _, predicted = torch.max(outputs, 1)
            label = CLASSES[predicted.item()]

        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        
        cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX,
                    0.9, (0, 255, 0), 2)

    cv2.imshow("MoodAI Emotion Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
