import cv2
import numpy as np
import tensorflow as tf
import face_recognition
import time

model = tf.keras.models.load_model('.models/model.h5')

def preprocess_image(image):
    img = cv2.resize(image, (224, 224))
    img = tf.keras.applications.efficientnet.preprocess_input(img)
    return img

def classify_face(img):
    img = preprocess_image(img)
    img = np.expand_dims(img, axis=0)
    prediction = model.predict(img)
    label = "Fake" if prediction[0, 0] > 0.5 else "Real"
    score = prediction[0, 0]
    return label, score

cap = cv2.VideoCapture('media/real.mp4')

WIDTH = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
HEIGHT = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
MY_WIDTH = 640
SCALE = MY_WIDTH / WIDTH
pTime = 0

while True:
    ret, frame = cap.read()
    if not ret: break
    frame = cv2.resize(frame, None, fx=SCALE, fy=SCALE)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    face_locations = face_recognition.face_locations(rgb_frame)
    for (top, right, bottom, left) in face_locations:
        face = frame[top:bottom, left:right]
        label, score = classify_face(face)
        color = (0, 0, 255) if label == "Fake" else (0, 255, 0)
        cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
        cv2.putText(frame, f'{label} ({score:.2f})', (left, bottom + 20), cv2.FONT_HERSHEY_DUPLEX, 0.5, color, 1)
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.imshow('Video Frame', frame)
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
