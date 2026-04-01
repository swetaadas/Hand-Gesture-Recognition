import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
import numpy as np
import math
import os
import tensorflow as tf

# Use tf_keras (legacy Keras 2) to load the old .h5 model
import tf_keras

print("Loading gesture model...")
model = tf_keras.models.load_model("keras_model.h5", compile=False)

with open("labels.txt", "r") as f:
    labels = [line.strip().split(" ", 1)[-1] for line in f if line.strip()]
print("Labels:", labels)

# ── MediaPipe Hand Landmarker (Tasks API) ────────────────────────────────────
TASK_FILE = "hand_landmarker.task"
if not os.path.exists(TASK_FILE):
    print(f"ERROR: '{TASK_FILE}' not found! Download it with:")
    print("  Invoke-WebRequest -Uri https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task -OutFile hand_landmarker.task")
    exit(1)

base_options = mp_python.BaseOptions(model_asset_path=TASK_FILE)
options = mp_vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=1,
    min_hand_detection_confidence=0.5,
    min_hand_presence_confidence=0.5,
    min_tracking_confidence=0.5,
)
landmarker = mp_vision.HandLandmarker.create_from_options(options)

# Landmark connections for drawing
HAND_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (5,9),(9,10),(10,11),(11,12),
    (9,13),(13,14),(14,15),(15,16),
    (13,17),(17,18),(18,19),(19,20),
    (0,17),
]

# ── Constants ────────────────────────────────────────────────────────────────
offset  = 20
imgSize = 300

def get_prediction(img_white):
    img_rgb = cv2.cvtColor(img_white, cv2.COLOR_BGR2RGB)
    img_res = cv2.resize(img_rgb, (224, 224))
    # Teachable Machine models expect normalization: pixel/127.5 - 1
    img_arr = np.expand_dims(img_res.astype("float32") / 127.5 - 1, axis=0)
    preds   = model.predict(img_arr, verbose=0)[0]
    index   = int(np.argmax(preds))
    return preds, index

def draw_landmarks(frame, landmarks, H, W):
    pts = [(int(lm.x * W), int(lm.y * H)) for lm in landmarks]
    for a, b in HAND_CONNECTIONS:
        cv2.line(frame, pts[a], pts[b], (0, 220, 0), 2)
    for pt in pts:
        cv2.circle(frame, pt, 4, (255, 255, 255), -1)
        cv2.circle(frame, pt, 4, (0, 180, 0), 1)

# ── Main loop ────────────────────────────────────────────────────────────────
cap = cv2.VideoCapture(0)
print("Running. Press 'q' to quit.")

while True:
    success, img = cap.read()
    if not success or img is None:
        continue

    imgOutput = img.copy()
    H, W = img.shape[:2]

    # Run MediaPipe Hand Landmarker
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
    result = landmarker.detect(mp_image)

    if result.hand_landmarks:
        for hand_landmarks in result.hand_landmarks:
            draw_landmarks(imgOutput, hand_landmarks, H, W)

            # Bounding box from landmarks
            xs = [lm.x * W for lm in hand_landmarks]
            ys = [lm.y * H for lm in hand_landmarks]
            x, y = int(min(xs)), int(min(ys))
            w, h = int(max(xs)) - x, int(max(ys)) - y

            # Clamped crop
            y1, y2 = max(0, y - offset), min(H, y + h + offset)
            x1, x2 = max(0, x - offset), min(W, x + w + offset)
            imgCrop = img[y1:y2, x1:x2]

            if imgCrop.size == 0:
                continue

            # Pad onto white square
            imgWhite    = np.ones((imgSize, imgSize, 3), np.uint8) * 255
            aspectRatio = h / (w + 1e-6)

            if aspectRatio > 1:
                k     = imgSize / h
                wCal  = min(math.ceil(k * w), imgSize)
                wGap  = math.ceil((imgSize - wCal) / 2)
                imgWhite[:, wGap:wCal + wGap] = cv2.resize(imgCrop, (wCal, imgSize))
            else:
                k     = imgSize / (w + 1e-6)
                hCal  = min(math.ceil(k * h), imgSize)
                hGap  = math.ceil((imgSize - hCal) / 2)
                imgWhite[hGap:hCal + hGap, :] = cv2.resize(imgCrop, (imgSize, hCal))

            # Predict
            _, index   = get_prediction(imgWhite)
            label_text = labels[index] if index < len(labels) else "Unknown"

            # Draw UI
            lx, ly = x - offset, y - offset
            cv2.rectangle(imgOutput, (lx, ly - 50), (lx + 200, ly), (138, 43, 226), cv2.FILLED)
            cv2.putText(imgOutput, label_text, (lx + 6, ly - 12),
                        cv2.FONT_HERSHEY_DUPLEX, 1.1, (255, 255, 255), 2)
            cv2.rectangle(imgOutput, (lx, ly), (x + w + offset, y + h + offset), (138, 43, 226), 3)

            # (debug windows removed)

    cv2.imshow("Hand Gesture Recognition", imgOutput)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
