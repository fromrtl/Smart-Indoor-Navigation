import cv2
import time
from ultralytics import YOLO

model = YOLO("best_final.pt")

cap = cv2.VideoCapture(0)

times = []

print("Starting benchmark...")

# Warm-up
for i in range(3):
    ret, frame = cap.read()
    if ret:
        model.predict(frame, imgsz=416, verbose=False)

# Measure 30 frames
for i in range(30):
    ret, frame = cap.read()

    if not ret:
        continue

    start = time.perf_counter()

    results = model.predict(
        frame,
        imgsz=416,
        verbose=False
    )

    end = time.perf_counter()

    elapsed = end - start
    times.append(elapsed)

    print(f"Frame {i+1}: {elapsed:.2f} seconds")

cap.release()

if times:
    print("\nAverage inference time:",
          round(sum(times)/len(times), 3), "seconds")

    print("Approximate inference FPS:",
          round(len(times)/sum(times), 2))
