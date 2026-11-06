import cv2
import time
import threading
import queue
from collections import deque, Counter

import pyttsx3
from ultralytics import YOLO

# ---------------- CONFIG ----------------
MODEL_PATH = "best_final.pt"   # was best.pt, the finetuned model was never used
IMGSZ = 416                            # 640 is too heavy for Pi 4 2GB, try 320 if still slow

# Per-class thresholds: strict on stairs (false positives), lenient on door (missed)
CLASS_CONF = {
    "door": 0.30,
    "stairs": 0.65,
    "person": 0.50,
    "window": 0.40,
    "wall": 0.60,
}
ANNOUNCE = {"door", "wall", "stairs", "person", "window"}

# Temporal confirmation: class must appear in K of the last M frames
WINDOW_M = 6
CONFIRM_K =3

# Stairs sanity check for a chest mounted camera
STAIRS_MIN_BOTTOM = 0.55   # box bottom edge must be in the lower 45% of frame
STAIRS_MIN_AREA = 0.04     # box must cover at least 4% of frame

COOLDOWN = 5
ABSENCE_TIME = 2

model = YOLO(MODEL_PATH)
MIN_CONF = min(CLASS_CONF.values())

# ---------------- SPEECH ----------------
speech_queue = queue.Queue()


def speech_worker():
    try:
        engine = pyttsx3.init()
        engine.setProperty("rate", 165)
        engine.setProperty("volume", 1.0)
        while True:
            message = speech_queue.get()
            if message is None:
                speech_queue.task_done()
                break
            try:
                engine.say(message)
                engine.runAndWait()
            except Exception as e:
                print("TTS ERROR:", e)
            finally:
                speech_queue.task_done()
        engine.stop()
    except Exception as e:
        print("Speech engine initialization failed:", e)


threading.Thread(target=speech_worker, daemon=True).start()


def speak(message):
    # Drop stale messages so speech never lags behind reality
    while not speech_queue.empty():
        try:
            speech_queue.get_nowait()
            speech_queue.task_done()
        except queue.Empty:
            break
    print("SPEAK:", message)
    speech_queue.put(message)


# ---------------- HELPERS ----------------
def get_zone(x, width):
    r = x / width
    if r < 1 / 3:
        return "left"
    if r > 2 / 3:
        return "right"
    return "center"


def make_message(name, zone):
    return f"{name} ahead" if zone == "center" else f"{name} on your {zone}"


def passes_filters(name, conf, box, w, h):
    if conf < CLASS_CONF.get(name, 0.5):
        return False
    x1, y1, x2, y2 = box
    if name == "stairs":
        if y2 < STAIRS_MIN_BOTTOM * h:
            return False
        if (x2 - x1) * (y2 - y1) < STAIRS_MIN_AREA * w * h:
            return False
    return True


# ---------------- MAIN ----------------
history = deque(maxlen=WINDOW_M)   # set of class names seen per frame
state = {}

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
if not cap.isOpened():
    raise RuntimeError("Cannot open webcam.")

print("Navigation started. Press Q to quit.")

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        now = time.time()
        h, w = frame.shape[:2]

        result = model.predict(frame, imgsz=IMGSZ, conf=MIN_CONF, verbose=False)[0]

        visible = {}
        for b in result.boxes:
            name = model.names[int(b.cls[0].item())]
            conf = float(b.conf[0].item())
            box = tuple(map(int, b.xyxy[0].tolist()))
            if not passes_filters(name, conf, box, w, h):
                continue
            if name not in visible or conf > visible[name]["conf"]:
                cx = (box[0] + box[2]) // 2
                visible[name] = {"conf": conf, "box": box, "zone": get_zone(cx, w)}

        history.append(set(visible))
        counts = Counter(n for s in history for n in s)
        confirmed = {n for n, c in counts.items() if c >= CONFIRM_K}

        for name, d in visible.items():
            x1, y1, x2, y2 = d["box"]
            color = (0, 255, 0) if name in confirmed else (0, 165, 255)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f"{name} {d['conf']:.2f} {d['zone']}",
                        (x1, max(20, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

            if name not in confirmed or name not in ANNOUNCE:
                continue

            zone = d["zone"]
            prev = state.get(name)
            announce = (
                prev is None
                or now - prev["last_seen"] >= ABSENCE_TIME
                or (zone != prev["zone"] and now - prev["last_spoken"] >= COOLDOWN)
            )
            if announce:
                speak(make_message(name, zone))
                state[name] = {"zone": zone, "last_seen": now, "last_spoken": now}
            else:
                state[name]["last_seen"] = now
                state[name]["zone"] = zone

        for name in list(state):
            if name not in confirmed and now - state[name]["last_seen"] >= ABSENCE_TIME:
                del state[name]

        cv2.imshow("Indoor Navigation", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
finally:
    cap.release()
    cv2.destroyAllWindows()
    speech_queue.put(None)
    print("Navigation stopped.")
