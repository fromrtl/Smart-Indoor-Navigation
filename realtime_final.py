import wave
from piper import PiperVoice
import cv2
import time
import threading
import queue
from collections import deque, Counter
import subprocess
from ultralytics import YOLO

SHOW_WINDOW = False

# ---------------- CONFIG ----------------
MODEL_PATH = "best_final.pt"
IMGSZ = 416

CLASS_CONF = {
    "door": 0.20,
    "stairs": 0.65,
    "person": 0.50,
    "window": 0.40,
    "wall": 0.30,
}

ANNOUNCE = {"door", "wall", "stairs", "person", "window"}

# Temporal confirmation
WINDOW_M = 6
CONFIRM_K = 3
CONFIRM_K_CLASS = {"door": 2}

# Stairs sanity check
STAIRS_MIN_BOTTOM = 0.55
STAIRS_MIN_AREA = 0.04

COOLDOWN = 1
ABSENCE_TIME = 2

model = YOLO(MODEL_PATH)
MIN_CONF = min(CLASS_CONF.values())

# ---------------- SPEECH ----------------
speech_queue = queue.Queue()

PIPER_MODEL = "voices/en_US-amy-medium.onnx"
AUDIO_FILE = "/tmp/realtime3_tts.wav"


def speech_worker():
    try:
        # Load Piper model only once
        voice = PiperVoice.load(PIPER_MODEL)

        while True:
            message = speech_queue.get()

            if message is None:
                speech_queue.task_done()
                break

            try:
                # Generate speech
                with wave.open(AUDIO_FILE, "wb") as wav_file:
                    voice.synthesize_wav(message, wav_file)

                # Play speech
                subprocess.run(
                    ["pw-play", AUDIO_FILE],
                    check=True
                )

            except Exception as e:
                print("TTS ERROR:", e)

            finally:
                speech_queue.task_done()

    except Exception as e:
        print("Speech engine initialization failed:", e)


threading.Thread(
    target=speech_worker,
    daemon=True
).start()


def speak(message):
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
history = deque(maxlen=WINDOW_M)
state = {}

cap = cv2.VideoCapture(1)
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

        result = model.predict(
            frame,
            imgsz=IMGSZ,
            conf=MIN_CONF,
            verbose=False
        )[0]

        visible = {}

        for b in result.boxes:
            name = model.names[int(b.cls[0].item())]
            conf = float(b.conf[0].item())
            box = b.xyxy[0].cpu().tolist()
            box = tuple(map(int, box))

            print(f"DETECTED: {name} | Confidence: {conf:.2f}")

            if not passes_filters(name, conf, box, w, h):
                continue

            if name not in visible or conf > visible[name]["conf"]:
                cx = (box[0] + box[2]) // 2

                visible[name] = {
                    "conf": conf,
                    "box": box,
                    "zone": get_zone(cx, w)
                }

        history.append(set(visible))

        counts = Counter(n for s in history for n in s)

        confirmed = {
            n for n, c in counts.items()
            if c >= CONFIRM_K
        }

        for name, d in visible.items():
            x1, y1, x2, y2 = d["box"]

            color = (0, 255, 0) if name in confirmed else (0, 165, 255)

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            cv2.putText(
                frame,
                f"{name} {d['conf']:.2f} {d['zone']}",
                (x1, max(20, y1 - 6)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                color,
                2
            )

            if name not in confirmed or name not in ANNOUNCE:
                continue

            zone = d["zone"]
            prev = state.get(name)

            announce = (
                prev is None
                or now - prev["last_seen"] >= ABSENCE_TIME
                or (
                    zone != prev["zone"]
                    and now - prev["last_spoken"] >= COOLDOWN
                )
            )

            if announce:
                speak(make_message(name, zone))

                state[name] = {
                    "zone": zone,
                    "last_seen": now,
                    "last_spoken": now
                }

            else:
                state[name]["last_seen"] = now
                state[name]["zone"] = zone

        for name in list(state):
            if (
                name not in confirmed
                and now - state[name]["last_seen"] >= ABSENCE_TIME
            ):
                del state[name]

        if SHOW_WINDOW:
            cv2.imshow("Indoor Navigation", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

finally:
    cap.release()

    if SHOW_WINDOW:
        cv2.destroyAllWindows()

    speech_queue.put(None)
    print("Navigation stopped.")
