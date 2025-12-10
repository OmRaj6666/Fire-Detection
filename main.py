import cv2
import numpy as np
import threading
import subprocess
import os

# ==============================
# CONFIG
# ==============================
ALARM_AUDIO_PATH = "alarm.mp3"   # make sure this exists in same folder

# 🔥 Fire detection tuning for small flames (matchstick / lighter)
FIRE_THRESHOLD = 0.0015      # ~0.15% of pixels
MIN_FIRE_PIXELS = 120        # very small blob allowed
CONSEC_FRAMES_FOR_ALARM = 5  # frames in a row to confirm fire

# Frame size (smaller = faster)
FRAME_WIDTH = 480
FRAME_HEIGHT = 360

# Human detection optimization
HUMAN_DETECT_EVERY = 5       # run heavy human detection every N frames

DEBUG_FIRE = False           # True = print fire pixel stats in console

# ==============================
# Alarm logic (macOS using afplay)
# ==============================

alarm_lock = threading.Lock()
alarm_playing = False  # To avoid starting multiple alarms at once


def play_alarm_sound():
    """
    Plays the alarm sound using macOS 'afplay'.
    Runs in a separate thread so video loop is not blocked.
    """
    global alarm_playing

    if not os.path.exists(ALARM_AUDIO_PATH):
        print(f"[ERROR] Alarm file not found: {ALARM_AUDIO_PATH}")
        alarm_playing = False
        return

    try:
        alarm_playing = True
        print("[ALARM] Fire near human confirmed! Playing alarm...")
        subprocess.call(["afplay", ALARM_AUDIO_PATH])
    except Exception as e:
        print(f"[ERROR] Could not play alarm sound: {e}")
    finally:
        alarm_playing = False


def trigger_alarm():
    """
    Starts the alarm if it's not already playing.
    """
    global alarm_playing
    with alarm_lock:
        if not alarm_playing:
            t = threading.Thread(target=play_alarm_sound, daemon=True)
            t.start()


# ==============================
# Human Detection (HOG + Face Cascade)
# ==============================

# Full-body HOG detector (heavier)
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

# Face detector (fast)
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)


def detect_humans(frame_resized, gray):
    """
    Detect both full-body persons (HOG) and faces (Haar cascade).

    Returns:
        human_detected (bool)
        body_boxes (list of (x, y, w, h))
        face_boxes (list of (x, y, w, h))
        all_boxes (union of body + face)
    """
    # --- HOG person detection (full body) ---
    rects, weights = hog.detectMultiScale(
        gray,
        winStride=(8, 8),
        padding=(16, 16),
        scale=1.05
    )
    body_boxes = [(x, y, w, h) for (x, y, w, h) in rects]

    # --- Face detection (frontal faces) ---
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(40, 40)
    )
    face_boxes = [(x, y, w, h) for (x, y, w, h) in faces]

    all_boxes = body_boxes + face_boxes
    human_detected = len(all_boxes) > 0

    return human_detected, body_boxes, face_boxes, all_boxes


# ==============================
# Fire Detection (Color-based, ignoring human boxes)
# ==============================

def detect_fire(frame_resized, ignore_boxes=None):
    """
    Fire detection based on color in HSV space.
    Ignores "fire-like" pixels inside human bounding boxes.

    Args:
        frame_resized: BGR image of size (FRAME_WIDTH, FRAME_HEIGHT)
        ignore_boxes: list of (x, y, w, h) to ignore (e.g., humans)

    Returns:
        fire_detected_raw (bool)
        mask (binary fire mask)
        fire_ratio (float)
    """
    hsv = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2HSV)

    # Color range tuned for small bright flames
    lower_fire = np.array([5, 90, 170])      # H, S, V lower
    upper_fire = np.array([40, 255, 255])    # H, S, V upper

    mask = cv2.inRange(hsv, lower_fire, upper_fire)

    # Clean up noise & enlarge small flames
    kernel = np.ones((5, 5), np.uint8)  # bigger kernel = expand small blobs
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE, kernel, iterations=2)

    # Ignore fire pixels inside human bounding boxes
    if ignore_boxes:
        h, w = mask.shape
        for (x, y, bw, bh) in ignore_boxes:
            x0 = max(0, x)
            y0 = max(0, y)
            x1 = min(w, x + bw)
            y1 = min(h, y + bh)
            mask[y0:y1, x0:x1] = 0

    fire_pixels = cv2.countNonZero(mask)
    total_pixels = mask.shape[0] * mask.shape[1]
    fire_ratio = fire_pixels / float(total_pixels)

    if DEBUG_FIRE:
        print(f"Fire pixels: {fire_pixels}  ratio: {fire_ratio:.5f}")

    # Raw fire detection for this frame only
    fire_detected_raw = (fire_ratio > FIRE_THRESHOLD) and (fire_pixels > MIN_FIRE_PIXELS)

    return fire_detected_raw, mask, fire_ratio


# ==============================
# Main camera loop
# ==============================

def main():
    cap = cv2.VideoCapture(0)  # 0 = default webcam

    if not cap.isOpened():
        print("[ERROR] Cannot open camera")
        return

    print("[INFO] Press 'q' to quit.")

    consecutive_fire_frames = 0           # how many frames in a row fire was seen
    frame_count = 0                       # for throttling human detection

    # Cache last human detection (for speed)
    cached_human_detected = False
    cached_body_boxes = []
    cached_face_boxes = []
    cached_all_boxes = []

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[ERROR] Failed to grab frame")
            break

        frame_resized = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))
        gray = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2GRAY)
        frame_count += 1

        # 1) Human detection (expensive) -> only every HUMAN_DETECT_EVERY frames
        if frame_count % HUMAN_DETECT_EVERY == 0:
            cached_human_detected, cached_body_boxes, cached_face_boxes, cached_all_boxes = \
                detect_humans(frame_resized, gray)

        human_detected = cached_human_detected
        body_boxes = cached_body_boxes
        face_boxes = cached_face_boxes
        all_boxes = cached_all_boxes

        # 2) Fire detection (every frame) but ignoring human areas
        fire_detected_raw, mask, fire_ratio = detect_fire(
            frame_resized,
            ignore_boxes=all_boxes
        )

        # Count consecutive fire frames (for stability)
        if fire_detected_raw:
            consecutive_fire_frames += 1
        else:
            consecutive_fire_frames = 0

        fire_confirmed = consecutive_fire_frames >= CONSEC_FRAMES_FOR_ALARM

        # Draw body boxes (blue)
        for (x, y, w, h) in body_boxes:
            cv2.rectangle(frame_resized, (x, y), (x + w, y + h), (255, 0, 0), 2)

        # Draw face boxes (green)
        for (x, y, w, h) in face_boxes:
            cv2.rectangle(frame_resized, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # --------- Decision: alarm only if FIRE + HUMAN ---------
        fire_and_human = fire_confirmed and human_detected

        # Show info on frame
        status_text = f"FireRatio: {fire_ratio:.4f}  FireFrames: {consecutive_fire_frames}"
        cv2.putText(frame_resized, status_text, (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        human_text = "Human: YES" if human_detected else "Human: NO"
        cv2.putText(frame_resized, human_text, (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        if fire_and_human:
            cv2.putText(frame_resized, "FIRE NEAR HUMAN - ALARM!", (10, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 3)
            trigger_alarm()
        elif fire_confirmed and not human_detected:
            cv2.putText(frame_resized, "FIRE DETECTED (NO HUMAN)", (10, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 165, 255), 2)
        else:
            cv2.putText(frame_resized, "NO FIRE", (10, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        cv2.imshow("Camera - Fire + Human Detection (Small Flame Tuned)", frame_resized)
        cv2.imshow("Fire Mask (Ignoring Humans)", mask)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
