import os
import time
import numpy as np
import torch
import cv2
 
from PIL import Image, ImageDraw
from facenet_pytorch import MTCNN
from sixdrepnet import SixDRepNet
 
# ---- VOICE MODULE (separate file) ----
from acoustic_capture import AcousticCapture
 
 
# ============================================================
# 1. MODEL PATHS
# ============================================================
 
HOME = os.path.expanduser("~")
 
HEAD_POSE_MODEL_PATH = os.path.join(HOME, "Downloads", "6DRepNet_300W_LP_AFLW2000.pth")
 
if not os.path.exists(HEAD_POSE_MODEL_PATH):
    raise FileNotFoundError("6DRepNet model not found:\n" + HEAD_POSE_MODEL_PATH)
 
print("Loading 6DRepNet (head pose)...")
device = "cpu"  # Raspberry Pi: no CUDA
print("Device:", device)
 
head_pose_model = SixDRepNet(
    gpu_id=-1,
    dict_path=HEAD_POSE_MODEL_PATH
)
print("[OK] 6DRepNet loaded")
 
print("Loading face detector (MTCNN)...")
mtcnn = MTCNN(keep_all=True, device=device, min_face_size=40)
print("[OK] MTCNN loaded")
 
print("Loading eye detector (OpenCV Haar Cascade)...")
 
# cv2.data.haarcascades only exists in pip-installed opencv-python.
# The apt-installed python3-opencv package stores cascade files on
# disk instead, so check both locations.
_candidate_paths = []
try:
    _candidate_paths.append(cv2.data.haarcascades + "haarcascade_eye.xml")
except AttributeError:
    pass
_candidate_paths += [
    "/usr/share/opencv4/haarcascades/haarcascade_eye.xml",
    "/usr/share/opencv/haarcascades/haarcascade_eye.xml",
    "/usr/local/share/opencv4/haarcascades/haarcascade_eye.xml",
]
 
eye_cascade_path = None
for _p in _candidate_paths:
    if os.path.exists(_p):
        eye_cascade_path = _p
        break
 
if eye_cascade_path is None:
    raise FileNotFoundError(
        "Could not find haarcascade_eye.xml in any known location.\n"
        "Tried: " + ", ".join(_candidate_paths) + "\n"
        "Run: sudo find / -name 'haarcascade_eye.xml' 2>/dev/null\n"
        "to locate it, then hardcode that path here."
    )
 
print("Using cascade file:", eye_cascade_path)
eye_cascade = cv2.CascadeClassifier(eye_cascade_path)
if eye_cascade.empty():
    raise RuntimeError("Failed to load Haar Cascade eye detector: " + eye_cascade_path)
print("[OK] Eye detector loaded")
 
 
# ============================================================
# 2. EYE STATE + EYE BOX FROM HAAR CASCADE
# ============================================================
# NOTE: Haar Cascade eye detection is a coarse method. It works by
# looking for the "open eye" pattern; when eyes are closed, the
# pattern usually is not found. So: eye(s) found => "EYES OPEN",
# no eyes found in the face region => "EYES CLOSED".
 
def get_eye_info(frame_pil, face_box):
    """Returns (eye_status, eye_box). eye_box is ALWAYS returned when a face
    exists: the real eye box if eyes are open, or the estimated eye area
    (upper part of the face) if eyes are closed, so a box never vanishes.
    """
    if face_box is None:
        return "NO FACE", None
 
    fx1, fy1, fx2, fy2 = face_box
    fh = fy2 - fy1
 
    # Only search the eye band of the face (top 15%-55%). Searching the whole
    # face makes Haar mistake nostrils/mouth for eyes, so "closed" is never seen.
    band_y1 = fy1 + int(fh * 0.15)
    band_y2 = fy1 + int(fh * 0.55)
    est_box = (fx1, band_y1, fx2, band_y2)
 
    frame_np = np.array(frame_pil)
    band_gray = cv2.cvtColor(frame_np[band_y1:band_y2, fx1:fx2], cv2.COLOR_RGB2GRAY)
 
    if band_gray.size == 0:
        return "NO EYES", None
 
    eyes = eye_cascade.detectMultiScale(
        band_gray,
        scaleFactor=1.1,
        minNeighbors=6,
        minSize=(20, 20)
    )
 
    if len(eyes) == 0:
        return "EYES CLOSED", est_box
 
    xs1 = [fx1 + ex for (ex, ey, ew, eh) in eyes]
    ys1 = [band_y1 + ey for (ex, ey, ew, eh) in eyes]
    xs2 = [fx1 + ex + ew for (ex, ey, ew, eh) in eyes]
    ys2 = [band_y1 + ey + eh for (ex, ey, ew, eh) in eyes]
 
    return "EYES OPEN", (min(xs1), min(ys1), max(xs2), max(ys2))
 
 
# ============================================================
# 3. HEAD POSE THRESHOLDS (degrees)
# ============================================================
 
YAW_THRESHOLD = 15    # lower = more sensitive to left/right turns
PITCH_THRESHOLD = 40  # higher = less sensitive to up/down tilt
ROLL_THRESHOLD = 20   # head tilt sideways
 
# Eyes must be "not found" this many frames in a row before we call them CLOSED
# (Haar sometimes misses open eyes for one frame; this stops flicker).
EYES_CLOSED_FRAMES = 2
 
 
# ============================================================
# 4. VOICE (MICROPHONE) SETUP
# ============================================================
 
VOICE_HOLD_SECONDS = 1.0   # keep showing "SPEAKING" this long after last speech
                           # (needed because the video loop is slow on a Pi)
 
print("Starting microphone (voice detection)...")
# vad_mode : 0-3, 3 = strictest
# min_rms  : loudness needed to count as speech. If it still says SPEAKING in
#            silence, raise it (0.02, 0.03...). If it never detects you, lower it.
#            The current loudness is shown on screen so you can tune it.
audio = AcousticCapture(vad_mode=3, min_rms=0.01, min_speech_frames=3)
mic_ok = audio.start()
if mic_ok:
    print(f"[OK] Microphone running at {audio.stream_sr} Hz")
else:
    print("[!] Microphone not available -> continuing with camera only.")
    print("    Reason:", audio.init_error)
 
last_speech_time = 0.0
last_voice_status = None
 
 
# ============================================================
# 5. CAMERA SETUP
# ============================================================
 
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
 
if not cap.isOpened():
    audio.stop()
    raise RuntimeError("Could not open camera. Check 'v4l2-ctl --list-devices'.")
 
HAS_DISPLAY = bool(os.environ.get("DISPLAY"))
if not HAS_DISPLAY:
    print("[!] No DISPLAY detected - will save frames to disk instead of showing a live window.")
 
print()
print("Starting live loop. Press 'q' in the video window to quit")
print("(or Ctrl+C in the terminal if running headless).")
print()
 
frame_count = 0
last_save_time = 0
eyes_missing_count = 0
 
try:
    while True:
        ok, frame_bgr = cap.read()
        if not ok:
            print("[X] Failed to read frame from camera.")
            break
 
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        frame = Image.fromarray(frame_rgb)
 
        boxes, probabilities = mtcnn.detect(frame)
 
        output = frame.copy()
        draw = ImageDraw.Draw(output)
 
        # ---- HEAD POSE (drawn on face box) ----
        if boxes is not None:
            best_index = int(np.argmax(probabilities))
            box = boxes[best_index]
            x1, y1, x2, y2 = box.astype(int)
 
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(frame.width, x2)
            y2 = min(frame.height, y2)
 
            w, h = x2 - x1, y2 - y1
            mx, my = int(w * 0.20), int(h * 0.20)
 
            fx1 = max(0, x1 - mx)
            fy1 = max(0, y1 - my)
            fx2 = min(frame.width, x2 + mx)
            fy2 = min(frame.height, y2 + my)
 
            face_np = np.array(frame.crop((fx1, fy1, fx2, fy2)))
 
            pitch, yaw, roll = head_pose_model.predict(face_np)
            pitch = float(np.asarray(pitch).reshape(-1)[0])
            yaw = float(np.asarray(yaw).reshape(-1)[0])
            roll = float(np.asarray(roll).reshape(-1)[0])
 
            if yaw > YAW_THRESHOLD:
                head_status = "RIGHT"
            elif yaw < -YAW_THRESHOLD:
                head_status = "LEFT"
            elif pitch > PITCH_THRESHOLD:
                head_status = "DOWN"
            elif pitch < -PITCH_THRESHOLD:
                head_status = "UP"
            elif abs(roll) > ROLL_THRESHOLD:
                head_status = "TILTED"
            else:
                head_status = "NORMAL"
 
            draw.rectangle([x1, y1, x2, y2], outline="lime", width=4)
            draw.text((x1, max(0, y1 - 22)), head_status, fill="lime")
            face_box_for_eyes = (x1, y1, x2, y2)
        else:
            draw.text((10, 10), "NO FACE", fill="red")
            face_box_for_eyes = None
 
        # ---- EYES (drawn on eye box) ----
        raw_eye_status, eye_box = get_eye_info(frame, face_box_for_eyes)
 
        # Debounce: only call it CLOSED after several missed frames in a row
        if raw_eye_status == "EYES CLOSED":
            eyes_missing_count += 1
        else:
            eyes_missing_count = 0
            
        if raw_eye_status == "EYES CLOSED" and eyes_missing_count < EYES_CLOSED_FRAMES:
            eye_status = "EYES OPEN"   # not confirmed closed yet
        else:
            eye_status = raw_eye_status
 
        if eye_box is not None:
            ex1, ey1, ex2, ey2 = eye_box
            eye_color = "red" if eye_status == "EYES CLOSED" else "cyan"
            draw.rectangle([ex1, ey1, ex2, ey2], outline=eye_color, width=3)
            draw.text((ex1, ey2 + 5), eye_status, fill=eye_color)
 
        # ---- VOICE (from acoustic_capture.py) ----
        has_speech, _audio_16k = audio.get_latest_capture()
        now = time.time()
 
        if has_speech is None:
            voice_status = "MIC OFF"
            voice_color = "red"
        else:
            if has_speech:
                last_speech_time = now
            if now - last_speech_time < VOICE_HOLD_SECONDS:
                voice_status = "SPEAKING"
                voice_color = "yellow"
            else:
                voice_status = "QUIET"
                voice_color = "white"
 
        draw.text((10, frame.height - 25),
                  f"VOICE: {voice_status}  (level {audio.last_rms:.3f})",
                  fill=voice_color)
 
        if voice_status != last_voice_status:
            print(f"[voice] {voice_status}")
            last_voice_status = voice_status
 
        output_bgr = cv2.cvtColor(np.array(output), cv2.COLOR_RGB2BGR)
 
        if HAS_DISPLAY:
            cv2.imshow("Head Pose + Eye State + Voice", output_bgr)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            # Headless fallback: save a frame every 2 seconds
            if now - last_save_time > 2:
                out_path = os.path.join(HOME, "live_frame.jpg")
                cv2.imwrite(out_path, output_bgr)
                last_save_time = now
                print(f"[saved] {out_path}  (frame {frame_count})")
 
        frame_count += 1
 
except KeyboardInterrupt:
    print("Stopped by user.")
 
finally:
    cap.release()
    audio.stop()
    if HAS_DISPLAY:
        cv2.destroyAllWindows()
    print("Camera and microphone stopped.")
 
