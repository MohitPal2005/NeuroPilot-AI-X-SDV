from flask import Flask, jsonify, Response
from flask_cors import CORS
import cv2
import threading
import time
import os
import csv
from modules.vision import AdvancedVisionEngine
from modules.engine import CognitiveStateEngine
from modules.acoustic import AcousticSensor
from modules.acoustic_capture import AcousticCapture
from modules.kinematic import KinematicSensor
from modules.models import SensorOutput
from modules.simulated_steering import SimulatedSteeringGenerator
from modules.recovery_engine import RecoveryEngine
import itertools

app = Flask(__name__)
CORS(app)

camera = cv2.VideoCapture(0)
vision_engine = AdvancedVisionEngine()
acoustic_capture = AcousticCapture(target_sr=16000)
if not acoustic_capture.start():
    print(f"WARNING: Failed to start microphone stream (Error: {acoustic_capture.init_error}). Acoustic pipeline will run with empty/dummy audio.")
acoustic_engine = AcousticSensor(acoustic_capture)
kinematic_engine = KinematicSensor()
cognitive_engine = CognitiveStateEngine()
recovery_engine = RecoveryEngine()

global_telemetry = {}
historical_buffer = []
global_frame = None  # New variable to hold the live video frame
global_ground_truth_label = "unlabeled"
ground_truth_lock = threading.Lock()

latest_acoustic_output = SensorOutput(
    score=0.0,
    confidence=0.0,
    timestamp=time.time(),
    source="acoustic_sensor",
    validity=False,
    raw_data={"arousal": 0.0, "dominance": 0.0, "valence": 0.0, "status": "initializing"}
)

def set_ground_truth(label):
    global global_ground_truth_label
    with ground_truth_lock:
        global_ground_truth_label = label
    print(f"[GROUND TRUTH] -> {label}")

def keyboard_listener_worker():
    try:
        import keyboard
        keyboard.add_hotkey('1', lambda: set_ground_truth("normal"))
        keyboard.add_hotkey('2', lambda: set_ground_truth("drowsy"))
        keyboard.add_hotkey('3', lambda: set_ground_truth("distracted"))
        keyboard.add_hotkey('0', lambda: set_ground_truth("unlabeled"))
        keyboard.wait()
    except ImportError:
        print("WARNING: 'keyboard' library not found. Ground truth labeling will not work. Please run 'pip install keyboard'.")
    except Exception as e:
        print(f"Keyboard listener error: {e}")

def background_acoustic_worker():
    global latest_acoustic_output
    while True:
        try:
            latest_acoustic_output = acoustic_engine.process_audio()
        except Exception as e:
            print(f"Acoustic worker error: {e}")
        # Process audio already takes ~1.5s, add small sleep to avoid tight loop on failure
        time.sleep(0.5)

def generate_sensor_status(visual_output, acoustic_output, kinematic_output):
    return {
        "visual": "active" if visual_output.validity else "degraded",
        "acoustic": "active" if acoustic_output.validity else "degraded",
        "kinematic": "active" if kinematic_output.validity else ("buffering" if kinematic_output.raw_data.get("status") == "buffering" else "degraded")
    }

def background_telemetry_worker():
    global global_telemetry, historical_buffer, global_frame
    
    log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'logs')
    os.makedirs(log_dir, exist_ok=True)
    log_file_path = os.path.join(log_dir, 'pipeline_log.csv')
    
    log_file = open(log_file_path, mode='a', newline='')
    csv_writer = csv.writer(log_file)
    if os.path.getsize(log_file_path) == 0:
        csv_writer.writerow([
            "timestamp", "vision_score", "vision_confidence", "vision_validity",
            "acoustic_score", "acoustic_confidence", "acoustic_validity",
            "kinematic_score", "kinematic_confidence", "kinematic_validity",
            "fused_csi", "zone", "recovery_state", "trigger_reason", "ground_truth_label"
        ])
        log_file.flush()
    
    row_count = 0

    # Create an infinite stream of normal SIMULATED steering data
    steering_gen = SimulatedSteeringGenerator(sample_rate_hz=15.0)
    simulated_normal_data = steering_gen.generate_simulated_normal_data(60.0)
    steering_stream = itertools.cycle(simulated_normal_data)
    
    while True:
        success, frame = camera.read()
        if not success:
            time.sleep(0.03)
            continue
        
        frame = cv2.flip(frame, 1)
        
        # Save the frame so the web can see it
        global_frame = frame.copy()
        
        visual_output = vision_engine.process_frame(frame)
        acoustic_output = latest_acoustic_output
        
        # Pull the next simulated angle and pass it in
        current_angle = next(steering_stream)
        kinematic_output = kinematic_engine.process_telemetry(current_angle)
        
        cognitive_assessment = cognitive_engine.compute_csi([visual_output, acoustic_output, kinematic_output])
        
        recovery_status = recovery_engine.process(cognitive_assessment, [visual_output, acoustic_output, kinematic_output])
        
        sensor_status = generate_sensor_status(visual_output, acoustic_output, kinematic_output)

        # Merge raw data from all sensors for backward compatibility and logging
        raw_telemetry = {
            **visual_output.raw_data,
            **acoustic_output.raw_data,
            **kinematic_output.raw_data,
            "sensor_status": sensor_status,
            "recovery_engine": recovery_status
        }
        
        unified_state = {**raw_telemetry, **cognitive_assessment, "timestamp": time.time()}
        global_telemetry = unified_state
        
        historical_buffer.append(unified_state)
        if len(historical_buffer) > 200:
            historical_buffer.pop(0)
            
        def safe_get(obj, attr):
            if obj is None: return ""
            val = getattr(obj, attr, None)
            return val if val is not None else ""

        current_time = time.time()
        
        with ground_truth_lock:
            current_label = global_ground_truth_label
            
        csv_writer.writerow([
            current_time,
            safe_get(visual_output, 'score'),
            safe_get(visual_output, 'confidence'),
            safe_get(visual_output, 'validity'),
            safe_get(acoustic_output, 'score'),
            safe_get(acoustic_output, 'confidence'),
            safe_get(acoustic_output, 'validity'),
            safe_get(kinematic_output, 'score'),
            safe_get(kinematic_output, 'confidence'),
            safe_get(kinematic_output, 'validity'),
            cognitive_assessment.get("csi", "") if cognitive_assessment else "",
            cognitive_assessment.get("zone", "") if cognitive_assessment else "",
            recovery_status.get("state", "") if recovery_status else "",
            recovery_status.get("trigger_reason", "") if recovery_status else "",
            current_label
        ])
        
        row_count += 1
        if row_count % 30 == 0:
            log_file.flush()
            
        time.sleep(0.03)

# --- NEW VIDEO STREAMING ENDPOINT ---
def generate_frames():
    global global_frame
    while True:
        if global_frame is None:
            time.sleep(0.1)
            continue
        # Encode the frame as a JPEG image
        ret, buffer = cv2.imencode('.jpg', global_frame)
        frame_bytes = buffer.tobytes()
        # Yield it in a continuous HTTP stream
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')
# ------------------------------------

@app.route('/driver/status', methods=['GET'])
def get_driver_status():
    return jsonify(global_telemetry)

@app.route('/analytics', methods=['GET'])
def get_analytics():
    return jsonify(historical_buffer[-40:])

if __name__ == '__main__':
    keyboard_worker = threading.Thread(target=keyboard_listener_worker, daemon=True)
    keyboard_worker.start()
    
    acoustic_worker = threading.Thread(target=background_acoustic_worker, daemon=True)
    acoustic_worker.start()
    
    worker = threading.Thread(target=background_telemetry_worker, daemon=True)
    worker.start()
    app.run(host='0.0.0.0', port=5000, debug=False)