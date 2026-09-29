from flask import Flask, jsonify, Response
from flask_cors import CORS
import cv2
import threading
import time
from modules.vision import AdvancedVisionEngine
from modules.engine import CognitiveStateEngine
from modules.acoustic import AcousticSensor
from modules.acoustic_capture import AcousticCapture
from modules.kinematic import KinematicSensor
from modules.models import SensorOutput
from modules.simulated_steering import SimulatedSteeringGenerator
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

global_telemetry = {}
historical_buffer = []
global_frame = None  # New variable to hold the live video frame

latest_acoustic_output = SensorOutput(
    score=0.0,
    confidence=0.0,
    timestamp=time.time(),
    source="acoustic_sensor",
    validity=False,
    raw_data={"arousal": 0.0, "dominance": 0.0, "valence": 0.0, "status": "initializing"}
)

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
        
        sensor_status = generate_sensor_status(visual_output, acoustic_output, kinematic_output)

        # Merge raw data from all sensors for backward compatibility and logging
        raw_telemetry = {
            **visual_output.raw_data,
            **acoustic_output.raw_data,
            **kinematic_output.raw_data,
            "sensor_status": sensor_status
        }
        
        unified_state = {**raw_telemetry, **cognitive_assessment, "timestamp": time.time()}
        global_telemetry = unified_state
        
        historical_buffer.append(unified_state)
        if len(historical_buffer) > 200:
            historical_buffer.pop(0)
            
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
    acoustic_worker = threading.Thread(target=background_acoustic_worker, daemon=True)
    acoustic_worker.start()
    
    worker = threading.Thread(target=background_telemetry_worker, daemon=True)
    worker.start()
    app.run(host='0.0.0.0', port=5000, debug=False)