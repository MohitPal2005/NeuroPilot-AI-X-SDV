from flask import Flask, jsonify, Response
from flask_cors import CORS
import cv2
import threading
import time
from modules.vision import AdvancedVisionEngine
from modules.engine import CognitiveStateEngine

app = Flask(__name__)
CORS(app)

camera = cv2.VideoCapture(0)
vision_engine = AdvancedVisionEngine()

global_telemetry = {}
historical_buffer = []
global_frame = None  # New variable to hold the live video frame

def background_telemetry_worker():
    global global_telemetry, historical_buffer, global_frame
    while True:
        success, frame = camera.read()
        if not success:
            time.sleep(0.03)
            continue
        
        frame = cv2.flip(frame, 1)
        
        # Save the frame so the web can see it
        global_frame = frame.copy()
        
        raw_telemetry = vision_engine.process_frame(frame)
        cognitive_assessment = CognitiveStateEngine.compute_csi(raw_telemetry)
        
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
    worker = threading.Thread(target=background_telemetry_worker, daemon=True)
    worker.start()
    app.run(host='0.0.0.0', port=5000, debug=False)