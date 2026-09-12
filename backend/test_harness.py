import cv2
import time
import csv
import sys
import numpy as np
from modules.vision import AdvancedVisionEngine

def run_harness(source=0, output_csv="vision_telemetry_log.csv"):
    print(f"Starting test harness on source: {source}")
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"Error: Could not open video source {source}")
        return

    engine = AdvancedVisionEngine()
    
    csv_file = open(output_csv, mode='w', newline='')
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow([
        "timestamp", "latency_ms", "fps", "face_detected", 
        "ear", "bpm", "pitch", "yaw", "gaze_direction", 
        "eye_closed_duration", "gaze_away_duration"
    ])

    frame_count = 0
    start_time = time.time()
    last_fps_time = start_time
    fps = 0

    print(f"Logging telemetry to {output_csv}. Press 'q' to stop.")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("End of video stream or cannot read frame.")
                break

            frame_start_time = time.time()

            # Process frame
            try:
                visual_output = engine.process_frame(frame)
                telemetry = visual_output.raw_data
                crash_status = "OK"
            except Exception as e:
                print(f"Exception during processing: {e}")
                telemetry = {
                    "face_detected": False, "ear": 0, "bpm": 0,
                    "pitch": 0, "yaw": 0, "gaze_direction": "Error",
                    "eye_closed_duration": 0, "gaze_away_duration": 0
                }
                crash_status = "CRASH"

            frame_end_time = time.time()
            latency_ms = (frame_end_time - frame_start_time) * 1000

            # Calculate FPS
            frame_count += 1
            if frame_end_time - last_fps_time >= 1.0:
                fps = frame_count / (frame_end_time - last_fps_time)
                frame_count = 0
                last_fps_time = frame_end_time

            # Log to CSV
            csv_writer.writerow([
                frame_end_time,
                round(latency_ms, 2),
                round(fps, 2),
                telemetry.get("face_detected", False),
                telemetry.get("ear", 0.0),
                telemetry.get("bpm", 0),
                telemetry.get("pitch", 0.0),
                telemetry.get("yaw", 0.0),
                telemetry.get("gaze_direction", "Unknown"),
                telemetry.get("eye_closed_duration", 0.0),
                telemetry.get("gaze_away_duration", 0.0)
            ])

            # Draw Overlay
            overlay = frame.copy()
            cv2.putText(overlay, f"FPS: {fps:.1f} | Latency: {latency_ms:.1f}ms", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(overlay, f"Face Detected: {telemetry.get('face_detected')}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0) if telemetry.get("face_detected") else (0, 0, 255), 2)
            cv2.putText(overlay, f"EAR: {telemetry.get('ear')} | BPM: {telemetry.get('bpm')}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            cv2.putText(overlay, f"Pitch: {telemetry.get('pitch')} | Yaw: {telemetry.get('yaw')}", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            cv2.putText(overlay, f"Gaze: {telemetry.get('gaze_direction')}", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            cv2.putText(overlay, f"Crash Status: {crash_status}", (10, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0) if crash_status == "OK" else (0, 0, 255), 2)

            # Display the resulting frame
            # cv2.imshow('Developer Overlay Test Harness', overlay)
            # if cv2.waitKey(1) & 0xFF == ord('q'):
            #     break

            # Let's break after 100 frames for a quick automated test if we don't have UI
            if frame_end_time - start_time > 5.0: # run for 5 seconds
                break

    finally:
        cap.release()
        # cv2.destroyAllWindows()
        csv_file.close()
        print("Harness finished. Data saved.")

if __name__ == "__main__":
    source = 0
    if len(sys.argv) > 1:
        source = sys.argv[1]
    run_harness(source)
