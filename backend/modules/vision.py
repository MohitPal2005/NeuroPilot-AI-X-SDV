import cv2
import numpy as np
import time

import mediapipe as mp
from mediapipe.python.solutions import face_mesh as mp_face_mesh

import config
from modules.models import SensorOutput

class AdvancedVisionEngine:
    def __init__(self):
        # Use the directly imported face_mesh module
        self.mp_face_mesh = mp_face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.6,
            min_tracking_confidence=0.6
        )
        
        # Landmark indices mapping
        self.LEFT_EYE = [362, 385, 387, 263, 373, 380]
        self.RIGHT_EYE = [33, 160, 158, 133, 153, 144]
        self.LEFT_IRIS = [474, 475, 476, 477]
        self.RIGHT_IRIS = [469, 470, 471, 472]
        
        # State variables for blink and gaze calculation
        self.blink_counter = 0
        self.eye_closed_start_time = None
        self.gaze_away_start_time = None
        self.last_blink_check_time = time.time()
        self.blinks_in_current_minute = 0
        self.current_bpm = 12

    def _calculate_ear(self, landmarks, eye_indices, width, height):
        coords = [np.array([landmarks[i].x * width, landmarks[i].y * height]) for i in eye_indices]
        # Distances between vertical eye landmarks
        v1 = np.linalg.norm(coords[1] - coords[5])
        v2 = np.linalg.norm(coords[2] - coords[4])
        # Distance between horizontal eye landmarks
        eye_width = np.linalg.norm(coords[0] - coords[3])
        if eye_width < 1e-6:
            return None
        return (v1 + v2) / (2.0 * eye_width)

    def _estimate_head_pose(self, landmarks, width, height):
        # Selected standard 3D model landmarks for 2D-to-3D projection matrix
        # Y-coordinates are negated to match OpenCV's Y-down image coordinate system
        model_points = np.array([
            (0.0, 0.0, 0.0),             # Nose tip
            (0.0, 330.0, -65.0),         # Chin
            (-225.0, -170.0, -135.0),    # Left eye corner
            (225.0, -170.0, -135.0),     # Right eye corner
            (-150.0, 150.0, -125.0),     # Left mouth corner
            (150.0, 150.0, -125.0)       # Right mouth corner
        ])
        
        # Map back to image space
        img_indices = [1, 152, 33, 263, 61, 291]
        image_points = np.array([ [landmarks[i].x * width, landmarks[i].y * height] for i in img_indices ], dtype="double")
        
        focal_length = width
        center = (width / 2, height / 2)
        camera_matrix = np.array([[focal_length, 0, center[0]], [0, focal_length, center[1]], [0, 0, 1]], dtype="double")
        dist_coeffs = np.zeros((4, 1)) 
        
        success, rotation_vector, translation_vector = cv2.solvePnP(model_points, image_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)
        rmat, _ = cv2.Rodrigues(rotation_vector)
        angles, _, _, _, _, _ = cv2.RQDecomp3x3(rmat)
        
        # RQDecomp3x3 returns angles in degrees directly
        return angles[0], angles[1] # Pitch, Yaw

    def process_frame(self, frame):
        h, w, _ = frame.shape
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_frame)
        
        current_time = time.time()
        # Track running window metrics for Blinks Per Minute (BPM)
        if current_time - self.last_blink_check_time >= 60.0:
            self.current_bpm = self.blinks_in_current_minute
            self.blinks_in_current_minute = 0
            self.last_blink_check_time = current_time

        # Default fallback states
        metrics = {
            "face_detected": False, "ear": 0.3, "bpm": self.current_bpm,
            "eye_closed_duration": 0.0, "pitch": 0.0, "yaw": 0.0,
            "gaze_direction": "Center", "gaze_away_duration": 0.0
        }

        if not results.multi_face_landmarks:
            if self.eye_closed_start_time: self.eye_closed_start_time = None
            if self.gaze_away_start_time: self.gaze_away_start_time = None
            return SensorOutput(
                score=0.0,
                confidence=0.0,
                timestamp=current_time,
                source="visual_engine",
                validity=False,
                raw_data=metrics
            )

        landmarks = results.multi_face_landmarks[0].landmark
        metrics["face_detected"] = True

        # 1. EAR & Eye Closure tracking
        ear_l = self._calculate_ear(landmarks, self.LEFT_EYE, w, h)
        ear_r = self._calculate_ear(landmarks, self.RIGHT_EYE, w, h)
        
        if ear_l is None or ear_r is None:
            if self.eye_closed_start_time: self.eye_closed_start_time = None
            if self.gaze_away_start_time: self.gaze_away_start_time = None
            return SensorOutput(
                score=0.0,
                confidence=0.0,
                timestamp=current_time,
                source="visual_engine",
                validity=False,
                raw_data=metrics
            )
            
        avg_ear = (ear_l + ear_r) / 2.0
        metrics["ear"] = round(avg_ear, 3)

        if avg_ear < config.EAR_THRESHOLD:
            if not self.eye_closed_start_time:
                self.eye_closed_start_time = current_time
            metrics["eye_closed_duration"] = round(current_time - self.eye_closed_start_time, 2)
        else:
            if self.eye_closed_start_time:
                duration = current_time - self.eye_closed_start_time
                if 0.08 <= duration <= 0.4: # Valid dynamic human blink length
                    self.blinks_in_current_minute += 1
                self.eye_closed_start_time = None

        # 2. Head Pose Calculations
        pitch, yaw = self._estimate_head_pose(landmarks, w, h)
        metrics["pitch"] = round(pitch, 1)
        metrics["yaw"] = round(yaw, 1)

        # 3. Micro-gaze tracking
        iris_center_x = landmarks[self.LEFT_IRIS[0]].x
        eye_left_edge_x = landmarks[self.LEFT_EYE[0]].x
        eye_right_edge_x = landmarks[self.LEFT_EYE[3]].x
        
        # Compute normalized positional index within eye contours
        gaze_ratio = (iris_center_x - eye_left_edge_x) / max((eye_right_edge_x - eye_left_edge_x), 0.001)
        
        # Determine macro-gaze (head orientation) first
        if yaw > config.GAZE_HEAD_YAW_THRESH:
            metrics["gaze_direction"] = "Left"
        elif yaw < -config.GAZE_HEAD_YAW_THRESH:
            metrics["gaze_direction"] = "Right"
        elif pitch > config.GAZE_HEAD_PITCH_THRESH:
            metrics["gaze_direction"] = "Down"
        elif pitch < -config.GAZE_HEAD_PITCH_THRESH:
            metrics["gaze_direction"] = "Up"
        else:
            # Fallback to micro-gaze (eye tracking) if head is relatively centered
            if gaze_ratio < config.GAZE_LEFT_THRESH:
                metrics["gaze_direction"] = "Left"
            elif gaze_ratio > config.GAZE_RIGHT_THRESH:
                metrics["gaze_direction"] = "Right"
            else:
                metrics["gaze_direction"] = "Center"

        # Track distraction intervals based on head pose or gaze vector deviation
        is_distracted = (metrics["gaze_direction"] != "Center") or (abs(yaw) > config.YAW_THRESHOLD) or (abs(pitch) > config.PITCH_THRESHOLD)
        if is_distracted:
            if not self.gaze_away_start_time:
                self.gaze_away_start_time = current_time
            metrics["gaze_away_duration"] = round(current_time - self.gaze_away_start_time, 2)
        else:
            self.gaze_away_start_time = None

        # 4. Calculate Risk Penalties
        risk_score = 0.0
        gaze_away = metrics.get("gaze_away_duration", 0.0)
        if gaze_away > 0:
            risk_score += min(gaze_away * config.CSI_DISTRACTION_MULTIPLIER, config.CSI_DISTRACTION_MAX)

        eye_closed = metrics.get("eye_closed_duration", 0.0)
        if eye_closed > config.CSI_DROWSINESS_MIN_DURATION:
            risk_score += min(eye_closed * config.CSI_DROWSINESS_MULTIPLIER, config.CSI_DROWSINESS_MAX)

        if abs(yaw) > config.CSI_YAW_THRESHOLD: risk_score += config.CSI_YAW_PENALTY
        if abs(pitch) > config.CSI_PITCH_THRESHOLD: risk_score += config.CSI_PITCH_PENALTY

        if self.current_bpm > config.CSI_BPM_HIGH or self.current_bpm < config.CSI_BPM_LOW:
            risk_score += config.CSI_BPM_PENALTY

        # Normalize score to 0.0 - 1.0 based on a max expected penalty of 100
        visual_score = min(risk_score / 100.0, 1.0)

        return SensorOutput(
            score=visual_score,
            confidence=0.9,
            timestamp=current_time,
            source="visual_engine",
            validity=True,
            raw_data=metrics
        )