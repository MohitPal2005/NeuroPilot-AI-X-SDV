import pytest
import cv2
import numpy as np

from modules.models import SensorOutput
from modules.vision import AdvancedVisionEngine
from modules.acoustic import AcousticSensor
from modules.kinematic import KinematicSensor

def test_visual_sensor_contract():
    engine = AdvancedVisionEngine()
    # Create a dummy frame (black image) to trigger face_detected=False path
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    output = engine.process_frame(frame)
    
    assert isinstance(output, SensorOutput)
    assert 0.0 <= output.score <= 1.0
    assert 0.0 <= output.confidence <= 1.0
    assert output.source == "visual_engine"
    assert isinstance(output.validity, bool)
    assert isinstance(output.raw_data, dict)

def test_visual_sensor_gaze_orientation(monkeypatch):
    engine = AdvancedVisionEngine()
    
    # We will mock _estimate_head_pose to simulate head turns without needing perfect 2D landmarks
    def mock_head_pose(landmarks, w, h):
        return pitch_val, yaw_val
    monkeypatch.setattr(engine, "_estimate_head_pose", mock_head_pose)
    
    # Generate a dummy frame with a generic white patch to fool face detection enough to pass if we mock face_mesh?
    # Actually, we can just mock face_mesh.process to return dummy landmarks
    import mediapipe as mp
    from unittest.mock import Mock
    
    dummy_results = Mock()
    dummy_landmarks = Mock()
    # Create 478 mock landmarks
    mock_lms = [Mock(x=0.5, y=0.5) for _ in range(478)]
    
    # Left eye edges
    mock_lms[362] = Mock(x=0.4, y=0.5) # left edge (LEFT_EYE[0])
    mock_lms[263] = Mock(x=0.6, y=0.5) # right edge (LEFT_EYE[3])
    # Left iris
    mock_lms[474] = Mock(x=0.5, y=0.5) # iris center (LEFT_IRIS[0])
    
    # Right eye edges (to avoid degenerate EAR division by zero)
    mock_lms[33] = Mock(x=0.7, y=0.5) # right edge (RIGHT_EYE[0])
    mock_lms[133] = Mock(x=0.9, y=0.5) # left edge (RIGHT_EYE[3])
    
    dummy_landmarks.landmark = mock_lms
    dummy_results.multi_face_landmarks = [dummy_landmarks]
    
    mock_face_mesh = Mock()
    mock_face_mesh.process.return_value = dummy_results
    engine.face_mesh = mock_face_mesh
    
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    
    # Test Left
    global pitch_val, yaw_val
    pitch_val, yaw_val = 0.0, 25.0
    output = engine.process_frame(frame)
    assert output.raw_data["gaze_direction"] == "Left"
    
    # Test Right
    pitch_val, yaw_val = 0.0, -25.0
    output = engine.process_frame(frame)
    assert output.raw_data["gaze_direction"] == "Right"
    
    # Test Up
    pitch_val, yaw_val = -25.0, 0.0
    output = engine.process_frame(frame)
    assert output.raw_data["gaze_direction"] == "Up"
    
    # Test Down
    pitch_val, yaw_val = 25.0, 0.0
    output = engine.process_frame(frame)
    assert output.raw_data["gaze_direction"] == "Down"
    
    # Test Center (fallback to eye gaze, which with x=0.5 will be Center)
    pitch_val, yaw_val = 0.0, 0.0
    output = engine.process_frame(frame)
    assert output.raw_data["gaze_direction"] == "Center"

def test_visual_sensor_degenerate_ear():
    engine = AdvancedVisionEngine()
    
    from unittest.mock import Mock
    
    dummy_results = Mock()
    dummy_landmarks = Mock()
    # Create mock landmarks where eye edges are identical (h = 0)
    mock_lms = [Mock(x=0.5, y=0.5) for _ in range(478)]
    
    dummy_landmarks.landmark = mock_lms
    dummy_results.multi_face_landmarks = [dummy_landmarks]
    
    mock_face_mesh = Mock()
    mock_face_mesh.process.return_value = dummy_results
    engine.face_mesh = mock_face_mesh
    
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    
    output = engine.process_frame(frame)
    # Confirm process_frame aborts gracefully and returns standard fallback
    assert output.validity is False
    assert output.confidence == 0.0

def test_acoustic_sensor_contract():
    sensor = AcousticSensor()
    output = sensor.process_audio()
    
    assert isinstance(output, SensorOutput)
    assert 0.0 <= output.score <= 1.0
    assert 0.0 <= output.confidence <= 1.0
    assert output.source == "acoustic_stub"
    assert output.validity is True
    assert isinstance(output.raw_data, dict)
    assert "arousal" in output.raw_data

def test_kinematic_sensor_contract():
    sensor = KinematicSensor()
    output = sensor.process_telemetry()
    
    assert isinstance(output, SensorOutput)
    assert 0.0 <= output.score <= 1.0
    assert 0.0 <= output.confidence <= 1.0
    assert output.source == "kinematic_stub"
    assert output.validity is True
    assert isinstance(output.raw_data, dict)
    assert "jerk" in output.raw_data

from modules.engine import CognitiveStateEngine

class TestConfidenceFusion:
    def setup_method(self):
        self.engine = CognitiveStateEngine()
        
    def _create_mock_output(self, score, confidence):
        import time
        return SensorOutput(
            score=score,
            confidence=confidence,
            timestamp=time.time(),
            source="mock_sensor",
            validity=True,
            raw_data={}
        )

    def test_all_channels_healthy_and_in_agreement(self):
        # Scenario 1: All three channels healthy (low risk) and in agreement
        out_v = self._create_mock_output(0.1, 0.9)
        out_a = self._create_mock_output(0.1, 0.8)
        out_k = self._create_mock_output(0.1, 0.85)
        
        result = self.engine.compute_csi([out_v, out_a, out_k])
        # Expected fused_score = 0.1
        # CSI = 10 (base) + 0.1 * 100 = 20
        assert result["csi"] == 20
        assert result["zone"] == "Safe"

    def test_one_channel_high_risk_others_normal(self):
        # Scenario 2: One channel high risk, others normal
        out_v = self._create_mock_output(0.9, 0.9)  # High risk
        out_a = self._create_mock_output(0.1, 0.8)
        out_k = self._create_mock_output(0.1, 0.85)
        
        result = self.engine.compute_csi([out_v, out_a, out_k])
        # Expected fused_score = (0.9*0.9 + 0.1*0.8 + 0.1*0.85) / (0.9 + 0.8 + 0.85)
        # = (0.81 + 0.08 + 0.085) / 2.55 = 0.975 / 2.55 ≈ 0.382
        # CSI = 10 + 38.2 = 48 -> Cognitive Overload (41-60)
        assert result["csi"] == 48
        assert result["zone"] == "Cognitive Overload"

    def test_one_channel_very_low_confidence(self):
        # Scenario 3: One channel high risk but very low confidence (should be ignored)
        out_v = self._create_mock_output(0.1, 0.9)
        out_a = self._create_mock_output(0.1, 0.8)
        out_k = self._create_mock_output(1.0, 0.05) # High risk, tiny confidence
        
        result = self.engine.compute_csi([out_v, out_a, out_k])
        # Expected fused_score = (0.1*0.9 + 0.1*0.8 + 1.0*0.05) / (1.75)
        # = (0.09 + 0.08 + 0.05) / 1.75 = 0.22 / 1.75 ≈ 0.125
        # CSI = 10 + 12.5 = 22. 
        # Since it starts at Safe (0-20), it needs to clear 20 + 3 (margin) = 23 to transition up.
        # So it remains in Safe.
        assert result["csi"] == 22
        assert result["zone"] == "Safe"

    def test_conflicting_signals(self):
        # Scenario 4: Conflicting signals (one high risk, one low risk, both moderate confidence)
        out_v = self._create_mock_output(0.9, 0.7)
        out_a = self._create_mock_output(0.1, 0.7)
        # Omit kinematic for simple 2-channel conflict
        
        result = self.engine.compute_csi([out_v, out_a])
        # Expected fused_score = (0.63 + 0.07) / 1.4 = 0.7 / 1.4 = 0.5
        # CSI = 10 + 50 = 60 -> Cognitive Overload
        assert result["csi"] == 60
        assert result["zone"] == "Cognitive Overload"

    def test_no_data_fallback(self):
        # Edge Case: zero confidence
        out_v = self._create_mock_output(0.5, 0.0)
        
        result = self.engine.compute_csi([out_v])
        assert result["csi"] == 0
        assert result["zone"] == "No Data (Calibrating)"

    def test_real_camera_occlusion(self):
        # Scenario: The real vision pipeline produces a no-face SensorOutput,
        # and it's fused with the acoustic and kinematic stubs.
        # This exercises the gap where the stubs' valid data caused the fusion
        # engine to produce CSI=34 instead of "No Data (Calibrating)".
        
        from modules.vision import AdvancedVisionEngine
        vision_engine = AdvancedVisionEngine()
        
        # Create a black frame simulating camera occlusion (no face)
        black_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        visual_output = vision_engine.process_frame(black_frame)
        
        # Verify vision pipeline semantics
        assert visual_output.validity is False
        assert visual_output.confidence == 0.0
        
        # Simulate the stubs
        acoustic_sensor = AcousticSensor()
        kinematic_sensor = KinematicSensor()
        acoustic_output = acoustic_sensor.process_audio()
        kinematic_output = kinematic_sensor.process_telemetry()
        
        # Pass all 3 to the fusion engine, just like app.py does
        result = self.engine.compute_csi([visual_output, acoustic_output, kinematic_output])
        
        # The fusion engine MUST produce No Data (Calibrating) because the primary 
        # visual sensor is invalid (even though stubs are present)
        assert result["csi"] == 0
        assert result["zone"] == "No Data (Calibrating)"
