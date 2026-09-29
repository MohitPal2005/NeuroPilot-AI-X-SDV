import sys
import os
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from modules.engine import CognitiveStateEngine
from modules.models import SensorOutput
import time

def test_3a_all_valid_agreement():
    engine = CognitiveStateEngine()
    outputs = [
        SensorOutput(score=0.1, confidence=0.9, source="visual_engine", validity=True),
        SensorOutput(score=0.1, confidence=0.4, source="acoustic_sensor", validity=True),
        SensorOutput(score=0.1, confidence=0.9, source="kinematic_sensor", validity=True)
    ]
    result = engine.compute_csi(outputs)
    # Expected weighted score = 0.1
    # Expected CSI = 10.0 + (0.1 * 100) = 20.0
    print(f"\n[3a] Actual Result: {result}")
    assert result["zone"] == "Safe"

def test_3b_all_valid_disagreement():
    engine = CognitiveStateEngine()
    outputs = [
        SensorOutput(score=0.0, confidence=0.9, source="visual_engine", validity=True),   # Visual Safe
        SensorOutput(score=0.1, confidence=0.9, source="acoustic_sensor", validity=True), # Acoustic Calm
        SensorOutput(score=0.9, confidence=0.9, source="kinematic_sensor", validity=True) # Kinematic Risk
    ]
    result = engine.compute_csi(outputs)
    # Total conf = 2.7
    # Weighted score sum = (0.0*0.9) + (0.1*0.9) + (0.9*0.9) = 0.0 + 0.09 + 0.81 = 0.9
    # Fused score = 0.9 / 2.7 = 0.333
    # CSI = 10 + 33.3 = 43
    print(f"\n[3b] Actual Result: {result}")
    assert result["csi"] > 20
    
    # Note: CSI=43 lands exactly on the hysteresis upper boundary (40 base + 3 margin) for the 
    # "Attention Required" zone. Because the transition logic requires `csi > bounds + margin` 
    # (i.e., > 43), it stays in "Attention Required" rather than crossing into "Cognitive Overload".
    assert result["zone"] == "Attention Required"

def test_3c_two_valid_one_invalid_buffering():
    engine = CognitiveStateEngine()
    # Visual and acoustic show REAL high risk
    # Kinematic is buffering (invalid)
    outputs = [
        SensorOutput(score=0.9, confidence=0.9, source="visual_engine", validity=True),
        SensorOutput(score=0.9, confidence=0.9, source="acoustic_sensor", validity=True),
        SensorOutput(score=0.0, confidence=0.0, source="kinematic_sensor", validity=False, raw_data={"status": "buffering"})
    ]
    result = engine.compute_csi(outputs)
    # With the all-invalid abort logic, the fused CSI should now reflect the valid sensors
    # Fused score = (0.9*0.9 + 0.9*0.9) / (0.9+0.9) = 1.62 / 1.8 = 0.9
    # CSI = 10 (base) + 90 = 100
    print(f"\n[3c] Actual Result: {result}")
    assert result["csi"] == 100
    assert result["zone"] == "Critical Risk"

def test_3d_all_invalid():
    engine = CognitiveStateEngine()
    outputs = [
        SensorOutput(score=0.0, confidence=0.0, source="visual_engine", validity=False),
        SensorOutput(score=0.0, confidence=0.0, source="acoustic_sensor", validity=False),
        SensorOutput(score=0.0, confidence=0.0, source="kinematic_sensor", validity=False)
    ]
    result = engine.compute_csi(outputs)
    print(f"\n[3d] Actual Result: {result}")
    assert result["csi"] == 0
    assert result["zone"] == "No Data (Calibrating)"
