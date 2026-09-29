import pytest
import time
from app import generate_sensor_status
from modules.models import SensorOutput

def test_generate_sensor_status_all_active():
    t = time.time()
    visual = SensorOutput(score=0.1, confidence=0.9, timestamp=t, source="visual_engine", validity=True, raw_data={})
    acoustic = SensorOutput(score=0.2, confidence=0.8, timestamp=t, source="acoustic_sensor", validity=True, raw_data={})
    kinematic = SensorOutput(score=0.0, confidence=0.9, timestamp=t, source="kinematic_sensor", validity=True, raw_data={})
    
    status = generate_sensor_status(visual, acoustic, kinematic)
    
    assert status["visual"] == "active"
    assert status["acoustic"] == "active"
    assert status["kinematic"] == "active"

def test_generate_sensor_status_degraded_and_buffering():
    t = time.time()
    # Visual is invalid -> degraded
    visual = SensorOutput(score=0.0, confidence=0.0, timestamp=t, source="visual_engine", validity=False, raw_data={"error": "no face"})
    # Acoustic is invalid but not buffering -> degraded
    acoustic = SensorOutput(score=0.0, confidence=0.0, timestamp=t, source="acoustic_sensor", validity=False, raw_data={"error": "model load failed"})
    # Kinematic is invalid due to buffering -> buffering
    kinematic = SensorOutput(score=0.0, confidence=0.0, timestamp=t, source="kinematic_sensor", validity=False, raw_data={"status": "buffering"})
    
    status = generate_sensor_status(visual, acoustic, kinematic)
    
    assert status["visual"] == "degraded"
    assert status["acoustic"] == "degraded"
    assert status["kinematic"] == "buffering"
