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
