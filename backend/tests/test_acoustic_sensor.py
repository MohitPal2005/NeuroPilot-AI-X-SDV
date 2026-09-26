import pytest
import numpy as np
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from unittest.mock import patch, MagicMock
from modules.acoustic import AcousticSensor
from modules.acoustic_capture import AcousticCapture

@pytest.fixture(scope="module")
def sensor():
    capture = MagicMock(spec=AcousticCapture)
    capture.target_sr = 16000
    
    # We load the real model once for the module to save time across tests
    return AcousticSensor(capture)

def test_acoustic_sensor_sane_output(sensor):
    # Fake audio input: 1.5 seconds of random noise
    fake_audio = np.random.uniform(-0.1, 0.1, 24000).astype(np.float32)
    
    # Mock capture to return this audio
    sensor.capture.get_latest_capture.return_value = (True, fake_audio)
    sensor.capture.detect_vad.return_value = True
    
    # Reset internal buffer just in case
    sensor.audio_buffer = np.array([], dtype=np.float32)
    
    output = sensor.process_audio()
    
    assert output.validity is True
    assert "arousal" in output.raw_data
    assert "dominance" in output.raw_data
    assert "valence" in output.raw_data
    
    # Sane range check for raw logits or normalized values. 
    # Usually, dimensional emotion outputs are roughly in the range [0, 1] or slightly outside.
    for key in ["arousal", "dominance", "valence"]:
        val = output.raw_data[key]
        assert -10.0 <= val <= 10.0  # Broad check to ensure it's a valid float output
        
    assert output.score == output.raw_data["arousal"]

def test_acoustic_sensor_confidence_vad(sensor):
    fake_audio = np.random.uniform(-0.1, 0.1, 24000).astype(np.float32)
    
    # Test with speech detected
    sensor.audio_buffer = np.array([], dtype=np.float32)
    sensor.capture.get_latest_capture.return_value = (True, fake_audio)
    sensor.capture.detect_vad.return_value = True
    
    out_speech = sensor.process_audio()
    
    # Test with no speech detected
    sensor.audio_buffer = np.array([], dtype=np.float32)
    sensor.capture.get_latest_capture.return_value = (False, fake_audio)
    sensor.capture.detect_vad.return_value = False
    
    out_no_speech = sensor.process_audio()
    
    assert out_speech.confidence > out_no_speech.confidence
    assert out_speech.confidence == 0.9
    assert out_no_speech.confidence == 0.4

@patch('modules.acoustic.EmotionModel.from_pretrained')
def test_acoustic_sensor_load_failure(mock_from_pretrained):
    mock_from_pretrained.side_effect = Exception("Simulated network failure")
    
    capture = MagicMock(spec=AcousticCapture)
    capture.target_sr = 16000
    
    # Initialize should not crash
    bad_sensor = AcousticSensor(capture)
    
    assert bad_sensor.load_failed is True
    
    # Process audio should return validity=False
    output = bad_sensor.process_audio()
    
    assert output.validity is False
    assert output.raw_data.get("error") == "model load failed"
