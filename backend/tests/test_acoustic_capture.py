import pytest
import numpy as np
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from unittest.mock import patch
from modules.acoustic_capture import AcousticCapture

def test_resampling():
    capture = AcousticCapture(target_sr=16000)
    # Generate 1 second of 44.1kHz audio
    raw_audio = np.random.uniform(-1, 1, 44100).astype(np.float32)
    audio_16k = capture.resample_to_target(raw_audio, 44100)
    
    # Should be exactly 16000 samples
    assert len(audio_16k) == 16000
    assert audio_16k.dtype == np.float32
    
def test_resampling_no_op():
    capture = AcousticCapture(target_sr=16000)
    raw_audio = np.random.uniform(-1, 1, 16000).astype(np.float32)
    audio_16k = capture.resample_to_target(raw_audio, 16000)
    # Should not be modified
    assert raw_audio is audio_16k

def test_vad_silence():
    capture = AcousticCapture(target_sr=16000)
    # Generate 1 second of silence
    audio_16k = np.zeros(16000, dtype=np.float32)
    has_speech = capture.detect_vad(audio_16k)
    assert has_speech is False

def test_vad_speech():
    capture = AcousticCapture(target_sr=16000)
    
    # We construct a deterministic 30ms frame simulating the spectral envelope of human speech
    # Fundamental at 120Hz (typical adult male), formants at 600Hz and 1200Hz
    t = np.linspace(0, 0.03, 480, endpoint=False)
    signal = (
        0.5 * np.sin(2 * np.pi * 120 * t) +
        0.2 * np.sin(2 * np.pi * 600 * t) +
        0.1 * np.sin(2 * np.pi * 1200 * t)
    )
    audio_16k = signal.astype(np.float32)
    
    has_speech = capture.detect_vad(audio_16k)
    # The GMM should confidently detect this harmonic energy as speech
    assert has_speech is True

@patch('sounddevice.query_devices')
def test_no_microphone_initialization(mock_query):
    # Simulate no microphone available
    mock_query.return_value = []
    
    capture = AcousticCapture()
    
    # Should not crash, but enter error state
    assert capture.init_error is not None
    assert capture.is_running is False
    
    # Calling start should gracefully fail
    assert capture.start() is False
    
    # get_latest_capture should return the fallback (None, None)
    has_speech, audio = capture.get_latest_capture()
    assert has_speech is None
    assert audio is None

@patch('sounddevice.InputStream')
@patch('sounddevice.query_devices')
def test_microphone_permission_failure(mock_query, mock_stream):
    # Microphone exists
    mock_query.return_value = {'default_samplerate': 44100}
    
    # But opening the stream throws a permission/OS error
    mock_stream.side_effect = Exception("OS Permission Denied")
    
    capture = AcousticCapture()
    
    # Starts okay, init didn't fail at query
    assert capture.init_error is None
    
    # But start() fails
    assert capture.start() is False
    assert "OS Permission Denied" in capture.init_error
    assert capture.is_running is False
