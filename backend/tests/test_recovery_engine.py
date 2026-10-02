import time
import pytest
from unittest.mock import patch
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from modules.recovery_engine import RecoveryEngine
from modules.models import SensorOutput

def create_sensor_outputs(
    vis_valid=True, eye_closed=0.0, gaze_away=0.0,
    kin_valid=True, is_anomaly=False,
    ac_valid=True, arousal=0.0
):
    visual = SensorOutput(score=0.0, confidence=0.0, timestamp=0.0, source="visual", validity=vis_valid, raw_data={"eye_closed_duration": eye_closed, "gaze_away_duration": gaze_away})
    acoustic = SensorOutput(score=0.0, confidence=0.0, timestamp=0.0, source="acoustic", validity=ac_valid, raw_data={"arousal": arousal})
    kinematic = SensorOutput(score=0.0, confidence=0.0, timestamp=0.0, source="kinematic", validity=kin_valid, raw_data={"is_anomaly": is_anomaly})
    return [visual, acoustic, kinematic]

def test_eye_trigger_fires():
    engine = RecoveryEngine()
    sensors = create_sensor_outputs(eye_closed=5.0)
    res = engine.process({"zone": "Safe"}, sensor_outputs=sensors, current_time=0.0)
    assert res["state"] == "INTERVENING"
    assert res["trigger_reason"] == "EYE"

def test_head_trigger_fires():
    engine = RecoveryEngine()
    sensors = create_sensor_outputs(gaze_away=4.0)
    res = engine.process({"zone": "Safe"}, sensor_outputs=sensors, current_time=0.0)
    assert res["state"] == "INTERVENING"
    assert res["trigger_reason"] == "HEAD"

def test_kinematic_trigger_fires_after_sustained():
    engine = RecoveryEngine()
    sensors = create_sensor_outputs(is_anomaly=True)
    
    res = engine.process({"zone": "Safe"}, sensor_outputs=sensors, current_time=0.0)
    assert res["state"] == "IDLE"
    
    res = engine.process({"zone": "Safe"}, sensor_outputs=sensors, current_time=2.9)
    assert res["state"] == "IDLE"
    
    res = engine.process({"zone": "Safe"}, sensor_outputs=sensors, current_time=3.1)
    assert res["state"] == "INTERVENING"
    assert res["trigger_reason"] == "KINEMATIC"

def test_acoustic_trigger_fires_after_sustained():
    engine = RecoveryEngine()
    sensors = create_sensor_outputs(arousal=0.66)
    
    res = engine.process({"zone": "Safe"}, sensor_outputs=sensors, current_time=0.0)
    assert res["state"] == "IDLE"
    
    res = engine.process({"zone": "Safe"}, sensor_outputs=sensors, current_time=3.1)
    assert res["state"] == "INTERVENING"
    assert res["trigger_reason"] == "ACOUSTIC"

def test_triggers_ignore_invalid_sensors():
    engine = RecoveryEngine()
    # EYE condition met but invalid
    sensors = create_sensor_outputs(vis_valid=False, eye_closed=5.0)
    res = engine.process({"zone": "Safe"}, sensor_outputs=sensors, current_time=0.0)
    assert res["state"] == "IDLE"

    # KINEMATIC condition met but invalid
    sensors = create_sensor_outputs(kin_valid=False, is_anomaly=True)
    engine.process({"zone": "Safe"}, sensor_outputs=sensors, current_time=0.0)
    res = engine.process({"zone": "Safe"}, sensor_outputs=sensors, current_time=3.1)
    assert res["state"] == "IDLE"

def test_priority_arbitration():
    engine = RecoveryEngine()
    # EYE and KINEMATIC both true. EYE (1) > KINEMATIC (3)
    sensors = create_sensor_outputs(eye_closed=5.0, is_anomaly=True)
    
    engine.process({"zone": "Safe"}, sensor_outputs=sensors, current_time=0.0)
    res = engine.process({"zone": "Safe"}, sensor_outputs=sensors, current_time=3.1)
    
    assert res["state"] == "INTERVENING"
    assert res["trigger_reason"] == "EYE"

def test_overall_trigger_regression():
    engine = RecoveryEngine()
    sensors = create_sensor_outputs()
    
    # t=0, starts risk
    engine.process({"zone": "Critical Risk"}, sensor_outputs=sensors, current_time=0.0)
    assert engine.state == "IDLE"
    
    # t=3.1, crosses 3s warning threshold
    res = engine.process({"zone": "High Risk"}, sensor_outputs=sensors, current_time=3.1)
    assert res["state"] == "WARNING"
    
    # t=5.2, crosses additional 2s intervening threshold (total 5s)
    res = engine.process({"zone": "Critical Risk"}, sensor_outputs=sensors, current_time=5.2)
    assert res["state"] == "INTERVENING"
    assert res["trigger_reason"] == "OVERALL"

def test_bridge_stale_reset_logic():
    engine = RecoveryEngine()
    sensors = create_sensor_outputs(is_anomaly=True)
    
    # KINEMATIC starts at t=0
    engine.process({"zone": "Safe"}, sensor_outputs=sensors, current_time=0.0)
    engine.process({"zone": "Safe"}, sensor_outputs=sensors, current_time=1.0)
    
    # Drops out at t=1.1 for 1.4 seconds (less than 2.0s bridge)
    sensors_safe = create_sensor_outputs(is_anomaly=False)
    engine.process({"zone": "Safe"}, sensor_outputs=sensors_safe, current_time=1.1)
    engine.process({"zone": "Safe"}, sensor_outputs=sensors_safe, current_time=2.5)
    
    # Returns at t=2.6
    engine.process({"zone": "Safe"}, sensor_outputs=sensors, current_time=2.6)
    
    # Total time from t=0 is 3.1s, should trigger because the gap was bridged!
    res = engine.process({"zone": "Safe"}, sensor_outputs=sensors, current_time=3.1)
    assert res["state"] == "INTERVENING"
    assert res["trigger_reason"] == "KINEMATIC"
    
    # Now test a gap > 2.0s
    engine2 = RecoveryEngine()
    engine2.process({"zone": "Safe"}, sensor_outputs=sensors, current_time=0.0)
    engine2.process({"zone": "Safe"}, sensor_outputs=sensors_safe, current_time=1.0)
    
    # Gap is 2.5s (> 2.0s), so the tracker should reset at t=3.5
    engine2.process({"zone": "Safe"}, sensor_outputs=sensors_safe, current_time=3.5)
    
    # Return at t=3.6
    engine2.process({"zone": "Safe"}, sensor_outputs=sensors, current_time=3.6)
    
    # Total time from t=0 is 4.0s, but because it reset, duration is only 0.4s
    res = engine2.process({"zone": "Safe"}, sensor_outputs=sensors, current_time=4.0)
    assert res["state"] == "IDLE"

def test_no_trigger_momentary_spike():
    engine = RecoveryEngine()
    sensors = create_sensor_outputs()
    
    # Send high risk at t=0
    engine.process({"zone": "High Risk"}, sensor_outputs=sensors, current_time=0.0)
    assert engine.state == "IDLE"
    
    # Send safe at t=2 (before IDLE_TO_WARNING_SEC which is 3.0)
    res = engine.process({"zone": "Safe"}, sensor_outputs=sensors, current_time=2.0)
    assert res["state"] == "IDLE"

def test_triggers_after_sustained_risk():
    engine = RecoveryEngine()
    sensors = create_sensor_outputs()
    
    # t=0, starts risk
    engine.process({"zone": "Critical Risk"}, sensor_outputs=sensors, current_time=0.0)
    assert engine.state == "IDLE"
    
    # t=3.1, crosses 3s warning threshold
    res = engine.process({"zone": "High Risk"}, sensor_outputs=sensors, current_time=3.1)
    assert res["state"] == "WARNING"
    
    # t=5.2, crosses additional 2s intervening threshold (total 5s)
    with patch.object(engine, '_speak_async') as mock_speak:
        res = engine.process({"zone": "Critical Risk"}, sensor_outputs=sensors, current_time=5.2)
        assert res["state"] == "INTERVENING"
        assert "SIMULATED" in res["ambient_action"]
        assert "Suggesting" in res["reroute_suggestion"]
        mock_speak.assert_called_once()
        
        # Another processing tick at t=6.0 should NOT call speak again
        res2 = engine.process({"zone": "High Risk"}, sensor_outputs=sensors, current_time=6.0)
        assert res2["state"] == "INTERVENING"
        mock_speak.assert_called_once()  # Call count remains 1

def test_recovery_transition():
    engine = RecoveryEngine()
    sensors = create_sensor_outputs()
    
    # Force into INTERVENING state
    engine.state = "INTERVENING"
    engine.ambient_action = "SIMULATED"
    engine.reroute_suggestion = "Route"
    
    # t=10.0, start safe
    engine.process({"zone": "Safe"}, sensor_outputs=sensors, current_time=10.0)
    assert engine.state == "INTERVENING"
    
    # t=14.0, still safe but not 5s yet
    engine.process({"zone": "Attention Required"}, sensor_outputs=sensors, current_time=14.0)
    assert engine.state == "INTERVENING"
    
    # t=15.1, crosses 5s recovery threshold
    res = engine.process({"zone": "Safe"}, sensor_outputs=sensors, current_time=15.1)
    assert res["state"] == "RECOVERED"
    assert res["ambient_action"] is None
    
    # Wait for IDLE cooldown (3s)
    res = engine.process({"zone": "Safe"}, sensor_outputs=sensors, current_time=18.2)
    assert res["state"] == "IDLE"

def test_tts_failure_graceful(caplog):
    engine = RecoveryEngine()
    engine.tts_enabled = True
    
    # Force _speak_async to actually run its thread but patch pyttsx3.init to fail
    with patch('pyttsx3.init', side_effect=Exception("Mocked TTS Failure")):
        engine._speak_async("Test")
        time.sleep(0.1) # Brief wait for the thread to catch the exception and log
        
    assert "TTS Failure" in caplog.text
    # Engine state should remain valid, no crash
    assert engine.state == "IDLE"

def test_bridge_brief_fluctuation():
    engine = RecoveryEngine()
    sensors = create_sensor_outputs()
    
    # Start high risk
    engine.process({"zone": "High Risk"}, sensor_outputs=sensors, current_time=0.0)
    
    # 2 seconds in, dip to Cognitive Overload (bridge state)
    res = engine.process({"zone": "Cognitive Overload"}, sensor_outputs=sensors, current_time=2.0)
    assert res["state"] == "IDLE"
    
    # 2.5 seconds, back to High Risk
    res = engine.process({"zone": "High Risk"}, sensor_outputs=sensors, current_time=2.5)
    
    # 3.1 seconds, 3s warning threshold reached despite the dip
    res = engine.process({"zone": "High Risk"}, sensor_outputs=sensors, current_time=3.1)
    assert res["state"] == "WARNING"

def test_bridge_long_dropout():
    engine = RecoveryEngine()
    sensors = create_sensor_outputs()
    
    # Start high risk
    engine.process({"zone": "High Risk"}, sensor_outputs=sensors, current_time=0.0)
    
    # Dropout to No Data for 3 seconds (> 2s bridge limit)
    res = engine.process({"zone": "No Data (Calibrating)"}, sensor_outputs=sensors, current_time=2.5)
    
    # Return to High Risk at 3.0s
    res = engine.process({"zone": "High Risk"}, sensor_outputs=sensors, current_time=3.0)
    assert res["state"] == "IDLE"  # Shouldn't trigger warning yet, needs fresh 3s
    
    # 5.5s (2.5s later), still IDLE
    res = engine.process({"zone": "High Risk"}, sensor_outputs=sensors, current_time=5.5)
    assert res["state"] == "IDLE"
    
    # 6.1s (3.1s since return), triggers WARNING
    res = engine.process({"zone": "High Risk"}, sensor_outputs=sensors, current_time=6.1)
    assert res["state"] == "WARNING"

def test_speech_cooldown_suppresses_rapid_fire():
    engine = RecoveryEngine()
    engine.SPEECH_COOLDOWN_SEC = 10.0
    sensors = create_sensor_outputs()
    
    with patch.object(engine, '_speak_async') as mock_speak:
        # 1. Trigger first intervention at t=5.2
        engine.process({"zone": "High Risk"}, sensor_outputs=sensors, current_time=0.0)
        engine.process({"zone": "High Risk"}, sensor_outputs=sensors, current_time=3.1)
        res1 = engine.process({"zone": "Critical Risk"}, sensor_outputs=sensors, current_time=5.2)
        
        assert res1["state"] == "INTERVENING"
        assert res1["trigger_reason"] == "OVERALL"
        assert mock_speak.call_count == 1
        
        # 2. Force state to WARNING to simulate dropping out of INTERVENING
        engine.state = "WARNING"
        
        # 3. Trigger intervention again at t=7.0 (within 10s cooldown)
        res2 = engine.process({"zone": "Critical Risk"}, sensor_outputs=sensors, current_time=7.0)
        
        assert res2["state"] == "INTERVENING"
        assert res2["trigger_reason"] == "OVERALL"
        assert "SIMULATED" in res2["ambient_action"]
        assert mock_speak.call_count == 1  # Cooldown active, no new speech

def test_speech_cooldown_allows_subsequent_speech():
    engine = RecoveryEngine()
    engine.SPEECH_COOLDOWN_SEC = 10.0
    sensors = create_sensor_outputs()
    
    with patch.object(engine, '_speak_async') as mock_speak:
        # 1. Trigger first intervention at t=5.2
        engine.process({"zone": "High Risk"}, sensor_outputs=sensors, current_time=0.0)
        engine.process({"zone": "High Risk"}, sensor_outputs=sensors, current_time=3.1)
        res1 = engine.process({"zone": "Critical Risk"}, sensor_outputs=sensors, current_time=5.2)
        
        assert res1["state"] == "INTERVENING"
        assert mock_speak.call_count == 1
        
        # 2. Force state to WARNING
        engine.state = "WARNING"
        
        # 3. Trigger intervention again at t=16.0 (past 10s cooldown)
        res2 = engine.process({"zone": "Critical Risk"}, sensor_outputs=sensors, current_time=16.0)
        
        assert res2["state"] == "INTERVENING"
        assert mock_speak.call_count == 2  # Cooldown expired, speech triggers again
