import pytest
import numpy as np
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from unittest.mock import patch

from modules.kinematic import KinematicSensor
from modules.models import SensorOutput
from modules.simulated_steering import SimulatedSteeringGenerator

def test_feature_engineering_exact_values():
    # SIMULATED input for deterministic feature test
    sensor = KinematicSensor()
    sensor.dt = 1/15.0
    
    # Simple explicit sequence
    window = np.array([0.0, 1.0, -1.0, 2.0, -2.0])
    features = sensor._extract_features(window)
    
    # Differences: [1.0, -2.0, 3.0, -4.0]
    # Velocities: [15.0, -30.0, 45.0, -60.0]
    # mean_abs_velocity = (15+30+45+60)/4 = 150/4 = 37.5
    assert np.isclose(features["velocity"], 37.5)
    
    # Jerk diffs: [-45.0, 75.0, -105.0]
    # Jerks: [-675.0, 1125.0, -1575.0]
    # mean_abs_jerk = (675+1125+1575)/3 = 3375/3 = 1125.0
    assert np.isclose(features["jerk"], 1125.0)
    
    # Rolling std of [0, 1, -1, 2, -2] (mean=0) -> vars [0, 1, 1, 4, 4] -> mean=10/5=2.0 -> std=sqrt(2)
    assert np.isclose(features["rolling_std"], np.sqrt(2.0))
    
    # Reversals: diffs [1, -2, 3, -4] -> 3 reversals
    # window_duration = 5 * (1/15.0) = 1/3
    # rate = 3 / (1/3) = 9.0
    assert np.isclose(features["reversal_rate"], 9.0)

def test_isolation_forest_normal_vs_erratic():
    sensor = KinematicSensor()
    generator = SimulatedSteeringGenerator(sample_rate_hz=15.0, random_seed=99)
    
    # 1. Fresh SIMULATED normal sequence
    normal_data = generator.generate_simulated_normal_data(3.0)
    
    # Feed data to fill window
    for angle in normal_data[:-1]:
        sensor.process_telemetry(angle)
        
    normal_output = sensor.process_telemetry(normal_data[-1])
    
    # 2. SIMULATED erratic test sequence
    erratic_data = generator.generate_simulated_erratic_test_data(3.0)
    
    # Reset sensor buffer for fair test
    sensor.steering_buffer = []
    for angle in erratic_data[:-1]:
        sensor.process_telemetry(angle)
        
    erratic_output = sensor.process_telemetry(erratic_data[-1])
    
    # Verifications
    print(f"\nSIMULATED Normal Score: {normal_output.score}")
    print(f"SIMULATED Erratic Score: {erratic_output.score}")
    
    assert normal_output.score < erratic_output.score
    assert erratic_output.raw_data["is_anomaly"] is True
    # The normal data could potentially trigger an anomaly if it hits an edge, 
    # but the erratic score should be meaningfully higher.
    assert (erratic_output.score - normal_output.score) > 0.3

def test_cold_start():
    sensor = KinematicSensor()
    # Provide only 5 readings (window is 30)
    output = None
    for i in range(5):
        output = sensor.process_telemetry(0.1 * i)
        
    assert output.validity is False
    assert output.score == 0.0
    assert output.confidence == 0.0
    assert output.raw_data.get("status") == "buffering"

@patch('modules.kinematic.IsolationForest.fit')
def test_model_failure(mock_fit):
    mock_fit.side_effect = Exception("Simulated training failure")
    
    sensor = KinematicSensor()
    assert sensor.model_load_failed is True
    
    output = sensor.process_telemetry(0.0)
    assert output.validity is False
    assert output.score == 0.0
    assert output.confidence == 0.0
    assert "error" in output.raw_data

def test_sensor_output_contract():
    sensor = KinematicSensor()
    # Fill window with zeroes
    for _ in range(sensor.window_size):
        output = sensor.process_telemetry(0.0)
        
    assert isinstance(output, SensorOutput)
    assert 0.0 <= output.score <= 1.0
    assert output.confidence == 0.9
    assert output.source == "kinematic_sensor"
    assert output.validity is True
    assert isinstance(output.raw_data, dict)
    assert "velocity" in output.raw_data
    assert "jerk" in output.raw_data
    assert "rolling_std" in output.raw_data
    assert "reversal_rate" in output.raw_data
    assert "raw_if_score" in output.raw_data
    assert output.raw_data.get("is_simulated") is True
