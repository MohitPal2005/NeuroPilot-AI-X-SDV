import pytest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from sklearn.ensemble import IsolationForest
from modules.kinematic import KinematicSensor
from modules.simulated_steering import SimulatedSteeringGenerator

def test_independent_dataset_robustness():
    """
    Test that the existing KinematicSensor (trained on seed=42)
    can properly differentiate normal vs erratic data on a completely
    unseen seed (seed=7) that wasn't used in training or original tests.
    """
    sensor = KinematicSensor()
    generator_7 = SimulatedSteeringGenerator(sample_rate_hz=15.0, random_seed=7)

    # 1. Evaluate normal data
    normal_data_7 = generator_7.generate_simulated_normal_data(3.0)
    
    # Pre-fill buffer for fair start
    sensor.steering_buffer = []
    for angle in normal_data_7[:-1]:
        sensor.process_telemetry(angle)
        
    normal_output_7 = sensor.process_telemetry(normal_data_7[-1])
    
    # 2. Evaluate erratic data
    erratic_data_7 = generator_7.generate_simulated_erratic_test_data(3.0)
    
    # Reset sensor buffer
    sensor.steering_buffer = []
    for angle in erratic_data_7[:-1]:
        sensor.process_telemetry(angle)
        
    erratic_output_7 = sensor.process_telemetry(erratic_data_7[-1])
    
    # Separation assertions
    margin = erratic_output_7.score - normal_output_7.score
    
    assert normal_output_7.score < erratic_output_7.score, "Erratic score should be higher than normal score."
    assert margin > 0.4, f"Separation margin ({margin:.2f}) is too small on independent data."


def test_contamination_sensitivity():
    """
    Run a side experiment to ensure our 0.05 contamination choice
    remains reasonable compared to tighter (0.01) and looser (0.15) bounds.
    """
    sensor = KinematicSensor()
    
    # Train 0.01 and 0.15 models using the same seed=42 normal driving data 
    # as the baseline model inside KinematicSensor uses.
    generator_train = SimulatedSteeringGenerator(sample_rate_hz=15.0, random_seed=42)
    simulated_normal_data_train = generator_train.generate_simulated_normal_data(300.0)
    
    X_train = []
    window_size = sensor.window_size
    for i in range(window_size, len(simulated_normal_data_train)):
        window = simulated_normal_data_train[i-window_size:i]
        features = sensor._extract_features(window)
        X_train.append([
            features["velocity"],
            features["jerk"],
            features["rolling_std"],
            features["reversal_rate"]
        ])
        
    models = {
        0.01: IsolationForest(contamination=0.01, random_state=42),
        0.05: sensor.model, # Use the shipped baseline model
        0.15: IsolationForest(contamination=0.15, random_state=42)
    }
    
    models[0.01].fit(X_train)
    models[0.15].fit(X_train)
    
    # Evaluate against the original test sequences (seed=99)
    generator_99 = SimulatedSteeringGenerator(sample_rate_hz=15.0, random_seed=99)
    normal_data_99 = generator_99.generate_simulated_normal_data(3.0)
    erratic_data_99 = generator_99.generate_simulated_erratic_test_data(3.0)
    
    def get_score_for_model(model, data):
        buffer = list(data[-window_size:])
        window_arr = np.array(buffer)
        features = sensor._extract_features(window_arr)
        X_test = [[
            features["velocity"],
            features["jerk"],
            features["rolling_std"],
            features["reversal_rate"]
        ]]
        raw_score = model.decision_function(X_test)[0]
        negated_score = -raw_score
        mapped_score = 1.0 / (1.0 + np.exp(-10.0 * negated_score))
        return float(mapped_score)

    scores = {}
    for c in [0.01, 0.05, 0.15]:
        scores[c] = {
            "normal": get_score_for_model(models[c], normal_data_99),
            "erratic": get_score_for_model(models[c], erratic_data_99)
        }
        
    # Assertions
    # 0.05 is our shipped configuration and should cleanly separate normal vs erratic
    assert scores[0.05]["normal"] < 0.35, "0.05 contamination flags normal data too harshly"
    assert scores[0.05]["erratic"] > 0.70, "0.05 contamination fails to strongly flag erratic data"
    
    # 0.15 should naturally score normal data higher than 0.05 (too many false positives)
    assert scores[0.15]["normal"] > scores[0.05]["normal"], "0.15 contamination should yield higher baseline scores"
    
    # 0.01 should score erratic data lower than 0.05 (less sensitive)
    assert scores[0.01]["erratic"] < scores[0.05]["erratic"], "0.01 contamination should be less sensitive to anomalies"
