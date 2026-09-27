import time
import numpy as np
from sklearn.ensemble import IsolationForest
from modules.models import SensorOutput
from modules.simulated_steering import SimulatedSteeringGenerator

class KinematicSensor:
    def __init__(self):
        self.sample_rate_hz = 15.0
        self.dt = 1.0 / self.sample_rate_hz
        self.window_size = 30 # 2 seconds of data
        self.steering_buffer = []
        self.model_load_failed = False
        
        try:
            # We train the baseline model in memory instead of loading a serialized model.
            generator = SimulatedSteeringGenerator(sample_rate_hz=self.sample_rate_hz, random_seed=42)
            # Generate 5 minutes of SIMULATED normal driving data for training the baseline
            simulated_normal_data = generator.generate_simulated_normal_data(300.0)
            
            # Extract features for training
            X_train = []
            for i in range(self.window_size, len(simulated_normal_data)):
                window = simulated_normal_data[i-self.window_size:i]
                features = self._extract_features(window)
                X_train.append([
                    features["velocity"],
                    features["jerk"],
                    features["rolling_std"],
                    features["reversal_rate"]
                ])
                
            # Initialize IsolationForest
            # Contamination is set to 0.05 (5%) as a standard initial design choice for an unsupervised 
            # anomaly detector flagging deviations from a baseline, assuming roughly 5% of natural 
            # variations in the normal simulated set might be extreme enough to mark as edges.
            self.model = IsolationForest(contamination=0.05, random_state=42)
            self.model.fit(X_train)
            
        except Exception as e:
            print(f"Failed to initialize SIMULATED kinematic model: {e}")
            self.model_load_failed = True

    def _extract_features(self, window: np.ndarray) -> dict:
        """
        Extracts 4 kinematic features from a window of steering angles using explicit formulas.
        """
        # 1. Velocity (mean absolute velocity)
        # Formula: v_i = (theta_i - theta_{i-1}) / dt
        differences = np.diff(window)
        velocities = differences / self.dt
        mean_abs_velocity = np.mean(np.abs(velocities)) if len(velocities) > 0 else 0.0
        
        # 2. Jerk (mean absolute jerk)
        # Formula: j_i = (v_i - v_{i-1}) / dt
        if len(velocities) > 1:
            jerks = np.diff(velocities) / self.dt
            mean_abs_jerk = np.mean(np.abs(jerks))
        else:
            mean_abs_jerk = 0.0
            
        # 3. Rolling standard deviation
        # Population standard deviation of the window (ddof=0)
        rolling_std = np.std(window, ddof=0)
        
        # 4. Reversal rate
        # Formula: d_i = theta_i - theta_{i-1}
        # Reversal occurs when sign changes between consecutive non-zero differences
        non_zero_diffs = differences[differences != 0]
        reversals = 0
        if len(non_zero_diffs) > 1:
            signs = np.sign(non_zero_diffs)
            sign_changes = np.where(np.diff(signs) != 0)[0]
            reversals = len(sign_changes)
            
        window_duration = len(window) * self.dt
        reversal_rate = reversals / window_duration if window_duration > 0 else 0.0
        
        return {
            "velocity": float(mean_abs_velocity),
            "jerk": float(mean_abs_jerk),
            "rolling_std": float(rolling_std),
            "reversal_rate": float(reversal_rate)
        }

    def process_telemetry(self, current_steering_angle: float = None) -> SensorOutput:
        """
        Processes real-time telemetry. For this session, we assume current_steering_angle 
        is provided (e.g. from SIMULATED test data).
        """
        current_time = time.time()
        
        if self.model_load_failed:
            return SensorOutput(
                score=0.0,
                confidence=0.0,
                timestamp=current_time,
                source="kinematic_sensor",
                validity=False,
                raw_data={"error": "Model initialization failed"}
            )
            
        if current_steering_angle is not None:
            self.steering_buffer.append(current_steering_angle)
            
        # Keep buffer at window_size
        if len(self.steering_buffer) > self.window_size:
            self.steering_buffer = self.steering_buffer[-self.window_size:]
            
        # Cold start handling
        if len(self.steering_buffer) < self.window_size:
            return SensorOutput(
                score=0.0,
                confidence=0.0,
                timestamp=current_time,
                source="kinematic_sensor",
                validity=False,
                raw_data={"status": "buffering"}
            )
            
        # Feature engineering
        window_arr = np.array(self.steering_buffer)
        features = self._extract_features(window_arr)
        
        # Inference
        X_test = [[
            features["velocity"],
            features["jerk"],
            features["rolling_std"],
            features["reversal_rate"]
        ]]
        
        # IsolationForest decision_function returns > 0 for normal, < 0 for anomalies.
        # We negate it so that higher positive values mean "more anomalous" (higher risk).
        # We then apply a sigmoid or min-max mapping. 
        # For simplicity and bound mapping [0, 1], we use a logistic sigmoid mapped around 0.
        # Normal (df > 0) -> negated df < 0 -> sigmoid < 0.5
        # Anomaly (df < 0) -> negated df > 0 -> sigmoid > 0.5
        raw_score = self.model.decision_function(X_test)[0]
        negated_score = -raw_score
        
        # Mapping to 0.0 - 1.0 (Higher means more anomalous)
        # Using a scaled sigmoid to push normal scores lower and anomalous scores higher
        mapped_score = 1.0 / (1.0 + np.exp(-10.0 * negated_score))
        
        is_anomaly = self.model.predict(X_test)[0] == -1
        
        raw_data = {
            "velocity": features["velocity"],
            "jerk": features["jerk"],
            "rolling_std": features["rolling_std"],
            "reversal_rate": features["reversal_rate"],
            "raw_if_score": float(raw_score),
            "is_anomaly": bool(is_anomaly),
            "is_simulated": True
        }
        
        return SensorOutput(
            score=float(mapped_score),
            confidence=0.9, # Valid feature calculation and healthy model
            timestamp=current_time,
            source="kinematic_sensor",
            validity=True,
            raw_data=raw_data
        )
