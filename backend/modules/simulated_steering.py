import numpy as np

class SimulatedSteeringGenerator:
    """
    Generates SIMULATED steering angle time series for baseline anomaly detection training and testing.
    This is strictly SIMULATED data and does not represent real IMU/steering sensor hardware.
    """
    
    def __init__(self, sample_rate_hz=15.0, random_seed=42):
        self.sample_rate_hz = sample_rate_hz
        self.dt = 1.0 / sample_rate_hz
        self.rng = np.random.default_rng(random_seed)

    def generate_simulated_normal_data(self, duration_sec: float) -> np.ndarray:
        """
        Generates SIMULATED normal driving behavior characterized by small corrections 
        and smooth transitions.
        """
        num_samples = int(duration_sec * self.sample_rate_hz)
        
        # Base wander (slow changing)
        time = np.linspace(0, duration_sec, num_samples)
        wander = 5.0 * np.sin(0.2 * np.pi * time) + 2.0 * np.cos(0.5 * np.pi * time)
        
        # Small high-frequency corrections (smooth noise)
        noise = self.rng.normal(0, 0.5, num_samples)
        
        # Smooth the noise slightly
        window = np.ones(5) / 5.0
        smoothed_noise = np.convolve(noise, window, mode='same')
        
        simulated_normal = wander + smoothed_noise
        return simulated_normal

    def generate_simulated_erratic_test_data(self, duration_sec: float) -> np.ndarray:
        """
        Generates SIMULATED ERRATIC TEST DATA — NOT TRAINING DATA.
        Characterized by large, frequent corrections and reduced smoothness.
        """
        num_samples = int(duration_sec * self.sample_rate_hz)
        time = np.linspace(0, duration_sec, num_samples)
        
        # Base erratic weaving
        erratic_base = 25.0 * np.sin(1.5 * np.pi * time) + 10.0 * np.sin(3.0 * np.pi * time)
        
        # Sharp jerky corrections
        jerks = self.rng.normal(0, 5.0, num_samples)
        
        simulated_erratic = erratic_base + jerks
        return simulated_erratic

if __name__ == "__main__":
    generator = SimulatedSteeringGenerator(sample_rate_hz=15.0, random_seed=42)
    simulated_normal_data = generator.generate_simulated_normal_data(2.0)
    simulated_erratic_test_data = generator.generate_simulated_erratic_test_data(2.0)
    
    print("SIMULATED NORMAL DATA (First 20 values):")
    print(np.round(simulated_normal_data[:20], 2))
    
    print("\nSIMULATED ERRATIC TEST DATA — NOT TRAINING DATA (First 20 values):")
    print(np.round(simulated_erratic_test_data[:20], 2))
