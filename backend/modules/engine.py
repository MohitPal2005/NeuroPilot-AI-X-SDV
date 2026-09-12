import config

class CognitiveStateEngine:
    """
    Stateful processing engine evaluating the driver's real-time Cognitive State Index (CSI).
    Maintains previous state to apply hysteresis to zone transitions.
    """
    def __init__(self):
        self.last_zone = "Safe"
        self.last_csi = config.CSI_BASE

    def compute_csi(self, telemetry):
        if not telemetry.get("face_detected", False):
            return {"csi": 50, "zone": "Calibrating (No Face)"}

        csi = config.CSI_BASE
        
        # Accumulate risk metrics via visual and attention feedback loops
        gaze_away = telemetry.get("gaze_away_duration", 0.0)
        eye_closed = telemetry.get("eye_closed_duration", 0.0)
        yaw = abs(telemetry.get("yaw", 0.0))
        pitch = abs(telemetry.get("pitch", 0.0))
        bpm = telemetry.get("bpm", 12)

        # 1. Distraction compounding logic
        if gaze_away > 0:
            csi += min(gaze_away * config.CSI_DISTRACTION_MULTIPLIER, config.CSI_DISTRACTION_MAX)

        # 2. Drowsiness/Fatigue compounding logic
        if eye_closed > config.CSI_DROWSINESS_MIN_DURATION:
            csi += min(eye_closed * config.CSI_DROWSINESS_MULTIPLIER, config.CSI_DROWSINESS_MAX)

        # 3. Posture deviation modifiers
        if yaw > config.CSI_YAW_THRESHOLD: csi += config.CSI_YAW_PENALTY
        if pitch > config.CSI_PITCH_THRESHOLD: csi += config.CSI_PITCH_PENALTY

        # 4. Stress indicators via physiological anomaly simulation (unusual blink patterns)
        if bpm > config.CSI_BPM_HIGH or bpm < config.CSI_BPM_LOW:
            csi += config.CSI_BPM_PENALTY

        # Cap output within explicit bounds [0 - 100]
        final_csi = min(max(int(csi), 0), 100)
        
        # Hysteresis Logic for Zone Transitions
        zones = ["Safe", "Attention Required", "Cognitive Overload", "High Risk", "Critical Risk"]
        zone_bounds = [20, 40, 60, 80] # Upper bounds for indices 0, 1, 2, 3
        margin = config.HYSTERESIS_MARGIN
        
        try:
            current_idx = zones.index(self.last_zone)
        except ValueError:
            current_idx = 0
            
        new_idx = current_idx
        
        # Check if we can move down to a safer zone
        while new_idx > 0 and final_csi <= zone_bounds[new_idx - 1] - margin:
            new_idx -= 1
            
        # Check if we should move up to a riskier zone
        while new_idx < 4 and final_csi > zone_bounds[new_idx] + margin:
            new_idx += 1
            
        zone = zones[new_idx]
        
        self.last_zone = zone
        self.last_csi = final_csi

        return {"csi": final_csi, "zone": zone}