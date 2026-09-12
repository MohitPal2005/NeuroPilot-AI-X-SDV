class CognitiveStateEngine:
    """
    Decoupled processing engine evaluating the driver's real-time Cognitive State Index (CSI).
    Designed to easily scale from a rule-based engine to a machine learning model.
    """
    @staticmethod
    def compute_csi(telemetry):
        if not telemetry.get("face_detected", False):
            return {"csi": 50, "zone": "Calibrating (No Face)"}

        csi = 10.0  # Base calibration score for normal operation
        
        # Accumulate risk metrics via visual and attention feedback loops
        gaze_away = telemetry.get("gaze_away_duration", 0.0)
        eye_closed = telemetry.get("eye_closed_duration", 0.0)
        yaw = abs(telemetry.get("yaw", 0.0))
        pitch = abs(telemetry.get("pitch", 0.0))
        bpm = telemetry.get("bpm", 12)

        # 1. Distraction compounding logic
        if gaze_away > 0:
            csi += min(gaze_away * 15, 45) # Max 45-point penalty for long distractions

        # 2. Drowsiness/Fatigue compounding logic
        if eye_closed > 0.5:
            csi += min(eye_closed * 25, 55)

        # 3. Posture deviation modifiers
        if yaw > 20: csi += 15
        if pitch > 15: csi += 10

        # 4. Stress indicators via physiological anomaly simulation (unusual blink patterns)
        if bpm > 28 or bpm < 6:
            csi += 10

        # Cap output within explicit bounds [0 - 100]
        final_csi = min(max(int(csi), 0), 100)
        
        # Map out standard descriptive risk zones
        if final_csi <= 20: zone = "Safe"
        elif final_csi <= 40: zone = "Attention Required"
        elif final_csi <= 60: zone = "Cognitive Overload"
        elif final_csi <= 80: zone = "High Risk"
        else: zone = "Critical Risk"

        return {"csi": final_csi, "zone": zone}