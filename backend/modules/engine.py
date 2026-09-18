import config

class CognitiveStateEngine:
    """
    Stateful processing engine evaluating the driver's real-time Cognitive State Index (CSI).
    Maintains previous state to apply hysteresis to zone transitions.
    """
    def __init__(self):
        self.last_zone = "Safe"
        self.last_csi = config.CSI_BASE

    def compute_csi(self, sensor_outputs):
        total_confidence = 0.0
        weighted_score_sum = 0.0
        
        for output in sensor_outputs:
            # TODO: revisit in Session 10 — sensor-degradation logic should redistribute confidence/weights when a sensor becomes invalid instead of unconditionally aborting to No Data.
            if not output.validity:
                # If any sensor is invalid (e.g. vision lost face tracking), 
                # we do not have enough calibrated data to produce a reliable fusion state.
                return {"csi": 0, "zone": "No Data (Calibrating)"}
                
            weight = output.confidence
            weighted_score_sum += output.score * weight
            total_confidence += weight

        if total_confidence == 0:
            return {"csi": 0, "zone": "No Data (Calibrating)"}
            
        fused_score = weighted_score_sum / total_confidence
        csi = config.CSI_BASE + (fused_score * 100.0)

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