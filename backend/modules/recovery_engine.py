import time
import threading
import pyttsx3
import logging
import os

class RecoveryEngine:
    def __init__(self, time_fn=time.time, tts_enabled=True):
        self.state = "IDLE"
        self.time_fn = time_fn
        # Disable real TTS in test environments to avoid blocking the test runner
        self.tts_enabled = tts_enabled and "PYTEST_CURRENT_TEST" not in os.environ
        
        self.trackers = {
            "EYE": {"start": None, "last": None},
            "HEAD": {"start": None, "last": None},
            "KINEMATIC": {"start": None, "last": None},
            "ACOUSTIC": {"start": None, "last": None},
            "OVERALL": {"start": None, "last": None}
        }
        
        self.sustained_safe_start = None
        self.last_safe_time = None
        self.recovered_start = None
        self.is_speaking = False

        self.last_speech_time = None
        self.SPEECH_COOLDOWN_SEC = 10.0

        # Tuning parameters for explicit state machine durations
        self.IDLE_TO_WARNING_SEC = 3.0
        self.WARNING_TO_INTERVENING_SEC = 2.0
        self.INTERVENING_TO_RECOVERED_SEC = 5.0
        self.RECOVERED_TO_IDLE_SEC = 3.0

        self.ambient_action = None
        self.reroute_suggestion = None

        self.suggestions = [
            "Suggesting nearest rest stop in 2 km",
            "Suggesting a coffee shop 5 minutes ahead",
            "Suggesting a scenic viewpoint to rest in 3 miles"
        ]
        self.suggestion_index = 0

    def _speak_async(self, message):
        if not self.tts_enabled:
            return

        def _speak():
            try:
                self.is_speaking = True
                engine = pyttsx3.init()
                engine.say(message)
                engine.runAndWait()
            except Exception as e:
                logging.error(f"TTS Failure: {e}")
            finally:
                self.is_speaking = False

        t = threading.Thread(target=_speak, daemon=True)
        t.start()

    def _update_tracker(self, key, is_active, current_time):
        t = self.trackers[key]
        if is_active:
            if t["start"] is None:
                t["start"] = current_time
            t["last"] = current_time
        else:
            if t["start"] is not None and t["last"] is not None and (current_time - t["last"] > 2.0):
                t["start"] = None

    def _get_duration(self, key, current_time):
        t = self.trackers[key]
        if t["start"] is not None:
            return current_time - t["start"]
        return -1.0

    def _trigger_intervention(self, reason, current_time):
        self.ambient_action = "SIMULATED: dimming cabin lights, reducing audio volume"
        self.reroute_suggestion = self.suggestions[self.suggestion_index]
        self.suggestion_index = (self.suggestion_index + 1) % len(self.suggestions)
        self.trigger_reason = reason
        
        if self.last_speech_time is None or (current_time - self.last_speech_time) >= self.SPEECH_COOLDOWN_SEC:
            messages = {
                "EYE": "You seem to be closing your eyes. Please stay alert.",
                "HEAD": "You seem to be looking away from the road for a while.",
                "KINEMATIC": "Your steering pattern seems unusual compared to your normal driving.",
                "ACOUSTIC": "You seem to be experiencing heightened stress. Please stay focused.",
                "OVERALL": "You seem to be losing focus. Please stay alert."
            }
            self._speak_async(messages.get(reason, messages["OVERALL"]))
            self.last_speech_time = current_time

    def process(self, cognitive_assessment, sensor_outputs=None, current_time=None):
        if current_time is None:
            current_time = self.time_fn()

        if sensor_outputs is None:
            from modules.models import SensorOutput
            visual_output = SensorOutput(score=0.0, confidence=0.0, timestamp=current_time, source="visual", validity=False, raw_data={})
            acoustic_output = SensorOutput(score=0.0, confidence=0.0, timestamp=current_time, source="acoustic", validity=False, raw_data={})
            kinematic_output = SensorOutput(score=0.0, confidence=0.0, timestamp=current_time, source="kinematic", validity=False, raw_data={})
        else:
            visual_output, acoustic_output, kinematic_output = sensor_outputs

        vis_valid = visual_output.validity
        eye_closed = visual_output.raw_data.get("eye_closed_duration", 0.0)
        gaze_away = visual_output.raw_data.get("gaze_away_duration", 0.0)
        
        kin_valid = kinematic_output.validity
        is_anomaly = kinematic_output.raw_data.get("is_anomaly", False)
        
        ac_valid = acoustic_output.validity
        arousal = acoustic_output.raw_data.get("arousal", 0.0)
        
        zone = cognitive_assessment.get("zone", "Safe")

        base_signals = {
            "OVERALL": zone in ["High Risk", "Critical Risk"],
            "EYE": vis_valid and eye_closed >= 5.0,
            "HEAD": vis_valid and gaze_away >= 4.0,
            "KINEMATIC": kin_valid and is_anomaly,
            "ACOUSTIC": ac_valid and arousal >= 0.65
        }

        for key, is_active in base_signals.items():
            self._update_tracker(key, is_active, current_time)

        warning_reasons = []
        intervening_reasons = []

        if self._get_duration("EYE", current_time) >= 0.0: intervening_reasons.append("EYE")
        if self._get_duration("HEAD", current_time) >= 0.0: intervening_reasons.append("HEAD")
        if self._get_duration("KINEMATIC", current_time) >= 3.0: intervening_reasons.append("KINEMATIC")
        if self._get_duration("ACOUSTIC", current_time) >= 3.0: intervening_reasons.append("ACOUSTIC")
        
        if self._get_duration("OVERALL", current_time) >= self.IDLE_TO_WARNING_SEC: warning_reasons.append("OVERALL")
        if self._get_duration("OVERALL", current_time) >= (self.IDLE_TO_WARNING_SEC + self.WARNING_TO_INTERVENING_SEC): intervening_reasons.append("OVERALL")

        is_high_risk = any(base_signals.values())
        is_safe = (zone in ["Safe", "Attention Required"]) and not is_high_risk

        if is_safe:
            if self.sustained_safe_start is None:
                self.sustained_safe_start = current_time
            self.last_safe_time = current_time
        else:
            if self.sustained_safe_start is not None and self.last_safe_time is not None and (current_time - self.last_safe_time > 2.0):
                self.sustained_safe_start = None

        priority = {"EYE": 1, "HEAD": 2, "KINEMATIC": 3, "ACOUSTIC": 4, "OVERALL": 5}
        active_reason = None
        if intervening_reasons:
            active_reason = sorted(intervening_reasons, key=lambda x: priority[x])[0]

        if self.state == "IDLE":
            self.ambient_action = None
            self.reroute_suggestion = None
            self.trigger_reason = None
            if intervening_reasons:
                self.state = "INTERVENING"
                self._trigger_intervention(active_reason, current_time)
            elif warning_reasons:
                self.state = "WARNING"
                
        elif self.state == "WARNING":
            if intervening_reasons:
                self.state = "INTERVENING"
                self._trigger_intervention(active_reason, current_time)
            elif not warning_reasons:
                self.state = "IDLE"

        elif self.state == "INTERVENING":
            if self.sustained_safe_start is not None and (current_time - self.sustained_safe_start >= self.INTERVENING_TO_RECOVERED_SEC):
                self.state = "RECOVERED"
                self.ambient_action = None
                self.reroute_suggestion = None
                self.trigger_reason = None
                self.recovered_start = current_time

        elif self.state == "RECOVERED":
            if intervening_reasons:
                self.state = "INTERVENING"
                self._trigger_intervention(active_reason, current_time)
            elif warning_reasons:
                self.state = "WARNING"
            elif self.recovered_start is not None and (current_time - self.recovered_start >= self.RECOVERED_TO_IDLE_SEC):
                self.state = "IDLE"

        return {
            "state": self.state,
            "ambient_action": self.ambient_action,
            "reroute_suggestion": self.reroute_suggestion,
            "trigger_reason": getattr(self, "trigger_reason", None)
        }
