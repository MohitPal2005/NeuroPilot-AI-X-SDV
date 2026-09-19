
import queue
 
import numpy as np
import sounddevice as sd
import webrtcvad
 
# librosa is optional: it is heavy on a Pi, so we fall back to numpy
try:
    import librosa
    HAS_LIBROSA = True
except ImportError:
    HAS_LIBROSA = False
 
 
class AcousticCapture:
    def __init__(self, target_sr=16000, frame_duration_ms=30,
                 vad_mode=3, device=None, min_rms=0.01, min_speech_frames=3):
        """
        target_sr         : sample rate used for VAD (WebRTC supports 8000/16000/32000/48000)
        frame_duration_ms : VAD frame size (10, 20 or 30 ms only)
        vad_mode          : 0-3, higher = stricter (fewer false "speech" detections)
        min_rms           : minimum loudness (0.0-1.0) before audio can count as speech.
                            Raise it if background noise still triggers "speaking".
        min_speech_frames : how many 30 ms frames in one chunk must look like speech
                            (filters out short clicks/pops). Raise to be stricter.
        device            : sounddevice input device index/name (None = system default).
                            Run  python3 -c "import sounddevice as sd; print(sd.query_devices())"
                            to see the list (useful for USB mics).
        """
        self.target_sr = target_sr
        self.frame_duration_ms = frame_duration_ms
        self.device = device
        self.min_rms = min_rms
        self.min_speech_frames = min_speech_frames
        self.last_rms = 0.0        # loudness of last chunk (handy for tuning min_rms)
        self.vad = webrtcvad.Vad(vad_mode)
 
        self.q = queue.Queue()
        self.stream = None
        self.is_running = False
        self.init_error = None
        self.native_sr = None      # native rate of the mic
        self.stream_sr = None      # rate the stream is actually opened at
        self.last_status = None    # last PortAudio overflow/underflow status
 
        try:
            device_info = sd.query_devices(device=self.device, kind='input')
            if not device_info:
                raise Exception("No input devices available")
            self.native_sr = int(device_info['default_samplerate'])
        except Exception as e:
            self.init_error = f"Audio device unavailable: {str(e)}"
            self.is_running = False
 
    # ------------------------------------------------------------------
    # Processing helpers
    # ------------------------------------------------------------------
    def resample_to_target(self, raw_audio, sr):
        """Resample 1D float32 audio to target_sr (no-op if already equal)."""
        if sr == self.target_sr:
            return raw_audio
 
        if HAS_LIBROSA:
            return librosa.resample(
                y=raw_audio, orig_sr=float(sr), target_sr=float(self.target_sr)
            )
 
        # Lightweight fallback (linear interpolation)
        new_len = int(round(len(raw_audio) * self.target_sr / float(sr)))
        if new_len <= 0:
            return np.array([], dtype=np.float32)
        old_x = np.linspace(0.0, 1.0, num=len(raw_audio), endpoint=False)
        new_x = np.linspace(0.0, 1.0, num=new_len, endpoint=False)
        return np.interp(new_x, old_x, raw_audio).astype(np.float32)
 
    def detect_vad(self, audio_16k):
        """
        Return True only if the chunk is loud enough AND enough 30 ms frames
        are classified as speech by WebRTC VAD.
        """
        if len(audio_16k) == 0:
            self.last_rms = 0.0
            return False
 
        # 1) Loudness gate: ignore quiet background noise
        self.last_rms = float(np.sqrt(np.mean(np.square(audio_16k))))
        if self.last_rms < self.min_rms:
            return False
 
        audio_clipped = np.clip(audio_16k, -1.0, 1.0)
        audio_int16 = (audio_clipped * 32767).astype(np.int16)
 
        frame_length = int(self.target_sr * (self.frame_duration_ms / 1000.0))
        # 2) Count speech frames; need several, not just one
        speech_frames = 0
        for i in range(0, len(audio_int16) - frame_length + 1, frame_length):
            frame = audio_int16[i:i + frame_length]
            try:
                if self.vad.is_speech(frame.tobytes(), self.target_sr):
                    speech_frames += 1
                    if speech_frames >= self.min_speech_frames:
                        return True
            except Exception:
                pass
        return False
 
    # ------------------------------------------------------------------
    # Stream handling
    # ------------------------------------------------------------------
    def _audio_callback(self, indata, frames, time_info, status):
        if status:
            self.last_status = str(status)
        self.q.put(indata[:, 0].copy())
 
    def start(self):
        """Open the mic. Returns True on success, False on failure."""
        if self.init_error:
            return False
 
        # Try 16 kHz directly first (no resampling needed), then native rate.
        rates_to_try = [self.target_sr]
        if self.native_sr and self.native_sr != self.target_sr:
            rates_to_try.append(self.native_sr)
 
        last_error = None
        for rate in rates_to_try:
            try:
                self.stream = sd.InputStream(
                    samplerate=rate,
                    channels=1,
                    callback=self._audio_callback,
                    dtype='float32',
                    device=self.device,
                )
                self.stream.start()
                self.stream_sr = rate
                self.is_running = True
                return True
            except Exception as e:
                last_error = e
                self.stream = None
 
        self.init_error = f"Stream open failed: {str(last_error)}"
        self.is_running = False
        return False
 
    def stop(self):
        if self.stream:
            try:
                self.stream.stop()
                self.stream.close()
            except Exception:
                pass
            self.stream = None
        self.is_running = False
 
    def get_latest_capture(self):
        """
        Drain the buffer and run VAD on everything captured since the last call.
        Returns (has_speech, audio_16k), or (None, None) on hardware error.
        """
        if self.init_error:
            return None, None
 
        frames = []
        while True:
            try:
                frames.append(self.q.get_nowait())
            except queue.Empty:
                break
 
        if not frames:
            return False, np.array([], dtype=np.float32)
 
        raw_audio = np.concatenate(frames)
        audio_16k = self.resample_to_target(raw_audio, self.stream_sr or self.native_sr)
        has_speech = self.detect_vad(audio_16k)
        return has_speech, audio_16k
 
