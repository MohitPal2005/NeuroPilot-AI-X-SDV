import numpy as np
import librosa
import webrtcvad
import sounddevice as sd
import queue
import threading

class AcousticCapture:
    """
    Independent acoustic capture component for Session 5.
    Handles microphone initialization, buffering, resampling to 16kHz, and WebRTC VAD.
    """
    def __init__(self, target_sr=16000, frame_duration_ms=30):
        self.target_sr = target_sr
        self.frame_duration_ms = frame_duration_ms
        # webrtcvad takes aggressiveness mode (0-3). 1 is standard.
        self.vad = webrtcvad.Vad(1)
        
        self.q = queue.Queue()
        self.stream = None
        self.is_running = False
        self.init_error = None
        self.native_sr = None
        
        try:
            # Query default input device to check availability and get native samplerate
            device_info = sd.query_devices(kind='input')
            if not device_info:
                raise Exception("No input devices available")
            self.native_sr = int(device_info['default_samplerate'])
        except Exception as e:
            self.init_error = f"Audio device unavailable: {str(e)}"
            self.is_running = False

    def resample_to_target(self, raw_audio, sr):
        """
        Resamples a 1D numpy float32 array to target_sr if needed.
        """
        if sr != self.target_sr:
            return librosa.resample(y=raw_audio, orig_sr=float(sr), target_sr=float(self.target_sr))
        return raw_audio
        
    def detect_vad(self, audio_16k):
        """
        Runs WebRTC VAD on 16kHz float32 audio array.
        Splits into 30ms frames and returns True if any frame contains speech.
        """
        if len(audio_16k) == 0:
            return False
            
        # Convert float32 [-1.0, 1.0] to int16 PCM expected by WebRTC VAD
        # Ensure we don't overflow
        audio_clipped = np.clip(audio_16k, -1.0, 1.0)
        audio_int16 = (audio_clipped * 32767).astype(np.int16)
        
        frame_length = int(self.target_sr * (self.frame_duration_ms / 1000.0))
        has_speech = False
        
        # Process in valid frame_length chunks
        for i in range(0, len(audio_int16) - frame_length + 1, frame_length):
            frame = audio_int16[i:i+frame_length]
            try:
                if self.vad.is_speech(frame.tobytes(), self.target_sr):
                    has_speech = True
                    break
            except Exception:
                pass
                
        return has_speech

    def _audio_callback(self, indata, frames, time, status):
        # Flatten the input since it might be 2D e.g., (frames, channels)
        # Assuming mono (1 channel)
        if status:
            pass # Handle overruns/underruns if needed
        self.q.put(indata[:, 0].copy())

    def start(self):
        """
        Attempts to open the microphone stream. Returns True on success, False on failure.
        """
        if self.init_error:
            return False
            
        try:
            self.stream = sd.InputStream(
                samplerate=self.native_sr, 
                channels=1, 
                callback=self._audio_callback,
                dtype='float32'
            )
            self.stream.start()
            self.is_running = True
            return True
        except Exception as e:
            self.init_error = f"Stream open failed: {str(e)}"
            self.is_running = False
            return False
            
    def stop(self):
        if self.stream:
            self.stream.stop()
            self.stream.close()
        self.is_running = False
        
    def get_latest_capture(self):
        """
        Retrieves accumulated audio from the buffer, resamples it, and returns the VAD status.
        Returns: (has_speech, audio_16k) or (None, None) if hardware error.
        """
        if self.init_error:
            # Deterministic failure state representation for hardware error
            return None, None
            
        frames = []
        while not self.q.empty():
            try:
                frames.append(self.q.get_nowait())
            except queue.Empty:
                break
                
        if not frames:
            return False, np.array([], dtype=np.float32)
            
        raw_audio = np.concatenate(frames)
        audio_16k = self.resample_to_target(raw_audio, self.native_sr)
        has_speech = self.detect_vad(audio_16k)
        
        return has_speech, audio_16k
