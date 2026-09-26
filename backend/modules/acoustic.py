import time
import numpy as np
import torch
import torch.nn as nn
from transformers import Wav2Vec2Processor
from transformers.models.wav2vec2.modeling_wav2vec2 import (
    Wav2Vec2Model,
    Wav2Vec2PreTrainedModel,
)

class RegressionHead(nn.Module):
    r"""Classification head."""
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.final_dropout)
        self.out_proj = nn.Linear(config.hidden_size, config.num_labels)

    def forward(self, features, **kwargs):
        x = features
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x

class EmotionModel(Wav2Vec2PreTrainedModel):
    r"""Speech emotion classifier."""
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.wav2vec2 = Wav2Vec2Model(config)
        self.classifier = RegressionHead(config)
        self.post_init()

    def forward(self, input_values):
        outputs = self.wav2vec2(input_values)
        hidden_states = outputs[0]
        hidden_states = torch.mean(hidden_states, dim=1)
        logits = self.classifier(hidden_states)
        return hidden_states, logits
from modules.models import SensorOutput
from modules.acoustic_capture import AcousticCapture

class AcousticSensor:
    def __init__(self, capture: AcousticCapture):
        self.capture = capture
        self.target_sr = capture.target_sr
        self.buffer_duration_sec = 3.0  # Max 3 seconds buffer
        self.min_duration_sec = 1.0     # Min 1 second for meaningful inference
        self.max_samples = int(self.target_sr * self.buffer_duration_sec)
        self.min_samples = int(self.target_sr * self.min_duration_sec)
        self.audio_buffer = np.array([], dtype=np.float32)
        
        self.load_failed = False
        try:
            # Load the model and processor
            # Model License: CC BY-NC-SA 4.0 (Non-commercial, Attribution, ShareAlike)
            model_name = "audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim"
            self.processor = Wav2Vec2Processor.from_pretrained(model_name)
            self.model = EmotionModel.from_pretrained(model_name)
            self.model.eval()
        except Exception as e:
            print(f"Failed to load acoustic model: {e}")
            self.load_failed = True
        
    def process_audio(self) -> SensorOutput:
        current_time = time.time()
        
        if self.load_failed:
            return SensorOutput(
                score=0.0,
                confidence=0.0,
                timestamp=current_time,
                source="acoustic_sensor",
                validity=False,
                raw_data={"error": "model load failed"}
            )
            
        # Pull new audio from capture
        new_speech_flag, new_audio = self.capture.get_latest_capture()
        
        if new_audio is not None and len(new_audio) > 0:
            self.audio_buffer = np.concatenate((self.audio_buffer, new_audio))
            
        # Truncate buffer to max 3 seconds
        if len(self.audio_buffer) > self.max_samples:
            self.audio_buffer = self.audio_buffer[-self.max_samples:]
            
        # Ensure we have at least 1 second of audio
        if len(self.audio_buffer) < self.min_samples:
            return SensorOutput(
                score=0.0,
                confidence=0.0,
                timestamp=current_time,
                source="acoustic_sensor",
                validity=False,
                raw_data={"arousal": 0.0, "dominance": 0.0, "valence": 0.0, "status": "buffering"}
            )
            
        # Re-evaluate VAD over the current buffer
        has_speech = self.capture.detect_vad(self.audio_buffer)
        confidence = 0.9 if has_speech else 0.4
        
        # Run model inference
        with torch.no_grad():
            inputs = self.processor(self.audio_buffer, sampling_rate=self.target_sr, return_tensors="pt")
            hidden_states, logits = self.model(inputs.input_values)
            logits = logits[0].cpu().numpy()
            
        results = {}
        for i, logit in enumerate(logits):
            label = self.model.config.id2label[i]
            results[label] = float(logit)
            
        arousal = results.get("arousal", 0.0)
        arousal = max(0.0, min(1.0, arousal))
        
        return SensorOutput(
            score=arousal,
            confidence=confidence,
            timestamp=current_time,
            source="acoustic_sensor",
            validity=True,
            raw_data=results
        )
