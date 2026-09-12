from pydantic import BaseModel, Field
from typing import Dict, Any, Optional
import time

class SensorOutput(BaseModel):
    score: float = Field(ge=0.0, le=1.0, description="Normalized risk or confidence score")
    confidence: float = Field(ge=0.0, le=1.0, description="Sensor confidence in the reading")
    timestamp: float = Field(default_factory=time.time, description="Unix timestamp of the reading")
    source: str = Field(..., description="Identifier for the sensor (e.g., 'visual_engine', 'acoustic_stub')")
    validity: bool = Field(..., description="Whether the data is valid or the sensor is degraded/offline")
    raw_data: Optional[Dict[str, Any]] = Field(default=None, description="Underlying raw telemetry for backward compatibility or downstream processing")
