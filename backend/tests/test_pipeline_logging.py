import pytest
import time
import os
import csv
import sys
from unittest.mock import patch, MagicMock
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

class StopLoopException(Exception):
    pass

@patch('app.time.sleep')
def test_pipeline_logging_csv(mock_sleep, tmpdir):
    class SleepCounter:
        def __init__(self):
            self.calls = 0
        def __call__(self, *args, **kwargs):
            self.calls += 1
            if self.calls >= 30:
                raise StopLoopException("Stop after 30 iterations to trigger flush")
                
    mock_sleep.side_effect = SleepCounter()
    
    test_log_dir = str(tmpdir.mkdir("test_logs"))
    test_log_path = os.path.join(test_log_dir, 'logs', 'pipeline_log.csv')
    
    with patch('app.os.path.dirname', return_value=test_log_dir):
        with patch('app.os.path.abspath', return_value=os.path.join(test_log_dir, 'app.py')):
            import app
            
            # Now that app is imported, mock its camera
            dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
            mock_cam = MagicMock()
            mock_cam.read.return_value = (True, dummy_frame)
            app.camera = mock_cam
            
            try:
                app.background_telemetry_worker()
            except StopLoopException:
                pass

    assert os.path.exists(test_log_path), "CSV log file was not created"
    
    with open(test_log_path, 'r', newline='') as f:
        reader = csv.reader(f)
        rows = list(reader)
        
    assert len(rows) >= 2, "Expected at least a header and one data row"
    
    header = rows[0]
    expected_header = [
        "timestamp", "vision_score", "vision_confidence", "vision_validity",
        "acoustic_score", "acoustic_confidence", "acoustic_validity",
        "kinematic_score", "kinematic_confidence", "kinematic_validity",
        "fused_csi", "zone", "recovery_state", "trigger_reason", "ground_truth_label"
    ]
    assert header == expected_header, f"Header mismatch. Got: {header}"
    
    data_row = rows[1]
    assert len(data_row) == len(expected_header), "Data row column count mismatch"
    
    # Assert numeric fields parse as floats (timestamp, scores, confidences)
    # timestamp is index 0
    float(data_row[0])
    
    # Validities are bools, but we'll just check if they are strings.
    # Scores and confidences should be parseable as floats if they are not empty strings.
    # In our dummy run, some might be empty or floats.
    for i in [1, 2, 4, 5, 7, 8, 10]:  # indices for scores, confidences, csi
        val = data_row[i]
        if val != "":
            float(val)  # Should not raise ValueError
