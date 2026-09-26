import os
import time
import psutil
import numpy as np
from modules.acoustic_capture import AcousticCapture
from modules.acoustic import AcousticSensor

def main():
    print("Initializing AcousticCapture...")
    capture = AcousticCapture(target_sr=16000)
    mic_working = capture.start()
    
    if not mic_working:
        print(f"Warning: Failed to start microphone stream. Error: {capture.init_error}")
        print("Will run a dummy benchmark for load time and latency anyway.")

    try:
        print("Loading AcousticSensor model...")
        load_start = time.time()
        sensor = AcousticSensor(capture)
        load_time = time.time() - load_start
        print(f"-> Model load time: {load_time:.2f} seconds")

        if mic_working:
            print("Buffering 2.0 seconds of audio from the microphone...")
            time.sleep(2.0)
        else:
            # Inject fake audio just for the benchmark
            print("Injecting 2 seconds of fake audio for benchmark...")
            fake_audio = np.random.uniform(-0.1, 0.1, 32000).astype(np.float32)
            sensor.audio_buffer = fake_audio

        process = psutil.Process(os.getpid())
        # First call to cpu_percent to initialize
        process.cpu_percent(interval=None)

        print("Running inference on buffered audio...")
        infer_start = time.time()
        sensor_output = sensor.process_audio()
        infer_time = time.time() - infer_start
        
        cpu_usage = process.cpu_percent(interval=None)
        ram_mb = process.memory_info().rss / (1024 * 1024)

        print(f"-> Per-inference latency: {infer_time * 1000:.1f} ms")
        print(f"-> CPU usage (this process): {cpu_usage:.1f}%")
        print(f"-> RAM usage (this process): {ram_mb:.1f} MB")

        print("\n--- Output ---")
        if not sensor_output.validity:
            print("Inference skipped (insufficient audio buffered).")
        else:
            print(f"Arousal (Risk Score): {sensor_output.score:.3f}")
            print(f"Confidence (VAD):   {sensor_output.confidence:.2f}")
            print("Raw Values:")
            print(f"  - Arousal:   {sensor_output.raw_data.get('arousal', 0.0):.3f}")
            print(f"  - Dominance: {sensor_output.raw_data.get('dominance', 0.0):.3f}")
            print(f"  - Valence:   {sensor_output.raw_data.get('valence', 0.0):.3f}")
            
    finally:
        if mic_working:
            capture.stop()
            print("Microphone stream stopped.")

if __name__ == '__main__':
    main()
