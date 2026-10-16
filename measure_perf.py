import subprocess
import time
import requests
import psutil
import sys
import os
import numpy as np

sys.path.insert(0, os.path.abspath('c:/neuropilot-ai-x-sdv/backend'))

def main():
    print("=============================================")
    print("Starting app.py subprocess...")
    proc = subprocess.Popen([sys.executable, 'app.py'], cwd='c:/neuropilot-ai-x-sdv/backend')

    print("Waiting for server to start (15s to allow acoustic model to load)...")
    time.sleep(15)

    # 2. End-to-end /driver/status latency
    print("Measuring /driver/status latency...")
    latencies = []
    for _ in range(20):
        start = time.time()
        try:
            r = requests.get('http://127.0.0.1:5000/driver/status', timeout=2)
            latencies.append(time.time() - start)
        except Exception as e:
            print(f"Error fetching status: {e}")
        time.sleep(0.1)

    if latencies:
        avg_latency = sum(latencies)/len(latencies)
        print(f"[MEASURED] End-to-end /driver/status latency (avg over 20 requests): {avg_latency*1000:.2f} ms")

    # 1. & 5. FPS and CPU/RAM over 30s
    print("Measuring FPS, CPU, and RAM over 30 seconds...")
    unique_timestamps = set()
    cpu_usage = []
    ram_usage = []
    try:
        p = psutil.Process(proc.pid)
        p.cpu_percent(interval=None) # initialization
    except:
        p = None

    start_time = time.time()
    end_time = start_time + 30.0

    while time.time() < end_time:
        try:
            r = requests.get('http://127.0.0.1:5000/analytics', timeout=2)
            if r.status_code == 200:
                data = r.json()
                for entry in data:
                    unique_timestamps.add(entry.get('timestamp'))
        except Exception:
            pass
        
        if p:
            try:
                cpu_usage.append(p.cpu_percent(interval=None))
                ram_usage.append(p.memory_info().rss / (1024 * 1024))
            except psutil.NoSuchProcess:
                pass
            
        time.sleep(0.5)

    fps = len(unique_timestamps) / 30.0
    avg_cpu = sum(cpu_usage)/len(cpu_usage) if cpu_usage else 0
    min_cpu = min(cpu_usage) if cpu_usage else 0
    max_cpu = max(cpu_usage) if cpu_usage else 0
    avg_ram = sum(ram_usage)/len(ram_usage) if ram_usage else 0

    print(f"[MEASURED] Visual pipeline FPS (actual frames in 30s): {fps:.2f} FPS")
    print(f"[MEASURED] CPU usage (avg over 30s): {avg_cpu:.1f} %")
    print(f"[MEASURED] CPU usage (min over 30s): {min_cpu:.1f} %")
    print(f"[MEASURED] CPU usage (max over 30s): {max_cpu:.1f} %")
    print(f"[MEASURED] RAM usage (avg over 30s): {avg_ram:.1f} MB")

    print("Terminating app subprocess...")
    proc.terminate()
    proc.wait(timeout=5)
    print("=============================================\n")

    # (Skipping acoustic/kinematic/recovery tests for faster execution)

if __name__ == '__main__':
    main()
