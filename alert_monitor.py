import os
import time
from datetime import datetime

ALERT_DIR = "alerts"

def ensure_alert_dir():
    if not os.path.exists(ALERT_DIR):
        os.makedirs(ALERT_DIR)

def log_alert(description, video_path):
    ensure_alert_dir()
    ts = time.time()
    human_time = datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S')
    alert = {
        "timestamp": ts,
        "human_time": human_time,
        "description": description,
        "video_path": video_path
    }
    alert_file = os.path.join(ALERT_DIR, f"alert_{int(ts)}.json")
    with open(alert_file, "w") as f:
        import json
        json.dump(alert, f, indent=2)
    print(f"[ALERT] {description} at {human_time} (see {alert_file})")

def check_for_cycle(detected_classes, video_path):
    """
    Checks if 'bicycle' is in detected_classes and logs an alert if so.
    """
    if "bicycle" in detected_classes:
        log_alert("Bicycle detected in frame.", video_path)

# Example usage in your video processing loop:
# from alert_monitor import check_for_cycle
# ...
# detected_classes = [...]  # list of detected object names for the frame
# check_for_cycle(detected_classes, video_path)
