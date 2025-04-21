import os
import json
from glob import glob

RECENT_DIR = "recent_indexes"
MASTER_LOG = "index_master.jsonl"
MAX_RECENT = 3

if not os.path.exists(RECENT_DIR):
    os.makedirs(RECENT_DIR)

def _get_next_run_number():
    files = sorted(glob(os.path.join(RECENT_DIR, "index_run_*.jsonl")))
    if not files:
        return 1
    nums = [int(os.path.splitext(os.path.basename(f))[0].split("_")[-1]) for f in files]
    return max(nums) + 1

def _cleanup_old_logs():
    files = sorted(glob(os.path.join(RECENT_DIR, "index_run_*.jsonl")))
    if len(files) > MAX_RECENT:
        for f in files[:-MAX_RECENT]:
            os.remove(f)

_RUN_NUMBER = _get_next_run_number()
SESSION_LOG = os.path.join(RECENT_DIR, f"index_run_{_RUN_NUMBER:03d}.jsonl")
_cleanup_old_logs()

def get_last_frame_id():
    """
    Returns the last frame_id used in index_master.jsonl, or 0 if none.
    """
    if not os.path.exists(MASTER_LOG):
        return 0
    try:
        with open(MASTER_LOG, "rb") as f:
            f.seek(0, os.SEEK_END)
            pos = f.tell() - 1
            while pos > 0:
                f.seek(pos)
                if f.read(1) == b"\n":
                    break
                pos -= 1
            f.seek(pos + 1)
            last_line = f.readline().decode()
        if last_line.strip():
            data = json.loads(last_line)
            return int(data.get("frame_id", 0))
        return 0
    except Exception:
        return 0

# --- GLOBAL FRAME ID TRACKER ---
_FRAME_ID = get_last_frame_id()

def log_frame_data(data: dict):
    """
    Assigns a unique, incrementing frame_id and logs the frame's data to both the current session log and the master log.
    Accepts and writes video_path, time_offset_sec, and speed fields.
    """
    global _FRAME_ID
    _FRAME_ID += 1
    data = dict(data)  # make a copy to avoid mutating caller's dict
    data["frame_id"] = _FRAME_ID

    required = [
        "frame_id", "timestamp", "lat", "lon", "alt", "heading", "speed", "source",
        "description", "video_path", "time_offset_sec"
    ]
    for k in required:
        if k not in data:
            data[k] = None
    line = json.dumps(data)
    with open(SESSION_LOG, "a") as f:
        f.write(line + "\n")
    with open(MASTER_LOG, "a") as f:
        f.write(line + "\n")
    return _FRAME_ID
