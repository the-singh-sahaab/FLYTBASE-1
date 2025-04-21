import time
import requests
from collections import deque
import threading

# --- CONFIGURATION ---
PHYPOX_URL = "http://192.168.0.107:8080/get?accX&accY&accZ&locLat&locLon&locZ"
TELEMETRY_POLL_HZ = 5  # 5 Hz = every 200ms
TELEMETRY_BUFFER_SIZE = 100

_telemetry_buffer = deque(maxlen=TELEMETRY_BUFFER_SIZE)
_latest_telemetry = {}
_lock = threading.Lock()

def poll_phyphox_telemetry(url):
    """
    Polls the Phyphox live data endpoint and parses telemetry.
    Returns a dict with timestamp, GPS, accel, alt.
    """
    try:
        resp = requests.get(url, timeout=2)
        resp.raise_for_status()
        data = resp.json()
        buffer = data.get("buffer", {})  # FIX: Access 'buffer' dict properly

        acc_x = buffer.get("accX", {}).get("buffer", [None])[-1]
        acc_y = buffer.get("accY", {}).get("buffer", [None])[-1]
        acc_z = buffer.get("accZ", {}).get("buffer", [None])[-1]
        lat = buffer.get("locLat", {}).get("buffer", [None])[-1]
        lon = buffer.get("locLon", {}).get("buffer", [None])[-1]
        alt = buffer.get("locZ", {}).get("buffer", [None])[-1]

        return {
            "timestamp": time.time(),
            "lat": float(lat) if lat is not None else None,
            "lon": float(lon) if lon is not None else None,
            "alt": float(alt) if alt is not None else None,
            "accel_x": float(acc_x) if acc_x is not None else None,
            "accel_y": float(acc_y) if acc_y is not None else None,
            "accel_z": float(acc_z) if acc_z is not None else None
        }
    except Exception as e:
        print(f"[WARN] Telemetry poll failed: {e}")
        return None


def telemetry_ingest_loop(url, poll_hz=5):
    """
    Polls telemetry and fills the buffer and latest reading.
    Runs in a background thread.
    """
    global _latest_telemetry
    interval = 1.0 / poll_hz
    while True:
        try:
            telem = poll_phyphox_telemetry(url)
        except requests.exceptions.ConnectionError:
            print("Telemetry malfunction. Drone returning to dock for inspection.")
            break
        if not telem or any(telem.get(k) is None for k in ("lat", "lon", "alt")):
            print("Telemetry data is unavailable. Drone returning to dock for inspection.")
            break
        with _lock:
            _telemetry_buffer.append(telem)
            _latest_telemetry = telem
        time.sleep(interval)

def get_latest_telemetry():
    """
    Returns the most recent telemetry reading as a dict.
    """
    with _lock:
        return dict(_latest_telemetry) if _latest_telemetry else None

def start_telemetry_thread(shared_state=None):
    """
    Starts the background telemetry polling thread (only once).
    If shared_state is provided, updates shared_state['last_telemetry_ts'] on each successful poll.
    """
    def loop():
        while True:
            telem = poll_phyphox_telemetry(PHYPOX_URL)
            if telem:
                with _lock:
                    _telemetry_buffer.append(telem)
                    global _latest_telemetry
                    _latest_telemetry = telem
                if shared_state is not None:
                    shared_state['last_telemetry_ts'] = time.time()
            time.sleep(1.0 / TELEMETRY_POLL_HZ)
    t = threading.Thread(target=loop, daemon=True)
    t.start()
    return t

def main(shared_state=None):
    print("[INFO] Starting real-time telemetry ingest from Phyphox...")
    start_telemetry_thread(shared_state=shared_state)
    while True:
        telem = get_latest_telemetry()
        print(telem)
        if shared_state and shared_state.get('shutdown', None) and shared_state['shutdown'].is_set():
            print("[INFO] Telemetry ingest shutting down due to system monitor.")
            break
        time.sleep(1)

if __name__ == "__main__":
    main()
