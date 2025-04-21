import time
import cv2
from indexer import log_frame_data, get_last_frame_id
import os
from telemetry_ingest import start_telemetry_thread, get_latest_telemetry

VIDEO_URL = "http://192.168.0.107:8081/video"  # Replace with your phone's IP

def video_frame_generator(video_url):
    cap = cv2.VideoCapture(video_url)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video stream: {video_url}")
    while True:
        ret, frame = cap.read()
        ts = time.time()
        if not ret:
            print("[WARN] Frame grab failed, retrying...")
            time.sleep(0.1)
            continue
        yield frame, ts

def main(shared_state=None):
    print("Select captioning pipeline:")
    print("1. YOLOv8-based (fast, bounding boxes, rule-based sentence)")
    print("2. CLIP+LLM-based (semantic, no boxes, VLM-style)")
    choice = input("Enter 1 or 2: ").strip()
    if choice == "1":
        from yolo_caption import YOLOCaptioner
        captioner = YOLOCaptioner()
        source = "YOLO+GPT2"
        print("[INFO] Using YOLOv8+GPT2 pipeline.")
    elif choice == "2":
        from clip_caption import CLIPCaptioner
        captioner = CLIPCaptioner()
        source = "CLIP+LLM"
        print("[INFO] Using CLIP+LLM pipeline.")
    else:
        print("Invalid choice.")
        return

    # --- VideoWriter setup ---
    session_dir = "video_sessions"
    if not os.path.exists(session_dir):
        os.makedirs(session_dir)
    session_num = len(os.listdir(session_dir)) + 1
    video_path = os.path.join(session_dir, f"session_{session_num:03d}.mp4")
    cap = cv2.VideoCapture(VIDEO_URL)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video stream: {VIDEO_URL}")
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(video_path, fourcc, fps, (width, height))
    session_start_time = time.time()

    if shared_state is not None:
        shared_state['last_video_ts'] = time.time()

    start_telemetry_thread()  # Start telemetry polling in the background

    print("[INFO] Starting video stream and captioning...")
    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            print("Camera connection lost. Drone returning to dock for inspection.")
            break
        frame_ts = time.time()
        try:
            telemetry = get_latest_telemetry()
        except Exception:
            print("Telemetry malfunction. Drone returning to dock for inspection.")
            break
        if (
            telemetry is None or
            any(telemetry.get(k) is None for k in ("lat", "lon", "alt"))
        ):
            print("Telemetry data is unavailable. Drone returning to dock for inspection.")
            break
        if not ret:
            print("[WARN] Frame grab failed, retrying...")
            time.sleep(0.1)
            continue
        if shared_state is not None:
            shared_state['last_video_ts'] = frame_ts
            if shared_state.get('shutdown', None) and shared_state['shutdown'].is_set():
                print("[INFO] Video ingest shutting down due to system monitor.")
                break
        description, annotated = captioner.caption(frame)
        overlay = annotated.copy()
        cv2.putText(
            overlay, description, (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA
        )
        out.write(frame)
        time_offset = frame_ts - session_start_time

        # --- Get real telemetry values ---
        telemetry = get_latest_telemetry() or {
            "lat": None, "lon": None, "alt": None,
            "accel_x": None, "accel_y": None, "accel_z": None
        }

        # --- Calculate speed from acceleration (magnitude of acceleration vector) ---
        if all(telemetry.get(k) is not None for k in ("accel_x", "accel_y", "accel_z")):
            speed = (telemetry["accel_x"]**2 + telemetry["accel_y"]**2 + telemetry["accel_z"]**2) ** 0.5
        else:
            speed = None

        log_frame_data({
            # "frame_id": ...assigned by indexer...
            "timestamp": frame_ts,
            "lat": telemetry["lat"],
            "lon": telemetry["lon"],
            "alt": telemetry["alt"],
            "heading": 0.0,  # If you have heading in telemetry, use telemetry["heading"]
            "speed": speed,
            "source": source,
            "description": description,
            "video_path": video_path,
            "time_offset_sec": time_offset
        })
        cv2.imshow("Captioned Video", overlay)
        print({
            # "frame_id": ...assigned by indexer...
            "timestamp": frame_ts,
            "lat": telemetry["lat"],
            "lon": telemetry["lon"],
            "alt": telemetry["alt"],
            "heading": 0.0,
            "speed": speed,
            "description": description,
            "video_path": video_path,
            "time_offset_sec": time_offset
        })
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    out.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
