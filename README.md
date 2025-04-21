# Drone Security Analyst Agent – Full System Guide

This project implements a modular, real-time drone (or phone-based) video analytics pipeline with:
- Live video and telemetry ingest from a mobile device
- Real-time object detection and scene captioning (YOLOv8 or CLIP+GPT-2)
- Frame-by-frame and session-level indexing for LangChain QA
- Fault detection and emergency stop logic

---

## 1. Prerequisites

- **Python 3.8–3.10** (recommended)
- **pip** (Python package manager)
- **A phone** running:
  - [IP Webcam](https://play.google.com/store/apps/details?id=com.pas.webcam) (video stream)
  - [Phyphox](https://play.google.com/store/apps/details?id=de.rwth_aachen.phyphox) (telemetry stream)
- **A computer** on the same WiFi network as the phone

---

## 2. Installation

### 2.1. Clone the repository

```sh
git clone <your-repo-url>
cd p31-flytbase/FLYTBASE-1
```

### 2.2. Install Python dependencies

```sh
pip install -r requirements.txt
```

---

## 3. Phone Setup

### 3.1. IP Webcam (Video)
- Start the app, set the port (e.g., 8081), and note the stream URL (e.g., `http://<phone_ip>:8081/video`).

### 3.2. Phyphox (Telemetry)
- Start the app, select an experiment with GPS and acceleration.
- Share via WiFi and note the endpoint (e.g., `http://<phone_ip>:8080/get?accX&accY&accZ&locLat&locLon&locZ`).

---

## 4. Configuration

- Edit `VIDEO_URL` in `video_ingest.py` to match your phone's IP Webcam stream.
- Edit `PHYPOX_URL` in `telemetry_ingest.py` to match your Phyphox endpoint.

---

## 5. Running the System

### 5.1. Start Telemetry Ingest (optional for testing)

```sh
python telemetry_ingest.py
```
- You should see live telemetry values printed.

### 5.2. Start the Main Video + Analytics Pipeline

```sh
python video_ingest.py
```
- Select the captioning pipeline:
  - `1` for YOLOv8 + GPT-2 (fast, bounding boxes, LLM captions)
  - `2` for CLIP + GPT-2 (semantic, VLM-style, LLM captions)
- The system will:
  - Show live annotated video with captions
  - Log every frame with real telemetry, video path, and time offset
  - Record the session video in `video_sessions/`
  - Index all events in `recent_indexes/` and `index_master.jsonl`
- If the camera or telemetry feed is lost, the system will print a message and exit gracefully.

---

## 6. Querying Session Logs with LangChain Agent

```sh
python langchain_agent.py
```
- Enter the path to a session log (e.g., `recent_indexes/index_run_007.jsonl`)
- Enter queries like:
  - `Show all bottle-related events.`
  - `What happened about car?`
- The agent will output a human-readable, time-stamped summary of matching events.

---

## 7. File Structure

- `video_ingest.py` – Main controller for video, telemetry, detection, and logging
- `telemetry_ingest.py` – Background telemetry polling from Phyphox
- `yolo_caption.py` – YOLOv8 detection and GPT-2-based captioning
- `clip_caption.py` – CLIP detection and GPT-2-based captioning
- `description_generator_yolo.py` / `description_generator_clip.py` – GPT-2 LLM wrappers
- `indexer.py` – Modular indexing and session/master log management
- `langchain_agent.py` – Query and summarize session logs
- `alert_monitor.py` – Real-time alerting for specific object detection
- `requirements.txt` – All dependencies
- `video_sessions/` – Saved session videos
- `recent_indexes/` – Last 3 session logs
- `index_master.jsonl` – Master log of all frames/events

---

## 8. Troubleshooting

- **Camera connection lost:** Check phone IP, WiFi, and IP Webcam app.
- **Telemetry malfunction:** Check Phyphox app, endpoint URL, and WiFi.
- **Dependencies:** Run `pip install -r requirements.txt` again if you see import errors.

---

## 9. Notes

- All logs are JSONL and ready for LangChain or other analytics.
- Frame IDs are globally unique and continuous across sessions.
- The system is modular: you can swap detection/captioning modules as needed.

---

## 10. Example Workflow

```sh
# 1. Start IP Webcam and Phyphox on your phone
# 2. Run the main pipeline:
python video_ingest.py
# 3. (Optional) Query session logs:
python langchain_agent.py
```

---

**Enjoy your modular, real-time drone analytics system!**