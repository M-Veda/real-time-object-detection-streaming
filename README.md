# 🚀 Real-Time Object Detection Streaming System

A production-grade **client-server object detection pipeline** built with **YOLOv8, FastAPI, and WebSockets**. Webcam frames are streamed from the client to the server, processed through YOLOv8 inference, and returned as annotated frames — all with sub-second latency.

---

## 📸 Features

- 🎥 **Live webcam streaming** — real-time capture via OpenCV
- 🧠 **YOLOv8 inference** — powered by Ultralytics (swap `yolov8n` → `s/m/l` freely)
- ⚡ **Low-latency WebSocket pipeline** — bidirectional async communication
- 🔄 **Frame skip + JPEG compression** — efficient bandwidth usage
- 🧵 **Thread-pool inference** — parallel processing with `ThreadPoolExecutor`
- 📊 **Live HUD overlay** — FPS, latency, inference time, object count
- 🔁 **Auto-reconnect** — client recovers from dropped connections automatically
- 🌐 **Multi-client ready** — session-based architecture, each client isolated
- 🛑 **Graceful shutdown** — press `Q` to cleanly stop the client

---

## 🏗️ System Architecture

```
┌─────────────────────────────────┐         ┌──────────────────────────────────┐
│           CLIENT                │         │             SERVER               │
│                                 │         │                                  │
│  Webcam → OpenCV Capture        │         │  FastAPI + Uvicorn               │
│  ↓                              │  Frame  │  ↓                               │
│  JPEG Encode + Base64           │ ──────► │  WebSocket Receiver              │
│  ↓                              │         │  ↓                               │
│  WebSocket Send (async)         │         │  asyncio.Queue (buffer)          │
│                                 │         │  ↓                               │
│  Display Thread                 │ ◄─────  │  ThreadPoolExecutor              │
│  ↓                              │Annotated│  ↓                               │
│  HUD Overlay + cv2.imshow       │  Frame  │  YOLOv8 Inference → Annotate     │
│                                 │         │  ↓                               │
│                                 │         │  JPEG Encode + Base64 → Send     │
└─────────────────────────────────┘         └──────────────────────────────────┘
```

---

## 🔄 Workflow

1. Client opens webcam and captures frames at up to 30 FPS
2. Every Nth frame is JPEG-compressed and Base64-encoded
3. Encoded frame is sent via WebSocket to the server with a timestamp
4. Server buffers frames in a per-client async queue (drops oldest on overflow)
5. `ThreadPoolExecutor` runs YOLOv8 inference on each frame
6. Annotated frame is encoded and sent back with detection metadata
7. Client decodes and displays the result with a live HUD overlay

---

## 🗂️ Project Structure

```
real-time-object-detection/
│
├── server.py                  # FastAPI + WebSocket server with YOLOv8 inference
├── client.py                  # Async WebSocket client with capture + display
├── yolov8n.pt                 # YOLOv8 nano model weights (swap for larger models)
├── requirements_server.txt    # Server-side Python dependencies
├── requirements_client.txt    # Client-side Python dependencies
└── README.md
```

---

## 🚀 Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/real-time-object-detection-streaming.git
cd real-time-object-detection-streaming
```

### 2. Create a Virtual Environment

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

### 3. Install Dependencies

**Server:**
```bash
pip install -r requirements_server.txt
```

**Client** (can be a different machine on the same network):
```bash
pip install -r requirements_client.txt
```

---

## ▶️ Running the Application

### Step 1 — Start the Server

```bash
python server.py
```

Server starts at:
```
ws://localhost:8765/ws/stream
http://localhost:8765/health
http://localhost:8765/stats
```

### Step 2 — Start the Client (new terminal)

```bash
python client.py
```

A window opens showing the webcam feed with live bounding boxes and HUD overlay. Press **`Q`** or **`ESC`** to quit.

---

## ⚙️ Configuration

### Client (`client.py`)

| Variable | Default | Description |
|---|---|---|
| `SERVER_URI` | `ws://localhost:8765/ws/stream` | Server WebSocket address |
| `CAMERA_INDEX` | `0` | Webcam device index |
| `CAPTURE_WIDTH / HEIGHT` | `640 x 480` | Capture resolution |
| `TARGET_FPS` | `30` | Webcam capture rate |
| `SEND_EVERY_N` | `2` | Send every Nth frame (higher = less bandwidth) |
| `JPEG_QUALITY` | `75` | Compression quality before sending (0–100) |
| `RECONNECT_DELAY` | `3.0` | Seconds to wait before reconnecting |

### Server (`server.py`)

| Variable | Default | Description |
|---|---|---|
| `MODEL_NAME` | `yolov8n.pt` | YOLO model file (`n/s/m/l/x`) |
| `CONFIDENCE_THRESHOLD` | `0.45` | Minimum detection confidence |
| `IOU_THRESHOLD` | `0.45` | Non-max suppression IoU threshold |
| `MAX_QUEUE_SIZE` | `10` | Frame buffer per client |
| `INFERENCE_WORKERS` | `2` | Parallel inference threads |
| `JPEG_QUALITY` | `80` | Annotated frame encode quality |

---

## 📡 API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/health` | Server status, connected clients, model info |
| `GET` | `/stats` | Per-client stats: frames processed, avg inference time, uptime |
| `WS` | `/ws/stream` | WebSocket stream endpoint for clients |

### Example `/health` Response

```json
{
  "status": "ok",
  "clients": 1,
  "model": "yolov8n.pt",
  "workers": 2
}
```

### Example `/stats` Response

```json
{
  "a3f1bc2d": {
    "frames_received": 412,
    "frames_processed": 410,
    "frames_dropped": 2,
    "total_inference_ms": 8540.3,
    "uptime_s": 38.2,
    "avg_inference_ms": 20.83
  }
}
```

---

## 📊 HUD Overlay Reference

The client overlays a live heads-up display on the video window:

| Metric | Description |
|---|---|
| `● LIVE / ● OFFLINE` | WebSocket connection status |
| `Send FPS` | Frames sent per second to server |
| `Recv FPS` | Annotated frames received per second |
| `Latency` | Rolling average round-trip time (ms) |
| `Inference` | Server-side YOLOv8 inference time (ms) |
| `Objects` | Number of detections in current frame |
| `Sent / Rcvd` | Total frame counters |
| `Skip / Quality` | Active send-every-N and JPEG quality settings |

---

## ⚡ Performance Optimizations

| Technique | Benefit |
|---|---|
| Frame skipping (`SEND_EVERY_N`) | Reduces bandwidth without stalling |
| JPEG compression | Shrinks payload size significantly |
| `asyncio.Queue` with overflow drop | Prevents server memory buildup |
| `ThreadPoolExecutor` for inference | Keeps async loop non-blocking |
| Single-slot frame buffer on client | Always sends the latest frame, not stale ones |
| Rolling latency average | Smooth HUD metrics, no jitter |

---

## 🧪 Tech Stack

| Layer | Technology |
|---|---|
| ML Inference | YOLOv8 (Ultralytics) |
| Server Framework | FastAPI + Uvicorn |
| Client Vision | OpenCV |
| Communication | WebSockets (bidirectional) |
| Concurrency | asyncio + threading + ThreadPoolExecutor |
| Encoding | JPEG + Base64 over JSON |

---

## 🔧 Swapping the YOLO Model

Replace `yolov8n.pt` with any variant for a speed/accuracy tradeoff:

| Model | Size | Speed | Accuracy |
|---|---|---|---|
| `yolov8n.pt` | ~6 MB | ⚡ Fastest | Good |
| `yolov8s.pt` | ~22 MB | Fast | Better |
| `yolov8m.pt` | ~50 MB | Moderate | High |
| `yolov8l.pt` | ~87 MB | Slower | Very High |
| `yolov8x.pt` | ~137 MB | Slowest | Best |

Update `MODEL_NAME` in `server.py` accordingly.

---

## 🚀 Future Improvements

- 🖥️ Browser-based frontend (WebRTC or canvas-based client)
- 🔥 GPU acceleration with CUDA / TensorRT
- 🔀 Load-balanced multi-server deployment
- 📹 Video file input support (not just webcam)
- 💾 Detection logging and analytics dashboard
- ☁️ Cloud deployment (AWS / GCP / Azure)

---

## ⚠️ Troubleshooting

**Camera not opening:**
```
Cannot open camera 0
```
Try `CAMERA_INDEX = 1` or check if another app is using the webcam.

**Connection refused:**
Make sure `server.py` is running before starting the client.

**YOLO not installed fallback:**
If `ultralytics` is missing, the server runs a `DummyDetector` and overlays a warning on frames. Fix with:
```bash
pip install ultralytics
```

**Slow inference:**
Switch to a smaller model (`yolov8n.pt`) or reduce `CAPTURE_WIDTH`/`CAPTURE_HEIGHT` in the client config.

---

## 👩‍💻 Author

**Veda** — B.Tech AI & ML Student

---

## 🙏 Acknowledgements

- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)
- [FastAPI](https://fastapi.tiangolo.com/)
- [OpenCV](https://opencv.org/)
- [websockets](https://websockets.readthedocs.io/)

---

## 📄 License

This project is intended for educational and research purposes.
