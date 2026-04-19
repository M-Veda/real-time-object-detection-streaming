# 🚀 Real-Time Object Detection Streaming System

A real-time client-server object detection system built using **YOLOv8, FastAPI, and WebSockets**, where video frames are streamed from a client to a server, processed for object detection, and returned as annotated frames with low latency.

---

## 📌 Features

* 🎥 Real-time video streaming from webcam
* 🧠 YOLOv8 object detection (Ultralytics)
* ⚡ Low-latency WebSocket communication
* 🔄 Bidirectional client-server pipeline
* 📉 Frame compression & skipping for efficiency
* 🧵 Asynchronous processing with thread pooling
* 📊 Live HUD display (FPS, latency, detections)
* 🌐 Scalable architecture (multi-client support ready)

---

## 🏗️ System Architecture

```
Client (Webcam) ──► WebSocket ──► Server (YOLOv8 Inference)
      ▲                                         │
      └──────── Annotated Frames ◄───────────────┘
```

---

## 🔄 Workflow

1. Client captures frames using OpenCV
2. Frames are compressed (JPEG) and encoded (Base64)
3. Frames are sent via WebSocket to the server
4. Server queues frames and performs YOLOv8 inference
5. Annotated frames are sent back to the client
6. Client displays results with real-time metrics

---

## ⚙️ Installation & Setup

### 1️⃣ Clone Repository

```
git clone https://github.com/your-username/real-time-object-detection-streaming.git
cd real-time-object-detection-streaming
```

---

### 2️⃣ Create Virtual Environment

```
python -m venv venv
```

Activate:

```
# Windows
venv\Scripts\activate

# Mac/Linux
source venv/bin/activate
```

---

### 3️⃣ Install Dependencies

#### Server:

```
pip install -r requirements_server.txt
```

#### Client:

```
pip install -r requirements_client.txt
```

---

## ▶️ Running the Application

### Step 1: Start Server

```
python server.py
```

Server runs at:

```
ws://localhost:8765/ws/stream
```

---

### Step 2: Start Client (New Terminal)

```
python client.py
```

---

## 🎯 Output

* Webcam stream opens
* Detected objects are highlighted with bounding boxes
* HUD displays:

  * FPS
  * Latency
  * Number of objects detected

---

## ⚡ Performance Optimizations

* Frame skipping to reduce load
* JPEG compression for bandwidth efficiency
* Async WebSocket communication
* ThreadPoolExecutor for parallel inference
* Queue-based buffering to avoid bottlenecks

---

## 🧪 Tech Stack

* **Client:** OpenCV, WebSockets
* **Server:** FastAPI, Uvicorn
* **ML Model:** YOLOv8 (Ultralytics)
* **Concurrency:** asyncio, threading
* **Communication:** WebSockets

---

## 📊 API Endpoints

| Endpoint  | Description                |
| --------- | -------------------------- |
| `/health` | Server health status       |
| `/stats`  | Client performance metrics |

---

## 🚀 Future Improvements

* Multi-client scaling with load balancing
* GPU acceleration (CUDA / TensorRT)
* Web-based frontend (browser client)
* Video recording & analytics
* Cloud deployment (AWS/GCP/Azure)

---

## 👩‍💻 Author

**Veda**
B.Tech AI & ML Student

---

## ⭐ Acknowledgements

* Ultralytics YOLOv8
* OpenCV
* FastAPI

---

## 📌 License

This project is for educational and research purposes.
