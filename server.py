"""
Real-Time Object Detection Streaming Server
Uses FastAPI + WebSockets + YOLOv8 for GPU-accelerated inference
"""

import asyncio
import base64
import json
import logging
import time
import uuid
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Dict, Optional

import cv2
import numpy as np
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

# ──────────────────────────────────────────────
# Logging
# ──────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("yolo-server")

# ──────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────
MAX_QUEUE_SIZE       = 10        # frames buffered per client
INFERENCE_WORKERS    = 2         # parallel inference threads
JPEG_QUALITY         = 80        # annotated-frame encode quality
CONFIDENCE_THRESHOLD = 0.45
IOU_THRESHOLD        = 0.45
MODEL_NAME           = "yolov8n.pt"   # swap for yolov8s/m/l as needed


# ──────────────────────────────────────────────
# YOLO Model (lazy-loaded singleton)
# ──────────────────────────────────────────────
class ModelManager:
    _model = None

    @classmethod
    def get_model(cls):
        if cls._model is None:
            try:
                from ultralytics import YOLO
                cls._model = YOLO(MODEL_NAME)
                logger.info("YOLOv8 model loaded (%s)", MODEL_NAME)
            except ImportError:
                logger.warning("ultralytics not installed – using dummy detector")
                cls._model = DummyDetector()
        return cls._model


class DummyDetector:
    """Fallback when ultralytics is absent (useful for testing the pipeline)."""

    def predict(self, frame, conf=0.45, iou=0.45, verbose=False):
        return [DummyResult(frame)]


@dataclass
class DummyResult:
    orig_img: np.ndarray

    def plot(self):
        img = self.orig_img.copy()
        cv2.putText(img, "YOLO not installed", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        return img


# ──────────────────────────────────────────────
# Per-client state
# ──────────────────────────────────────────────
@dataclass
class ClientSession:
    client_id: str
    websocket: WebSocket
    frame_queue: asyncio.Queue = field(default_factory=lambda: asyncio.Queue(maxsize=MAX_QUEUE_SIZE))
    stats: Dict = field(default_factory=lambda: {
        "frames_received": 0,
        "frames_processed": 0,
        "frames_dropped": 0,
        "total_inference_ms": 0.0,
        "connected_at": time.time(),
    })
    active: bool = True


# ──────────────────────────────────────────────
# Connection Manager
# ──────────────────────────────────────────────
class ConnectionManager:
    def __init__(self):
        self.sessions: Dict[str, ClientSession] = {}

    def register(self, ws: WebSocket) -> ClientSession:
        cid = str(uuid.uuid4())[:8]
        session = ClientSession(client_id=cid, websocket=ws)
        self.sessions[cid] = session
        logger.info("Client %s connected  (total=%d)", cid, len(self.sessions))
        return session

    def remove(self, client_id: str):
        self.sessions.pop(client_id, None)
        logger.info("Client %s disconnected (total=%d)", client_id, len(self.sessions))

    @property
    def count(self) -> int:
        return len(self.sessions)


manager = ConnectionManager()
executor = ThreadPoolExecutor(max_workers=INFERENCE_WORKERS, thread_name_prefix="yolo")


# ──────────────────────────────────────────────
# Inference (runs in thread-pool)
# ──────────────────────────────────────────────
def run_inference(frame_bytes: bytes) -> tuple[bytes, dict]:
    """
    Decode → infer → annotate → re-encode.
    Returns (jpeg_bytes, metadata_dict).
    """
    t0 = time.perf_counter()

    # Decode JPEG
    arr  = np.frombuffer(frame_bytes, dtype=np.uint8)
    img  = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Could not decode frame")

    # YOLOv8 inference
    model   = ModelManager.get_model()
    results = model.predict(img, conf=CONFIDENCE_THRESHOLD,
                            iou=IOU_THRESHOLD, verbose=False)
    annotated = results[0].plot()          # BGR with boxes + labels

    # Collect detection summary
    detections = []
    try:
        boxes = results[0].boxes
        for box in boxes:
            detections.append({
                "class": int(box.cls[0]),
                "label": results[0].names[int(box.cls[0])],
                "conf":  round(float(box.conf[0]), 3),
                "xyxy":  [round(v, 1) for v in box.xyxy[0].tolist()],
            })
    except Exception:
        pass

    # Encode annotated frame
    ok, buf = cv2.imencode(".jpg", annotated,
                           [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
    if not ok:
        raise RuntimeError("Failed to encode annotated frame")

    inference_ms = (time.perf_counter() - t0) * 1000
    meta = {
        "inference_ms": round(inference_ms, 2),
        "detections":   detections,
        "num_objects":  len(detections),
    }
    return buf.tobytes(), meta


# ──────────────────────────────────────────────
# FastAPI App
# ──────────────────────────────────────────────
app = FastAPI(title="YOLO Stream Server", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health():
    return {
        "status":   "ok",
        "clients":  manager.count,
        "model":    MODEL_NAME,
        "workers":  INFERENCE_WORKERS,
    }


@app.get("/stats")
async def stats():
    out = {}
    for cid, s in manager.sessions.items():
        elapsed = time.time() - s.stats["connected_at"]
        proc    = s.stats["frames_processed"] or 1
        out[cid] = {
            **s.stats,
            "uptime_s":       round(elapsed, 1),
            "avg_inference_ms": round(s.stats["total_inference_ms"] / proc, 2),
        }
    return out


@app.websocket("/ws/stream")
async def stream_endpoint(ws: WebSocket):
    await ws.accept()
    session = manager.register(ws)
    loop    = asyncio.get_event_loop()

    # ── Consumer: pull frames from queue, run inference, send result ──
    async def consumer():
        while session.active:
            try:
                item = await asyncio.wait_for(session.frame_queue.get(), timeout=5.0)
            except asyncio.TimeoutError:
                continue

            frame_bytes, client_ts = item
            t_recv = time.perf_counter()

            try:
                annotated_bytes, meta = await loop.run_in_executor(
                    executor, run_inference, frame_bytes
                )
            except Exception as exc:
                logger.warning("Inference error for %s: %s", session.client_id, exc)
                session.frame_queue.task_done()
                continue

            server_ts  = time.perf_counter()
            round_trip = round((server_ts - t_recv) * 1000, 2)

            b64 = base64.b64encode(annotated_bytes).decode()
            payload = json.dumps({
                "type":          "annotated_frame",
                "client_id":     session.client_id,
                "image":         b64,
                "inference_ms":  meta["inference_ms"],
                "round_trip_ms": round_trip,
                "num_objects":   meta["num_objects"],
                "detections":    meta["detections"],
                "client_ts":     client_ts,
                "server_ts":     server_ts,
            })

            try:
                await ws.send_text(payload)
            except Exception:
                session.active = False
                break

            session.stats["frames_processed"]    += 1
            session.stats["total_inference_ms"]  += meta["inference_ms"]
            session.frame_queue.task_done()

    consumer_task = asyncio.create_task(consumer())

    # ── Producer: receive frames from client ──
    try:
        while True:
            raw = await ws.receive_text()
            msg = json.loads(raw)

            if msg.get("type") != "frame":
                continue

            session.stats["frames_received"] += 1
            frame_bytes = base64.b64decode(msg["image"])
            client_ts   = msg.get("timestamp", time.time())

            if session.frame_queue.full():
                # Drop oldest frame (non-blocking)
                try:
                    session.frame_queue.get_nowait()
                    session.stats["frames_dropped"] += 1
                except asyncio.QueueEmpty:
                    pass

            await session.frame_queue.put((frame_bytes, client_ts))

    except WebSocketDisconnect:
        logger.info("Client %s disconnected cleanly", session.client_id)
    except Exception as exc:
        logger.error("Client %s error: %s", session.client_id, exc)
    finally:
        session.active = False
        consumer_task.cancel()
        manager.remove(session.client_id)


# ──────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────
if __name__ == "__main__":
    uvicorn.run(
        "server:app",
        host="0.0.0.0",
        port=8765,
        log_level="info",
        ws_ping_interval=20,
        ws_ping_timeout=30,
    )